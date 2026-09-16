//! Group: negative results are first-class.
//!
//! A row is negative iff any of:
//!   * `results["passed"] == false`
//!   * `results["negative"] == true`
//!   * `replication.status == Failed`
//! Incomplete rows are not negatives.

mod common;

use prometheus_ledger::{Ledger, Replication, ReplicationStatus};
use serde_json::json;

fn ids(rows: &[prometheus_ledger::Record]) -> Vec<&str> {
    rows.iter().map(|r| r.experiment_id.as_str()).collect()
}

#[test]
fn list_negatives_returns_only_negative_rows() {
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");

    let mut positive = common::sample_record("exp-positive");
    positive.results = Some(json!({"passed": true, "loss": 1.0}));
    positive.replication = Some(Replication {
        status: ReplicationStatus::Matched,
        n: 2,
        notes: None,
    });
    positive.delta_rci = Some(0.02);

    let mut neg_passed = common::sample_record("exp-neg-passed");
    neg_passed.results = Some(json!({"passed": false, "loss": 9.9}));

    let mut neg_flag = common::sample_record("exp-neg-flag");
    neg_flag.results = Some(json!({"negative": true, "reason": "null result"}));

    let mut neg_repl = common::sample_record("exp-neg-repl");
    neg_repl.results = Some(json!({"passed": true}));
    neg_repl.replication = Some(Replication {
        status: ReplicationStatus::Failed,
        n: 3,
        notes: Some("did not match".into()),
    });

    let mut incomplete = common::sample_record("exp-incomplete");
    incomplete.replication = Some(Replication {
        status: ReplicationStatus::Unreplicated,
        n: 0,
        notes: None,
    });

    ledger.append(positive).unwrap();
    ledger.append(neg_passed).unwrap();
    ledger.append(neg_flag).unwrap();
    ledger.append(neg_repl).unwrap();
    ledger.append(incomplete).unwrap();

    let negs = ledger.list_negatives().expect("list_negatives");
    let got = ids(&negs);
    assert!(
        got.contains(&"exp-neg-passed"),
        "results.passed == false is negative, got {got:?}"
    );
    assert!(
        got.contains(&"exp-neg-flag"),
        "results.negative == true is negative, got {got:?}"
    );
    assert!(
        got.contains(&"exp-neg-repl"),
        "replication.status == failed is negative, got {got:?}"
    );
    assert!(
        !got.contains(&"exp-positive"),
        "positives must not appear, got {got:?}"
    );
    assert!(
        !got.contains(&"exp-incomplete"),
        "incomplete (no failed/null outcome) must not appear, got {got:?}"
    );
    assert_eq!(negs.len(), 3, "exactly the three negatives, got {got:?}");
}

#[test]
fn list_negatives_on_empty_store_is_empty() {
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");
    let negs = ledger.list_negatives().expect("list_negatives");
    assert!(negs.is_empty());
}
