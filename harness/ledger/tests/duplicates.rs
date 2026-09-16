//! Group: append-only uniqueness of experiment_id.

mod common;

use prometheus_ledger::{Error, Ledger};

#[test]
fn second_append_same_experiment_id_without_supersedes_is_duplicate() {
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");
    ledger
        .append(common::sample_record("exp-dup"))
        .expect("first append");

    match ledger.append(common::sample_record("exp-dup")) {
        Err(Error::Duplicate(id)) => assert_eq!(id, "exp-dup"),
        other => panic!("expected Error::Duplicate(\"exp-dup\"), got {other:?}"),
    }

    assert_eq!(ledger.len().expect("len"), 1, "duplicate must not insert");
}

#[test]
fn same_experiment_id_with_supersedes_set_is_still_duplicate() {
    // Spec: when supersedes is set, experiment_id must still be *new*.
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");
    ledger
        .append(common::sample_record("exp-orig"))
        .expect("first append");

    let mut correction = common::sample_record("exp-orig");
    correction.supersedes = Some("exp-orig".to_string());
    match ledger.append(correction) {
        Err(Error::Duplicate(id)) => assert_eq!(id, "exp-orig"),
        other => {
            panic!("expected Error::Duplicate because experiment_id is not new, got {other:?}")
        }
    }
}
