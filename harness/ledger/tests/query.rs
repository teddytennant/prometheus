//! Group: parameterized SQL query. Params must be bound, never interpolated.

mod common;

use prometheus_ledger::Ledger;
use serde_json::json;

#[test]
fn query_by_author_role_returns_matching_subset() {
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");

    let mut a = common::sample_record("exp-a");
    a.author_role = "researcher".to_string();
    a.rung = Some(1);
    let mut b = common::sample_record("exp-b");
    b.author_role = "program_lead".to_string();
    b.rung = Some(1);
    let mut c = common::sample_record("exp-c");
    c.author_role = "researcher".to_string();
    c.rung = Some(2);

    ledger.append(a).unwrap();
    ledger.append(b).unwrap();
    ledger.append(c).unwrap();

    let hits = ledger
        .query("author_role = ?", &[json!("researcher")])
        .expect("query author_role");
    let ids: Vec<_> = hits.iter().map(|r| r.experiment_id.as_str()).collect();
    assert_eq!(hits.len(), 2, "expected two researchers, got {ids:?}");
    assert!(ids.contains(&"exp-a"));
    assert!(ids.contains(&"exp-c"));
    assert!(!ids.contains(&"exp-b"));
}

#[test]
fn query_by_rung_returns_matching_subset() {
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");

    let mut a = common::sample_record("exp-r1");
    a.rung = Some(1);
    let mut b = common::sample_record("exp-r2");
    b.rung = Some(2);
    let mut c = common::sample_record("exp-r1b");
    c.rung = Some(1);
    let d = common::sample_record("exp-none"); // rung = None

    ledger.append(a).unwrap();
    ledger.append(b).unwrap();
    ledger.append(c).unwrap();
    ledger.append(d).unwrap();

    let hits = ledger.query("rung = ?", &[json!(1)]).expect("query rung");
    let ids: Vec<_> = hits.iter().map(|r| r.experiment_id.as_str()).collect();
    assert_eq!(hits.len(), 2, "expected two rung=1 rows, got {ids:?}");
    assert!(ids.contains(&"exp-r1"));
    assert!(ids.contains(&"exp-r1b"));
}

#[test]
fn query_by_author_role_and_rung() {
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");

    let mut a = common::sample_record("exp-and-hit");
    a.author_role = "researcher".to_string();
    a.rung = Some(3);
    let mut b = common::sample_record("exp-and-role-only");
    b.author_role = "researcher".to_string();
    b.rung = Some(1);
    let mut c = common::sample_record("exp-and-rung-only");
    c.author_role = "program_lead".to_string();
    c.rung = Some(3);

    ledger.append(a).unwrap();
    ledger.append(b).unwrap();
    ledger.append(c).unwrap();

    let hits = ledger
        .query(
            "author_role = ? AND rung = ?",
            &[json!("researcher"), json!(3)],
        )
        .expect("query AND");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].experiment_id, "exp-and-hit");
}

#[test]
fn query_no_match_returns_empty_vec() {
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");
    ledger.append(common::sample_record("exp-only")).unwrap();
    let hits = ledger
        .query("author_role = ?", &[json!("does-not-exist")])
        .expect("query");
    assert!(hits.is_empty());
}

#[test]
fn query_binds_params_and_does_not_interpolate_sql() {
    // Fault injection: a param that looks like SQL must be a *value*, not syntax.
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");

    let mut alice = common::sample_record("exp-alice");
    alice.author_role = "alice".to_string();
    let mut bob = common::sample_record("exp-bob");
    bob.author_role = "bob".to_string();
    ledger.append(alice).unwrap();
    ledger.append(bob).unwrap();

    let malicious = json!("alice' OR '1'='1");
    let hits = ledger
        .query("author_role = ?", &[malicious])
        .expect("bound query must succeed");
    assert!(
        hits.is_empty(),
        "interpolating the param into SQL would match every row; bound params match none. got {:?}",
        hits.iter()
            .map(|r| r.experiment_id.as_str())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        ledger.len().expect("len"),
        2,
        "query must not mutate the store"
    );
}

#[test]
fn query_sql_metacharacters_in_params_do_not_drop_rows() {
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");
    ledger.append(common::sample_record("exp-keep")).unwrap();

    let _ = ledger.query("author_role = ?", &[json!("1; DROP TABLE records; --")]);
    assert_eq!(
        ledger.len().expect("len after injection attempt"),
        1,
        "bound params must not execute extra SQL"
    );
    ledger
        .get_by_experiment("exp-keep")
        .expect("row still present after injection attempt");
}
