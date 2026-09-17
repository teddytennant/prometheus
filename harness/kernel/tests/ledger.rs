//! Group: ledger_append / ledger_query.
//!
//! Query language (implementer must match; also documented in `common/mod.rs`):
//!
//! ```text
//! <empty> | * | all
//! predicate ( AND predicate)*
//! predicate:
//!   experiment_id=<exact>
//!   author_role=<exact>
//!   rung=<u32> | rung=none
//!   hypothesis~<substr>
//!   kind=check | kind=prereg
//! values may be "double quoted"
//! ```
//!
//! Not SQL. Not embedding search. Unknown / SQL-looking → Error::Message.

mod common;
mod reference;

use common::{assert_records_eq, fresh_world, open_kernel, rec, unwrap_err};
use prometheus_kernel::Error;
use reference::RefKernel;

fn seed(k: &mut prometheus_kernel::Kernel, r: &mut RefKernel) {
    let rows = [
        rec("e1", "CHECK:run-a prior art"),
        rec("e2", "PREREG:run-a will beat baseline"),
        rec("e3", "unrelated hypothesis"),
    ];
    let mut rows = rows.map(|mut row| {
        row.author_role = "researcher".into();
        row
    });
    rows[2].author_role = "engineer".into();
    rows[2].rung = Some(1);
    for row in rows {
        k.ledger_append(row.clone()).expect("k append");
        r.ledger_append(row).expect("r append");
    }
}

#[test]
fn append_then_query_all() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let mut r = RefKernel::open(&world.cfg);
    seed(&mut k, &mut r);
    for q in ["", "*", "all", "  *  "] {
        let got = k
            .ledger_query(q)
            .unwrap_or_else(|e| panic!("query {q:?}: {e}"));
        let exp = r.ledger_query(q).expect("ref");
        assert_eq!(got.len(), 3, "query {q:?}");
        assert_eq!(got.len(), exp.len());
        for (a, b) in got.iter().zip(exp.iter()) {
            assert_records_eq(a, b);
        }
    }
}

#[test]
fn query_experiment_id() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let mut r = RefKernel::open(&world.cfg);
    seed(&mut k, &mut r);
    let got = k.ledger_query("experiment_id=e2").expect("q");
    let exp = r.ledger_query("experiment_id=e2").expect("rq");
    assert_eq!(got.len(), 1);
    assert_eq!(exp.len(), 1);
    assert_eq!(got[0].hypothesis, "PREREG:run-a will beat baseline");
}

#[test]
fn query_and_kind_and_role() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let mut r = RefKernel::open(&world.cfg);
    seed(&mut k, &mut r);
    let got = k
        .ledger_query("kind=check AND author_role=researcher")
        .expect("q");
    let exp = r
        .ledger_query("kind=check AND author_role=researcher")
        .expect("rq");
    assert_eq!(got.len(), 1);
    assert_eq!(exp.len(), 1);
    assert!(got[0].hypothesis.starts_with("CHECK:"));
}

#[test]
fn query_kind_prereg() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let got = {
        let mut r = RefKernel::open(&world.cfg);
        seed(&mut k, &mut r);
        k.ledger_query("kind=prereg").expect("q")
    };
    assert_eq!(got.len(), 1);
    assert!(got[0].hypothesis.starts_with("PREREG:"));
}

#[test]
fn query_rung_none_and_some() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let mut r = RefKernel::open(&world.cfg);
    seed(&mut k, &mut r);
    let none = k.ledger_query("rung=none").expect("none");
    let some = k.ledger_query("rung=1").expect("1");
    assert_eq!(none.len(), 2);
    assert_eq!(some.len(), 1);
    assert_eq!(some[0].experiment_id, "e3");
    assert_eq!(r.ledger_query("rung=1").unwrap().len(), 1);
}

#[test]
fn query_hypothesis_contains_quoted() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let mut r = RefKernel::open(&world.cfg);
    seed(&mut k, &mut r);
    let got = k.ledger_query("hypothesis~\"prior art\"").expect("q");
    let exp = r.ledger_query("hypothesis~\"prior art\"").expect("rq");
    assert_eq!(got.len(), 1);
    assert_eq!(exp.len(), 1);
    assert_eq!(got[0].experiment_id, "e1");
}

#[test]
fn query_sql_is_rejected() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    k.ledger_append(rec("e1", "h")).expect("append");
    for q in [
        "SELECT * FROM records",
        "DROP TABLE records",
        "author_role=x AND DROP TABLE records",
    ] {
        let err = unwrap_err(k.ledger_query(q), q);
        match err {
            Error::Message(s) => assert!(!s.is_empty(), "message for {q}"),
            other => panic!("expected Message for {q}, got {other:?}"),
        }
    }
}

#[test]
fn query_unknown_predicate_is_message() {
    let world = fresh_world();
    let k = open_kernel(&world);
    let err = unwrap_err(k.ledger_query("nope=1"), "nope");
    match err {
        Error::Message(s) => assert!(!s.is_empty()),
        other => panic!("expected Message, got {other:?}"),
    }
}

#[test]
fn duplicate_experiment_id_is_message() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    k.ledger_append(rec("dup", "h1")).expect("first");
    let err = unwrap_err(k.ledger_append(rec("dup", "h2")), "dup");
    match err {
        Error::Message(s) => assert!(!s.is_empty()),
        other => panic!("expected Message, got {other:?}"),
    }
}

#[test]
fn append_order_is_preserved() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    for i in 0..5 {
        k.ledger_append(rec(&format!("n{i}"), &format!("h{i}")))
            .expect("append");
    }
    let got = k.ledger_query("*").expect("all");
    let ids: Vec<_> = got.iter().map(|r| r.experiment_id.as_str()).collect();
    assert_eq!(ids, ["n0", "n1", "n2", "n3", "n4"]);
}
