//! Group: Registry lookup, schema mismatch, unknown id, dispatch into checkers.

mod common;
mod reference;

use common::{answer_ok, reward_request, CpuRegistry, SCORED_AT};
use prometheus_verifiers::{
    Answer, AnyVerifier, Error, Grid, GridMatch, GridTask, LeanCheck, LeanTask, MarketResolution,
    MarketTask, MathTask, SymbolicMath, TinyKernel,
};

#[test]
fn unknown_verifier_id_is_unknown_verifier() {
    let mut reg = CpuRegistry::new();
    let req = reward_request("no-such-verifier");
    let err = reg
        .verify(&req, &Answer::Math { latex: "1".into() }, SCORED_AT, 0)
        .unwrap_err();
    assert_eq!(err, Error::UnknownVerifier("no-such-verifier".into()));
}

#[test]
fn schema_id_mismatch_is_schema_error() {
    let mut reg = CpuRegistry::new();
    reg.insert(AnyVerifier::Math(SymbolicMath::new(
        "sympy-exact",
        MathTask::new("3"),
    )))
    .unwrap();
    let mut req = reward_request("sympy-exact");
    req.schema_id = "prometheus.not_a_reward_request".into();
    let err = reg
        .verify(&req, &Answer::Math { latex: "1".into() }, SCORED_AT, 0)
        .unwrap_err();
    match err {
        Error::Schema(s) => assert!(!s.is_empty()),
        other => panic!("expected Schema, got {other:?}"),
    }
}

#[test]
fn schema_version_mismatch_is_schema_error() {
    let mut reg = CpuRegistry::new();
    reg.insert(AnyVerifier::Math(SymbolicMath::new(
        "sympy-exact",
        MathTask::new("3"),
    )))
    .unwrap();
    let mut req = reward_request("sympy-exact");
    req.schema_version = 99;
    let err = reg
        .verify(&req, &Answer::Math { latex: "3".into() }, SCORED_AT, 0)
        .unwrap_err();
    match err {
        Error::Schema(s) => assert!(!s.is_empty()),
        other => panic!("expected Schema, got {other:?}"),
    }
}

#[test]
fn registry_lookup_by_f1_verifier_id_dispatches_symbolic() {
    let mut reg = CpuRegistry::new();
    let task = MathTask::new("3");
    reg.insert(AnyVerifier::Math(SymbolicMath::new(
        "sympy-exact",
        task.clone(),
    )))
    .unwrap();
    let req = reward_request("sympy-exact");
    let ans = Answer::Math {
        latex: "1+2".into(),
    };
    let got = reg.verify(&req, &ans, SCORED_AT, common::NOW_MS).unwrap();
    let exp = reference::verify_symbolic(&task, &req, &ans, SCORED_AT).unwrap();
    assert_eq!(got.passed, exp.passed);
    assert_eq!(got.score, exp.score);
    assert_eq!(got.request_id, req.request_id);
    assert_eq!(got.scored_at, SCORED_AT);
}

#[test]
fn registry_get_mut_none_for_missing() {
    let mut reg = CpuRegistry::new();
    match reg.get_mut("missing") {
        Err(Error::UnknownVerifier(id)) => assert_eq!(id, "missing"),
        Ok(_) => panic!("expected UnknownVerifier, got Ok"),
        Err(other) => panic!("expected UnknownVerifier, got {other:?}"),
    }
}

#[test]
fn wrong_kind_on_each_checker() {
    let req = reward_request("x");
    let code = answer_ok();
    let math = Answer::Math { latex: "1".into() };

    let err = SymbolicMath::new("x", MathTask::new("1"))
        .verify(&req, &code, SCORED_AT, 0)
        .unwrap_err();
    assert_eq!(err, Error::WrongKind);

    let err = LeanCheck::new("x", LeanTask::new("true"), TinyKernel::new())
        .verify(&req, &math, SCORED_AT, 0)
        .unwrap_err();
    assert_eq!(err, Error::WrongKind);

    let err = GridMatch::new("x", GridTask::new(Grid::new(vec![vec![1]])))
        .verify(&req, &math, SCORED_AT, 0)
        .unwrap_err();
    assert_eq!(err, Error::WrongKind);

    let err = MarketResolution::new("x", MarketTask::new(0.5, true))
        .verify(&req, &math, SCORED_AT, 0)
        .unwrap_err();
    assert_eq!(err, Error::WrongKind);
}

#[test]
fn duplicate_insert_is_schema_error() {
    let mut reg = CpuRegistry::new();
    let v = AnyVerifier::Math(SymbolicMath::new("sympy-exact", MathTask::new("3")));
    reg.insert(v).unwrap();
    let v2 = AnyVerifier::Math(SymbolicMath::new("sympy-exact", MathTask::new("4")));
    match reg.insert(v2) {
        Err(Error::Schema(s)) => assert!(s.contains("duplicate"), "{s}"),
        other => panic!("{other:?}"),
    }
}
