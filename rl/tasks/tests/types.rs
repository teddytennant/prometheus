//! Group: constructors, getters, ScriptedSolver, schema_ok, SolveRate helpers,
//! Horizon::new, Factory domain/id, hygiene. These may pass against the stub.

mod common;
mod reference;

use common::{src, CREATED, MATH_ID, NOW};
use prometheus_tasks::{
    CodeFactory, Error, Factory, Horizon, MathFactory, Provenance, ScriptedSolver, SolveRate,
    Solver, Source, Split, SweFactory, TaskDomain, TaskSpec, RAISE_THRESHOLD, SCHEMA_TASK_SPEC,
    SCHEMA_VERSION, START_TOOL_CALLS,
};

#[test]
fn constants_match_spec() {
    assert_eq!(START_TOOL_CALLS, 10);
    assert_eq!(RAISE_THRESHOLD, 0.5);
    assert_eq!(SCHEMA_TASK_SPEC, "prometheus.task_spec");
    assert_eq!(SCHEMA_VERSION, 1);
    assert_eq!(reference::CODE_HORIZON_S, 30);
    assert_eq!(reference::SWE_HORIZON_S, 60);
    assert_eq!(reference::SWE_HORIZON_S, reference::CODE_HORIZON_S * 2);
    assert_eq!(reference::CODE_MAX_TOOL_CALLS, START_TOOL_CALLS);
    assert_eq!(
        reference::SWE_MAX_TOOL_CALLS,
        reference::CODE_MAX_TOOL_CALLS * 2
    );
    assert_eq!(reference::MATH_MAX_TOOL_CALLS, START_TOOL_CALLS);
}

#[test]
fn horizon_new_starts_at_ten() {
    let h = Horizon::new();
    assert_eq!(h.tool_calls(), START_TOOL_CALLS);
    assert_eq!(Horizon::default().tool_calls(), START_TOOL_CALLS);
}

#[test]
fn solve_rate_helpers() {
    let half = SolveRate::new(1, 2);
    assert_eq!(half.rate(), 0.5);
    assert!(half.in_open_interval());

    let zero = SolveRate::new(0, 4);
    assert_eq!(zero.rate(), 0.0);
    assert!(!zero.in_open_interval());

    let all = SolveRate::new(4, 4);
    assert_eq!(all.rate(), 1.0);
    assert!(!all.in_open_interval());

    let bad = SolveRate::new(0, 0);
    assert_eq!(bad.rate(), 0.0);
    assert!(!bad.in_open_interval());

    let n1_pass = SolveRate::new(1, 1);
    assert!(!n1_pass.in_open_interval());
    let n1_fail = SolveRate::new(0, 1);
    assert!(!n1_fail.in_open_interval());
}

#[test]
fn schema_ok_accepts_v1_and_rejects_mismatch() {
    let mut spec = TaskSpec {
        schema_id: SCHEMA_TASK_SPEC.to_string(),
        schema_version: SCHEMA_VERSION,
        task_id: "t".into(),
        domain: TaskDomain::Math,
        split: Split::Train,
        statement_hash: reference::sha256_hex(b""),
        hidden_tests_hash: reference::sha256_hex(b""),
        verifier_id: "v".into(),
        env_image: None,
        horizon_s: 30,
        max_tool_calls: 10,
        provenance: Provenance::new("src"),
        created_at: CREATED.into(),
    };
    assert!(spec.schema_ok().is_ok());
    spec.schema_id = "other".into();
    assert!(matches!(spec.schema_ok(), Err(Error::Schema(_))));
    spec.schema_id = SCHEMA_TASK_SPEC.into();
    spec.schema_version = 99;
    assert!(matches!(spec.schema_ok(), Err(Error::Schema(_))));
}

#[test]
fn scripted_solver_missing_statement_is_unverifiable() {
    let mut s = ScriptedSolver::new();
    match s.attempt("nope", NOW) {
        Err(Error::Unverifiable(msg)) => assert!(msg.contains("nope"), "{msg}"),
        other => panic!("expected Unverifiable, got {other:?}"),
    }
}

#[test]
fn scripted_solver_returns_inserted_reply() {
    let mut s = ScriptedSolver::new();
    s.insert("What is 2+2?", "4");
    assert_eq!(s.attempt("What is 2+2?", NOW).unwrap(), "4");
    assert_eq!(s.attempt("What is 2+2?", NOW + 1).unwrap(), "4");
}

#[test]
fn scripted_solver_default_is_empty() {
    let mut s = ScriptedSolver::default();
    assert!(matches!(s.attempt("x", NOW), Err(Error::Unverifiable(_))));
}

#[test]
fn factory_constructors_and_getters() {
    let rows = vec![src("a", "body")];
    let math = MathFactory::new(MATH_ID, rows.clone());
    assert_eq!(math.id().0, MATH_ID);
    assert_eq!(math.sources(), rows.as_slice());
    assert_eq!(math.domain(), TaskDomain::Math);

    let code = CodeFactory::new("code-1", rows.clone());
    assert_eq!(code.id().0, "code-1");
    assert_eq!(code.domain(), TaskDomain::Code);

    let swe = SweFactory::new("swe-1", rows);
    assert_eq!(swe.id().0, "swe-1");
    assert_eq!(swe.domain(), TaskDomain::Code);
    assert_ne!(swe.domain(), TaskDomain::Agent);
}

#[test]
fn source_and_provenance_constructors() {
    let s = Source::new("id", "gsm8k", "body");
    assert_eq!(s.id, "id");
    assert_eq!(s.provenance.source, "gsm8k");
    assert_eq!(s.provenance.license, None);
    assert_eq!(s.body, "body");
}

#[test]
fn error_display_pins() {
    assert_eq!(Error::EmptyCatalog.to_string(), "empty factory catalog");
    assert_eq!(Error::EmptyStatement.to_string(), "empty public statement");
    assert_eq!(
        Error::BadN.to_string(),
        "solve-rate sample count must be > 0"
    );
    assert_eq!(Error::Impossible.to_string(), "solve rate 0%");
    assert_eq!(Error::Trivial.to_string(), "solve rate 100%");
}

#[test]
fn src_does_not_import_tests() {
    common::src_does_not_import_tests();
}

#[test]
fn src_does_not_read_wall_clock() {
    common::src_does_not_read_wall_clock();
}

#[test]
fn created_at_fixture_is_not_now_to_string() {
    assert_ne!(CREATED, NOW.to_string());
}
