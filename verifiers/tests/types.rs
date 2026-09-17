//! Group: constants, constructors, hygiene. Some of these pass against the stub.

mod common;
mod reference;

use common::{
    code_task, cpu_pool, reward_request, src_does_not_import_tests, src_does_not_read_wall_clock,
    strong_mutant,
};
use prometheus_verifiers::{
    Grid, GridMatch, GridTask, LeanCheck, LeanTask, MarketResolution, MarketTask, MathTask,
    SandboxedTests, SymbolicMath, TinyKernel, VerifierKind, GRID_PASS_K, PROB_EPS,
    SCHEMA_REWARD_REQUEST, SCHEMA_REWARD_RESPONSE, SCHEMA_VERSION,
};

#[test]
fn constants_match_spec() {
    assert_eq!(SCHEMA_VERSION, 1);
    assert_eq!(SCHEMA_REWARD_REQUEST, "prometheus.reward_request");
    assert_eq!(SCHEMA_REWARD_RESPONSE, "prometheus.reward_response");
    assert_eq!(GRID_PASS_K, 2);
    assert_eq!(PROB_EPS, 1e-12);
}

#[test]
fn constructors_set_ids_and_kinds() {
    let math = SymbolicMath::new("sympy-exact", MathTask::new("3"));
    assert_eq!(math.id.0, "sympy-exact");
    assert_eq!(math.kind(), VerifierKind::SymbolicMath);

    let grid = GridMatch::new(
        "grid-arc",
        GridTask::new(Grid::new(vec![vec![1, 2], vec![3, 4]])),
    );
    assert_eq!(grid.id.0, "grid-arc");
    assert_eq!(grid.kind(), VerifierKind::GridMatch);

    let lean = LeanCheck::new("lean-tiny", LeanTask::new("true"), TinyKernel::new());
    assert_eq!(lean.id.0, "lean-tiny");
    assert_eq!(lean.kind(), VerifierKind::LeanKernel);

    let market = MarketResolution::new("market-log", MarketTask::new(0.5, true));
    assert_eq!(market.id.0, "market-log");
    assert_eq!(market.kind(), VerifierKind::MarketResolution);

    let v = SandboxedTests::new(
        "code-py",
        code_task("img-t", vec![strong_mutant()]),
        cpu_pool(),
    );
    assert_eq!(v.id.0, "code-py");
    assert_eq!(v.kind(), VerifierKind::SandboxedTests);
}

#[test]
fn reward_request_new_fills_schema() {
    let req = reward_request("sympy-exact");
    common::assert_request_schema(&req);
    assert_eq!(req.request_id, "rew-0001");
    assert_eq!(req.verifier_id, "sympy-exact");
}

#[test]
fn production_src_does_not_import_tests_or_reference() {
    src_does_not_import_tests();
}

#[test]
fn production_src_does_not_read_wall_clock() {
    src_does_not_read_wall_clock();
}

#[test]
fn production_canonicalize_matches_pinned_reference() {
    let s = "$1 + 2$";
    assert_eq!(reference::canonicalize(s), "1+2");
    assert_eq!(SymbolicMath::canonicalize(s), "1+2");
}
