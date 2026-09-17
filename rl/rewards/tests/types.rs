//! Group: constants, constructors, getters, hygiene, ScriptedJudge, recall.

mod common;
mod reference;

use common::{
    assert_request_schema, criterion, reward_request, src_does_not_import_tests,
    src_does_not_read_wall_clock, NOW_MS,
};
use prometheus_rewards::{
    ContrastivePair, DetectReport, Error, HeldOutHacks, Judge, RescoreFilter, Rubric,
    ScriptedJudge, TamperPairFilter, TamperingDetector, DISAGREE_EPS,
};
use prometheus_verifiers::{SCHEMA_REWARD_REQUEST, SCHEMA_REWARD_RESPONSE, SCHEMA_VERSION};

#[test]
fn constants_match_spec() {
    assert_eq!(DISAGREE_EPS, 0.25);
    assert_eq!(SCHEMA_VERSION, 1);
    assert_eq!(SCHEMA_REWARD_REQUEST, "prometheus.reward_request");
    assert_eq!(SCHEMA_REWARD_RESPONSE, "prometheus.reward_response");
}

#[test]
fn constructors_set_ids_and_fields() {
    let c = criterion("clarity", "Is it clear?", 2.5);
    assert_eq!(c.id.0, "clarity");
    assert_eq!(c.prompt, "Is it clear?");
    assert_eq!(c.weight, 2.5);

    let r = Rubric::new("r1", vec![c]);
    assert_eq!(r.id.0, "r1");
    assert_eq!(r.criteria.len(), 1);
    assert_eq!(r.total_weight(), 2.5);

    let pair = ContrastivePair::new("in", "pp", "np", "good", "hack");
    assert_eq!(pair.input, "in");
    assert_eq!(pair.positive_prompt, "pp");
    assert_eq!(pair.negative_prompt, "np");
    assert_eq!(pair.positive, "good");
    assert_eq!(pair.negative, "hack");

    let _ = TamperPairFilter::new();
    let _ = RescoreFilter::new();
    let det = TamperingDetector::new("tamper");
    assert_eq!(det.id, "tamper");
    let _ = HeldOutHacks::new();
}

#[test]
fn empty_rubric_total_weight_is_zero() {
    let r = Rubric::new("empty", vec![]);
    assert_eq!(r.total_weight(), 0.0);
}

#[test]
fn held_out_insert_contains_get() {
    let mut set = HeldOutHacks::new();
    let view = common::test_write_view();
    set.insert("planted-a", view.clone());
    assert!(set.contains("planted-a"));
    assert!(!set.contains("missing"));
    assert_eq!(set.get("planted-a"), Some(&view));
    assert_eq!(set.get("missing"), None);
}

#[test]
fn detect_report_recall_is_flagged_over_n() {
    let full = DetectReport { n: 4, flagged: 3 };
    assert!((full.recall() - 0.75).abs() < 1e-12);
    let empty = DetectReport { n: 0, flagged: 0 };
    assert_eq!(empty.recall(), 0.0);
    let none = DetectReport { n: 5, flagged: 0 };
    assert_eq!(none.recall(), 0.0);
    let all = DetectReport { n: 3, flagged: 3 };
    assert_eq!(all.recall(), 1.0);
}

#[test]
fn scripted_judge_returns_reply_or_unverifiable() {
    let mut j = ScriptedJudge::new();
    j.insert("p1", "0.5");
    assert_eq!(j.complete("p1", NOW_MS).unwrap(), "0.5");
    let err = j.complete("missing", NOW_MS).unwrap_err();
    assert!(matches!(err, Error::Unverifiable(_)));
    assert!(err.to_string().contains("no scripted reply"));
}

#[test]
fn reward_request_new_fills_schema() {
    let req = reward_request("panel-1");
    assert_request_schema(&req);
    assert_eq!(req.request_id, "rew-0001");
    assert_eq!(req.verifier_id, "panel-1");
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
fn held_out_leak_error_displays_training_invariant() {
    let err = Error::HeldOutLeak;
    assert_eq!(err.to_string(), "held-out hack used as training pair");
}

#[test]
fn error_displays_match_iface() {
    assert_eq!(Error::EmptyRubric.to_string(), "empty rubric");
    assert_eq!(Error::EmptyPanel.to_string(), "empty grader panel");
    assert_eq!(
        Error::BadWeight.to_string(),
        "criterion weight must be finite and > 0"
    );
    assert_eq!(
        Error::UnconfirmedPair.to_string(),
        "pair labels not confirmed"
    );
    assert_eq!(
        Error::ExactVerifier("sympy-exact".into()).to_string(),
        "exact verifier sympy-exact already scores this task"
    );
}

#[test]
fn disagree_eps_is_a_named_threshold_not_a_drop_switch() {
    assert_eq!(DISAGREE_EPS, 0.25);
    let scores = common::weighted_scores(&[("g0", 1.0), ("g1", 0.7)]);
    let got = reference::combine(&scores).unwrap();
    assert!(got.is_finite());
    assert_ne!(got, 0.0);
}

#[test]
fn one_criterion_panel_constructor_keeps_grader_order() {
    let panel = common::panel_with_graders(2);
    assert_eq!(panel.id, "panel");
    assert_eq!(panel.grader_ids.len(), 2);
    assert_eq!(panel.grader_ids[0].0, "g0");
    assert_eq!(panel.grader_ids[1].0, "g1");
}
