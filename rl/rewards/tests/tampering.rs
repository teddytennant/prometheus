//! Group: TamperingDetector verify and HeldOutHacks eval.

mod common;
mod reference;

use common::{
    all_hack_kinds_view, assert_close, assert_meta, clean_view, hidden_pass_visible_fail_view,
    reward_request, special_case_view, test_write_view, visible_pass_hidden_fail_view, NOW_MS,
    SCORED_AT,
};
use prometheus_rewards::{Error, Flag, HeldOutHacks, TamperingDetector};

#[test]
fn confirmed_hack_verify_is_zero_failed_tampering_flag() {
    let det = TamperingDetector::new("tamper");
    let req = reward_request("tamper");
    for view in [
        test_write_view(),
        special_case_view(),
        visible_pass_hidden_fail_view(),
        all_hack_kinds_view(),
    ] {
        let got = det.verify(&req, &view, SCORED_AT, NOW_MS).unwrap();
        let exp = reference::verify_tamper(&req, &view, SCORED_AT, NOW_MS).unwrap();
        assert_meta(&got, &req, SCORED_AT);
        assert_eq!(got.score, 0.0);
        assert!(!got.passed);
        assert!(got.flags.contains(&Flag::Tampering));
        assert_eq!(got.score, exp.score);
        assert_eq!(got.passed, exp.passed);
        assert_eq!(got.flags, exp.flags);
    }
}

#[test]
fn clean_view_verify_is_one_passed_no_tampering_flag() {
    let det = TamperingDetector::new("tamper");
    let req = reward_request("tamper");
    for view in [clean_view(), hidden_pass_visible_fail_view()] {
        let got = det.verify(&req, &view, SCORED_AT, NOW_MS).unwrap();
        let exp = reference::verify_tamper(&req, &view, SCORED_AT, NOW_MS).unwrap();
        assert_meta(&got, &req, SCORED_AT);
        assert_eq!(got.score, 1.0);
        assert!(got.passed);
        assert!(!got.flags.contains(&Flag::Tampering));
        assert_eq!(got.flags, exp.flags);
        assert!(got.flags.is_empty());
    }
}

#[test]
fn detector_schema_mismatch() {
    let det = TamperingDetector::new("tamper");
    let mut req = reward_request("tamper");
    req.schema_id = "nope".into();
    assert!(matches!(
        det.verify(&req, &clean_view(), SCORED_AT, NOW_MS)
            .unwrap_err(),
        Error::Schema(_)
    ));
    let mut req = reward_request("tamper");
    req.schema_version = 0;
    assert!(matches!(
        det.verify(&req, &test_write_view(), SCORED_AT, NOW_MS)
            .unwrap_err(),
        Error::Schema(_)
    ));
}

#[test]
fn detector_scored_at_is_caller_string() {
    let det = TamperingDetector::new("tamper");
    let req = reward_request("tamper");
    let resp = det
        .verify(&req, &clean_view(), "caller-ts", NOW_MS)
        .unwrap();
    assert_eq!(resp.scored_at, "caller-ts");
    assert_ne!(resp.scored_at, NOW_MS.to_string());
}

#[test]
fn eval_empty_set_recall_is_zero() {
    let set = HeldOutHacks::new();
    let det = TamperingDetector::new("tamper");
    let report = set.eval(&det).unwrap();
    let exp = reference::eval_ids(&set, &[]);
    assert_eq!(report.n, 0);
    assert_eq!(report.flagged, 0);
    assert_eq!(report.recall(), 0.0);
    assert_eq!(report.n, exp.n);
    assert_eq!(report.flagged, exp.flagged);
}

#[test]
fn planted_three_hack_kinds_are_all_flagged() {
    let mut set = HeldOutHacks::new();
    set.insert("planted-test-write", test_write_view());
    set.insert("planted-special-case", special_case_view());
    set.insert("planted-visible-hidden", visible_pass_hidden_fail_view());
    let det = TamperingDetector::new("tamper");
    assert!(det.is_hack(&test_write_view()));
    assert!(det.is_hack(&special_case_view()));
    assert!(det.is_hack(&visible_pass_hidden_fail_view()));
    let report = set.eval(&det).unwrap();
    assert_eq!(report.n, 3);
    assert_eq!(report.flagged, 3);
    assert_eq!(report.recall(), 1.0);
    let exp = reference::eval_ids(
        &set,
        &[
            "planted-test-write",
            "planted-special-case",
            "planted-visible-hidden",
        ],
    );
    assert_eq!(report.n, exp.n);
    assert_eq!(report.flagged, exp.flagged);
}

#[test]
fn clean_view_in_held_out_set_is_not_flagged() {
    let mut set = HeldOutHacks::new();
    set.insert("planted-test-write", test_write_view());
    set.insert("planted-special-case", special_case_view());
    set.insert("planted-visible-hidden", visible_pass_hidden_fail_view());
    set.insert("clean", clean_view());
    let det = TamperingDetector::new("tamper");
    assert!(!det.is_hack(&clean_view()));
    let report = set.eval(&det).unwrap();
    assert_eq!(report.n, 4);
    assert_eq!(report.flagged, 3);
    assert_close(report.recall(), 0.75);
}

#[test]
fn hidden_pass_visible_fail_is_not_flagged_in_eval() {
    let mut set = HeldOutHacks::new();
    set.insert("not-a-hack", hidden_pass_visible_fail_view());
    let det = TamperingDetector::new("tamper");
    let report = set.eval(&det).unwrap();
    assert_eq!(report.n, 1);
    assert_eq!(report.flagged, 0);
    assert_eq!(report.recall(), 0.0);
}

#[test]
fn eval_overwrite_same_id_does_not_double_count() {
    let mut set = HeldOutHacks::new();
    set.insert("x", test_write_view());
    set.insert("x", special_case_view());
    let det = TamperingDetector::new("tamper");
    let report = set.eval(&det).unwrap();
    assert_eq!(report.n, 1);
    assert_eq!(report.flagged, 1);
}

#[test]
fn eval_is_the_scoring_entry_pair_filters_take_views() {
    let mut set = HeldOutHacks::new();
    set.insert("held-out", test_write_view());
    assert!(set.contains("held-out"));
    assert!(set.get("train-id").is_none());
    let det = TamperingDetector::new("tamper");
    let report = set.eval(&det).unwrap();
    assert_eq!(report.n, 1);
}
