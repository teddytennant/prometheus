//! Group: escape / leak probes. Tasks never leave the gate.
//!
//! `request` / `rotate` / `suite_hash` return values, serde of Scores, and
//! error Display/Debug must not contain planted prompts, answers, or bodies.

mod common;
mod reference;

use prometheus_eval_gate::{Error, EvalGate, Subject, ROTATION_MS};

use common::{
    assert_no_leak, cfg, fresh_root, leak_scan_scores, pending_items, seed_standard, CANARY_ANSWER,
    CANARY_BODY, CANARY_PROMPT, HV,
};

fn scan_err(err: &Error, where_: &str) {
    assert_no_leak(&format!("{err}"), &format!("{where_} display"));
    assert_no_leak(&format!("{err:?}"), &format!("{where_} debug"));
}

#[test]
fn request_scores_serde_has_no_task_text() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let scores = gate
        .request(Subject::GenomeRev("perfect".into()), "re_bench", 0)
        .unwrap();
    leak_scan_scores(&scores, "request Scores");
    let json = serde_json::to_string(&scores).unwrap();
    assert!(!json.contains(CANARY_PROMPT));
    assert!(!json.contains(CANARY_ANSWER));
    assert!(!json.contains(CANARY_BODY));
    assert!(!json.contains("alpha"), "item answer leaked in {json}");
}

#[test]
fn request_error_on_missing_subject_does_not_include_prompts() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let err = gate
        .request(Subject::GenomeRev("ghost".into()), "re_bench", 0)
        .unwrap_err();
    scan_err(&err, "SubjectNotFound");
}

#[test]
fn suite_hash_is_hex_not_task_text() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let h = gate.suite_hash("re_bench").unwrap();
    assert_no_leak(&h, "suite_hash");
    common::assert_hash_shape(&h);
}

#[test]
fn rotate_errors_do_not_include_pending_bodies() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    reference::install_pending(&root, "re_bench", pending_items("re_bench", 2), 10, 1).unwrap();
    let mut gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let err = gate.rotate("re_bench", 10, ROTATION_MS).unwrap_err();
    scan_err(&err, "rotate cutoff refuse");
    let err = gate.rotate("re_bench", 1, 0).unwrap_err();
    scan_err(&err, "rotate early refuse");
}

#[test]
fn rotate_success_and_followup_request_have_no_task_text() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    reference::install_pending(&root, "re_bench", pending_items("re_bench", 2), 20, 1).unwrap();
    let mut gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let h = gate.rotate("re_bench", 1, ROTATION_MS).unwrap();
    assert_no_leak(&h, "rotate hash");
    let err = gate
        .request(Subject::GenomeRev("ghost".into()), "re_bench", ROTATION_MS)
        .unwrap_err();
    scan_err(&err, "request after rotate");
}

#[test]
fn unknown_suite_error_does_not_grow_task_fields() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let err = gate
        .request(Subject::GenomeRev("perfect".into()), "nope", 0)
        .unwrap_err();
    scan_err(&err, "UnknownSuite");
    match err {
        Error::UnknownSuite(_) | Error::NotHeldOut(_) | Error::Message(_) => {}
        other => panic!("unexpected variant {other:?}"),
    }
}

#[test]
fn scores_debug_does_not_pretty_print_items() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let scores = gate
        .request(Subject::Checkpoint("ckpt-perfect".into()), "re_bench", 0)
        .unwrap();
    let dbg = format!("{scores:#?}");
    assert_no_leak(&dbg, "pretty debug");
    assert!(
        !dbg.to_lowercase().contains("prompt"),
        "debug mentioned prompt: {dbg}"
    );
}
