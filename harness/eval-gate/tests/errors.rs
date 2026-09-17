//! Group: UnknownSuite, NotHeldOut, SubjectNotFound, open validation.
//! Every test calls an `EvalGate` method and must fail on the stub.

mod common;
mod reference;

use prometheus_eval_gate::{Error, EvalGate, Subject, HELD_OUT_SUITES};

use common::{cfg, fresh_root, seed_standard, HV, PUBLIC_NOT_HELD_OUT};

#[test]
fn open_refuses_missing_suite_root() {
    let (_tmp, root) = fresh_root();
    let missing = root.join("no-such-dir");
    match EvalGate::open(cfg(&missing, HV)) {
        Err(Error::Message(_)) => {}
        Err(e) => panic!("expected Message, got {e:?}"),
        Ok(_) => panic!("expected Message, got Ok"),
    }
}

#[test]
fn open_refuses_suite_root_that_is_a_file() {
    let (_tmp, root) = fresh_root();
    let file = root.join("not-a-dir");
    std::fs::write(&file, b"x").unwrap();
    match EvalGate::open(cfg(&file, HV)) {
        Err(Error::Message(_)) => {}
        Err(e) => panic!("expected Message, got {e:?}"),
        Ok(_) => panic!("expected Message, got Ok"),
    }
}

#[test]
fn open_refuses_empty_harness_version() {
    let (_tmp, root) = fresh_root();
    match EvalGate::open(cfg(&root, "")) {
        Err(Error::Message(_)) => {}
        Err(e) => panic!("expected Message, got {e:?}"),
        Ok(_) => panic!("expected Message, got Ok"),
    }
}

#[test]
fn open_succeeds_on_empty_directory() {
    let (_tmp, root) = fresh_root();
    let gate = EvalGate::open(cfg(&root, HV)).expect("open empty root");
    drop(gate);
}

#[test]
fn request_unknown_suite_is_unknown_suite() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).expect("open");
    let subject = Subject::GenomeRev("perfect".into());
    match gate.request(subject, "not_a_real_suite", 0) {
        Err(Error::UnknownSuite(s)) => assert_eq!(s, "not_a_real_suite"),
        other => panic!("expected UnknownSuite, got {other:?}"),
    }
}

#[test]
fn request_wrong_case_held_out_is_unknown_suite() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).expect("open");
    let subject = Subject::GenomeRev("perfect".into());
    match gate.request(subject, "RE_BENCH", 0) {
        Err(Error::UnknownSuite(s)) => assert_eq!(s, "RE_BENCH"),
        other => panic!("expected UnknownSuite, got {other:?}"),
    }
}

#[test]
fn request_public_not_held_out_is_not_held_out() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).expect("open");
    for slug in PUBLIC_NOT_HELD_OUT {
        let subject = Subject::GenomeRev("perfect".into());
        match gate.request(subject, slug, 0) {
            Err(Error::NotHeldOut(s)) => assert_eq!(&s, slug),
            other => panic!("{slug}: expected NotHeldOut, got {other:?}"),
        }
    }
}

#[test]
fn request_missing_subject_is_subject_not_found() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).expect("open");
    match gate.request(Subject::GenomeRev("missing-rev".into()), "re_bench", 0) {
        Err(Error::SubjectNotFound(s)) => assert_eq!(s, "missing-rev"),
        other => panic!("expected SubjectNotFound, got {other:?}"),
    }
}

#[test]
fn request_missing_checkpoint_is_subject_not_found() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).expect("open");
    match gate.request(Subject::Checkpoint("no-ckpt".into()), "re_bench", 0) {
        Err(Error::SubjectNotFound(s)) => assert_eq!(s, "no-ckpt"),
        other => panic!("expected SubjectNotFound, got {other:?}"),
    }
}

#[test]
fn suite_hash_unknown_and_not_held_out() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).expect("open");
    match gate.suite_hash("zzz_unknown") {
        Err(Error::UnknownSuite(s)) => assert_eq!(s, "zzz_unknown"),
        other => panic!("expected UnknownSuite, got {other:?}"),
    }
    match gate.suite_hash("gpqa") {
        Err(Error::NotHeldOut(s)) => assert_eq!(s, "gpqa"),
        other => panic!("expected NotHeldOut, got {other:?}"),
    }
}

#[test]
fn rotate_unknown_and_not_held_out() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let mut gate = EvalGate::open(cfg(&root, HV)).expect("open");
    match gate.rotate("zzz_unknown", 0, prometheus_eval_gate::ROTATION_MS) {
        Err(Error::UnknownSuite(s)) => assert_eq!(s, "zzz_unknown"),
        other => panic!("expected UnknownSuite, got {other:?}"),
    }
    match gate.rotate("gpqa", 0, prometheus_eval_gate::ROTATION_MS) {
        Err(Error::NotHeldOut(s)) => assert_eq!(s, "gpqa"),
        other => panic!("expected NotHeldOut, got {other:?}"),
    }
}

#[test]
fn suite_hash_missing_current_is_message() {
    let (_tmp, root) = fresh_root();
    let gate = EvalGate::open(cfg(&root, HV)).expect("open");
    match gate.suite_hash(HELD_OUT_SUITES[0]) {
        Err(Error::Message(_)) => {}
        other => panic!("expected Message for uninstalled suite, got {other:?}"),
    }
}

#[test]
fn request_path_traversal_subject_is_subject_not_found() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).expect("open");
    for bad in ["../etc/passwd", "foo/bar", "..", ".", "a\\b", ""] {
        match gate.request(Subject::GenomeRev(bad.into()), "re_bench", 0) {
            Err(Error::SubjectNotFound(s)) => assert_eq!(s, bad),
            other => panic!("{bad:?}: expected SubjectNotFound, got {other:?}"),
        }
    }
}
