//! Group: `request` returns Scores only; exact values vs the independent
//! reference (rci_milli, n_items, suite_hash, compute_ms, harness_version).

mod common;
mod reference;

use prometheus_eval_gate::{EvalGate, Subject, HELD_OUT_SUITES};

use common::{
    assert_hash_shape, assert_scores_eq, cfg, fresh_root, leak_scan_scores, seed_standard, HV,
};

#[test]
fn request_perfect_genome_matches_reference() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let cfg = cfg(&root, HV);
    let gate = EvalGate::open(cfg.clone()).expect("open");
    let refer = reference::RefGate::open(cfg).expect("ref open");
    let subject = Subject::GenomeRev("perfect".into());
    let got = gate
        .request(subject.clone(), "re_bench", 0)
        .expect("request");
    let exp = refer.request(subject, "re_bench", 0).expect("ref request");
    assert_scores_eq(&got, &exp);
    assert_eq!(got.rci_milli, 1000);
    assert_eq!(got.n_items, 2);
    assert_eq!(got.compute_ms, 5_000);
    assert_eq!(got.harness_version, HV);
    assert_eq!(got.suite, "re_bench");
    assert_hash_shape(&got.suite_hash);
    leak_scan_scores(&got, "perfect genome");
}

#[test]
fn request_partial_and_zero_rci() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let cfg = cfg(&root, HV);
    let gate = EvalGate::open(cfg.clone()).expect("open");
    let refer = reference::RefGate::open(cfg).expect("ref open");
    for name in ["partial", "zero"] {
        let subject = Subject::GenomeRev(name.into());
        let got = gate
            .request(subject.clone(), "re_bench", 0)
            .expect("request");
        let exp = refer.request(subject, "re_bench", 0).expect("ref request");
        assert_scores_eq(&got, &exp);
    }
    let partial = gate
        .request(Subject::GenomeRev("partial".into()), "re_bench", 0)
        .unwrap();
    let zero = gate
        .request(Subject::GenomeRev("zero".into()), "re_bench", 0)
        .unwrap();
    assert_eq!(partial.rci_milli, 400);
    assert_eq!(zero.rci_milli, 0);
    assert_eq!(partial.n_items, zero.n_items);
    assert_eq!(partial.suite_hash, zero.suite_hash);
}

#[test]
fn request_checkpoint_matches_equivalent_genome() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).expect("open");
    let g = gate
        .request(Subject::GenomeRev("perfect".into()), "re_bench", 0)
        .unwrap();
    let c = gate
        .request(Subject::Checkpoint("ckpt-perfect".into()), "re_bench", 0)
        .unwrap();
    assert_eq!(g.rci_milli, c.rci_milli);
    assert_eq!(g.n_items, c.n_items);
    assert_eq!(g.suite_hash, c.suite_hash);
    assert_ne!(g.subject, c.subject);
    match c.subject {
        Subject::Checkpoint(id) => assert_eq!(id, "ckpt-perfect"),
        other => panic!("expected Checkpoint, got {other:?}"),
    }
}

#[test]
fn request_all_held_out_suites_match_reference() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let cfg = cfg(&root, HV);
    let gate = EvalGate::open(cfg.clone()).expect("open");
    let refer = reference::RefGate::open(cfg).expect("ref open");
    for suite in HELD_OUT_SUITES {
        let subject = if *suite == "re_bench" {
            Subject::GenomeRev("perfect".into())
        } else {
            Subject::GenomeRev(format!("perfect-{suite}"))
        };
        let got = gate.request(subject.clone(), suite, 0).expect(suite);
        let exp = refer.request(subject, suite, 0).expect("ref");
        assert_scores_eq(&got, &exp);
        assert_eq!(&got.suite, suite);
        leak_scan_scores(&got, suite);
    }
}

#[test]
fn request_copies_harness_version_and_ignores_it_for_rci() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let a = EvalGate::open(cfg(&root, "hv-a")).expect("open a");
    let b = EvalGate::open(cfg(&root, "hv-b")).expect("open b");
    let subject = Subject::GenomeRev("perfect".into());
    let sa = a.request(subject.clone(), "re_bench", 0).unwrap();
    let sb = b.request(subject, "re_bench", 0).unwrap();
    assert_eq!(sa.harness_version, "hv-a");
    assert_eq!(sb.harness_version, "hv-b");
    assert_eq!(sa.rci_milli, sb.rci_milli);
    assert_eq!(sa.n_items, sb.n_items);
    assert_eq!(sa.suite_hash, sb.suite_hash);
}

#[test]
fn request_suite_hash_matches_independent_hash_of_items() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).expect("open");
    let got = gate
        .request(Subject::GenomeRev("perfect".into()), "re_bench", 0)
        .unwrap();
    let expect = reference::hash_items(&common::canary_items());
    assert_eq!(got.suite_hash, expect);
    assert_eq!(gate.suite_hash("re_bench").unwrap(), expect);
}

#[test]
fn request_does_not_mutate_suite_hash() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).expect("open");
    let before = gate.suite_hash("re_bench").unwrap();
    let _ = gate
        .request(Subject::GenomeRev("perfect".into()), "re_bench", 0)
        .unwrap();
    let after = gate.suite_hash("re_bench").unwrap();
    assert_eq!(before, after);
}

#[test]
fn extra_subject_keys_are_ignored() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let mut answers = std::collections::BTreeMap::new();
    answers.insert("i1".into(), "alpha".into());
    answers.insert("i2".into(), common::CANARY_ANSWER.into());
    answers.insert("extra".into(), "no-such-item".into());
    reference::install_subject(&root, &Subject::GenomeRev("extra".into()), &answers).unwrap();
    let gate = EvalGate::open(cfg(&root, HV)).expect("open");
    let got = gate
        .request(Subject::GenomeRev("extra".into()), "re_bench", 0)
        .unwrap();
    assert_eq!(got.rci_milli, 1000);
}
