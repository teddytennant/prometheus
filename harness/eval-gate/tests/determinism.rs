//! Group: same subject+suite+harness_version is deterministic
//! (rci_milli, n_items, suite_hash). `now` must not change those fields.

mod common;
mod reference;

use prometheus_eval_gate::{EvalGate, Subject};

use common::{assert_scores_eq, cfg, fresh_root, seed_standard, HV};

#[test]
fn request_twice_same_inputs_same_scores() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).expect("open");
    let subject = Subject::GenomeRev("partial".into());
    let a = gate.request(subject.clone(), "re_bench", 0).unwrap();
    let b = gate.request(subject, "re_bench", 0).unwrap();
    assert_scores_eq(&a, &b);
}

#[test]
fn request_independent_of_now() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).expect("open");
    let subject = Subject::Checkpoint("ckpt-perfect".into());
    let a = gate.request(subject.clone(), "re_bench", 0).unwrap();
    let b = gate.request(subject.clone(), "re_bench", 1).unwrap();
    let c = gate.request(subject, "re_bench", u64::MAX).unwrap();
    assert_eq!(a.rci_milli, b.rci_milli);
    assert_eq!(a.n_items, b.n_items);
    assert_eq!(a.suite_hash, b.suite_hash);
    assert_eq!(a.rci_milli, c.rci_milli);
    assert_eq!(a.n_items, c.n_items);
    assert_eq!(a.suite_hash, c.suite_hash);
    assert_eq!(a.compute_ms, b.compute_ms);
    assert_eq!(a.compute_ms, c.compute_ms);
}

#[test]
fn reopen_is_deterministic() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let subject = Subject::GenomeRev("perfect".into());
    let first = {
        let gate = EvalGate::open(cfg(&root, HV)).expect("open");
        gate.request(subject.clone(), "re_bench", 42).unwrap()
    };
    let second = {
        let gate = EvalGate::open(cfg(&root, HV)).expect("reopen");
        gate.request(subject, "re_bench", 99).unwrap()
    };
    assert_scores_eq(&first, &second);
}

#[test]
fn suite_hash_stable_across_opens() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let h1 = EvalGate::open(cfg(&root, HV))
        .unwrap()
        .suite_hash("mle_bench")
        .unwrap();
    let h2 = EvalGate::open(cfg(&root, "other-hv"))
        .unwrap()
        .suite_hash("mle_bench")
        .unwrap();
    assert_eq!(h1, h2);
    assert_eq!(
        h1,
        reference::hash_items(&common::bland_items("mle_bench", 3))
    );
}

#[test]
fn two_gates_same_root_same_scores() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let a = EvalGate::open(cfg(&root, HV)).unwrap();
    let b = EvalGate::open(cfg(&root, HV)).unwrap();
    let subject = Subject::GenomeRev("partial".into());
    let sa = a.request(subject.clone(), "re_bench", 7).unwrap();
    let sb = b.request(subject, "re_bench", 7).unwrap();
    assert_scores_eq(&sa, &sb);
}
