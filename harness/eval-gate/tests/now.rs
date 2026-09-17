//! Group: `NowMs` is injected. The gate must not sleep or read the wall clock
//! for correctness. `compute_ms` is the suite's fixed budget, not elapsed time.

mod common;
mod reference;

use std::time::Instant;

use prometheus_eval_gate::{EvalGate, Subject, ROTATION_MS};

use common::{cfg, fresh_root, seed_standard, HV};

#[test]
fn request_returns_quickly_and_ignores_wall_clock() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let subject = Subject::GenomeRev("perfect".into());
    let t0 = Instant::now();
    let a = gate.request(subject.clone(), "re_bench", 0).unwrap();
    let b = gate.request(subject, "re_bench", ROTATION_MS).unwrap();
    let elapsed = t0.elapsed();
    assert!(
        elapsed.as_secs() < 2,
        "request slept or blocked on wall clock: {elapsed:?}"
    );
    assert_eq!(a.rci_milli, b.rci_milli);
    assert_eq!(a.n_items, b.n_items);
    assert_eq!(a.suite_hash, b.suite_hash);
    assert_eq!(a.compute_ms, 5_000, "compute_ms is the fixture budget");
    assert_eq!(a.compute_ms, b.compute_ms);
}

#[test]
fn rotate_uses_injected_now_not_wall_clock() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    reference::install_pending(
        &root,
        "re_bench",
        common::pending_items("re_bench", 1),
        10,
        1,
    )
    .unwrap();
    let mut gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let t0 = Instant::now();
    let err = gate.rotate("re_bench", 1, 0).unwrap_err();
    match err {
        prometheus_eval_gate::Error::RotationRefused(_) => {}
        other => panic!("expected RotationRefused at now=0, got {other:?}"),
    }
    let h = gate
        .rotate("re_bench", 1, ROTATION_MS)
        .expect("rotate at injected now");
    let elapsed = t0.elapsed();
    assert!(
        elapsed.as_secs() < 2,
        "rotate slept for the quarter: {elapsed:?}"
    );
    common::assert_hash_shape(&h);
}

#[test]
fn open_does_not_sleep() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let t0 = Instant::now();
    let _ = EvalGate::open(cfg(&root, HV)).unwrap();
    assert!(t0.elapsed().as_secs() < 2, "open blocked on wall clock");
}
