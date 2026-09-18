//! Group 7: `run_parity` — pass, equal-to-threshold pass, above-threshold
//! HALT, NaN/inf InvalidLogprob without halt, empty pass, Unpublished,
//! then Halted on later mutators. CPU analog of V7 halt.

mod common;
mod reference;

use common::*;
use prometheus_coordinator::{CoordError, CoordinatorState, ParityReport};

fn parity_both(
    prod: &mut prometheus_coordinator::Coordinator,
    refer: &mut reference::RefCoordinator,
    version: u64,
    samples: &[prometheus_coordinator::ParitySample],
) -> prometheus_coordinator::Result<ParityReport> {
    let p = prod.run_parity(version, samples);
    let r = refer.run_parity(version, samples);
    match (p, r) {
        (Ok(a), Ok(b)) => {
            assert_eq!(a, b, "ParityReport");
            Ok(a)
        }
        (Err(e), Err(re)) => {
            assert_err_eq(&e, &re);
            Err(e)
        }
        other => panic!("run_parity mismatch {other:?}"),
    }
}

#[test]
fn parity_pass_under_threshold() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    let samples = [sample(-1.0, -1.1), sample(-2.0, -2.05)];
    // max |diff| = 0.1 < 0.25
    let report = parity_both(&mut prod, &mut refer, 0, &samples).unwrap();
    assert_eq!(report.policy_version, 0);
    assert_eq!(report.n_samples, 2);
    assert!((report.max_abs_err - 0.1).abs() < 1e-12);
    assert!(!prod.halted().unwrap());
    assert_eq!(prod.state().unwrap(), CoordinatorState::Running);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn parity_equal_to_threshold_does_not_halt() {
    let mut cfg = default_config();
    cfg.parity_threshold = 0.5;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    let samples = [sample(0.0, 0.5), sample(-1.0, -1.25)];
    // max = 0.5 == threshold
    let report = parity_both(&mut prod, &mut refer, 0, &samples).unwrap();
    assert_eq!(report.max_abs_err, 0.5);
    assert_eq!(report.n_samples, 2);
    assert!(!prod.halted().unwrap());
    assert_eq!(prod.state().unwrap(), CoordinatorState::Running);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn parity_above_threshold_halts() {
    let mut cfg = default_config();
    cfg.parity_threshold = 0.25;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    let samples = [sample(0.0, 0.5)];
    let err = match parity_both(&mut prod, &mut refer, 0, &samples) {
        Err(e) => e,
        Ok(r) => panic!("expected ParityHalt, got Ok {r:?}"),
    };
    match err {
        CoordError::ParityHalt {
            version,
            max_abs_err,
            threshold,
        } => {
            assert_eq!(version, 0);
            assert_eq!(max_abs_err, 0.5);
            assert_eq!(threshold, 0.25);
        }
        other => panic!("expected ParityHalt, got {other:?}"),
    }
    assert!(prod.halted().unwrap());
    assert_eq!(prod.state().unwrap(), CoordinatorState::Halted);
    assert!(refer.halted().unwrap());
    assert_eq!(refer.state().unwrap(), CoordinatorState::Halted);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn nan_is_invalid_logprob_without_halt() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    for samples in [
        vec![sample(f64::NAN, 0.0)],
        vec![sample(0.0, f64::NAN)],
        vec![sample(f64::NAN, f64::NAN)],
    ] {
        let err = match parity_both(&mut prod, &mut refer, 0, &samples) {
            Err(e) => e,
            Ok(r) => panic!("expected InvalidLogprob, got Ok {r:?}"),
        };
        match err {
            CoordError::InvalidLogprob => {}
            other => panic!("expected InvalidLogprob, got {other:?}"),
        }
        assert!(!prod.halted().unwrap());
        assert_eq!(prod.state().unwrap(), CoordinatorState::Running);
    }
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn inf_is_invalid_logprob_without_halt() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    for samples in [
        vec![sample(f64::INFINITY, 0.0)],
        vec![sample(0.0, f64::NEG_INFINITY)],
        vec![sample(f64::INFINITY, f64::NEG_INFINITY)],
        vec![sample(-1.0, f64::INFINITY)],
    ] {
        let err = match parity_both(&mut prod, &mut refer, 0, &samples) {
            Err(e) => e,
            Ok(r) => panic!("expected InvalidLogprob, got Ok {r:?}"),
        };
        match err {
            CoordError::InvalidLogprob => {}
            other => panic!("expected InvalidLogprob, got {other:?}"),
        }
        assert!(!prod.halted().unwrap());
    }
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn nan_does_not_halt_even_if_other_samples_would_drift() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    let samples = [sample(0.0, 10.0), sample(f64::NAN, 0.0)];
    match parity_both(&mut prod, &mut refer, 0, &samples) {
        Err(CoordError::InvalidLogprob) => {}
        other => panic!("expected InvalidLogprob before halt, got {other:?}"),
    }
    assert!(!prod.halted().unwrap());
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn empty_samples_pass_with_zero_err() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    let report = parity_both(&mut prod, &mut refer, 0, &[]).unwrap();
    assert_eq!(
        report,
        ParityReport {
            policy_version: 0,
            max_abs_err: 0.0,
            n_samples: 0,
        }
    );
    assert!(!prod.halted().unwrap());
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn parity_unpublished_version() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    let err = match parity_both(&mut prod, &mut refer, 1, &[sample(0.0, 0.0)]) {
        Err(e) => e,
        Ok(r) => panic!("expected Unpublished, got Ok {r:?}"),
    };
    match err {
        CoordError::Unpublished(v) => assert_eq!(v, 1),
        other => panic!("expected Unpublished(1), got {other:?}"),
    }
    assert!(!prod.halted().unwrap());
    assert_both_ok(prod.publish_weights(1), refer.publish_weights(1));
    // Now version 0 is unpublished.
    let err = match parity_both(&mut prod, &mut refer, 0, &[]) {
        Err(e) => e,
        Ok(r) => panic!("expected Unpublished, got Ok {r:?}"),
    };
    match err {
        CoordError::Unpublished(v) => assert_eq!(v, 0),
        other => panic!("expected Unpublished(0), got {other:?}"),
    }
    let report = parity_both(&mut prod, &mut refer, 1, &[]).unwrap();
    assert_eq!(report.policy_version, 1);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn halt_then_every_mutator_returns_halted_reads_still_work() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    assert_both_ok(
        prod.enqueue_batch(batch("queued", "p", 0, 2, false)),
        refer.enqueue_batch(batch("queued", "p", 0, 2, false)),
    );
    let depth_before = prod.queue_depth().unwrap();
    match parity_both(&mut prod, &mut refer, 0, &[sample(0.0, 1.0)]) {
        Err(CoordError::ParityHalt { .. }) => {}
        other => panic!("expected ParityHalt, got {other:?}"),
    }
    assert!(prod.halted().unwrap());
    assert_eq!(prod.state().unwrap(), CoordinatorState::Halted);

    assert_halted(&assert_both_err(
        prod.enqueue_batch(batch("nope", "p", 0, 1, false)),
        refer.enqueue_batch(batch("nope", "p", 0, 1, false)),
    ));
    assert_halted(&assert_both_err(
        prod.consume_batch(),
        refer.consume_batch(),
    ));
    assert_eq!(prod.queue_depth().unwrap(), depth_before);
    assert_halted(&assert_both_err(prod.rebalance(), refer.rebalance()));
    assert_halted(&assert_both_err(
        prod.publish_weights(1),
        refer.publish_weights(1),
    ));
    assert_halted(&assert_both_err(
        prod.sync_rack(&rid("r00"), 0),
        refer.sync_rack(&rid("r00"), 0),
    ));
    assert_halted(&assert_both_err(
        prod.run_parity(0, &[]),
        refer.run_parity(0, &[]),
    ));
    // Wrong version after halt is still Halted, not Unpublished.
    assert_halted(&assert_both_err(
        prod.run_parity(9, &[sample(0.0, 0.0)]),
        refer.run_parity(9, &[sample(0.0, 0.0)]),
    ));

    // Reads still work.
    assert_eq!(prod.state().unwrap(), CoordinatorState::Halted);
    assert!(prod.halted().unwrap());
    assert_eq!(prod.trainer_version().unwrap(), 0);
    let (ro, tr) = prod.split_gpus().unwrap();
    assert_eq!(ro + tr, 32);
    assert_eq!(prod.n_rollout_racks().unwrap(), 2);
    assert_eq!(prod.n_trainer_racks().unwrap(), 2);
    assert_eq!(prod.queue_depth().unwrap(), depth_before);
    assert_eq!(
        prod.rack_role(&rid("r00")).unwrap(),
        prometheus_coordinator::RackRole::Rollout
    );
    assert_eq!(prod.rack_version(&rid("r00")).unwrap(), 0);
    // Unknown rack still RackNotFound, not Halted.
    match prod.rack_role(&rid("ghost")) {
        Err(CoordError::RackNotFound(_)) => {}
        other => panic!("read must not return Halted, got {other:?}"),
    }
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn max_abs_err_is_max_over_samples() {
    let mut cfg = default_config();
    cfg.parity_threshold = 10.0;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    let samples = [
        sample(-1.0, -1.2), // 0.2
        sample(3.0, 3.5),   // 0.5
        sample(0.0, -0.1),  // 0.1
    ];
    let report = parity_both(&mut prod, &mut refer, 0, &samples).unwrap();
    assert_eq!(report.max_abs_err, 0.5);
    assert_eq!(report.n_samples, 3);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn halt_is_sticky_across_matching_parity_retry() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    match parity_both(&mut prod, &mut refer, 0, &[sample(0.0, 9.0)]) {
        Err(CoordError::ParityHalt { .. }) => {}
        other => panic!("{other:?}"),
    }
    // Perfect match would have passed, but we are halted.
    assert_halted(&assert_both_err(
        prod.run_parity(0, &[sample(0.0, 0.0)]),
        refer.run_parity(0, &[sample(0.0, 0.0)]),
    ));
    assert_prod_matches_ref(&prod, &refer, &racks);
}
