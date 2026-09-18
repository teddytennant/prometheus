//! Group: `validate_observation` first-hit error order.
//!
//! 1. `spec.id == Zero` → Rung0NotUsed
//! 2. `active_params == 0` or `tokens == 0` → InvalidCompute
//! 3. non-finite loss → NonFiniteLoss
//! 4. `loss <= 0` → NonPositiveLoss

mod reference;

use prometheus_control::rung::{RungId, RungSpec};
use prometheus_control::scale::{validate_observation, RungObservation, ScaleError};
use reference::scale as ref_scale;

fn obs(id: RungId, active: u64, tokens: u64, loss: f64) -> RungObservation {
    RungObservation {
        spec: RungSpec {
            id,
            active_params: active,
            total_params: 0,
            tokens,
            gpus: 0,
        },
        loss,
    }
}

fn assert_both(prod: Result<(), ScaleError>, refer: Result<(), ScaleError>) {
    assert_eq!(prod, refer);
}

#[test]
fn rung_zero_wins_over_zero_compute_and_bad_loss() {
    let o = obs(RungId::Zero, 0, 0, f64::NAN);
    assert_eq!(validate_observation(&o), Err(ScaleError::Rung0NotUsed));
    assert_both(
        validate_observation(&o),
        ref_scale::validate_observation(&o),
    );

    let o = obs(RungId::Zero, 1, 1, f64::NEG_INFINITY);
    assert_eq!(validate_observation(&o), Err(ScaleError::Rung0NotUsed));

    let o = obs(RungId::Zero, 1, 1, -1.0);
    assert_eq!(validate_observation(&o), Err(ScaleError::Rung0NotUsed));

    let o = obs(RungId::Zero, 1, 1, 1.0);
    assert_eq!(validate_observation(&o), Err(ScaleError::Rung0NotUsed));
}

#[test]
fn invalid_compute_wins_over_non_finite_and_non_positive_loss() {
    for id in [RungId::One, RungId::Two, RungId::Three] {
        let o = obs(id, 0, 0, f64::NAN);
        assert_eq!(validate_observation(&o), Err(ScaleError::InvalidCompute));
        assert_both(
            validate_observation(&o),
            ref_scale::validate_observation(&o),
        );

        let o = obs(id, 0, 10, 1.0);
        assert_eq!(validate_observation(&o), Err(ScaleError::InvalidCompute));

        let o = obs(id, 10, 0, -1.0);
        assert_eq!(validate_observation(&o), Err(ScaleError::InvalidCompute));
    }
}

#[test]
fn non_finite_loss_wins_over_non_positive() {
    for id in [RungId::One, RungId::Two, RungId::Three] {
        for loss in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let o = obs(id, 2, 3, loss);
            assert_eq!(
                validate_observation(&o),
                Err(ScaleError::NonFiniteLoss),
                "id={id:?} loss={loss}"
            );
            assert_both(
                validate_observation(&o),
                ref_scale::validate_observation(&o),
            );
        }
    }
}

#[test]
fn non_positive_finite_loss_is_rejected() {
    for id in [RungId::One, RungId::Two, RungId::Three] {
        for loss in [0.0, -0.0, -1e-9, -1.0, f64::MIN] {
            let o = obs(id, 2, 3, loss);
            assert_eq!(
                validate_observation(&o),
                Err(ScaleError::NonPositiveLoss),
                "id={id:?} loss={loss}"
            );
            assert_both(
                validate_observation(&o),
                ref_scale::validate_observation(&o),
            );
        }
    }
}

#[test]
fn valid_tiny_observation_is_ok() {
    for id in [RungId::One, RungId::Two, RungId::Three] {
        let o = obs(id, 1, 1, 1e-9);
        assert_eq!(validate_observation(&o), Ok(()));
        assert_both(
            validate_observation(&o),
            ref_scale::validate_observation(&o),
        );

        let o = obs(id, 7, 11, 2.5);
        assert_eq!(validate_observation(&o), Ok(()));
    }
}

#[test]
fn unused_spec_fields_are_not_validated() {
    let o = RungObservation {
        spec: RungSpec {
            id: RungId::One,
            active_params: 3,
            total_params: 0,
            tokens: 5,
            gpus: 0,
        },
        loss: 0.25,
    };
    assert_eq!(validate_observation(&o), Ok(()));
    assert_eq!(ref_scale::validate_observation(&o), Ok(()));
}

#[test]
fn production_matches_reference_on_the_error_ladder() {
    let cases = [
        obs(RungId::Zero, 1, 1, 1.0),
        obs(RungId::One, 0, 1, 1.0),
        obs(RungId::Two, 1, 0, 1.0),
        obs(RungId::Three, 1, 1, f64::NAN),
        obs(RungId::One, 1, 1, f64::INFINITY),
        obs(RungId::Two, 1, 1, 0.0),
        obs(RungId::Three, 1, 1, -0.5),
        obs(RungId::One, 4, 8, 1.5),
    ];
    for o in cases {
        assert_eq!(
            validate_observation(&o),
            ref_scale::validate_observation(&o),
            "obs={o:?}"
        );
    }
}
