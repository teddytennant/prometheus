//! Group: 5% stop boundary, 10% fraction constant, NaN/inf first-hit order.
//!
//! `should_stop`:
//! 1. predicted non-finite or `<= 0` → InvalidPredicted
//! 2. observed non-finite → InvalidObserved
//! 3. else `observed >= predicted * (1 + STOP_RELATIVE_EXCESS)`

mod reference;

use prometheus_control::scale::{should_stop, ScaleError, STOP_FRACTION, STOP_RELATIVE_EXCESS};
use reference::scale as ref_scale;

#[test]
fn boundary_equal_to_five_percent_excess_stops() {
    let predicted = 1.0;
    let observed = predicted * (1.0 + STOP_RELATIVE_EXCESS);
    assert_eq!(should_stop(predicted, observed), Ok(true));
    assert_eq!(ref_scale::should_stop(predicted, observed), Ok(true));

    let predicted = 20.0;
    let observed = predicted * (1.0 + 0.05);
    assert_eq!(should_stop(predicted, observed), Ok(true));
}

#[test]
fn just_below_five_percent_does_not_stop() {
    let predicted = 1.0;
    let threshold = predicted * (1.0 + STOP_RELATIVE_EXCESS);
    let observed = threshold.next_down();
    assert!(observed < threshold);
    assert_eq!(should_stop(predicted, observed), Ok(false));
    assert_eq!(ref_scale::should_stop(predicted, observed), Ok(false));
}

#[test]
fn exact_match_to_prediction_does_not_stop() {
    assert_eq!(should_stop(1.0, 1.0), Ok(false));
    assert_eq!(should_stop(2.5, 2.5), Ok(false));
    assert_eq!(ref_scale::should_stop(2.5, 2.5), Ok(false));
}

#[test]
fn well_above_five_percent_stops() {
    assert_eq!(should_stop(1.0, 1.06), Ok(true));
    assert_eq!(should_stop(100.0, 200.0), Ok(true));
    assert_eq!(ref_scale::should_stop(100.0, 200.0), Ok(true));
}

#[test]
fn four_percent_excess_does_not_stop() {
    assert_eq!(should_stop(1.0, 1.04), Ok(false));
    assert_eq!(ref_scale::should_stop(1.0, 1.04), Ok(false));
}

#[test]
fn invalid_predicted_wins_over_invalid_observed() {
    for predicted in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0, -1.0] {
        for observed in [f64::NAN, f64::INFINITY, 1.0, -1.0] {
            assert_eq!(
                should_stop(predicted, observed),
                Err(ScaleError::InvalidPredicted),
                "predicted={predicted} observed={observed}"
            );
            assert_eq!(
                should_stop(predicted, observed),
                ref_scale::should_stop(predicted, observed)
            );
        }
    }
}

#[test]
fn invalid_observed_when_predicted_is_usable() {
    for observed in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            should_stop(1.0, observed),
            Err(ScaleError::InvalidObserved),
            "observed={observed}"
        );
        assert_eq!(
            should_stop(1.0, observed),
            ref_scale::should_stop(1.0, observed)
        );
    }
}

#[test]
fn finite_negative_observed_is_not_an_error_and_does_not_stop() {
    // Only observed non-finite is InvalidObserved. Negative finite compares.
    assert_eq!(should_stop(1.0, -1.0), Ok(false));
    assert_eq!(should_stop(1.0, 0.0), Ok(false));
    assert_eq!(ref_scale::should_stop(1.0, -1.0), Ok(false));
}

#[test]
fn stop_fraction_is_not_an_argument_to_should_stop() {
    // Caller passes the loss at STOP_FRACTION of the token budget. The
    // predicate itself is only the 5% relative excess.
    let _ = STOP_FRACTION;
    assert_eq!(should_stop(10.0, 10.0), Ok(false));
    assert_eq!(
        should_stop(10.0, 10.0 * (1.0 + STOP_RELATIVE_EXCESS)),
        Ok(true)
    );
}
