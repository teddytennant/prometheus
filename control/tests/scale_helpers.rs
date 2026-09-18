//! Group: associated constants, `flagship_spec`, `compute_flops`, Display.
//!
//! Constants, Display, and type-trait checks inspect frozen iface data only
//! and may pass against the I1 stub. `flagship_spec` and `compute_flops` call
//! unimplemented bodies and must fail on the stub.

mod reference;

use prometheus_control::scale::{
    compute_flops, flagship_spec, FlagshipSpec, RungObservation, ScaleError, ScalingFit,
    FLAGSHIP_ACTIVE_PARAMS, FLAGSHIP_GPUS, FLAGSHIP_TOKENS, FLAGSHIP_TOTAL_PARAMS, FLOP_COEFF,
    STOP_FRACTION, STOP_RELATIVE_EXCESS,
};
use reference::scale as ref_scale;

fn assert_debug_clone_partial_eq<T: std::fmt::Debug + Clone + PartialEq>() {}
fn assert_debug_partial_eq<T: std::fmt::Debug + PartialEq>() {}
fn assert_eq_trait<T: Eq>() {}
fn assert_error<T: std::error::Error>() {}
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn flagship_constants_match_spec_6_table() {
    assert_eq!(FLAGSHIP_ACTIVE_PARAMS, 400_000_000_000);
    assert_eq!(FLAGSHIP_TOTAL_PARAMS, 7_400_000_000_000);
    assert_eq!(FLAGSHIP_TOKENS, 180_000_000_000_000);
    assert_eq!(FLAGSHIP_GPUS, 100_000);
    assert_eq!(FLAGSHIP_ACTIVE_PARAMS, ref_scale::FLAGSHIP_ACTIVE_PARAMS);
    assert_eq!(FLAGSHIP_TOTAL_PARAMS, ref_scale::FLAGSHIP_TOTAL_PARAMS);
    assert_eq!(FLAGSHIP_TOKENS, ref_scale::FLAGSHIP_TOKENS);
    assert_eq!(FLAGSHIP_GPUS, ref_scale::FLAGSHIP_GPUS);
}

#[test]
fn flop_coeff_is_six() {
    assert_eq!(FLOP_COEFF, 6.0);
    assert_eq!(FLOP_COEFF, ref_scale::FLOP_COEFF);
}

#[test]
fn stop_relative_excess_is_five_percent() {
    assert_eq!(STOP_RELATIVE_EXCESS, 0.05);
    assert_eq!(STOP_RELATIVE_EXCESS, ref_scale::STOP_RELATIVE_EXCESS);
}

#[test]
fn stop_fraction_is_ten_percent_of_training() {
    assert_eq!(STOP_FRACTION, 0.10);
    assert_eq!(STOP_FRACTION, ref_scale::STOP_FRACTION);
}

#[test]
fn scale_error_display_strings_match_iface() {
    assert_eq!(
        ScaleError::Rung0NotUsed.to_string(),
        "rung 0 is not a scaling-law point"
    );
    assert_eq!(
        ScaleError::NeedTwoRungs.to_string(),
        "need at least two rungs from 1 to 3"
    );
    assert_eq!(
        ScaleError::DuplicateRung.to_string(),
        "duplicate rung in the fit"
    );
    assert_eq!(ScaleError::NonFiniteLoss.to_string(), "loss must be finite");
    assert_eq!(ScaleError::NonPositiveLoss.to_string(), "loss must be > 0");
    assert_eq!(
        ScaleError::InvalidCompute.to_string(),
        "active_params and tokens must be > 0"
    );
    assert_eq!(
        ScaleError::DegenerateFit.to_string(),
        "rungs do not determine a unique (A, alpha)"
    );
    assert_eq!(
        ScaleError::HeldOutInTrain.to_string(),
        "held-out rung was in the train set"
    );
    assert_eq!(
        ScaleError::NonFinitePrediction.to_string(),
        "prediction must be finite"
    );
    assert_eq!(
        ScaleError::InvalidPredicted.to_string(),
        "predicted loss must be finite and > 0"
    );
    assert_eq!(
        ScaleError::InvalidObserved.to_string(),
        "observed loss must be finite"
    );
}

#[test]
fn type_traits_on_public_surface() {
    assert_debug_clone_partial_eq::<FlagshipSpec>();
    assert_eq_trait::<FlagshipSpec>();
    assert_debug_clone_partial_eq::<RungObservation>();
    assert_debug_clone_partial_eq::<ScalingFit>();
    assert_debug_partial_eq::<ScaleError>();
    assert_eq_trait::<ScaleError>();
    assert_error::<ScaleError>();
    assert_send_sync::<FlagshipSpec>();
    assert_send_sync::<ScaleError>();
    assert_send_sync::<RungObservation>();
}

#[test]
fn flagship_spec_matches_spec_6_literals() {
    let s = flagship_spec();
    assert_eq!(s.active_params, 400_000_000_000);
    assert_eq!(s.total_params, 7_400_000_000_000);
    assert_eq!(s.tokens, 180_000_000_000_000);
    assert_eq!(s.gpus, 100_000);
    assert_eq!(s.active_params, FLAGSHIP_ACTIVE_PARAMS);
    assert_eq!(s.total_params, FLAGSHIP_TOTAL_PARAMS);
    assert_eq!(s.tokens, FLAGSHIP_TOKENS);
    assert_eq!(s.gpus, FLAGSHIP_GPUS);
    assert_eq!(s, ref_scale::flagship_spec());
}

#[test]
fn compute_flops_rejects_either_arg_zero() {
    assert_eq!(compute_flops(0, 1), Err(ScaleError::InvalidCompute));
    assert_eq!(compute_flops(1, 0), Err(ScaleError::InvalidCompute));
    assert_eq!(compute_flops(0, 0), Err(ScaleError::InvalidCompute));
    assert_eq!(
        compute_flops(0, FLAGSHIP_TOKENS),
        Err(ScaleError::InvalidCompute)
    );
    assert_eq!(
        compute_flops(FLAGSHIP_ACTIVE_PARAMS, 0),
        Err(ScaleError::InvalidCompute)
    );
    assert_eq!(compute_flops(0, 1), ref_scale::compute_flops(0, 1));
}

#[test]
fn compute_flops_tiny_values_are_six_n_d() {
    assert_eq!(compute_flops(1, 1).unwrap(), 6.0);
    assert_eq!(compute_flops(2, 3).unwrap(), 36.0);
    assert_eq!(compute_flops(10, 20).unwrap(), 1_200.0);
    assert_eq!(
        compute_flops(1, 1).unwrap(),
        ref_scale::compute_flops(1, 1).unwrap()
    );
    assert_eq!(
        compute_flops(2, 3).unwrap(),
        ref_scale::compute_flops(2, 3).unwrap()
    );
}

#[test]
fn compute_flops_casts_n_and_d_to_f64_before_multiply() {
    // 2^40 * 2^40 overflows u64 (wraps to 0). The f64 product is exact.
    let n = 1u64 << 40;
    let d = 1u64 << 40;
    let got = compute_flops(n, d).expect("fits in f64");
    let expect = 6.0 * (n as f64) * (d as f64);
    assert_eq!(got, expect);
    assert_eq!(got, ref_scale::compute_flops(n, d).unwrap());
    assert!(got > 0.0 && got.is_finite());
    // Integer wrap would have produced 0.
    assert_ne!(n.wrapping_mul(d), n.saturating_mul(d).max(1));
    assert_eq!(n.wrapping_mul(d), 0);
}

#[test]
fn compute_flops_flagship_does_not_use_u64_product() {
    let got = compute_flops(FLAGSHIP_ACTIVE_PARAMS, FLAGSHIP_TOKENS).expect("flagship C fits f64");
    let expect = 6.0 * (400_000_000_000f64) * (180_000_000_000_000f64);
    assert_eq!(got, expect);
    assert_eq!(
        got,
        ref_scale::compute_flops(FLAGSHIP_ACTIVE_PARAMS, FLAGSHIP_TOKENS).unwrap()
    );
    assert!(got.is_finite() && got > 0.0);
    // 0.4T * 180T overflows u64; a checked integer mul must not become InvalidCompute.
    assert!(FLAGSHIP_ACTIVE_PARAMS
        .checked_mul(FLAGSHIP_TOKENS)
        .is_none());
}
