//! Group: two-point exact, three-point OLS, duplicates, rung 0, degenerate.
//!
//! `fit` first-hit order:
//! validate each observation in slice order, then DuplicateRung, then
//! NeedTwoRungs, then DegenerateFit.

mod reference;

use prometheus_control::rung::{RungId, RungSpec};
use prometheus_control::scale::{
    fit, ScaleError, ScalingFit, FLAGSHIP_ACTIVE_PARAMS, FLAGSHIP_TOKENS,
};
use reference::scale as ref_scale;

fn obs(
    id: RungId,
    active: u64,
    tokens: u64,
    loss: f64,
) -> prometheus_control::scale::RungObservation {
    ref_scale::observation(id, active, tokens, loss)
}

fn assert_fit_matches(
    observations: &[prometheus_control::scale::RungObservation],
    a: f64,
    alpha: f64,
    tol: f64,
) {
    let prod = fit(observations).expect("prod fit");
    let refer = ref_scale::fit(observations).expect("ref fit");
    ref_scale::assert_rel_close(prod.a(), refer.a(), tol, "A vs ref");
    ref_scale::assert_rel_close(prod.alpha(), refer.alpha(), tol, "alpha vs ref");
    ref_scale::assert_rel_close(prod.a(), a, tol, "A vs expected");
    ref_scale::assert_rel_close(prod.alpha(), alpha, tol, "alpha vs expected");
}

#[test]
fn two_point_exact_recovers_a_three_alpha_one() {
    // L = 3 / C, C = 6 N D. Exact in f64: C=6,12 → L=0.5, 0.25.
    let train = [obs(RungId::One, 1, 1, 0.5), obs(RungId::Two, 2, 1, 0.25)];
    assert_fit_matches(&train, 3.0, 1.0, 1e-9);

    let swapped = [obs(RungId::Two, 2, 1, 0.25), obs(RungId::One, 1, 1, 0.5)];
    assert_fit_matches(&swapped, 3.0, 1.0, 1e-9);
}

#[test]
fn two_point_closed_form_matches_ols() {
    let l1 = 0.5_f64;
    let l2 = 0.25_f64;
    let c1 = 6.0_f64;
    let c2 = 12.0_f64;
    let alpha = (l1.ln() - l2.ln()) / (c2.ln() - c1.ln());
    let a = l1 * c1.powf(alpha);
    let train = [obs(RungId::One, 1, 1, l1), obs(RungId::Three, 2, 1, l2)];
    assert_fit_matches(&train, a, alpha, 1e-12);
}

#[test]
fn three_point_colinear_ols_recovers_the_law() {
    let train = [
        obs(RungId::One, 1, 1, 0.5),
        obs(RungId::Two, 2, 1, 0.25),
        obs(RungId::Three, 4, 1, 0.125),
    ];
    assert_fit_matches(&train, 3.0, 1.0, 1e-9);
}

#[test]
fn three_point_ols_uses_all_rungs_not_the_first_two() {
    // Not on L = 3/C. OLS must move off the two-point (1,2) solution.
    let train = [
        obs(RungId::One, 1, 1, 2.0),
        obs(RungId::Two, 2, 1, 1.0),
        obs(RungId::Three, 4, 1, 1.0),
    ];
    let prod = fit(&train).expect("prod");
    let refer = ref_scale::fit(&train).expect("ref");
    ref_scale::assert_rel_close(prod.a(), refer.a(), 1e-12, "A");
    ref_scale::assert_rel_close(prod.alpha(), refer.alpha(), 1e-12, "alpha");

    let two = ref_scale::fit(&[train[0].clone(), train[1].clone()]).expect("two-point");
    assert!(
        ref_scale::rel_err(prod.a(), two.a()) > 1e-6
            || ref_scale::rel_err(prod.alpha(), two.alpha()) > 1e-6,
        "three-point OLS must not equal the first-two interpolation: prod A={} alpha={} two A={} alpha={}",
        prod.a(),
        prod.alpha(),
        two.a(),
        two.alpha()
    );

    // Hand OLS: mean ln C = ln 12, slope = -1/2, alpha = 1/2.
    ref_scale::assert_rel_close(prod.alpha(), 0.5, 1e-12, "hand alpha");
    let b0 = 2.0_f64.ln() / 3.0 + 12.0_f64.ln() / 2.0;
    ref_scale::assert_rel_close(prod.a(), b0.exp(), 1e-12, "hand A");
}

#[test]
fn fit_empty_and_single_are_need_two_rungs() {
    assert_eq!(fit(&[]), Err(ScaleError::NeedTwoRungs));
    assert_eq!(ref_scale::fit(&[]), Err(ScaleError::NeedTwoRungs));

    let one = [obs(RungId::One, 1, 1, 0.5)];
    assert_eq!(fit(&one), Err(ScaleError::NeedTwoRungs));
    assert_eq!(ref_scale::fit(&one), Err(ScaleError::NeedTwoRungs));
}

#[test]
fn duplicate_rung_wins_over_need_two_and_over_a_unique_fit() {
    let dup_pair = [obs(RungId::One, 1, 1, 0.5), obs(RungId::One, 2, 1, 0.25)];
    assert_eq!(fit(&dup_pair), Err(ScaleError::DuplicateRung));
    assert_eq!(ref_scale::fit(&dup_pair), Err(ScaleError::DuplicateRung));

    let dup_in_three = [
        obs(RungId::One, 1, 1, 0.5),
        obs(RungId::Two, 2, 1, 0.25),
        obs(RungId::One, 4, 1, 0.125),
    ];
    assert_eq!(fit(&dup_in_three), Err(ScaleError::DuplicateRung));
    assert_eq!(
        ref_scale::fit(&dup_in_three),
        Err(ScaleError::DuplicateRung)
    );
}

#[test]
fn rung_zero_is_never_a_fit_point() {
    let with_zero = [obs(RungId::Zero, 1, 1, 0.5), obs(RungId::One, 2, 1, 0.25)];
    assert_eq!(fit(&with_zero), Err(ScaleError::Rung0NotUsed));

    let zero_second = [obs(RungId::One, 1, 1, 0.5), obs(RungId::Zero, 2, 1, 0.25)];
    assert_eq!(fit(&zero_second), Err(ScaleError::Rung0NotUsed));
}

#[test]
fn fit_validates_in_slice_order_before_duplicate() {
    let nan_on_repeat = [
        obs(RungId::One, 1, 1, 0.5),
        obs(RungId::One, 2, 1, f64::NAN),
    ];
    assert_eq!(fit(&nan_on_repeat), Err(ScaleError::NonFiniteLoss));
    assert_eq!(
        ref_scale::fit(&nan_on_repeat),
        Err(ScaleError::NonFiniteLoss)
    );

    let zero_compute_first = [obs(RungId::Two, 0, 1, 0.5), obs(RungId::Two, 1, 1, 0.25)];
    assert_eq!(fit(&zero_compute_first), Err(ScaleError::InvalidCompute));
}

#[test]
fn degenerate_when_c_values_are_not_all_distinct() {
    let same_nd = [obs(RungId::One, 3, 5, 0.5), obs(RungId::Two, 3, 5, 0.4)];
    assert_eq!(fit(&same_nd), Err(ScaleError::DegenerateFit));
    assert_eq!(ref_scale::fit(&same_nd), Err(ScaleError::DegenerateFit));

    let same_product = [obs(RungId::One, 2, 6, 0.5), obs(RungId::Three, 4, 3, 0.4)];
    assert_eq!(fit(&same_product), Err(ScaleError::DegenerateFit));

    let two_of_three_share_c = [
        obs(RungId::One, 1, 4, 0.5),
        obs(RungId::Two, 2, 2, 0.4),
        obs(RungId::Three, 8, 1, 0.3),
    ];
    assert_eq!(fit(&two_of_three_share_c), Err(ScaleError::DegenerateFit));
}

#[test]
fn degenerate_when_f64_compute_collides() {
    // 2^53 and 2^53+1 are the same f64, so C is not distinct as f64.
    let n = 1u64 << 53;
    let train = [obs(RungId::One, n, 1, 0.5), obs(RungId::Two, n + 1, 1, 0.4)];
    assert_eq!((n as f64), ((n + 1) as f64));
    assert_eq!(fit(&train), Err(ScaleError::DegenerateFit));
    assert_eq!(ref_scale::fit(&train), Err(ScaleError::DegenerateFit));
}

#[test]
fn total_params_and_gpus_do_not_enter_the_fit() {
    let mut a = obs(RungId::One, 1, 1, 0.5);
    let mut b = obs(RungId::Two, 2, 1, 0.25);
    a.spec.total_params = 1;
    a.spec.gpus = 1;
    b.spec.total_params = 10_000;
    b.spec.gpus = 99;
    let prod = fit(&[a, b]).expect("fit");
    ref_scale::assert_rel_close(prod.a(), 3.0, 1e-9, "A");
    ref_scale::assert_rel_close(prod.alpha(), 1.0, 1e-9, "alpha");
}

#[test]
fn predict_rejects_zero_n_or_d() {
    let train = [obs(RungId::One, 1, 1, 0.5), obs(RungId::Two, 2, 1, 0.25)];
    let prod = fit(&train).unwrap();
    assert_eq!(prod.predict(0, 1), Err(ScaleError::InvalidCompute));
    assert_eq!(prod.predict(1, 0), Err(ScaleError::InvalidCompute));
    assert_eq!(prod.predict(0, 0), Err(ScaleError::InvalidCompute));
}

#[test]
fn predict_spec_rejects_rung_zero_even_with_positive_nd() {
    let train = [obs(RungId::One, 1, 1, 0.5), obs(RungId::Two, 2, 1, 0.25)];
    let prod = fit(&train).unwrap();
    let zero = RungSpec {
        id: RungId::Zero,
        active_params: 1,
        total_params: 1,
        tokens: 1,
        gpus: 1,
    };
    assert_eq!(prod.predict_spec(&zero), Err(ScaleError::Rung0NotUsed));
    let refer = ref_scale::fit(&train).unwrap();
    assert_eq!(refer.predict_spec(&zero), Err(ScaleError::Rung0NotUsed));

    let zero_and_zero_n = RungSpec {
        id: RungId::Zero,
        active_params: 0,
        total_params: 0,
        tokens: 0,
        gpus: 0,
    };
    assert_eq!(
        prod.predict_spec(&zero_and_zero_n),
        Err(ScaleError::Rung0NotUsed)
    );
}

#[test]
fn predict_spec_invalid_compute_on_zero_n_for_nonzero_rung() {
    let train = [obs(RungId::One, 1, 1, 0.5), obs(RungId::Two, 2, 1, 0.25)];
    let prod = fit(&train).unwrap();
    let bad = RungSpec {
        id: RungId::Three,
        active_params: 0,
        total_params: 1,
        tokens: 4,
        gpus: 1,
    };
    assert_eq!(prod.predict_spec(&bad), Err(ScaleError::InvalidCompute));
}

#[test]
fn predict_matches_a_times_c_to_the_minus_alpha() {
    let train = [obs(RungId::One, 1, 1, 0.5), obs(RungId::Two, 2, 1, 0.25)];
    let prod = fit(&train).unwrap();
    let got = prod.predict(4, 1).unwrap();
    let c: f64 = 6.0 * 4.0 * 1.0;
    let expect = prod.a() * c.powf(-prod.alpha());
    ref_scale::assert_rel_close(got, expect, 1e-12, "predict");
    ref_scale::assert_rel_close(got, 0.125, 1e-9, "L(24)");
    assert_eq!(
        prod.predict_spec(&obs(RungId::Three, 4, 1, 0.125).spec)
            .unwrap(),
        got
    );
}

#[test]
fn predict_flagship_equals_predict_of_flagship_nd() {
    let train = [obs(RungId::One, 1, 1, 0.5), obs(RungId::Two, 2, 1, 0.25)];
    let prod = fit(&train).unwrap();
    let via_flagship = prod.predict_flagship().expect("flagship predict");
    let via_nd = prod
        .predict(FLAGSHIP_ACTIVE_PARAMS, FLAGSHIP_TOKENS)
        .expect("nd predict");
    assert_eq!(via_flagship, via_nd);
    let refer = ref_scale::fit(&train).unwrap();
    ref_scale::assert_rel_close(
        via_flagship,
        refer.predict_flagship().unwrap(),
        1e-12,
        "flagship vs ref",
    );
}

#[test]
fn predict_non_finite_power_is_non_finite_prediction() {
    // Rising loss with C → large A, large negative alpha; flagship C overflows.
    let train = [obs(RungId::One, 1, 1, 1.0), obs(RungId::Two, 2, 1, 1e100)];
    let prod = fit(&train).unwrap();
    assert_eq!(
        prod.predict_flagship(),
        Err(ScaleError::NonFinitePrediction)
    );
    assert_eq!(
        ref_scale::fit(&train).unwrap().predict_flagship(),
        Err(ScaleError::NonFinitePrediction)
    );
}

#[test]
fn scaling_fit_serde_exposes_a_and_alpha() {
    let fit_json = r#"{"a":3.0,"alpha":1.0}"#;
    let loaded: ScalingFit = serde_json::from_str(fit_json).unwrap();
    ref_scale::assert_rel_close(loaded.a(), 3.0, 0.0, "serde A");
    ref_scale::assert_rel_close(loaded.alpha(), 1.0, 0.0, "serde alpha");
}
