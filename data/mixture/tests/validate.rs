//! Group: validate_mix — empty id/bucket, non-finite / non-positive weights, sum.

mod common;
mod reference;

use common::{mix, single_web, two_source, uniform8};
use prometheus_mixture::{validate_mix, Mix, Phase, Source, WEIGHT_SUM_TOL};
use std::collections::BTreeMap;

#[test]
fn uniform8_is_valid_and_matches_reference() {
    let mix = uniform8();
    validate_mix(&mix).expect("public");
    reference::validate_mix(&mix).expect("reference");
}

#[test]
fn two_source_and_single_source_are_valid() {
    validate_mix(&two_source()).expect("two");
    validate_mix(&single_web()).expect("one");
    reference::validate_mix(&two_source()).expect("ref two");
    reference::validate_mix(&single_web()).expect("ref one");
}

#[test]
fn empty_mix_id_is_config() {
    let mut mix = two_source();
    mix.mix_id.clear();
    common::assert_config(validate_mix(&mix));
    common::assert_config(reference::validate_mix(&mix));
}

#[test]
fn empty_mix_bucket_is_config() {
    let mut mix = two_source();
    mix.mix_bucket.clear();
    common::assert_config(validate_mix(&mix));
    common::assert_config(reference::validate_mix(&mix));
}

#[test]
fn empty_weights_is_config() {
    let mix = Mix {
        mix_id: "m".into(),
        mix_bucket: "b".into(),
        phase: Phase::Pretrain,
        weights: BTreeMap::new(),
    };
    common::assert_config(validate_mix(&mix));
    common::assert_config(reference::validate_mix(&mix));
}

#[test]
fn zero_weight_is_config() {
    let mix = mix(
        "m",
        "b",
        Phase::Pretrain,
        &[(Source::Web, 1.0), (Source::Code, 0.0)],
    );
    common::assert_config(validate_mix(&mix));
    common::assert_config(reference::validate_mix(&mix));
}

#[test]
fn negative_weight_is_config() {
    let mix = mix(
        "m",
        "b",
        Phase::Pretrain,
        &[(Source::Web, 1.5), (Source::Code, -0.5)],
    );
    common::assert_config(validate_mix(&mix));
    common::assert_config(reference::validate_mix(&mix));
}

#[test]
fn nan_weight_is_config() {
    let mix = mix(
        "m",
        "b",
        Phase::Pretrain,
        &[(Source::Web, f64::NAN), (Source::Code, 1.0)],
    );
    common::assert_config(validate_mix(&mix));
    common::assert_config(reference::validate_mix(&mix));
}

#[test]
fn inf_weight_is_config() {
    let mix = mix(
        "m",
        "b",
        Phase::Pretrain,
        &[(Source::Web, f64::INFINITY), (Source::Code, 1.0)],
    );
    common::assert_config(validate_mix(&mix));
    common::assert_config(reference::validate_mix(&mix));
}

#[test]
fn sum_far_from_one_is_config() {
    let mix = mix(
        "m",
        "b",
        Phase::Pretrain,
        &[(Source::Web, 0.5), (Source::Code, 0.5 + 1e-6)],
    );
    common::assert_config(validate_mix(&mix));
    common::assert_config(reference::validate_mix(&mix));
}

#[test]
fn sum_within_weight_sum_tol_is_ok() {
    // 1e-10 is inside WEIGHT_SUM_TOL = 1e-9.
    let mix = mix(
        "m",
        "b",
        Phase::Pretrain,
        &[(Source::Web, 0.5), (Source::Code, 0.5 + 1e-10)],
    );
    let delta = mix.weights.values().sum::<f64>() - 1.0;
    assert!(delta.abs() <= WEIGHT_SUM_TOL);
    validate_mix(&mix).expect("public within tol");
    reference::validate_mix(&mix).expect("ref within tol");
}

#[test]
fn thirds_sum_is_within_tol() {
    let mix = mix(
        "thirds",
        "b",
        Phase::Pretrain,
        &[
            (Source::Web, 1.0 / 3.0),
            (Source::Code, 1.0 / 3.0),
            (Source::MathScienceArxiv, 1.0 / 3.0),
        ],
    );
    validate_mix(&mix).expect("public thirds");
    reference::validate_mix(&mix).expect("ref thirds");
}
