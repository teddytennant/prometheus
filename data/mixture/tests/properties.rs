//! Group: properties — hash stability, decay share, drop-one sum, sample CDF.

mod common;
mod reference;

use common::{rung2_like, two_source, uniform8};
use prometheus_mixture::{
    decay_reweight, drop_source, mix_hash, sample_source, validate_mix, Mix, Phase, Source,
    WEIGHT_SUM_TOL, DECAY_SOURCES,
};
use std::collections::BTreeMap;

#[test]
fn mix_hash_is_stable_across_calls() {
    let mix = rung2_like();
    let a = mix_hash(&mix).expect("a");
    let b = mix_hash(&mix).expect("b");
    assert_eq!(a, b);
    assert_eq!(a, reference::mix_hash(&mix).expect("ref"));
    common::assert_sha256_hex(&a);
}

#[test]
fn mix_hash_changes_when_any_weight_changes() {
    let mut other = two_source();
    other.weights.insert(Source::Web, 0.74);
    other.weights.insert(Source::Code, 0.26);
    assert_ne!(
        mix_hash(&two_source()).expect("a"),
        mix_hash(&other).expect("b")
    );
}

#[test]
fn decay_increases_decay_sources_share() {
    for factor in [1.000001, 1.5, 2.0, 5.0] {
        for base in [uniform8(), rung2_like()] {
            let got = decay_reweight(&base, factor).expect("decay");
            let want = reference::decay_reweight(&base, factor).expect("ref");
            common::assert_mix_close(&got, &want);
            let before = common::decay_share(&base);
            let after = common::decay_share(&got);
            assert!(
                after > before,
                "factor={factor} share {before} -> {after}"
            );
            let sum: f64 = got.weights.values().copied().sum();
            assert!((sum - 1.0).abs() <= WEIGHT_SUM_TOL);
            validate_mix(&got).expect("valid");
        }
    }
}

#[test]
fn drop_one_weights_still_sum_to_one() {
    let base = uniform8();
    for src in Source::all() {
        let got = drop_source(&base, src).expect("drop");
        let want = reference::drop_source(&base, src).expect("ref");
        common::assert_mix_close(&got, &want);
        let sum: f64 = got.weights.values().copied().sum();
        assert!(
            (sum - 1.0).abs() <= WEIGHT_SUM_TOL,
            "after dropping {src:?} sum={sum}"
        );
        validate_mix(&got).expect("valid");
    }
}

#[test]
fn sample_source_matches_cdf_over_unit_interval() {
    let mix = uniform8();
    let n = 400;
    let mut counts = BTreeMap::<Source, usize>::new();
    for i in 0..n {
        let u = i as f64 / n as f64; // [0, 1)
        let got = sample_source(&mix, u).expect("public");
        let want = reference::sample_source(&mix, u).expect("ref");
        assert_eq!(got, want, "u={u}");
        *counts.entry(got).or_insert(0) += 1;
    }
    // Uniform 8: each source owns 1/8 of [0,1). 400/8 = 50.
    for src in Source::all() {
        assert_eq!(counts[&src], n / 8, "{src:?}");
    }
}

#[test]
fn sample_source_cdf_is_monotone_in_btree_order() {
    let mix = rung2_like();
    let mut last_idx = 0usize;
    let order: Vec<Source> = mix.weights.keys().copied().collect();
    for i in 0..200 {
        let u = i as f64 / 200.0;
        let src = sample_source(&mix, u).expect("public");
        assert_eq!(src, reference::sample_source(&mix, u).expect("ref"));
        let idx = order.iter().position(|s| *s == src).expect("in mix");
        assert!(idx >= last_idx, "u={u} walked backwards {last_idx} -> {idx}");
        last_idx = idx;
    }
}

#[test]
fn decay_then_hash_matches_reference_hash() {
    let got = decay_reweight(&uniform8(), 2.0).expect("decay");
    let want = reference::decay_reweight(&uniform8(), 2.0).expect("ref decay");
    assert_eq!(mix_hash(&got).expect("public hash"), mix_hash(&want).expect("hash want"));
    assert_eq!(
        mix_hash(&got).expect("public hash"),
        reference::mix_hash(&got).expect("ref hash")
    );
}

#[test]
fn drop_preserves_relative_ratios() {
    let base = Mix {
        mix_id: "r".into(),
        mix_bucket: "b".into(),
        phase: Phase::Pretrain,
        weights: [(Source::Web, 0.2), (Source::Code, 0.3), (Source::MathScienceArxiv, 0.5)]
            .into_iter()
            .collect(),
    };
    let got = drop_source(&base, Source::Web).expect("drop");
    let code = got.weights[&Source::Code];
    let math = got.weights[&Source::MathScienceArxiv];
    assert!((code / math - 0.3 / 0.5).abs() < 1e-12);
    assert!(DECAY_SOURCES.contains(&Source::Code));
}
