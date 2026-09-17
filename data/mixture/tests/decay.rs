//! Group: decay_reweight — boost DECAY_SOURCES, renormalize, phase=Decay.

mod common;
mod reference;

use common::{rung2_like, two_source, uniform8};
use prometheus_mixture::{decay_reweight, validate_mix, Phase, Source, DECAY_SOURCES};

#[test]
fn uniform8_factor_two_matches_reference() {
    let base = uniform8();
    let got = decay_reweight(&base, 2.0).expect("public");
    let want = reference::decay_reweight(&base, 2.0).expect("reference");
    common::assert_mix_close(&got, &want);
    assert_eq!(got.phase, Phase::Decay);
    assert_eq!(got.mix_id, base.mix_id);
    assert_eq!(got.mix_bucket, base.mix_bucket);
    validate_mix(&got).expect("decayed mix is valid");
}

#[test]
fn rung2_factor_matches_reference() {
    let base = rung2_like();
    let got = decay_reweight(&base, 1.5).expect("public");
    let want = reference::decay_reweight(&base, 1.5).expect("reference");
    common::assert_mix_close(&got, &want);
    assert_eq!(got.phase, Phase::Decay);
}

#[test]
fn decay_sources_are_boosted_others_shrink() {
    let base = uniform8();
    let before = common::decay_share(&base);
    let got = decay_reweight(&base, 3.0).expect("public");
    let after = common::decay_share(&got);
    assert!(after > before, "decay share {after} should exceed {before}");
    for src in DECAY_SOURCES {
        assert!(
            got.weights[&src] > base.weights[&src],
            "{src:?} should increase"
        );
    }
    for src in Source::all() {
        if DECAY_SOURCES.contains(&src) {
            continue;
        }
        assert!(
            got.weights[&src] < base.weights[&src],
            "{src:?} should shrink after renormalize"
        );
    }
}

#[test]
fn factor_one_is_config() {
    common::assert_config(decay_reweight(&uniform8(), 1.0));
    common::assert_config(reference::decay_reweight(&uniform8(), 1.0));
}

#[test]
fn factor_below_one_is_config() {
    common::assert_config(decay_reweight(&uniform8(), 0.5));
    common::assert_config(reference::decay_reweight(&two_source(), 0.0));
}

#[test]
fn factor_nan_or_inf_is_config() {
    common::assert_config(decay_reweight(&uniform8(), f64::NAN));
    common::assert_config(decay_reweight(&uniform8(), f64::INFINITY));
    common::assert_config(decay_reweight(&uniform8(), f64::NEG_INFINITY));
    common::assert_config(reference::decay_reweight(&uniform8(), f64::NAN));
    common::assert_config(reference::decay_reweight(&uniform8(), f64::INFINITY));
}

#[test]
fn invalid_input_mix_is_config() {
    let mut bad = two_source();
    bad.mix_id.clear();
    common::assert_config(decay_reweight(&bad, 2.0));
    common::assert_config(reference::decay_reweight(&bad, 2.0));
}

#[test]
fn missing_decay_source_is_still_ok() {
    // two_source has code (decay) and web (not). math/reasoning absent.
    let base = two_source();
    let got = decay_reweight(&base, 2.0).expect("public");
    let want = reference::decay_reweight(&base, 2.0).expect("reference");
    common::assert_mix_close(&got, &want);
    assert!(got.weights[&Source::Code] > got.weights[&Source::Web]);
    assert_eq!(got.weights.len(), 2);
}

#[test]
fn already_decay_phase_still_reweights() {
    let mut base = uniform8();
    base.phase = Phase::Decay;
    let got = decay_reweight(&base, 2.0).expect("public");
    let want = reference::decay_reweight(&base, 2.0).expect("reference");
    common::assert_mix_close(&got, &want);
    assert_eq!(got.phase, Phase::Decay);
}
