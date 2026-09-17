//! Group: drop_source — remove one source, renormalize; missing / last fail.

mod common;
mod reference;

use common::{single_web, two_source, uniform8};
use prometheus_mixture::{drop_source, validate_mix, Source};

#[test]
fn drop_web_from_uniform8_matches_reference() {
    let base = uniform8();
    let got = drop_source(&base, Source::Web).expect("public");
    let want = reference::drop_source(&base, Source::Web).expect("reference");
    common::assert_mix_close(&got, &want);
    assert!(!got.weights.contains_key(&Source::Web));
    assert_eq!(got.weights.len(), 7);
    assert_eq!(got.phase, base.phase);
    assert_eq!(got.mix_id, base.mix_id);
    validate_mix(&got).expect("dropped mix is valid");
}

#[test]
fn drop_each_source_matches_reference() {
    let base = uniform8();
    for src in Source::all() {
        let got = drop_source(&base, src).expect("public");
        let want = reference::drop_source(&base, src).expect("reference");
        common::assert_mix_close(&got, &want);
        assert!(!got.weights.contains_key(&src));
    }
}

#[test]
fn remaining_weights_are_proportional() {
    let base = two_source();
    let got = drop_source(&base, Source::Code).expect("public");
    assert_eq!(got.weights.len(), 1);
    assert!((got.weights[&Source::Web] - 1.0).abs() < common::WEIGHT_TOL);
    common::assert_mix_close(&got, &reference::drop_source(&base, Source::Code).unwrap());
}

#[test]
fn missing_source_is_unknown_source() {
    let base = two_source();
    common::assert_unknown(drop_source(&base, Source::BooksPapers));
    common::assert_unknown(reference::drop_source(&base, Source::BooksPapers));
}

#[test]
fn last_source_is_config() {
    common::assert_config(drop_source(&single_web(), Source::Web));
    common::assert_config(reference::drop_source(&single_web(), Source::Web));
}

#[test]
fn last_source_missing_is_unknown_not_last() {
    common::assert_unknown(drop_source(&single_web(), Source::Code));
    common::assert_unknown(reference::drop_source(&single_web(), Source::Code));
}

#[test]
fn empty_id_is_config_when_source_present() {
    let mut bad = two_source();
    bad.mix_id.clear();
    common::assert_config(drop_source(&bad, Source::Web));
    common::assert_config(reference::drop_source(&bad, Source::Web));
}
