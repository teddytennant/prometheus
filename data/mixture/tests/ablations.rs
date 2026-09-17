//! Group: rung2_ablations — one drop-one mix per source in base (spec 6).

mod common;
mod reference;

use common::{single_web, two_source, uniform8};
use prometheus_mixture::{rung2_ablations, Source};

#[test]
fn uniform8_has_eight_ablations_matching_reference() {
    let base = uniform8();
    let got = rung2_ablations(&base).expect("public");
    let want = reference::rung2_ablations(&base).expect("reference");
    assert_eq!(got.len(), 8);
    assert_eq!(want.len(), 8);
    for (g, w) in got.iter().zip(want.iter()) {
        common::assert_mix_close(g, w);
        assert_eq!(g.weights.len(), 7);
    }
}

#[test]
fn ablations_are_btree_order_drop_one() {
    let base = uniform8();
    let got = rung2_ablations(&base).expect("public");
    let dropped: Vec<Source> = Source::all()
        .into_iter()
        .zip(got.iter())
        .map(|(src, mix)| {
            assert!(
                !mix.weights.contains_key(&src),
                "ablation should drop {src:?}"
            );
            for other in Source::all() {
                if other != src {
                    assert!(mix.weights.contains_key(&other), "kept {other:?}");
                }
            }
            src
        })
        .collect();
    assert_eq!(dropped, Source::all().to_vec());
}

#[test]
fn two_source_ablations_match_reference() {
    let base = two_source();
    let got = rung2_ablations(&base).expect("public");
    let want = reference::rung2_ablations(&base).expect("reference");
    assert_eq!(got.len(), 2);
    for (g, w) in got.iter().zip(want.iter()) {
        common::assert_mix_close(g, w);
        assert_eq!(g.weights.len(), 1);
    }
}

#[test]
fn single_source_cannot_ablate() {
    common::assert_config(rung2_ablations(&single_web()));
    common::assert_config(reference::rung2_ablations(&single_web()));
}

#[test]
fn invalid_base_is_config() {
    let mut bad = uniform8();
    bad.mix_bucket.clear();
    common::assert_config(rung2_ablations(&bad));
    common::assert_config(reference::rung2_ablations(&bad));
}
