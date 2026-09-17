//! Group: sample_source — CDF walk in BTreeMap order, u in [0, 1).

mod common;
mod reference;

use common::{two_source, uniform8};
use prometheus_mixture::{sample_source, Source};

#[test]
fn u_zero_is_first_btree_source() {
    let mix = two_source();
    assert_eq!(sample_source(&mix, 0.0).expect("public"), Source::Web);
    assert_eq!(
        reference::sample_source(&mix, 0.0).expect("ref"),
        Source::Web
    );
}

#[test]
fn two_source_buckets() {
    let mix = two_source();
    // web=0.75, code=0.25. [0, 0.75) -> web; [0.75, 1) -> code.
    for u in [0.0, 0.1, 0.749999] {
        assert_eq!(
            sample_source(&mix, u).expect("public"),
            Source::Web,
            "u={u}"
        );
        assert_eq!(
            reference::sample_source(&mix, u).expect("ref"),
            Source::Web,
            "ref u={u}"
        );
    }
    for u in [0.75, 0.9, 0.999999] {
        assert_eq!(
            sample_source(&mix, u).expect("public"),
            Source::Code,
            "u={u}"
        );
        assert_eq!(
            reference::sample_source(&mix, u).expect("ref"),
            Source::Code,
            "ref u={u}"
        );
    }
}

#[test]
fn uniform8_boundaries_match_reference() {
    let mix = uniform8();
    let edges = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875];
    for (i, u) in edges.into_iter().enumerate() {
        let got = sample_source(&mix, u).expect("public");
        let want = reference::sample_source(&mix, u).expect("ref");
        assert_eq!(got, want, "u={u}");
        assert_eq!(got, Source::all()[i], "left edge of bucket {i}");
    }
    let last = sample_source(&mix, 0.999999999).expect("last");
    assert_eq!(last, Source::AgenticTrajectories);
    assert_eq!(last, reference::sample_source(&mix, 0.999999999).unwrap());
}

#[test]
fn u_one_is_config() {
    common::assert_config(sample_source(&two_source(), 1.0));
    common::assert_config(reference::sample_source(&two_source(), 1.0));
}

#[test]
fn u_negative_or_nonfinite_is_config() {
    common::assert_config(sample_source(&two_source(), -1e-12));
    common::assert_config(sample_source(&two_source(), f64::NAN));
    common::assert_config(sample_source(&two_source(), f64::INFINITY));
    common::assert_config(reference::sample_source(&two_source(), -0.1));
    common::assert_config(reference::sample_source(&two_source(), f64::NAN));
}

#[test]
fn invalid_mix_is_config() {
    let mut bad = two_source();
    bad.mix_id.clear();
    common::assert_config(sample_source(&bad, 0.3));
    common::assert_config(reference::sample_source(&bad, 0.3));
}
