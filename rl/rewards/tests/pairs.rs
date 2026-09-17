//! Group: TamperPairFilter and RescoreFilter.

mod common;
mod reference;

use common::{
    clean_view, hidden_pass_visible_fail_view, pair, special_case_view, test_write_view,
    visible_pass_hidden_fail_view,
};
use prometheus_rewards::{Error, RescoreFilter, TamperPairFilter, TrajectoryView};

fn assert_keep_matches(pos: &TrajectoryView, neg: &TrajectoryView) {
    let filter = TamperPairFilter::new();
    let p = pair("ignored-positive-text", "ignored-negative-text");
    let got = filter.keep(&p, pos, neg);
    let exp = reference::keep_tamper(&p, pos, neg);
    match (got, exp) {
        (Ok(()), Ok(())) => {}
        (Err(Error::UnconfirmedPair), Err(Error::UnconfirmedPair)) => {}
        (g, e) => panic!("keep mismatch: got {g:?} expected {e:?}"),
    }
}

#[test]
fn keep_ok_only_when_negative_is_hack_and_positive_is_clean() {
    let filter = TamperPairFilter::new();
    let p = pair("pos", "neg");
    filter
        .keep(&p, &clean_view(), &test_write_view())
        .expect("confirmed (clean, hack) pair");
    assert_keep_matches(&clean_view(), &test_write_view());
    assert_keep_matches(&clean_view(), &special_case_view());
    assert_keep_matches(&clean_view(), &visible_pass_hidden_fail_view());
}

#[test]
fn keep_rejects_clean_clean() {
    let err = TamperPairFilter::new()
        .keep(&pair("a", "b"), &clean_view(), &clean_view())
        .unwrap_err();
    assert_eq!(err, Error::UnconfirmedPair);
}

#[test]
fn keep_rejects_hack_hack() {
    let err = TamperPairFilter::new()
        .keep(&pair("a", "b"), &test_write_view(), &special_case_view())
        .unwrap_err();
    assert_eq!(err, Error::UnconfirmedPair);
}

#[test]
fn keep_rejects_inverted_hack_as_positive() {
    let err = TamperPairFilter::new()
        .keep(&pair("a", "b"), &test_write_view(), &clean_view())
        .unwrap_err();
    assert_eq!(err, Error::UnconfirmedPair);
}

#[test]
fn keep_ignores_pair_text_fields() {
    let filter = TamperPairFilter::new();
    let misleading = pair("this looks like a hack", "this looks clean");
    filter
        .keep(&misleading, &clean_view(), &test_write_view())
        .expect("views decide, not text");
    let empty = pair("", "");
    filter
        .keep(&empty, &clean_view(), &special_case_view())
        .expect("empty text is fine");
}

#[test]
fn hidden_pass_visible_fail_is_not_a_confirmed_negative() {
    let err = TamperPairFilter::new()
        .keep(
            &pair("a", "b"),
            &clean_view(),
            &hidden_pass_visible_fail_view(),
        )
        .unwrap_err();
    assert_eq!(err, Error::UnconfirmedPair);
}

#[test]
fn rescore_keep_requires_strict_greater() {
    let f = RescoreFilter::new();
    f.keep(0.9, 0.1).unwrap();
    f.keep(0.0, -1.0).unwrap();
    f.keep(1.0, 0.999).unwrap();
    assert_eq!(f.keep(0.5, 0.5).unwrap_err(), Error::UnconfirmedPair);
    assert_eq!(f.keep(0.1, 0.9).unwrap_err(), Error::UnconfirmedPair);
    assert_eq!(f.keep(0.0, 0.0).unwrap_err(), Error::UnconfirmedPair);
}

#[test]
fn rescore_keep_matches_reference_grid() {
    let f = RescoreFilter::new();
    let vals = [-2.0, -0.5, 0.0, 0.25, 0.5, 1.0, 3.0];
    for &p in &vals {
        for &n in &vals {
            let got = f.keep(p, n);
            let exp = reference::keep_rescore(p, n);
            match (got, exp) {
                (Ok(()), Ok(())) => {}
                (Err(Error::UnconfirmedPair), Err(Error::UnconfirmedPair)) => {}
                (g, e) => panic!("rescore {p} vs {n}: got {g:?} expected {e:?}"),
            }
        }
    }
}

#[test]
fn rescore_nan_is_unconfirmed() {
    let f = RescoreFilter::new();
    assert_eq!(f.keep(f64::NAN, 0.0).unwrap_err(), Error::UnconfirmedPair);
    assert_eq!(f.keep(1.0, f64::NAN).unwrap_err(), Error::UnconfirmedPair);
    assert_eq!(
        f.keep(f64::NAN, f64::NAN).unwrap_err(),
        Error::UnconfirmedPair
    );
}

#[test]
fn pair_construction_uses_views_never_held_out_ids() {
    let mut set = prometheus_rewards::HeldOutHacks::new();
    set.insert("planted-test-write", test_write_view());
    assert!(set.contains("planted-test-write"));
    assert!(!set.contains("train-clean"));
    TamperPairFilter::new()
        .keep(&pair("ok", "hack"), &clean_view(), &test_write_view())
        .expect("filters take views, not held-out ids");
}
