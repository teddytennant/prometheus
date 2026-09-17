//! Group: `is_planted_bad` vs the independent reference.

mod common;
mod reference;

use common::good_patch;
use prometheus_pipeline::{is_planted_bad, PLANTED_BAD_MARKER};
use reference::ref_is_planted_bad;

fn assert_same(patch: &[u8], expect: bool) {
    let prod = is_planted_bad(patch);
    let refer = ref_is_planted_bad(patch);
    assert_eq!(prod, refer, "prod vs ref for {patch:?}");
    assert_eq!(prod, expect, "expected {expect} for {patch:?}");
}

#[test]
fn empty_is_not_planted() {
    assert_same(b"", false);
}

#[test]
fn unrelated_bytes_are_not_planted() {
    assert_same(b"hello world", false);
    assert_same(good_patch(), false);
    assert_same(b"PLANTED_BAD", false);
    assert_same(b"PLANTED_BAD_PATC", false);
}

#[test]
fn exact_marker_is_planted() {
    assert_same(PLANTED_BAD_MARKER, true);
}

#[test]
fn marker_as_substring_is_planted() {
    let mut mid = b"prefix-".to_vec();
    mid.extend_from_slice(PLANTED_BAD_MARKER);
    mid.extend_from_slice(b"-suffix");
    assert_same(&mid, true);

    let mut start = PLANTED_BAD_MARKER.to_vec();
    start.extend_from_slice(b" after");
    assert_same(&start, true);

    let mut end = b"before ".to_vec();
    end.extend_from_slice(PLANTED_BAD_MARKER);
    assert_same(&end, true);
}

#[test]
fn truncated_marker_is_not_planted() {
    let m = PLANTED_BAD_MARKER;
    assert!(m.len() > 1, "marker must be non-trivial");
    for n in 1..m.len() {
        assert_same(&m[..n], false);
        assert_same(&m[m.len() - n..], false);
    }
}

#[test]
fn one_byte_short_at_either_end() {
    let m = PLANTED_BAD_MARKER;
    assert_same(&m[..m.len() - 1], false);
    assert_same(&m[1..], false);
}

#[test]
fn case_differs_is_not_planted() {
    assert_same(b"planted_bad_patch", false);
    assert_same(b"PLANTED_BAD_patch", false);
    assert_same(b"PLANTED_BAD_PATCH\x00", true);
}

#[test]
fn overlapping_and_double_marker() {
    let mut twice = PLANTED_BAD_MARKER.to_vec();
    twice.extend_from_slice(PLANTED_BAD_MARKER);
    assert_same(&twice, true);

    let mut overlap = PLANTED_BAD_MARKER.to_vec();
    overlap.extend_from_slice(&PLANTED_BAD_MARKER[1..]);
    assert_same(&overlap, true);
}

#[test]
fn marker_split_by_one_byte_is_not_planted() {
    let m = PLANTED_BAD_MARKER;
    let mid = m.len() / 2;
    let mut split = m[..mid].to_vec();
    split.push(b'X');
    split.extend_from_slice(&m[mid..]);
    assert_same(&split, false);
}

#[test]
fn binary_noise_around_marker() {
    let mut v = vec![0u8, 255, 1, 2];
    v.extend_from_slice(PLANTED_BAD_MARKER);
    v.extend_from_slice(&[9, 8, 7]);
    assert_same(&v, true);
    assert_same(&[0xff, 0x00, 0x01], false);
}
