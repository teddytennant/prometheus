//! Group: payload_hash / event_hash vs the independent reference goldens.
//!
//! Production functions must match `obs/tests/reference` (and the Python twin
//! `tests/reference/event_log.py`). Fixed payload goldens are locked here.

mod common;
mod reference;

use prometheus_obs::{event_hash, payload_hash, GENESIS_HASH};
use serde_json::json;

use common::is_sha256_hex;

#[test]
fn payload_hash_matches_reference_and_locked_golden() {
    let payload = reference::fixed_payload();
    let independent = reference::payload_hash(&payload);
    assert_eq!(independent, reference::GOLDEN_FIXED_PAYLOAD_HASH);
    let got = payload_hash(&payload);
    assert_eq!(got, independent);
    assert_eq!(got, reference::GOLDEN_FIXED_PAYLOAD_HASH);
    assert!(is_sha256_hex(&got));
}

#[test]
fn event_hash_matches_reference_and_locked_golden() {
    let ph = reference::GOLDEN_FIXED_PAYLOAD_HASH;
    let independent = reference::event_hash(
        1,
        GENESIS_HASH,
        ph,
        reference::GOLDEN_FIXED_TIMESTAMP,
        reference::GOLDEN_FIXED_EVENT_TYPE,
    );
    assert_eq!(independent, reference::GOLDEN_FIXED_EVENT_HASH);
    let got = event_hash(
        1,
        GENESIS_HASH,
        ph,
        reference::GOLDEN_FIXED_TIMESTAMP,
        reference::GOLDEN_FIXED_EVENT_TYPE,
    );
    assert_eq!(got, independent);
    assert_eq!(got, reference::GOLDEN_FIXED_EVENT_HASH);
    assert!(is_sha256_hex(&got));
}

#[test]
fn payload_hash_empty_object_golden() {
    let got = payload_hash(&json!({}));
    assert_eq!(got, reference::payload_hash(&json!({})));
    assert_eq!(got, reference::GOLDEN_EMPTY_OBJECT_HASH);
}

#[test]
fn payload_hash_unicode_is_utf8_not_u_escape() {
    let payload = json!({"note": "ΔRCI café"});
    let got = payload_hash(&payload);
    assert_eq!(got, reference::payload_hash(&payload));
    assert_eq!(got, reference::GOLDEN_UNICODE_PAYLOAD_HASH);
}

#[test]
fn payload_hash_is_independent_of_object_key_order() {
    let a = json!({"z": 1, "a": 2, "m": {"b": 0, "a": 1}});
    let b = json!({"a": 2, "m": {"a": 1, "b": 0}, "z": 1});
    let ha = payload_hash(&a);
    let hb = payload_hash(&b);
    assert_eq!(ha, hb);
    assert_eq!(ha, reference::payload_hash(&a));
    assert_eq!(ha, reference::payload_hash(&b));
}

#[test]
fn payload_hash_differs_when_payload_differs() {
    let h1 = payload_hash(&json!({"n": 1}));
    let h2 = payload_hash(&json!({"n": 2}));
    assert_ne!(h1, h2);
    assert_eq!(h1, reference::payload_hash(&json!({"n": 1})));
    assert_eq!(h2, reference::payload_hash(&json!({"n": 2})));
}

#[test]
fn event_hash_uses_pipe_concatenation() {
    // Property: flipping any one field changes the digest.
    let base = event_hash(1, GENESIS_HASH, "ab", "2026-09-16T00:00:00Z", "t");
    assert_eq!(
        base,
        reference::event_hash(1, GENESIS_HASH, "ab", "2026-09-16T00:00:00Z", "t")
    );
    assert_ne!(
        base,
        event_hash(2, GENESIS_HASH, "ab", "2026-09-16T00:00:00Z", "t")
    );
    assert_ne!(
        base,
        event_hash(1, &"1".repeat(64), "ab", "2026-09-16T00:00:00Z", "t")
    );
    assert_ne!(
        base,
        event_hash(1, GENESIS_HASH, "cd", "2026-09-16T00:00:00Z", "t")
    );
    assert_ne!(
        base,
        event_hash(1, GENESIS_HASH, "ab", "2026-09-16T00:00:01Z", "t")
    );
    assert_ne!(
        base,
        event_hash(1, GENESIS_HASH, "ab", "2026-09-16T00:00:00Z", "u")
    );
}

#[test]
fn event_hash_seq_decimal_has_no_leading_zeros() {
    // seq=10 must not hash as "010|..."
    let h10 = event_hash(10, "p", "h", "t", "e");
    assert_eq!(h10, reference::event_hash(10, "p", "h", "t", "e"));
    let h010_if_padded_would_differ = reference::event_hash(10, "p", "h", "t", "e");
    assert_eq!(h10, h010_if_padded_would_differ);
    assert_ne!(h10, event_hash(1, "p", "h", "t", "e"));
}

#[test]
fn hashes_are_64_lowercase_hex() {
    for payload in [json!(null), json!(true), json!([]), json!({"k": [1, 2, 3]})] {
        let h = payload_hash(&payload);
        assert!(is_sha256_hex(&h), "payload_hash {h} for {payload}");
        assert_eq!(h, reference::payload_hash(&payload));
    }
    let eh = event_hash(0, GENESIS_HASH, &"a".repeat(64), "T", "e");
    assert!(is_sha256_hex(&eh));
    assert_eq!(
        eh,
        reference::event_hash(0, GENESIS_HASH, &"a".repeat(64), "T", "e")
    );
}
