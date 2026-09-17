//! Group: payload_hash, event_hash, canonical_json vs the independent reference
//! and vs independently recomputed F1-field goldens.

mod common;
mod reference;

use prometheus_log::{canonical_json, event_hash, payload_hash, GENESIS_PREV_HASH};
use serde_json::{json, Value};

fn assert_payload_eq(payload: &Value) {
    let got = payload_hash(payload);
    let expect = reference::payload_hash(payload);
    assert_eq!(got, expect, "payload_hash vs tests/reference");
    assert!(common::is_sha256_hex(&got), "payload_hash hex {got}");
}

#[test]
fn reference_goldens_recompute() {
    reference::assert_self_consistent_goldens();
    // Production must be invoked so a stub cannot pass this file.
    assert_payload_eq(&json!({}));
}

#[test]
fn payload_hash_empty_object() {
    let payload = json!({});
    assert_payload_eq(&payload);
    assert_eq!(payload_hash(&payload), reference::GOLDEN_EMPTY_OBJECT_HASH);
}

#[test]
fn payload_hash_f1_cycle_payload() {
    let payload = json!({"cycle": 1});
    assert_payload_eq(&payload);
    assert_eq!(payload_hash(&payload), reference::F1_GENESIS_PAYLOAD_HASH);
}

#[test]
fn payload_hash_fixed_payload() {
    let payload = reference::fixed_payload();
    assert_payload_eq(&payload);
    assert_eq!(payload_hash(&payload), reference::GOLDEN_FIXED_PAYLOAD_HASH);
}

#[test]
fn payload_hash_unicode() {
    let payload = reference::unicode_payload();
    assert_payload_eq(&payload);
    assert_eq!(payload_hash(&payload), reference::GOLDEN_UNICODE_PAYLOAD_HASH);
    let bytes = canonical_json(&payload);
    assert_eq!(bytes, reference::canonical_json_bytes(&payload));
    let s = std::str::from_utf8(&bytes).expect("utf-8");
    assert!(s.contains('Δ'), "non-ascii must be UTF-8, not \\uXXXX: {s}");
    assert!(!s.contains("\\u"), "must not escape unicode: {s}");
}

#[test]
fn payload_hash_independent_of_key_order() {
    let a = json!({"z": 1, "a": 2, "m": {"y": true, "x": false}});
    let b = json!({"a": 2, "m": {"x": false, "y": true}, "z": 1});
    assert_eq!(payload_hash(&a), payload_hash(&b));
    assert_eq!(payload_hash(&a), reference::payload_hash(&a));
    assert_eq!(payload_hash(&a), reference::payload_hash(&b));
}

#[test]
fn canonical_json_compact_sorted_no_spaces() {
    let payload = json!({"b": 2, "a": 1});
    let bytes = canonical_json(&payload);
    assert_eq!(bytes, reference::canonical_json_bytes(&payload));
    assert_eq!(bytes, br#"{"a":1,"b":2}"#);
    assert!(!bytes.contains(&b' '));
}

#[test]
fn event_hash_genesis_seq_zero_f1_fields() {
    reference::assert_self_consistent_goldens();
    let ph = payload_hash(&json!({"cycle": 1}));
    let got = event_hash(
        0,
        GENESIS_PREV_HASH,
        &ph,
        reference::F1_TIMESTAMP,
        reference::F1_EVENT_TYPE,
    );
    let expect = reference::event_hash(
        0,
        reference::GENESIS_PREV_HASH,
        &ph,
        reference::F1_TIMESTAMP,
        reference::F1_EVENT_TYPE,
    );
    assert_eq!(got, expect);
    assert_eq!(got, reference::F1_GENESIS_EVENT_HASH);
    assert_eq!(GENESIS_PREV_HASH, reference::GENESIS_PREV_HASH);
}

#[test]
fn event_hash_seq_decimal_no_leading_zeros() {
    let ph = "a".repeat(64);
    let prev = "b".repeat(64);
    let h0 = event_hash(0, &prev, &ph, "t", "e");
    let h10 = event_hash(10, &prev, &ph, "t", "e");
    assert_eq!(h0, reference::event_hash(0, &prev, &ph, "t", "e"));
    assert_eq!(h10, reference::event_hash(10, &prev, &ph, "t", "e"));
    assert_ne!(h0, h10);
    assert_ne!(
        event_hash(10, &prev, &ph, "t", "e"),
        event_hash(1, &prev, &ph, "t", "e")
    );
}

#[test]
fn f1_golden_shape_and_recomputed_hashes() {
    let raw = std::fs::read_to_string(common::golden_path()).expect("read F1 golden");
    let v: Value = serde_json::from_str(&raw).expect("parse F1 golden");
    assert_eq!(v["schema_id"], "prometheus.event_log");
    assert_eq!(v["schema_version"], 1);
    assert_eq!(v["seq"], 0);
    assert_eq!(v["prev_hash"], "0".repeat(64));
    assert_eq!(v["timestamp"], "2026-09-16T12:00:00Z");
    assert_eq!(v["event_type"], "session.start");
    assert!(v["task_id"].is_null());
    assert!(v["attempt"].is_null());
    assert_eq!(v["node_id"], "local");
    assert_eq!(v["payload"], json!({"cycle": 1}));
    for key in common::F1_FIELDS {
        assert!(v.get(*key).is_some(), "golden missing {key}");
    }

    let payload = v["payload"].clone();
    let ph = payload_hash(&payload);
    assert_eq!(ph, reference::payload_hash(&payload));
    assert_eq!(ph, reference::F1_GENESIS_PAYLOAD_HASH);
    let eh = event_hash(
        0,
        GENESIS_PREV_HASH,
        &ph,
        v["timestamp"].as_str().unwrap(),
        v["event_type"].as_str().unwrap(),
    );
    assert_eq!(eh, reference::F1_GENESIS_EVENT_HASH);
}
