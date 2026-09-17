//! Independent SHA-256 reference for F2 `payload_hash` / `event_hash`.
//!
//! Slow and obvious: sort keys, compact JSON, then SHA-256. This module is the
//! oracle; production `prometheus_obs::{payload_hash,event_hash}` must match it.
//! A Python twin lives at repo `tests/reference/event_log.py` (hashlib).
//!
//! # `payload_hash`
//! 1. Recursively rewrite every JSON object so keys are in UTF-8 byte order.
//! 2. Serialize with `serde_json` compact encoding (no whitespace). Non-ASCII
//!    is emitted as UTF-8, not `\uXXXX`. Integers have no decimal point.
//! 3. SHA-256 those bytes.
//! 4. Lowercase hex (64 chars).
//!
//! # `event_hash`
//! SHA-256 of the UTF-8 concatenation
//! `{seq}|{prev_hash}|{payload_hash}|{timestamp}|{event_type}`
//! where `{seq}` is the decimal `u64` with no leading zeros (`0` for zero).
//! Digest is lowercase hex (64 chars).
//!
//! Contracts goldens under `contracts/goldens/v1/event_log.default.json` are
//! schema-shape fixtures, **not** hash oracles.

#![allow(dead_code)]

use serde_json::Value;
use sha2::{Digest, Sha256};

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

/// Recursively sort object keys so `serde_json` emits a stable encoding.
pub fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            let mut out = serde_json::Map::new();
            for k in keys {
                out.insert(k.clone(), canonicalize(&map[k]));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonicalize).collect()),
        other => other.clone(),
    }
}

/// Canonical compact JSON bytes (sorted keys, no whitespace).
pub fn canonical_json_bytes(payload: &Value) -> Vec<u8> {
    serde_json::to_vec(&canonicalize(payload)).expect("canonical JSON is serializable")
}

/// Independent `payload_hash`: SHA-256 of canonical JSON payload bytes.
pub fn payload_hash(payload: &Value) -> String {
    sha256_hex(&canonical_json_bytes(payload))
}

/// Independent `event_hash`: SHA-256 of the documented `|` concatenation.
pub fn event_hash(
    seq: u64,
    prev_hash: &str,
    payload_hash: &str,
    timestamp: &str,
    event_type: &str,
) -> String {
    let concat = format!("{seq}|{prev_hash}|{payload_hash}|{timestamp}|{event_type}");
    sha256_hex(concat.as_bytes())
}

/// Fixed payload used as a golden in `hashes.rs` and `tests/reference/event_log.py`.
pub fn fixed_payload() -> Value {
    serde_json::json!({
        "msg": "hello",
        "n": 1,
        "ok": true
    })
}

/// Golden `payload_hash(fixed_payload)` from the Python hashlib twin.
pub const GOLDEN_FIXED_PAYLOAD_HASH: &str =
    "2fa4793603fe17e52b611317579148d3b47e527b3f6f8163240c0d734ca6273c";

pub const GOLDEN_FIXED_TIMESTAMP: &str = "2026-09-16T12:00:00Z";
pub const GOLDEN_FIXED_EVENT_TYPE: &str = "session.start";

/// Golden `event_hash(1, GENESIS, payload_hash, timestamp, event_type)`.
pub const GOLDEN_FIXED_EVENT_HASH: &str =
    "167b92870604c9cb29fff1d41e47af8b8de79f7618939267b2e6b586e7616272";

pub const GOLDEN_EMPTY_OBJECT_HASH: &str =
    "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a";

pub const GOLDEN_UNICODE_PAYLOAD_HASH: &str =
    "e8d13b8f2f569be2e3ef0a2803761f39ed4d93cf167636d3e3c7a6d3c7562e00";
