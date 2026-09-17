//! Independent H2 reference: hashing, chain linking, JSONL parse, torn lines.
//!
//! Slow and obvious. Compiled only as a submodule of the integration tests.
//! Production `src/lib.rs` must never import this module.
//!
//! # Canonical JSON
//! 1. Recursively rewrite every JSON object so keys are in UTF-8 byte order.
//! 2. Serialize with `serde_json` compact encoding (no whitespace). Non-ASCII
//!    is emitted as UTF-8, not `\uXXXX`. Integers have no decimal point.
//!
//! # `payload_hash`
//! SHA-256 of those UTF-8 bytes, lowercase hex (64 chars).
//!
//! # `event_hash`
//! SHA-256 of the UTF-8 concatenation
//! `{seq}|{prev_hash}|{payload_hash}|{timestamp}|{event_type}`
//! where `{seq}` is the decimal `u64` with no leading zeros (`0` for zero).
//!
//! The F1 file `contracts/goldens/v1/event_log.default.json` is a schema-shape
//! fixture. Its `payload_hash` / `hash` strings are not the SHA-256 of this
//! algorithm applied to `payload = {"cycle": 1}`; tests lock the independently
//! recomputed digests below.

#![allow(dead_code)]

use prometheus_log::{Event, SCHEMA_ID, SCHEMA_VERSION};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

pub const GENESIS_PREV_HASH: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// Independently computed `payload_hash({"cycle": 1})`.
pub const F1_GENESIS_PAYLOAD_HASH: &str =
    "9ea9926526cd16b40c70cd8988bfa74bb6c857dbb64faf55ea20a1c42d5ab122";

/// Independently computed `event_hash` for the F1 golden fields at seq 0.
pub const F1_GENESIS_EVENT_HASH: &str =
    "c975ddd2d64d8781552891f9b1b5241208b703554c6f323a47336d2b5de5d74c";

pub const GOLDEN_FIXED_PAYLOAD_HASH: &str =
    "2fa4793603fe17e52b611317579148d3b47e527b3f6f8163240c0d734ca6273c";

pub const GOLDEN_EMPTY_OBJECT_HASH: &str =
    "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a";

pub const GOLDEN_UNICODE_PAYLOAD_HASH: &str =
    "e8d13b8f2f569be2e3ef0a2803761f39ed4d93cf167636d3e3c7a6d3c7562e00";

pub const F1_TIMESTAMP: &str = "2026-09-16T12:00:00Z";
pub const F1_EVENT_TYPE: &str = "session.start";

#[derive(Debug)]
pub enum ReplayError {
    Io(String),
    TornLastLine,
    InvalidJson { line: usize, msg: String },
    Chain(String),
}

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

pub fn fixed_payload() -> Value {
    serde_json::json!({
        "msg": "hello",
        "n": 1,
        "ok": true
    })
}

pub fn unicode_payload() -> Value {
    serde_json::json!({"note": "ΔRCI café"})
}

pub fn f1_payload() -> Value {
    serde_json::json!({"cycle": 1})
}

/// Build a fully hashed `Event` using this reference, not production hashing.
pub fn make_event(
    seq: u64,
    prev_hash: &str,
    timestamp: &str,
    event_type: &str,
    task_id: Option<String>,
    attempt: Option<u64>,
    node_id: Option<String>,
    payload: Value,
) -> Event {
    let payload_hash = payload_hash(&payload);
    let hash = event_hash(seq, prev_hash, &payload_hash, timestamp, event_type);
    Event {
        schema_id: SCHEMA_ID.to_string(),
        schema_version: SCHEMA_VERSION,
        seq,
        prev_hash: prev_hash.to_string(),
        hash,
        timestamp: timestamp.to_string(),
        event_type: event_type.to_string(),
        task_id,
        attempt,
        node_id,
        payload,
        payload_hash,
    }
}

pub fn verify_event(ev: &Event, expected_prev: &str) -> Result<(), String> {
    if ev.schema_id != SCHEMA_ID {
        return Err(format!("schema_id {}", ev.schema_id));
    }
    if ev.schema_version != SCHEMA_VERSION {
        return Err(format!("schema_version {}", ev.schema_version));
    }
    if ev.prev_hash != expected_prev {
        return Err(format!(
            "prev_hash mismatch seq {}: got {} expected {}",
            ev.seq, ev.prev_hash, expected_prev
        ));
    }
    let ph = payload_hash(&ev.payload);
    if ev.payload_hash != ph {
        return Err(format!(
            "payload_hash mismatch seq {}: stored {} computed {}",
            ev.seq, ev.payload_hash, ph
        ));
    }
    let eh = event_hash(
        ev.seq,
        &ev.prev_hash,
        &ev.payload_hash,
        &ev.timestamp,
        &ev.event_type,
    );
    if ev.hash != eh {
        return Err(format!(
            "event hash mismatch seq {}: stored {} computed {}",
            ev.seq, ev.hash, eh
        ));
    }
    Ok(())
}

pub fn verify_chain(events: &[Event]) -> Result<(), String> {
    let mut prev = GENESIS_PREV_HASH.to_string();
    for (i, ev) in events.iter().enumerate() {
        let expect_seq = i as u64;
        if ev.seq != expect_seq {
            return Err(format!("seq gap: index {i} has seq {}", ev.seq));
        }
        verify_event(ev, &prev)?;
        prev = ev.hash.clone();
    }
    Ok(())
}

/// Parse JSONL. A last line that is not valid JSON is a torn write.
pub fn parse_jsonl_bytes(bytes: &[u8]) -> Result<Vec<Event>, ReplayError> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let text = std::str::from_utf8(bytes).map_err(|e| ReplayError::InvalidJson {
        line: 0,
        msg: format!("not utf-8: {e}"),
    })?;
    let mut parts: Vec<&str> = text.split('\n').collect();
    if parts.last() == Some(&"") {
        parts.pop();
    }
    let mut events = Vec::new();
    for (i, line) in parts.iter().enumerate() {
        match serde_json::from_str::<Event>(line) {
            Ok(ev) => events.push(ev),
            Err(e) => {
                if i + 1 == parts.len() {
                    return Err(ReplayError::TornLastLine);
                }
                return Err(ReplayError::InvalidJson {
                    line: i,
                    msg: e.to_string(),
                });
            }
        }
    }
    Ok(events)
}

pub fn replay_jsonl(path: &Path) -> Result<Vec<Event>, ReplayError> {
    let bytes = fs::read(path).map_err(|e| ReplayError::Io(e.to_string()))?;
    let events = parse_jsonl_bytes(&bytes)?;
    verify_chain(&events).map_err(ReplayError::Chain)?;
    Ok(events)
}

pub fn event_to_jsonl_line(ev: &Event) -> String {
    serde_json::to_string(ev).expect("serialize Event")
}

/// Recompute F1 / fixed goldens. Panics if this reference disagrees with itself.
pub fn assert_self_consistent_goldens() {
    let cycle = f1_payload();
    let ph = payload_hash(&cycle);
    assert_eq!(ph, F1_GENESIS_PAYLOAD_HASH, "F1 payload_hash recompute");
    let eh = event_hash(0, GENESIS_PREV_HASH, &ph, F1_TIMESTAMP, F1_EVENT_TYPE);
    assert_eq!(eh, F1_GENESIS_EVENT_HASH, "F1 event_hash recompute");
    assert_eq!(
        payload_hash(&serde_json::json!({})),
        GOLDEN_EMPTY_OBJECT_HASH
    );
    assert_eq!(payload_hash(&fixed_payload()), GOLDEN_FIXED_PAYLOAD_HASH);
    assert_eq!(
        payload_hash(&unicode_payload()),
        GOLDEN_UNICODE_PAYLOAD_HASH
    );
    assert_eq!(GENESIS_PREV_HASH.len(), 64);
    assert!(GENESIS_PREV_HASH.chars().all(|c| c == '0'));
}
