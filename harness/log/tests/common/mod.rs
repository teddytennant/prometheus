//! Shared builders and assertions for H2 `prometheus-log` oracle tests.
//!
//! Documented rules the implementer must match:
//! - First seq is 0. `prev_hash` of seq 0 is 64 zero hex chars (`GENESIS_PREV_HASH`).
//! - Durable file is `<dir>/events.jsonl` (one JSON object per line).
//! - `create` fails if `dir` already exists. `open` fail-closes on a missing file,
//!   torn last line, seq gap, hash mismatch, or broken chain.
//! - After `append` returns, the record is durable (fsync before return).
//! - Hashing matches `tests/reference` (canonical JSON + SHA-256), not the
//!   placeholder digest strings in `contracts/goldens/v1/event_log.default.json`.
//!
//! No GPU coverage in H2; these tests are CPU-only (no `gpu` marker).

#![allow(dead_code)]

use prometheus_log::{Append, Event, EventLog, Result as LogResult};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

pub const EVENTS_JSONL: &str = "events.jsonl";
pub const F1_TIMESTAMP: &str = "2026-09-16T12:00:00Z";
pub const F1_EVENT_TYPE: &str = "session.start";
pub const F1_NODE_ID: &str = "local";

pub const F1_FIELDS: &[&str] = &[
    "schema_id",
    "schema_version",
    "seq",
    "prev_hash",
    "hash",
    "timestamp",
    "event_type",
    "task_id",
    "attempt",
    "node_id",
    "payload",
    "payload_hash",
];

/// Parent temp dir plus a not-yet-created `log/` child for `EventLog::create`.
pub fn fresh_log_dir() -> (tempfile::TempDir, PathBuf) {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("log");
    (parent, dir)
}

pub fn events_path(dir: &Path) -> PathBuf {
    dir.join(EVENTS_JSONL)
}

pub fn f1_payload() -> Value {
    serde_json::json!({"cycle": 1})
}

pub fn append_input(event_type: &str, payload: Value) -> Append {
    Append {
        timestamp: F1_TIMESTAMP.to_string(),
        event_type: event_type.to_string(),
        task_id: None,
        attempt: None,
        node_id: Some(F1_NODE_ID.to_string()),
        payload,
    }
}

pub fn f1_append() -> Append {
    append_input(F1_EVENT_TYPE, f1_payload())
}

pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

pub fn assert_err<T: std::fmt::Debug>(r: LogResult<T>, what: &str) {
    assert!(r.is_err(), "expected fail-closed ({what}), got {r:?}");
}

pub fn assert_event_fields(ev: &Event) {
    assert_eq!(ev.schema_id, prometheus_log::SCHEMA_ID);
    assert_eq!(ev.schema_id, "prometheus.event_log");
    assert_eq!(ev.schema_version, prometheus_log::SCHEMA_VERSION);
    assert_eq!(ev.schema_version, 1);
    assert!(is_sha256_hex(&ev.prev_hash), "prev_hash {}", ev.prev_hash);
    assert!(is_sha256_hex(&ev.hash), "hash {}", ev.hash);
    assert!(
        is_sha256_hex(&ev.payload_hash),
        "payload_hash {}",
        ev.payload_hash
    );
    assert!(!ev.event_type.is_empty(), "event_type must be non-empty");
    assert!(ev.payload.is_object(), "payload must be a JSON object");
}

pub fn assert_f1_field_names(ev: &Event) {
    let value = serde_json::to_value(ev).expect("serialize Event");
    let obj = value.as_object().expect("Event JSON object");
    for key in F1_FIELDS {
        assert!(
            obj.contains_key(*key),
            "missing F1 field `{key}` in {value}"
        );
    }
    assert!(
        !obj.contains_key("schemaId")
            && !obj.contains_key("schemaVersion")
            && !obj.contains_key("prevHash")
            && !obj.contains_key("eventType")
            && !obj.contains_key("taskId")
            && !obj.contains_key("nodeId")
            && !obj.contains_key("payloadHash"),
        "Event JSON must use snake_case F1 names, got keys {:?}",
        obj.keys().collect::<Vec<_>>()
    );
}

pub fn write_jsonl_lines(dir: &Path, lines: &[String]) {
    fs::create_dir_all(dir).expect("mkdir log dir");
    let mut body = lines.join("\n");
    if !body.is_empty() {
        body.push('\n');
    }
    fs::write(events_path(dir), body).expect("write events.jsonl");
}

pub fn read_jsonl_bytes(dir: &Path) -> Vec<u8> {
    fs::read(events_path(dir)).expect("read events.jsonl")
}

pub fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts/goldens/v1/event_log.default.json")
}

/// Open must not treat torn bytes as a committed record.
/// Recovering the complete prefix or returning Err are both fail-closed.
pub fn assert_torn_does_not_invent(dir: &Path, committed: u64) {
    match EventLog::open(dir) {
        Err(_) => {}
        Ok(log) => {
            assert_eq!(
                log.len() as u64,
                committed,
                "recovered log must contain only complete records"
            );
            if committed == 0 {
                assert!(log.is_empty());
                assert_eq!(log.last_index(), None);
                assert!(log.get(0).is_none());
            } else {
                assert_eq!(log.last_index(), Some(committed - 1));
                assert!(log.get(0).is_some());
                assert!(log.get(committed).is_none());
            }
            log.verify().expect("recovered prefix must verify");
        }
    }
}
