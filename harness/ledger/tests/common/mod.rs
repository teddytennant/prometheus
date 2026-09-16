//! Shared builders and assertions for implementer-facing ledger tests.
#![allow(dead_code)]
//!
//! Negative-result rule (must be implemented by production `list_negatives`):
//! a row is negative iff any of
//!   * `results.passed == false` (JSON boolean),
//!   * `results.negative == true` (JSON boolean),
//!   * `replication.status == failed`.
//! Incomplete rows (`results` absent, replication unreplicated/absent) are
//! **not** negatives.
//!
//! Missing `supersedes` target → `Error::NotFound`.
//! `query` WHERE bodies use SQLite `?` placeholders; `params` are bound, never
//! interpolated.

use prometheus_ledger::{Record, Replication, ReplicationStatus};
use serde_json::Value;
use std::path::PathBuf;

pub const CONFIG_HASH: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

pub const F1_FIELDS: &[&str] = &[
    "experiment_id",
    "hypothesis",
    "prediction",
    "config_hash",
    "results",
    "delta_rci",
    "replication",
    "gpu_hours",
    "author_role",
    "rung",
    "created_at",
    "closed_at",
];

pub fn temp_ledger_path() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ledger.db");
    (dir, path)
}

/// Sparse record: every `Option` field is `None`.
pub fn sample_record(experiment_id: &str) -> Record {
    Record {
        experiment_id: experiment_id.to_string(),
        hypothesis: format!("hypothesis for {experiment_id}"),
        prediction: "pre-registered: ΔRCI will not decrease".to_string(),
        config_hash: CONFIG_HASH.to_string(),
        results: None,
        delta_rci: None,
        replication: None,
        gpu_hours: None,
        author_role: "researcher".to_string(),
        rung: None,
        created_at: "2026-10-01T00:00:00Z".to_string(),
        closed_at: None,
        supersedes: None,
    }
}

/// Fully populated record (every `Option` is `Some`, including nested notes).
pub fn full_record(experiment_id: &str) -> Record {
    Record {
        experiment_id: experiment_id.to_string(),
        hypothesis: "MuonClip beats AdamW at equal compute on rung 1".to_string(),
        prediction: "eval loss −0.04 ± 0.01 versus AdamW".to_string(),
        config_hash: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        results: Some(serde_json::json!({
            "passed": true,
            "loss": 1.23,
            "notes": "seed 7"
        })),
        delta_rci: Some(0.04),
        replication: Some(Replication {
            status: ReplicationStatus::Matched,
            n: 3,
            notes: Some("three new seeds".to_string()),
        }),
        gpu_hours: Some(12.5),
        author_role: "program_lead".to_string(),
        rung: Some(1),
        created_at: "2026-10-02T12:34:56Z".to_string(),
        closed_at: Some("2026-10-03T01:00:00Z".to_string()),
        supersedes: None,
    }
}

pub fn assert_records_eq(got: &Record, expected: &Record) {
    assert_eq!(got.experiment_id, expected.experiment_id, "experiment_id");
    assert_eq!(got.hypothesis, expected.hypothesis, "hypothesis");
    assert_eq!(got.prediction, expected.prediction, "prediction");
    assert_eq!(got.config_hash, expected.config_hash, "config_hash");
    assert_eq!(got.results, expected.results, "results");
    assert_eq!(got.delta_rci, expected.delta_rci, "delta_rci");
    assert_eq!(got.gpu_hours, expected.gpu_hours, "gpu_hours");
    assert_eq!(got.author_role, expected.author_role, "author_role");
    assert_eq!(got.rung, expected.rung, "rung");
    assert_eq!(got.created_at, expected.created_at, "created_at");
    assert_eq!(got.closed_at, expected.closed_at, "closed_at");
    assert_eq!(got.supersedes, expected.supersedes, "supersedes");
    match (&got.replication, &expected.replication) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            assert_eq!(a.status, b.status, "replication.status");
            assert_eq!(a.n, b.n, "replication.n");
            assert_eq!(a.notes, b.notes, "replication.notes");
        }
        _ => panic!(
            "replication mismatch: got {:?} expected {:?}",
            got.replication.is_some(),
            expected.replication.is_some()
        ),
    }
}

/// Serde field names must match F1 `prometheus.ledger_record` (snake_case).
/// `supersedes` may be present as an extra key.
pub fn assert_f1_field_names(record: &Record) {
    let value = serde_json::to_value(record).expect("serialize Record");
    let obj = value.as_object().expect("Record JSON must be an object");
    for key in F1_FIELDS {
        assert!(
            obj.contains_key(*key),
            "missing F1 field name `{key}` in {}",
            value
        );
    }
    assert!(
        !obj.contains_key("experimentId")
            && !obj.contains_key("configHash")
            && !obj.contains_key("deltaRci")
            && !obj.contains_key("gpuHours")
            && !obj.contains_key("authorRole")
            && !obj.contains_key("createdAt")
            && !obj.contains_key("closedAt"),
        "Record JSON must use snake_case F1 names, got keys {:?}",
        obj.keys().collect::<Vec<_>>()
    );

    if let Some(repl) = obj.get("replication") {
        if !repl.is_null() {
            let status = repl
                .get("status")
                .and_then(Value::as_str)
                .expect("replication.status string");
            assert!(
                matches!(status, "unreplicated" | "matched" | "failed" | "running"),
                "replication.status must be snake_case, got {status}"
            );
        }
    }

    let back: Record = serde_json::from_value(value).expect("deserialize Record");
    assert_records_eq(&back, record);
}
