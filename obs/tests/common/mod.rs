//! Shared helpers for F2 `prometheus-obs` oracle tests.
//!
//! # Documented rules the implementer must match
//!
//! ## Metrics
//! - `inc` is a **counter**: amounts **accumulate** for `(name, labels)`.
//! - `set` is a **gauge**: each call **overwrites** the series value.
//! - `observe` is a **histogram**. `get` on a histogram series returns the
//!   **sum** of every observed value (not the count, not the last sample).
//! - `get` of a name+labels pair that has never been written is `Ok(None)`
//!   (not an error). A counter incremented by `0.0` is present (`Some(0.0)`).
//! - Kind is per **name**, not per labels. Using the same name as two of
//!   `{inc, set, observe}` is `Error::KindMismatch(name)` even if labels differ.
//! - Labels distinguish series of the same kind. The empty map is the unlabeled
//!   series and is distinct from any non-empty label set.
//! - `get` snapshots are compared at 1e-5 relative to the written f64 values.
//!
//! ## Tracer
//! - `end` returns elapsed time in **milliseconds** as `u128`
//!   (`Duration::as_millis()`). Sub-millisecond spans may be `0`.
//! - `current()` is the **innermost** still-open span name, or `None`.
//! - A span is closed **only** by `Tracer::end`. Dropping a `Span` without
//!   `end` does **not** close it: `current()` still reports it. That is the
//!   leak signal these tests assert.
//! - `end` is LIFO. Ending a span that is not the innermost open span errors.
//! - Ending a span never started on this tracer (constructed by the caller, or
//!   started on a different `Tracer`) errors.
//!
//! ## Run registry
//! - `register` then `get` / `list` / `set_status`.
//! - Second `register` of the same `run_id` is `Error::DuplicateRun`.
//! - `get` / `set_status` / `events` / `append_event` on an unknown id is
//!   `Error::UnknownRun`.
//!
//! ## Event log
//! - First event of a run: `seq == 1`, `prev_hash == GENESIS_HASH` (64 `'0'`).
//! - Later events: `seq` increases by 1; `prev_hash` is the previous event's
//!   `hash`.
//! - `payload_hash` / `hash` match `tests/reference` (see `reference/mod.rs`).
//! - `schema_id == "prometheus.event_log"`, `schema_version == 1`.
//! - `timestamp` is RFC3339-ish: non-empty and contains `'T'`.
//!
//! No GPU coverage in F2; these tests are CPU-only (no `gpu` marker).

#![allow(dead_code)]

use prometheus_obs::{Error, Labels, Result, RunRecord, RunStatus};

/// Absolute tolerance used for metric snapshots (spec FP32-style 1e-5).
pub const METRIC_TOL: f64 = 1e-5;

pub fn labels(pairs: &[(&str, &str)]) -> Labels {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

pub fn empty_labels() -> Labels {
    Labels::new()
}

pub fn assert_f64_close(got: f64, expected: f64) {
    let diff = (got - expected).abs();
    assert!(
        diff <= METRIC_TOL,
        "metric snapshot {got} not within {METRIC_TOL} of {expected} (diff {diff})"
    );
}

pub fn assert_some_f64(got: Result<Option<f64>>, expected: f64) {
    match got {
        Ok(Some(v)) => assert_f64_close(v, expected),
        other => panic!("expected Ok(Some({expected})), got {other:?}"),
    }
}

pub fn assert_kind_mismatch(err: Result<()>, name: &str) {
    match err {
        Err(Error::KindMismatch(n)) => {
            assert_eq!(n, name, "KindMismatch name");
        }
        other => panic!("expected Error::KindMismatch({name:?}), got {other:?}"),
    }
}

pub fn assert_unknown_run<T: std::fmt::Debug>(err: Result<T>, run_id: &str) {
    match err {
        Err(Error::UnknownRun(id)) => {
            assert_eq!(id, run_id, "UnknownRun id");
        }
        other => panic!("expected Error::UnknownRun({run_id:?}), got {other:?}"),
    }
}

pub fn assert_duplicate_run(err: Result<()>, run_id: &str) {
    match err {
        Err(Error::DuplicateRun(id)) => {
            assert_eq!(id, run_id, "DuplicateRun id");
        }
        other => panic!("expected Error::DuplicateRun({run_id:?}), got {other:?}"),
    }
}

pub fn assert_rfc3339ish(ts: &str) {
    assert!(!ts.is_empty(), "timestamp must be non-empty");
    assert!(
        ts.contains('T'),
        "timestamp must contain 'T' (RFC3339-ish), got {ts:?}"
    );
}

pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

pub fn sample_run(run_id: &str) -> RunRecord {
    RunRecord {
        run_id: run_id.to_string(),
        kind: "train".to_string(),
        status: RunStatus::Pending,
        config_hash: "c".repeat(64),
        created_at: "2026-09-16T12:00:00Z".to_string(),
        parent_run_id: None,
    }
}
