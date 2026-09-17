//! Minimal observability: metrics, tracing, run registry (spec 15.5 F2).
//!
//! F1 contracts this crate speaks:
//! - run records are identified by a `run_id` string
//! - the event log (when a run emits one) matches `prometheus.event_log`
//!   (`contracts/schemas/v1/event_log.schema.json`): hash-chained, seq,
//!   prev_hash, hash, timestamp, event_type, payload, payload_hash
//!
//! This is the in-process side. Dashboards (I10) come later. Nothing here
//! talks to the network.
//!
//! Nothing here runs. Types are real; every mutating function is
//! `unimplemented!`.

use std::collections::BTreeMap;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

/// Failures from the registry or a metric name that is already a different kind.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unknown run {0}")]
    UnknownRun(String),
    #[error("run {0} already registered")]
    DuplicateRun(String),
    #[error("metric {0} already registered as a different kind")]
    KindMismatch(String),
    #[error("obs: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Sorted label map. Empty map is the unlabeled series.
pub type Labels = BTreeMap<String, String>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricKind {
    Counter,
    Gauge,
    Histogram,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// One registered training / eval / verify run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunRecord {
    pub run_id: String,
    pub kind: String,
    pub status: RunStatus,
    pub config_hash: String,
    pub created_at: String,
    pub parent_run_id: Option<String>,
}

/// Hash-chained event. Field names match the F1 `event_log` schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    pub schema_id: String,
    pub schema_version: u32,
    pub seq: u64,
    pub prev_hash: String,
    pub hash: String,
    pub timestamp: String,
    pub event_type: String,
    pub payload: serde_json::Value,
    pub payload_hash: String,
    pub task_id: Option<String>,
    pub attempt: Option<u64>,
    pub node_id: Option<String>,
}

pub const EVENT_SCHEMA_ID: &str = "prometheus.event_log";
pub const EVENT_SCHEMA_VERSION: u32 = 1;
/// Genesis prev_hash: 64 zero hex chars.
pub const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// In-process metric registry. Names are dotted (`train.loss.ce`).
#[derive(Debug, Default)]
pub struct Metrics {
    _private: (),
}

impl Metrics {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn inc(&mut self, name: &str, labels: &Labels, amount: f64) -> Result<()> {
        let _ = (name, labels, amount);
        unimplemented!("F2 Metrics::inc")
    }

    pub fn set(&mut self, name: &str, labels: &Labels, value: f64) -> Result<()> {
        let _ = (name, labels, value);
        unimplemented!("F2 Metrics::set")
    }

    pub fn observe(&mut self, name: &str, labels: &Labels, value: f64) -> Result<()> {
        let _ = (name, labels, value);
        unimplemented!("F2 Metrics::observe")
    }

    /// Snapshot of one series. Missing name is `Ok(None)`.
    pub fn get(&self, name: &str, labels: &Labels) -> Result<Option<f64>> {
        let _ = (name, labels);
        unimplemented!("F2 Metrics::get")
    }
}

/// One open span. Dropping it without `end` is a leak the tests catch.
#[derive(Debug)]
pub struct Span {
    pub name: String,
    pub start: SystemTime,
}

/// In-process tracer. Spans nest; `current` is the innermost open span.
#[derive(Debug, Default)]
pub struct Tracer {
    _private: (),
}

impl Tracer {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn start(&mut self, name: &str) -> Result<Span> {
        let _ = name;
        unimplemented!("F2 Tracer::start")
    }

    pub fn end(&mut self, span: Span) -> Result<u128> {
        let _ = span;
        unimplemented!("F2 Tracer::end")
    }

    pub fn current(&self) -> Option<&str> {
        unimplemented!("F2 Tracer::current")
    }
}

/// In-process run registry. `run_id` is unique.
#[derive(Debug, Default)]
pub struct RunRegistry {
    _private: (),
}

impl RunRegistry {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn register(&mut self, record: RunRecord) -> Result<()> {
        let _ = record;
        unimplemented!("F2 RunRegistry::register")
    }

    pub fn get(&self, run_id: &str) -> Result<RunRecord> {
        let _ = run_id;
        unimplemented!("F2 RunRegistry::get")
    }

    pub fn set_status(&mut self, run_id: &str, status: RunStatus) -> Result<()> {
        let _ = (run_id, status);
        unimplemented!("F2 RunRegistry::set_status")
    }

    pub fn list(&self) -> Result<Vec<RunRecord>> {
        unimplemented!("F2 RunRegistry::list")
    }

    /// Append a hash-chained event for `run_id`. First event uses `GENESIS_HASH`.
    pub fn append_event(
        &mut self,
        run_id: &str,
        event_type: &str,
        payload: serde_json::Value,
    ) -> Result<Event> {
        let _ = (run_id, event_type, payload);
        unimplemented!("F2 RunRegistry::append_event")
    }

    pub fn events(&self, run_id: &str) -> Result<Vec<Event>> {
        let _ = run_id;
        unimplemented!("F2 RunRegistry::events")
    }
}

/// SHA-256 of canonical JSON payload bytes, lowercase hex.
pub fn payload_hash(payload: &serde_json::Value) -> String {
    let _ = payload;
    unimplemented!("F2 payload_hash")
}

/// SHA-256 of `seq | prev_hash | payload_hash | timestamp | event_type`.
pub fn event_hash(
    seq: u64,
    prev_hash: &str,
    payload_hash: &str,
    timestamp: &str,
    event_type: &str,
) -> String {
    let _ = (seq, prev_hash, payload_hash, timestamp, event_type);
    unimplemented!("F2 event_hash")
}
