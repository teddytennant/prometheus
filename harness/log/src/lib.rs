//! Durable hash-chained event log (spec 15.2, 15.5 H2).
//!
//! Source of truth for tasks, leases, agent messages, tool calls, results,
//! and ledger entries. Crash-only: every state change is an append; startup
//! is replay. Raft-ready: the on-disk sequence is what a later H4 Raft group
//! replicates. This crate does not run Raft.
//!
//! Record shape matches F1 `prometheus.event_log`. Genesis `prev_hash` is 64
//! zero hex chars. The first record is `seq = 0` (see
//! `contracts/goldens/v1/event_log.default.json`). Readers must accept every
//! old `schema_version`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Fail-closed log errors. A broken chain is not skipped.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("corrupt log: {0}")]
    Corrupt(String),
    #[error("broken hash chain at seq {0}")]
    BrokenChain(u64),
    #[error("schema: {0}")]
    Schema(String),
    #[error("io: {0}")]
    Io(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub const SCHEMA_ID: &str = "prometheus.event_log";
pub const SCHEMA_VERSION: u32 = 1;
/// Genesis `prev_hash`: 64 zero hex chars.
pub const GENESIS_PREV_HASH: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// One F1 `event_log` record. Field names match the schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub schema_id: String,
    pub schema_version: u32,
    pub seq: u64,
    pub prev_hash: String,
    pub hash: String,
    pub timestamp: String,
    pub event_type: String,
    #[serde(default)]
    pub task_id: Option<String>,
    #[serde(default)]
    pub attempt: Option<u64>,
    #[serde(default)]
    pub node_id: Option<String>,
    pub payload: Value,
    pub payload_hash: String,
}

/// Fields the caller supplies on append. `seq`, hashes, and schema ids are
/// filled by the log.
#[derive(Debug, Clone, PartialEq)]
pub struct Append {
    pub event_type: String,
    pub payload: Value,
    /// RFC 3339 UTC. Tests pin this so hashes match F1 goldens.
    pub timestamp: String,
    pub task_id: Option<String>,
    pub attempt: Option<u64>,
    pub node_id: Option<String>,
}

/// Durable JSONL log at `dir/events.jsonl`. Each append fsyncs.
#[derive(Debug)]
pub struct EventLog {
    dir: PathBuf,
    events: Vec<Event>,
}

impl EventLog {
    /// Create an empty log directory. Fails if it already exists.
    pub fn create(_dir: impl AsRef<Path>) -> Result<Self> {
        unimplemented!("H2 event log: create")
    }

    /// Open and replay. Verifies the chain. Fails closed on a broken hash.
    pub fn open(_dir: impl AsRef<Path>) -> Result<Self> {
        unimplemented!("H2 event log: open")
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Append one record, fsync, return the filled Event. First seq is 0.
    pub fn append(&mut self, _record: Append) -> Result<Event> {
        unimplemented!("H2 event log: append")
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn last(&self) -> Option<&Event> {
        self.events.last()
    }

    pub fn get(&self, _seq: u64) -> Option<&Event> {
        unimplemented!("H2 event log: get")
    }

    pub fn iter(&self) -> impl Iterator<Item = &Event> {
        self.events.iter()
    }

    /// Recompute hashes and prev links. Used after replay and by tests.
    pub fn verify(&self) -> Result<()> {
        unimplemented!("H2 event log: verify")
    }

    /// Raft-ready last index: `seq` of the last record, or `None` if empty.
    pub fn last_index(&self) -> Option<u64> {
        self.last().map(|e| e.seq)
    }
}

/// SHA-256 of canonical JSON payload bytes, lowercase hex.
pub fn payload_hash(_payload: &Value) -> String {
    unimplemented!("H2 event log: payload_hash")
}

/// SHA-256 of `seq|prev_hash|payload_hash|timestamp|event_type`, lowercase hex.
pub fn event_hash(
    _seq: u64,
    _prev_hash: &str,
    _payload_hash: &str,
    _timestamp: &str,
    _event_type: &str,
) -> String {
    unimplemented!("H2 event log: event_hash")
}

/// Canonical JSON bytes of `payload` (sorted object keys, compact).
pub fn canonical_json(_payload: &Value) -> Vec<u8> {
    unimplemented!("H2 event log: canonical_json")
}
