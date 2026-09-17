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
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
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

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

pub const SCHEMA_ID: &str = "prometheus.event_log";
pub const SCHEMA_VERSION: u32 = 1;
/// Genesis `prev_hash`: 64 zero hex chars.
pub const GENESIS_PREV_HASH: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

const EVENTS_JSONL: &str = "events.jsonl";

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
    /// Kept open so append does not reopen the JSONL file.
    file: File,
}

impl EventLog {
    /// Create an empty log directory. Fails if it already exists.
    pub fn create(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir(&dir)?;
        let path = dir.join(EVENTS_JSONL);
        let file = OpenOptions::new()
            .read(true)
            .create_new(true)
            .append(true)
            .open(&path)?;
        file.sync_all()?;
        fsync_dir(&dir)?;
        if let Some(parent) = dir.parent() {
            if parent != Path::new("") {
                fsync_dir(parent)?;
            }
        }
        Ok(Self {
            dir,
            events: Vec::new(),
            file,
        })
    }

    /// Open and replay. Verifies the chain. Fails closed on a broken hash.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        if !dir.is_dir() {
            return Err(Error::Io(format!(
                "log dir missing or not a directory: {}",
                dir.display()
            )));
        }
        let path = dir.join(EVENTS_JSONL);
        let mut file = OpenOptions::new().read(true).append(true).open(&path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let (events, complete_end, needs_newline) = replay_bytes(&bytes)?;
        let on_disk = bytes.len() as u64;
        if complete_end < on_disk {
            file.set_len(complete_end)?;
            file.sync_all()?;
        }
        if needs_newline {
            file.write_all(b"\n")?;
            file.sync_all()?;
        }
        Ok(Self { dir, events, file })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Append one record, fsync, return the filled Event. First seq is 0.
    pub fn append(&mut self, record: Append) -> Result<Event> {
        if record.event_type.is_empty() {
            return Err(Error::Schema(
                "event_type must be non-empty (F1 minLength 1)".into(),
            ));
        }
        let seq = self.events.len() as u64;
        let prev_hash = match self.events.last() {
            Some(ev) => ev.hash.clone(),
            None => GENESIS_PREV_HASH.to_string(),
        };
        let payload_hash_hex = payload_hash(&record.payload);
        let hash = event_hash(
            seq,
            &prev_hash,
            &payload_hash_hex,
            &record.timestamp,
            &record.event_type,
        );
        let event = Event {
            schema_id: SCHEMA_ID.to_string(),
            schema_version: SCHEMA_VERSION,
            seq,
            prev_hash,
            hash,
            timestamp: record.timestamp,
            event_type: record.event_type,
            task_id: record.task_id,
            attempt: record.attempt,
            node_id: record.node_id,
            payload: record.payload,
            payload_hash: payload_hash_hex,
        };
        write_record(&mut self.file, &event)?;
        self.events.push(event.clone());
        Ok(event)
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

    pub fn get(&self, seq: u64) -> Option<&Event> {
        let i = usize::try_from(seq).ok()?;
        self.events.get(i).filter(|ev| ev.seq == seq)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Event> {
        self.events.iter()
    }

    /// Recompute hashes and prev links. Used after replay and by tests.
    pub fn verify(&self) -> Result<()> {
        verify_events(&self.events)
    }

    /// Raft-ready last index: `seq` of the last record, or `None` if empty.
    pub fn last_index(&self) -> Option<u64> {
        self.last().map(|e| e.seq)
    }
}

/// SHA-256 of canonical JSON payload bytes, lowercase hex.
pub fn payload_hash(payload: &Value) -> String {
    format!("{:x}", Sha256::digest(canonical_json(payload)))
}

/// SHA-256 of `seq|prev_hash|payload_hash|timestamp|event_type`, lowercase hex.
pub fn event_hash(
    seq: u64,
    prev_hash: &str,
    payload_hash: &str,
    timestamp: &str,
    event_type: &str,
) -> String {
    let concat = format!("{seq}|{prev_hash}|{payload_hash}|{timestamp}|{event_type}");
    format!("{:x}", Sha256::digest(concat.as_bytes()))
}

/// Canonical JSON bytes of `payload` (sorted object keys, compact).
pub fn canonical_json(payload: &Value) -> Vec<u8> {
    serde_json::to_vec(&canonicalize(payload)).expect("serde_json::Value is always serializable")
}

fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            let mut out = serde_json::Map::with_capacity(map.len());
            for k in keys {
                out.insert(k.clone(), canonicalize(&map[k]));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonicalize).collect()),
        other => other.clone(),
    }
}

fn fsync_dir(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn write_record(file: &mut File, event: &Event) -> Result<()> {
    let mut line = serde_json::to_vec(event).map_err(|e| Error::Other(e.to_string()))?;
    line.push(b'\n');
    let start = file.metadata()?.len();
    if let Err(e) = file.write_all(&line).and_then(|()| file.sync_all()) {
        let _ = file.set_len(start);
        let _ = file.sync_all();
        return Err(e.into());
    }
    Ok(())
}

/// Replay JSONL bytes. A torn last line (incomplete JSON, no inventing a
/// record) yields the complete prefix. `complete_end` is the durable byte
/// length of that prefix; `needs_newline` is set when the last accepted
/// record was missing its trailing newline.
fn replay_bytes(bytes: &[u8]) -> Result<(Vec<Event>, u64, bool)> {
    let mut events = Vec::new();
    let mut offset = 0usize;
    let mut complete_end = 0u64;
    let mut needs_newline = false;

    while offset < bytes.len() {
        match bytes[offset..].iter().position(|&b| b == b'\n') {
            Some(i) => {
                let line = &bytes[offset..offset + i];
                let ev: Event = serde_json::from_slice(line)
                    .map_err(|e| Error::Corrupt(format!("invalid json: {e}")))?;
                events.push(ev);
                offset += i + 1;
                complete_end = offset as u64;
            }
            None => {
                let line = &bytes[offset..];
                if let Ok(ev) = serde_json::from_slice::<Event>(line) {
                    events.push(ev);
                    complete_end = bytes.len() as u64;
                    needs_newline = true;
                }
                break;
            }
        }
    }

    verify_events(&events)?;
    Ok((events, complete_end, needs_newline))
}

fn verify_events(events: &[Event]) -> Result<()> {
    let mut prev: &str = GENESIS_PREV_HASH;
    for (i, ev) in events.iter().enumerate() {
        let seq = i as u64;
        if ev.seq != seq {
            return Err(Error::BrokenChain(ev.seq));
        }
        if ev.schema_id != SCHEMA_ID {
            return Err(Error::Schema(format!("schema_id {}", ev.schema_id)));
        }
        if ev.schema_version < 1 || ev.schema_version > SCHEMA_VERSION {
            return Err(Error::Schema(format!(
                "unknown schema_version {}",
                ev.schema_version
            )));
        }
        if ev.event_type.is_empty() {
            return Err(Error::Schema("event_type must be non-empty".into()));
        }
        if ev.prev_hash != prev {
            return Err(Error::BrokenChain(ev.seq));
        }
        let ph = payload_hash(&ev.payload);
        if ev.payload_hash != ph {
            return Err(Error::Corrupt(format!(
                "payload_hash mismatch at seq {seq}"
            )));
        }
        let eh = event_hash(
            ev.seq,
            &ev.prev_hash,
            &ev.payload_hash,
            &ev.timestamp,
            &ev.event_type,
        );
        if ev.hash != eh {
            return Err(Error::Corrupt(format!("hash mismatch at seq {seq}")));
        }
        prev = ev.hash.as_str();
    }
    Ok(())
}
