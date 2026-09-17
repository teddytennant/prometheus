//! Content-addressed artifact store (spec 15.2, 15.5 H8).
//!
//! Blob bytes are SHA-256 addressed and replicated to at least two backends.
//! Named pins keep blobs alive; [`Store::gc`] deletes unpinned blobs from every
//! replica. Pin set is an H2 `EventLog` at `<dir>/log/` so a kill -9 is a replay.
//!
//! CPU tests use two local directories as backends. Production can point those
//! at NCShare `/work` and a second disk (git remotes and object buckets come
//! later without changing [`Store::put`] / [`Store::get`]).

use prometheus_log::{Append, EventLog};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest as _, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const MIN_REPLICAS: usize = 2;

pub const EVENT_PUT: &str = "cas.put";
pub const EVENT_PIN: &str = "cas.pin";
pub const EVENT_UNPIN: &str = "cas.unpin";
pub const EVENT_GC: &str = "cas.gc";

const EVENT_TIMESTAMP: &str = "1970-01-01T00:00:00Z";

/// Lowercase hex SHA-256 of the raw bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Digest(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PinName(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreConfig {
    pub min_replicas: usize,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            min_replicas: MIN_REPLICAS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GcReport {
    pub blobs_removed: u64,
    pub bytes_freed: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("need at least {MIN_REPLICAS} backends, have {0}")]
    TooFewBackends(usize),
    #[error("put wrote {wrote} replicas, need {need}")]
    UnderReplicated { wrote: usize, need: usize },
    #[error("digest mismatch: expected {expected} got {got}")]
    DigestMismatch { expected: String, got: String },
    #[error("duplicate pin {0}")]
    DuplicatePin(String),
    #[error("{0}")]
    Log(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<prometheus_log::Error> for Error {
    fn from(err: prometheus_log::Error) -> Self {
        Error::Log(err.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Other(err.to_string())
    }
}

/// Content-addressed store. Blobs live as `<backend>/<digest-hex>` files.
pub struct Store {
    dir: PathBuf,
    log: EventLog,
    config: StoreConfig,
    backends: Vec<PathBuf>,
    /// Pin name → digest hex. Replayed from the event log on [`Store::open`].
    pins: HashMap<String, String>,
}

impl Store {
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config(&self) -> &StoreConfig {
        &self.config
    }

    pub fn log(&self) -> &EventLog {
        &self.log
    }

    pub fn backends(&self) -> &[PathBuf] {
        &self.backends
    }

    /// Fails if `dir` exists or `backends.len() < config.min_replicas`.
    /// Creates `dir/log/` as an H2 EventLog. No events on create.
    pub fn create(
        dir: impl AsRef<Path>,
        backends: Vec<PathBuf>,
        config: StoreConfig,
    ) -> Result<Self> {
        if backends.len() < config.min_replicas {
            return Err(Error::TooFewBackends(backends.len()));
        }
        let dir = dir.as_ref().to_path_buf();
        if dir.exists() {
            return Err(Error::Other(format!("already exists: {}", dir.display())));
        }
        fs::create_dir(&dir)?;
        let log = match EventLog::create(dir.join("log")) {
            Ok(log) => log,
            Err(err) => {
                let _ = fs::remove_dir_all(&dir);
                return Err(err.into());
            }
        };
        Ok(Self {
            dir,
            log,
            config,
            backends,
            pins: HashMap::new(),
        })
    }

    /// Replay pins from `dir/log/`. Backend dirs are caller-provided, not on disk
    /// in the log (a lost replica is replaced by passing a new path).
    pub fn open(
        dir: impl AsRef<Path>,
        backends: Vec<PathBuf>,
        config: StoreConfig,
    ) -> Result<Self> {
        if backends.len() < config.min_replicas {
            return Err(Error::TooFewBackends(backends.len()));
        }
        let dir = dir.as_ref().to_path_buf();
        let log = EventLog::open(dir.join("log"))?;
        let pins = replay_pins(&log);
        Ok(Self {
            dir,
            log,
            config,
            backends,
            pins,
        })
    }

    /// SHA-256 the bytes, write to every backend, succeed if at least
    /// `min_replicas` writes match. Emit `cas.put`. Partial writes of a failed
    /// put are orphans for GC.
    pub fn put(&mut self, bytes: &[u8]) -> Result<Digest> {
        let hex = digest_hex(bytes);
        let mut wrote = 0usize;
        for backend in &self.backends {
            if write_replica(backend, &hex, bytes).is_ok() {
                wrote += 1;
            }
        }
        if wrote < self.config.min_replicas {
            return Err(Error::UnderReplicated {
                wrote,
                need: self.config.min_replicas,
            });
        }
        self.append_event(
            EVENT_PUT,
            json!({
                "digest": hex,
            }),
        )?;
        Ok(Digest(hex))
    }

    /// Read from any replica. Verify SHA-256. Fail closed on mismatch.
    /// Prefers a matching replica if one exists.
    pub fn get(&self, digest: &Digest) -> Result<Vec<u8>> {
        let mut last_bad: Option<String> = None;
        for backend in &self.backends {
            let bytes = match fs::read(blob_path(backend, &digest.0)) {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };
            let got = digest_hex(&bytes);
            if got == digest.0 {
                return Ok(bytes);
            }
            last_bad = Some(got);
        }
        if let Some(got) = last_bad {
            return Err(Error::DigestMismatch {
                expected: digest.0.clone(),
                got,
            });
        }
        Err(Error::NotFound(digest.0.clone()))
    }

    /// Named pin. Unknown digest is NotFound. Duplicate name is DuplicatePin.
    pub fn pin(&mut self, digest: &Digest, name: PinName) -> Result<()> {
        if self.replica_count(digest)? == 0 {
            return Err(Error::NotFound(digest.0.clone()));
        }
        if self.pins.contains_key(&name.0) {
            return Err(Error::DuplicatePin(name.0));
        }
        self.append_event(
            EVENT_PIN,
            json!({
                "name": name.0,
                "digest": digest.0,
            }),
        )?;
        self.pins.insert(name.0, digest.0.clone());
        Ok(())
    }

    pub fn unpin(&mut self, name: &PinName) -> Result<()> {
        if !self.pins.contains_key(&name.0) {
            return Err(Error::NotFound(name.0.clone()));
        }
        self.append_event(
            EVENT_UNPIN,
            json!({
                "name": name.0,
            }),
        )?;
        self.pins.remove(&name.0);
        Ok(())
    }

    /// Delete blobs that have no pin from every backend that has them.
    pub fn gc(&mut self) -> Result<GcReport> {
        let pinned: HashSet<&str> = self.pins.values().map(String::as_str).collect();
        let mut sizes: HashMap<String, u64> = HashMap::new();
        let mut paths: Vec<(String, PathBuf)> = Vec::new();
        for backend in &self.backends {
            for path in walk_files(backend) {
                let Some(digest) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                let digest = digest.to_string();
                if pinned.contains(digest.as_str()) {
                    continue;
                }
                let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                *sizes.entry(digest.clone()).or_insert(0) += size;
                paths.push((digest, path));
            }
        }
        for (_digest, path) in &paths {
            fs::remove_file(path)?;
        }
        let blobs_removed = sizes.len() as u64;
        let bytes_freed = sizes.values().copied().sum();
        self.append_event(
            EVENT_GC,
            json!({
                "blobs_removed": blobs_removed,
                "bytes_freed": bytes_freed,
            }),
        )?;
        Ok(GcReport {
            blobs_removed,
            bytes_freed,
        })
    }

    /// How many backends currently have this digest.
    pub fn replica_count(&self, digest: &Digest) -> Result<usize> {
        Ok(self
            .backends
            .iter()
            .filter(|backend| blob_path(backend, &digest.0).is_file())
            .count())
    }

    fn append_event(&mut self, event_type: &str, payload: Value) -> Result<()> {
        self.log.append(Append {
            event_type: event_type.to_string(),
            payload,
            timestamp: EVENT_TIMESTAMP.to_string(),
            task_id: None,
            attempt: None,
            node_id: None,
        })?;
        Ok(())
    }
}

fn blob_path(backend: &Path, digest: &str) -> PathBuf {
    backend.join(digest)
}

fn digest_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_replica(backend: &Path, digest: &str, bytes: &[u8]) -> std::io::Result<()> {
    let path = blob_path(backend, digest);
    let mut file = File::create(&path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    if let Ok(dir) = File::open(backend) {
        let _ = dir.sync_all();
    }
    Ok(())
}

fn walk_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if !dir.exists() {
        return out;
    }
    fn rec(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                rec(&path, out);
            } else if path.is_file() {
                out.push(path);
            }
        }
    }
    rec(dir, &mut out);
    out
}

fn replay_pins(log: &EventLog) -> HashMap<String, String> {
    let mut pins = HashMap::new();
    for event in log.iter() {
        match event.event_type.as_str() {
            EVENT_PIN => {
                let Some(name) = event.payload.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let Some(digest) = event.payload.get("digest").and_then(Value::as_str) else {
                    continue;
                };
                pins.insert(name.to_string(), digest.to_string());
            }
            EVENT_UNPIN => {
                if let Some(name) = event.payload.get("name").and_then(Value::as_str) {
                    pins.remove(name);
                }
            }
            _ => {}
        }
    }
    pins
}
