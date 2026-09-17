//! Content-addressed artifact store (spec 15.2, 15.5 H8).
//!
//! Blob bytes are SHA-256 addressed and replicated to at least two backends.
//! Named pins keep blobs alive; [`Store::gc`] deletes unpinned blobs from every
//! replica. Pin set is an H2 `EventLog` at `<dir>/log/` so a kill -9 is a replay.
//!
//! CPU tests use two local directories as backends. Production can point those
//! at NCShare `/work` and a second disk (git remotes and object buckets come
//! later without changing [`Store::put`] / [`Store::get`]).

use prometheus_log::EventLog;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const MIN_REPLICAS: usize = 2;

pub const EVENT_PUT: &str = "cas.put";
pub const EVENT_PIN: &str = "cas.pin";
pub const EVENT_UNPIN: &str = "cas.unpin";
pub const EVENT_GC: &str = "cas.gc";

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

/// Content-addressed store. Bodies of create/open/put/get/pin/unpin/gc are unimplemented.
pub struct Store {
    dir: PathBuf,
    log: EventLog,
    config: StoreConfig,
    backends: Vec<PathBuf>,
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
        let _ = (dir, backends, config);
        unimplemented!("H8: Store::create")
    }

    /// Replay pins from `dir/log/`. Backend dirs are caller-provided, not on disk
    /// in the log (a lost replica is replaced by passing a new path).
    pub fn open(
        dir: impl AsRef<Path>,
        backends: Vec<PathBuf>,
        config: StoreConfig,
    ) -> Result<Self> {
        let _ = (dir, backends, config);
        unimplemented!("H8: Store::open")
    }

    /// SHA-256 the bytes, write to every backend, succeed if at least
    /// `min_replicas` writes match. Emit `cas.put`. Partial writes of a failed
    /// put are orphans for GC.
    pub fn put(&mut self, bytes: &[u8]) -> Result<Digest> {
        let _ = bytes;
        unimplemented!("H8: Store::put")
    }

    /// Read from any replica. Verify SHA-256. Fail closed on mismatch.
    pub fn get(&self, digest: &Digest) -> Result<Vec<u8>> {
        let _ = digest;
        unimplemented!("H8: Store::get")
    }

    /// Named pin. Unknown digest is NotFound. Duplicate name is DuplicatePin.
    pub fn pin(&mut self, digest: &Digest, name: PinName) -> Result<()> {
        let _ = (digest, name);
        unimplemented!("H8: Store::pin")
    }

    pub fn unpin(&mut self, name: &PinName) -> Result<()> {
        let _ = name;
        unimplemented!("H8: Store::unpin")
    }

    /// Delete blobs that have no pin from every backend that has them.
    pub fn gc(&mut self) -> Result<GcReport> {
        unimplemented!("H8: Store::gc")
    }

    /// How many backends currently have this digest.
    pub fn replica_count(&self, digest: &Digest) -> Result<usize> {
        let _ = digest;
        unimplemented!("H8: Store::replica_count")
    }
}
