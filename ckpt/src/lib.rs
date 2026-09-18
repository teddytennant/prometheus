//! Async sharded checkpoint, in-memory redundancy (spec 5.2, 5.5, 15.4, 15.5 A5).
//!
//! Persistent checkpoints go to an object-store [`Store`] (CPU tests use a
//! directory). In-memory checkpoints go to host RAM, the stand-in for Grace
//! offload, and are replicated [`MEMORY_REPLICAS`] times (spec 5.5: two other
//! racks). Only one DP replica's shards are written.
//!
//! A checkpoint holds sharded weights, optimizer state, loader state, and RNG
//! (spec 15.4). The JSON envelope is F1 `prometheus.checkpoint` v1. Blob bytes
//! live next to the manifest, addressed by `content_hash`.
//!
//! `created_at` is supplied by the caller. Nothing here reads the wall clock.
//!
//! Gate: V4 (kill -9 mid-step, in-memory restore, SDC hash, bad shard, host-RAM
//! offload). Elastic DP, SDC detection, and spike rollback live in `control/`
//! (A6). This crate is save, restore, and the in-memory replicas those use.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

pub const SCHEMA_ID: &str = "prometheus.checkpoint";
pub const SCHEMA_VERSION: u32 = 1;

/// Spec 5.5: in-memory copy on two other racks. Host RAM stands in for Grace.
pub const MEMORY_REPLICAS: usize = 2;

pub type Result<T> = std::result::Result<T, CkptError>;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CkptError {
    #[error("checkpoint not found: {0}")]
    NotFound(String),
    #[error("shard missing: {name} rank {shard_rank}")]
    MissingShard { name: String, shard_rank: u64 },
    #[error("hash mismatch: expected {expected} got {got}")]
    HashMismatch { expected: String, got: String },
    #[error("schema: {0}")]
    Schema(String),
    #[error("only one DP replica's shards may be written")]
    MultipleReplicas,
    #[error("in-memory replica {0} missing")]
    ReplicaLost(usize),
    #[error("{0}")]
    Message(String),
}

/// Lowercase hex SHA-256 of raw bytes. F1 `content_hash` / combined hashes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    F32,
    Bf16,
    Fp8,
    Fp4,
    Nvfp4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerKind {
    Adamw,
    Muon,
    Soap,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mesh {
    pub dp: u32,
    pub fsdp: u32,
    pub ep: u32,
    pub pp: u32,
    pub cp: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Precision {
    pub param_dtype: Dtype,
    pub remat: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightMeta {
    pub name: String,
    pub shard_rank: u64,
    pub shape: Vec<u64>,
    pub dtype: Dtype,
    pub content_hash: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mesh_axes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizerMeta {
    pub name: String,
    pub shard_rank: u64,
    pub kind: OptimizerKind,
    pub content_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RngState {
    pub scope: String,
    pub rank: u32,
    pub state: String,
}

/// F1 `prometheus.checkpoint` v1 envelope payload (hashes, not blob bytes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub checkpoint_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    pub step: u64,
    pub rung: u32,
    pub weights_hash: String,
    pub optimizer_hash: String,
    pub loader_hash: String,
    pub rng_hash: String,
    pub tokenizer_id: String,
    pub precision: Precision,
    pub mesh: Mesh,
    pub weights: Vec<WeightMeta>,
    pub optimizer: Vec<OptimizerMeta>,
    pub loader_state: Value,
    pub rng: Vec<RngState>,
    pub created_at: String,
    pub git_commit: String,
}

impl Manifest {
    pub fn to_json(&self) -> Result<Value> {
        let mut payload =
            serde_json::to_value(self).map_err(|err| CkptError::Schema(err.to_string()))?;
        let obj = payload
            .as_object_mut()
            .ok_or_else(|| CkptError::Schema("manifest is not an object".into()))?;
        let mut envelope = Map::new();
        envelope.insert("schema".into(), json!(SCHEMA_ID));
        envelope.insert("schema_version".into(), json!(SCHEMA_VERSION));
        envelope.append(obj);
        Ok(Value::Object(envelope))
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        let obj = value
            .as_object()
            .ok_or_else(|| CkptError::Schema("checkpoint is not an object".into()))?;
        let schema = obj
            .get("schema")
            .and_then(Value::as_str)
            .ok_or_else(|| CkptError::Schema("missing schema".into()))?;
        if schema != SCHEMA_ID {
            return Err(CkptError::Schema(format!(
                "expected {SCHEMA_ID}, got {schema}"
            )));
        }
        let version = obj
            .get("schema_version")
            .and_then(Value::as_u64)
            .ok_or_else(|| CkptError::Schema("missing schema_version".into()))?;
        if version != u64::from(SCHEMA_VERSION) {
            return Err(CkptError::Schema(format!(
                "expected schema_version {SCHEMA_VERSION}, got {version}"
            )));
        }
        let mut payload = obj.clone();
        payload.remove("schema");
        payload.remove("schema_version");
        serde_json::from_value(Value::Object(payload))
            .map_err(|err| CkptError::Schema(err.to_string()))
    }
}

/// One shard's bytes. `content_hash` is `sha256_hex(bytes)` when filled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardBlob {
    pub name: String,
    pub shard_rank: u64,
    pub bytes: Vec<u8>,
}

/// In-memory checkpoint: F1 manifest plus shard bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    pub manifest: Manifest,
    pub weights: Vec<ShardBlob>,
    pub optimizer: Vec<ShardBlob>,
}

/// Persistent or in-memory blob backend. CPU tests use [`MemoryStore`] and
/// [`DirStore`]. Production persistent store is an object bucket.
pub trait Store {
    fn put(&mut self, key: &str, bytes: &[u8]) -> Result<()>;
    fn get(&self, key: &str) -> Result<Vec<u8>>;
    fn contains(&self, key: &str) -> Result<bool>;
}

/// Host-RAM blob map. Also the Grace-offload stand-in (spec 5.2, V4).
pub struct MemoryStore {
    _private: (),
}

impl MemoryStore {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Store for MemoryStore {
    fn put(&mut self, _key: &str, _bytes: &[u8]) -> Result<()> {
        unimplemented!("A5: MemoryStore::put")
    }

    fn get(&self, _key: &str) -> Result<Vec<u8>> {
        unimplemented!("A5: MemoryStore::get")
    }

    fn contains(&self, _key: &str) -> Result<bool> {
        unimplemented!("A5: MemoryStore::contains")
    }
}

/// Directory-backed persistent store (object-store stand-in for CPU tests).
pub struct DirStore {
    _root: std::path::PathBuf,
}

impl DirStore {
    pub fn open(root: std::path::PathBuf) -> Result<Self> {
        let _ = root;
        unimplemented!("A5: DirStore::open")
    }
}

impl Store for DirStore {
    fn put(&mut self, _key: &str, _bytes: &[u8]) -> Result<()> {
        unimplemented!("A5: DirStore::put")
    }

    fn get(&self, _key: &str) -> Result<Vec<u8>> {
        unimplemented!("A5: DirStore::get")
    }

    fn contains(&self, _key: &str) -> Result<bool> {
        unimplemented!("A5: DirStore::contains")
    }
}

/// Host-RAM stand-in for streaming optimizer state off Grace (spec 5.2, V4).
pub struct HostRamOffload {
    _private: (),
}

impl HostRamOffload {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn put(&mut self, _name: &str, _bytes: Vec<u8>) -> Result<()> {
        unimplemented!("A5: HostRamOffload::put")
    }

    pub fn get(&self, _name: &str) -> Result<Vec<u8>> {
        unimplemented!("A5: HostRamOffload::get")
    }

    pub fn remove(&mut self, _name: &str) -> Result<()> {
        unimplemented!("A5: HostRamOffload::remove")
    }
}

impl Default for HostRamOffload {
    fn default() -> Self {
        Self::new()
    }
}

/// Save and restore. In-memory path keeps [`MEMORY_REPLICAS`] copies. Persistent
/// path writes one DP replica's shards through [`Store`].
pub struct Checkpointer {
    _private: (),
}

impl Checkpointer {
    pub fn new(_persistent: Box<dyn Store>) -> Self {
        Self { _private: () }
    }

    /// Write `ckpt` into host RAM, replicated [`MEMORY_REPLICAS`] times.
    pub fn save_in_memory(&mut self, _ckpt: &Checkpoint) -> Result<String> {
        unimplemented!("A5: Checkpointer::save_in_memory")
    }

    /// Write `ckpt` through the persistent store. Spec 5.5 is async (does not
    /// stall the train step); a CPU implementation may finish the write before
    /// returning.
    pub fn save_persistent(&mut self, _ckpt: &Checkpoint) -> Result<String> {
        unimplemented!("A5: Checkpointer::save_persistent")
    }

    pub fn restore(&self, _checkpoint_id: &str) -> Result<Checkpoint> {
        unimplemented!("A5: Checkpointer::restore")
    }

    pub fn restore_latest_memory(&self) -> Result<Checkpoint> {
        unimplemented!("A5: Checkpointer::restore_latest_memory")
    }

    /// Bytes held by in-memory replica `index` in `0..MEMORY_REPLICAS`.
    pub fn replica_len(&self, _index: usize) -> Result<usize> {
        unimplemented!("A5: Checkpointer::replica_len")
    }
}
