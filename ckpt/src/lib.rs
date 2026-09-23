//! Async sharded checkpoint, in-memory redundancy (spec 5.2, 5.5, 15.4, 15.5 A5).
//!
//! Persistent checkpoints go to an object-store [`Store`] (CPU tests use a
//! directory). In-memory checkpoints go to host RAM, the stand-in for Grace
//! offload, and are replicated [`MEMORY_REPLICAS`] times (spec 5.5: two other
//! racks). Only one DP replica's shards are written.
//!
//! A checkpoint holds sharded weights, optimizer state, loader state, and RNG
//! (spec 15.4). The JSON envelope is F1 `prometheus.checkpoint` v1 (`schema_id`,
//! not `schema`). Blob bytes live next to the manifest, addressed by
//! `content_hash`. Combined hashes are not in F1; integrity is per-shard.
//!
//! `created_at` is supplied by the caller. Nothing here reads the wall clock.
//!
//! Gate: V4 (kill -9 mid-step, in-memory restore, SDC hash, bad shard, host-RAM
//! offload). Elastic DP, SDC detection, and spike rollback live in `control/`
//! (A6). This crate is save, restore, and the in-memory replicas those use.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;

pub const SCHEMA_ID: &str = "prometheus.checkpoint";
pub const SCHEMA_VERSION: u32 = 1;

/// Spec 5.5: in-memory copy on two other racks. Host RAM stands in for Grace.
pub const MEMORY_REPLICAS: usize = 2;

/// Spec 5.5: copies live on this many racks other than the source.
pub const CROSS_RACK_COPIES: usize = 2;

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
    #[error("persistent save already in flight")]
    SaveInFlight,
    #[error("rack placement: {0}")]
    Rack(String),
    #[error("{0}")]
    Message(String),
}

/// Shared by [`Checkpointer`] and the in-flight [`PendingSave`].
///
/// The in-flight flag has to live here, not only on the handle: a second
/// `save_persistent_async` must see a save that has already returned. One
/// worker, one finished flag, one error slot — no job channel.
struct SaveState {
    in_flight: AtomicBool,
    finished: AtomicBool,
    error: Mutex<Option<CkptError>>,
    cv: Condvar,
}

impl SaveState {
    fn new() -> Self {
        Self {
            in_flight: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            error: Mutex::new(None),
            cv: Condvar::new(),
        }
    }
}

/// Handle for a persistent save that has not finished.
///
/// Spec 5.5: the train step must not stall on the object-store write.
/// `Checkpointer::save_persistent_async` returns this before the store's
/// `put` calls finish. Drop detaches; it does not join the worker, so a
/// blocked `put` cannot stall `Drop`.
pub struct PendingSave {
    checkpoint_id: String,
    state: Arc<SaveState>,
}

impl PendingSave {
    /// Id `restore` will accept after `wait` returns Ok.
    pub fn checkpoint_id(&self) -> &str {
        &self.checkpoint_id
    }

    /// True once every shard and the manifest are durable in the store.
    pub fn is_finished(&self) -> bool {
        // Finished is stored with Release after the error slot is published,
        // so Acquire here sees that write. A store error is not durable.
        self.state.finished.load(Ordering::Acquire) && lock_error(&self.state.error).is_none()
    }

    /// Block until the save finishes.
    ///
    /// Store errors surface here, not from `save_persistent_async`. After
    /// Ok, `Checkpointer::restore` of `checkpoint_id` matches the checkpoint
    /// a synchronous `save_persistent` would have written: same hashes, same
    /// bytes.
    pub fn wait(self) -> Result<()> {
        let mut slot = lock_error(&self.state.error);
        while !self.state.finished.load(Ordering::Acquire) {
            slot = self
                .state
                .cv
                .wait(slot)
                .unwrap_or_else(PoisonError::into_inner);
        }
        let err = slot.take();
        drop(slot);
        // Cleared only once this call observes completion, so a second save
        // before `wait` returns still sees the in-flight flag.
        self.state.in_flight.store(false, Ordering::Release);
        match err {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

/// Lowercase hex SHA-256 of raw bytes. F1 `content_hash`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    Fp32,
    Bf16,
    Fp16,
    Fp8,
    Nvfp4,
    Int8,
    Int32,
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
    pub remat: String,
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
    pub state_hash: String,
}

/// F1 `prometheus.checkpoint` v1 envelope payload (hashes, not blob bytes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub checkpoint_id: String,
    #[serde(default)]
    pub parent_checkpoint_id: Option<String>,
    pub step: u64,
    pub rung: u32,
    pub model_config_hash: String,
    pub data_mix_hash: String,
    pub tokenizer_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<Precision>,
    pub mesh: Mesh,
    pub weights: Vec<WeightMeta>,
    pub optimizer: Vec<OptimizerMeta>,
    pub loader_state: Value,
    pub rng: Vec<RngState>,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
}

impl Manifest {
    pub fn to_json(&self) -> Result<Value> {
        let mut payload =
            serde_json::to_value(self).map_err(|err| CkptError::Schema(err.to_string()))?;
        let obj = payload
            .as_object_mut()
            .ok_or_else(|| CkptError::Schema("manifest is not an object".into()))?;
        let mut envelope = Map::new();
        envelope.insert("schema_id".into(), json!(SCHEMA_ID));
        envelope.insert("schema_version".into(), json!(SCHEMA_VERSION));
        envelope.append(obj);
        Ok(Value::Object(envelope))
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        let obj = value
            .as_object()
            .ok_or_else(|| CkptError::Schema("checkpoint is not an object".into()))?;
        let schema = obj
            .get("schema_id")
            .and_then(Value::as_str)
            .ok_or_else(|| CkptError::Schema("missing schema_id".into()))?;
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
        payload.remove("schema_id");
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
/// `Send` so a persistent save can run on another thread (spec 5.5).
pub trait Store: Send {
    fn put(&mut self, key: &str, bytes: &[u8]) -> Result<()>;
    fn get(&self, key: &str) -> Result<Vec<u8>>;
    fn contains(&self, key: &str) -> Result<bool>;
}

/// Host-RAM blob map. Also the Grace-offload stand-in (spec 5.2, V4).
pub struct MemoryStore {
    map: HashMap<String, Vec<u8>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Store for MemoryStore {
    fn put(&mut self, key: &str, bytes: &[u8]) -> Result<()> {
        self.map.insert(key.to_string(), bytes.to_vec());
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        self.map
            .get(key)
            .cloned()
            .ok_or_else(|| CkptError::NotFound(key.to_string()))
    }

    fn contains(&self, key: &str) -> Result<bool> {
        Ok(self.map.contains_key(key))
    }
}

/// Directory-backed persistent store (object-store stand-in for CPU tests).
pub struct DirStore {
    root: PathBuf,
}

impl DirStore {
    pub fn open(root: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&root)
            .map_err(|err| CkptError::Message(format!("mkdir {}: {err}", root.display())))?;
        Ok(Self { root })
    }
}

impl Store for DirStore {
    fn put(&mut self, key: &str, bytes: &[u8]) -> Result<()> {
        let path = self.root.join(key);
        std::fs::write(&path, bytes)
            .map_err(|err| CkptError::Message(format!("write {}: {err}", path.display())))
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        let path = self.root.join(key);
        match std::fs::read(&path) {
            Ok(bytes) => Ok(bytes),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                Err(CkptError::NotFound(key.to_string()))
            }
            Err(err) => Err(CkptError::Message(format!(
                "read {}: {err}",
                path.display()
            ))),
        }
    }

    fn contains(&self, key: &str) -> Result<bool> {
        Ok(self.root.join(key).is_file())
    }
}

/// Host-RAM stand-in for streaming optimizer state off Grace (spec 5.2, V4).
pub struct HostRamOffload {
    map: HashMap<String, Vec<u8>>,
}

impl HostRamOffload {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn put(&mut self, name: &str, bytes: Vec<u8>) -> Result<()> {
        self.map.insert(name.to_string(), bytes);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Result<Vec<u8>> {
        self.map
            .get(name)
            .cloned()
            .ok_or_else(|| CkptError::NotFound(name.to_string()))
    }

    pub fn remove(&mut self, name: &str) -> Result<()> {
        self.map
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| CkptError::NotFound(name.to_string()))
    }
}

impl Default for HostRamOffload {
    fn default() -> Self {
        Self::new()
    }
}

fn find_blob<'a>(blobs: &'a [ShardBlob], name: &str, shard_rank: u64) -> Option<&'a ShardBlob> {
    blobs
        .iter()
        .find(|blob| blob.name == name && blob.shard_rank == shard_rank)
}

fn payload_len(ckpt: &Checkpoint) -> usize {
    ckpt.weights
        .iter()
        .map(|blob| blob.bytes.len())
        .sum::<usize>()
        + ckpt
            .optimizer
            .iter()
            .map(|blob| blob.bytes.len())
            .sum::<usize>()
}

fn prepare_checkpoint(ckpt: &Checkpoint) -> Result<Checkpoint> {
    let mut out = ckpt.clone();
    for meta in &mut out.manifest.weights {
        let blob = find_blob(&ckpt.weights, &meta.name, meta.shard_rank).ok_or_else(|| {
            CkptError::MissingShard {
                name: meta.name.clone(),
                shard_rank: meta.shard_rank,
            }
        })?;
        let got = sha256_hex(&blob.bytes);
        if !meta.content_hash.is_empty() && meta.content_hash != got {
            return Err(CkptError::HashMismatch {
                expected: meta.content_hash.clone(),
                got,
            });
        }
        meta.content_hash = got;
    }
    for meta in &mut out.manifest.optimizer {
        let blob = find_blob(&ckpt.optimizer, &meta.name, meta.shard_rank).ok_or_else(|| {
            CkptError::MissingShard {
                name: meta.name.clone(),
                shard_rank: meta.shard_rank,
            }
        })?;
        let got = sha256_hex(&blob.bytes);
        if !meta.content_hash.is_empty() && meta.content_hash != got {
            return Err(CkptError::HashMismatch {
                expected: meta.content_hash.clone(),
                got,
            });
        }
        meta.content_hash = got;
    }
    Ok(out)
}

fn verify_checkpoint(ckpt: &Checkpoint) -> Result<()> {
    for meta in &ckpt.manifest.weights {
        let blob = find_blob(&ckpt.weights, &meta.name, meta.shard_rank).ok_or_else(|| {
            CkptError::MissingShard {
                name: meta.name.clone(),
                shard_rank: meta.shard_rank,
            }
        })?;
        let got = sha256_hex(&blob.bytes);
        if meta.content_hash != got {
            return Err(CkptError::HashMismatch {
                expected: meta.content_hash.clone(),
                got,
            });
        }
    }
    for meta in &ckpt.manifest.optimizer {
        let blob = find_blob(&ckpt.optimizer, &meta.name, meta.shard_rank).ok_or_else(|| {
            CkptError::MissingShard {
                name: meta.name.clone(),
                shard_rank: meta.shard_rank,
            }
        })?;
        let got = sha256_hex(&blob.bytes);
        if meta.content_hash != got {
            return Err(CkptError::HashMismatch {
                expected: meta.content_hash.clone(),
                got,
            });
        }
    }
    Ok(())
}

fn load_shard(
    store: &dyn Store,
    name: &str,
    shard_rank: u64,
    content_hash: &str,
) -> Result<ShardBlob> {
    let bytes = match store.get(content_hash) {
        Ok(bytes) => bytes,
        Err(CkptError::NotFound(_)) => {
            return Err(CkptError::MissingShard {
                name: name.to_string(),
                shard_rank,
            });
        }
        Err(err) => return Err(err),
    };
    let got = sha256_hex(&bytes);
    if got != content_hash {
        return Err(CkptError::HashMismatch {
            expected: content_hash.to_string(),
            got,
        });
    }
    Ok(ShardBlob {
        name: name.to_string(),
        shard_rank,
        bytes,
    })
}

fn lock_store(store: &Mutex<Box<dyn Store>>) -> MutexGuard<'_, Box<dyn Store>> {
    store.lock().unwrap_or_else(PoisonError::into_inner)
}

fn lock_error(slot: &Mutex<Option<CkptError>>) -> MutexGuard<'_, Option<CkptError>> {
    slot.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Same puts as the synchronous path: shard bytes under their sha256, then the
/// manifest under `checkpoint_id`. The prepared checkpoint is borrowed, not
/// cloned — `prepare_checkpoint` already copied shard bytes once.
fn persist(store: &Mutex<Box<dyn Store>>, prepared: &Checkpoint) -> Result<()> {
    for blob in prepared.weights.iter().chain(prepared.optimizer.iter()) {
        let key = sha256_hex(&blob.bytes);
        lock_store(store).put(&key, &blob.bytes)?;
    }
    let envelope = prepared.manifest.to_json()?;
    let bytes = serde_json::to_vec(&envelope).map_err(|err| CkptError::Schema(err.to_string()))?;
    lock_store(store).put(&prepared.manifest.checkpoint_id, &bytes)?;
    Ok(())
}

/// Save and restore. In-memory path keeps [`MEMORY_REPLICAS`] copies. Persistent
/// path writes one DP replica's shards through [`Store`].
pub struct Checkpointer {
    /// Shared with the save worker. `put` takes `&mut self`, and `restore`
    /// after `wait` must see those puts, so the store is not moved out.
    store: Arc<Mutex<Box<dyn Store>>>,
    replicas: Vec<Option<Checkpoint>>,
    save: Arc<SaveState>,
}

impl Checkpointer {
    pub fn new(persistent: Box<dyn Store>) -> Self {
        Self {
            store: Arc::new(Mutex::new(persistent)),
            replicas: vec![None; MEMORY_REPLICAS],
            save: Arc::new(SaveState::new()),
        }
    }

    /// Write `ckpt` into host RAM, replicated [`MEMORY_REPLICAS`] times.
    pub fn save_in_memory(&mut self, ckpt: &Checkpoint) -> Result<String> {
        let prepared = prepare_checkpoint(ckpt)?;
        let id = prepared.manifest.checkpoint_id.clone();
        for slot in &mut self.replicas {
            *slot = Some(prepared.clone());
        }
        Ok(id)
    }

    /// Write `ckpt` through the persistent store and return after the write
    /// finishes. Existing callers keep this synchronous path.
    pub fn save_persistent(&mut self, ckpt: &Checkpoint) -> Result<String> {
        let prepared = prepare_checkpoint(ckpt)?;
        let id = prepared.manifest.checkpoint_id.clone();
        persist(&self.store, &prepared)?;
        Ok(id)
    }

    /// Start a persistent save and return before the store finishes.
    ///
    /// Preparation failures (schema, hash, multiple replicas) return Err and
    /// start no write. A store whose `put` blocks cannot delay this return:
    /// the call returns while the first `put` is still blocked. At most one
    /// async save is in flight on this checkpointer; a second call before
    /// `PendingSave::wait` returns `CkptError::SaveInFlight` and starts
    /// nothing. Dropping the handle does not cancel the write. `wait` is how
    /// the caller observes completion and store errors.
    ///
    /// `restore` during the write may return `NotFound`. After `wait` returns
    /// Ok, `restore` of the id matches `save_persistent` of the same
    /// checkpoint.
    pub fn save_persistent_async(&mut self, ckpt: &Checkpoint) -> Result<PendingSave> {
        // Claim the slot before prepare so a second call starts nothing — no
        // hash, no put. Prepare still runs on this thread, before any put.
        if self.save.in_flight.swap(true, Ordering::AcqRel) {
            return Err(CkptError::SaveInFlight);
        }
        let prepared = match prepare_checkpoint(ckpt) {
            Ok(prepared) => prepared,
            Err(err) => {
                self.save.in_flight.store(false, Ordering::Release);
                return Err(err);
            }
        };
        let id = prepared.manifest.checkpoint_id.clone();
        {
            let mut slot = lock_error(&self.save.error);
            *slot = None;
        }
        self.save.finished.store(false, Ordering::Release);

        let store = Arc::clone(&self.store);
        let state = Arc::clone(&self.save);
        // Detach on purpose: `PendingSave` does not hold the `JoinHandle`, so
        // its Drop cannot join a worker blocked inside `put`. One thread does
        // every put; the caller never enters `put`.
        let spawned = thread::Builder::new()
            .name("ckpt-save".into())
            .spawn(move || {
                let write = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    persist(&store, &prepared)
                }));
                let mut slot = lock_error(&state.error);
                match write {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => *slot = Some(err),
                    Err(_) => {
                        *slot = Some(CkptError::Message(
                            "persistent save worker panicked".to_string(),
                        ));
                    }
                }
                state.finished.store(true, Ordering::Release);
                state.cv.notify_all();
            });
        if let Err(err) = spawned {
            self.save.in_flight.store(false, Ordering::Release);
            return Err(CkptError::Message(format!(
                "failed to start save worker: {err}"
            )));
        }
        Ok(PendingSave {
            checkpoint_id: id,
            state: Arc::clone(&self.save),
        })
    }

    pub fn restore(&self, checkpoint_id: &str) -> Result<Checkpoint> {
        let guard = lock_store(&self.store);
        let store: &dyn Store = &**guard;
        let bytes = match store.get(checkpoint_id) {
            Ok(bytes) => bytes,
            Err(CkptError::NotFound(_)) => {
                return Err(CkptError::NotFound(checkpoint_id.to_string()));
            }
            Err(err) => return Err(err),
        };
        let value: Value =
            serde_json::from_slice(&bytes).map_err(|err| CkptError::Schema(err.to_string()))?;
        let manifest = Manifest::from_json(&value)?;
        let mut weights = Vec::with_capacity(manifest.weights.len());
        for meta in &manifest.weights {
            weights.push(load_shard(
                store,
                &meta.name,
                meta.shard_rank,
                &meta.content_hash,
            )?);
        }
        let mut optimizer = Vec::with_capacity(manifest.optimizer.len());
        for meta in &manifest.optimizer {
            optimizer.push(load_shard(
                store,
                &meta.name,
                meta.shard_rank,
                &meta.content_hash,
            )?);
        }
        let ckpt = Checkpoint {
            manifest,
            weights,
            optimizer,
        };
        verify_checkpoint(&ckpt)?;
        Ok(ckpt)
    }

    pub fn restore_latest_memory(&self) -> Result<Checkpoint> {
        let ckpt = self
            .replicas
            .first()
            .and_then(|slot| slot.clone())
            .ok_or_else(|| CkptError::NotFound("latest".to_string()))?;
        verify_checkpoint(&ckpt)?;
        Ok(ckpt)
    }

    /// Bytes held by in-memory replica `index` in `0..MEMORY_REPLICAS`.
    pub fn replica_len(&self, index: usize) -> Result<usize> {
        if index >= MEMORY_REPLICAS {
            return Err(CkptError::Message(format!(
                "replica index {index} out of range (MEMORY_REPLICAS={MEMORY_REPLICAS})"
            )));
        }
        match self.replicas.get(index).and_then(|slot| slot.as_ref()) {
            Some(ckpt) => Ok(payload_len(ckpt)),
            None => Err(CkptError::ReplicaLost(index)),
        }
    }

    /// Place `ckpt` in host RAM on two racks other than `source`.
    ///
    /// Spec 5.5: in-memory checkpoints go to Grace RAM on two other racks.
    /// Host RAM and the caller-supplied rack list stand in for that. This
    /// does not write the [`Store`], and it does not touch
    /// [`Self::save_in_memory`] or [`MEMORY_REPLICAS`].
    ///
    /// `racks` is the fleet list, in order. Each id is a non-empty opaque
    /// string (no trimming: `"a"` and `"a "` differ). `source` must occur in
    /// `racks`. Destinations are [`plan_cross_rack`]: the next two distinct
    /// racks after the first occurrence of `source`, wrapping, skipping
    /// `source`. Fewer than three distinct racks is [`CkptError::Rack`].
    ///
    /// The stored value is the prepared checkpoint, the same bytes
    /// [`Self::save_in_memory`] would keep for this input. Both destinations
    /// hold independent clones. Mutating one copy, or the value returned by
    /// [`Self::restore_from_rack`], must not change the other.
    ///
    /// A re-save from the same source replaces that source's copies and drops
    /// a previous destination that is no longer in the pair, but only if that
    /// rack still holds this source. A destination that already holds a
    /// different source is [`CkptError::Rack`], and the call changes nothing.
    /// Two sources whose pairs do not overlap both remain restorable.
    ///
    /// `restore_from_rack(source)` is [`CkptError::NotFound`] after a
    /// successful save: the source rack has no copy.
    ///
    /// On a checkpoint that fails `prepare` / `verify`, the error is the same
    /// one [`Self::save_in_memory`] would return, and no rack is written.
    pub fn save_cross_rack(
        &mut self,
        ckpt: &Checkpoint,
        source: &str,
        racks: &[String],
    ) -> Result<(String, String)> {
        let _ = (ckpt, source, racks);
        unimplemented!("save_cross_rack")
    }

    /// Clone the copy on `rack` and verify it.
    ///
    /// [`CkptError::NotFound`] if that rack holds nothing (never written, or
    /// [`Self::lose_rack`]). [`CkptError::HashMismatch`] if the bytes were
    /// corrupted. Does not consult the other rack.
    pub fn restore_from_rack(&self, rack: &str) -> Result<Checkpoint> {
        let _ = rack;
        unimplemented!("restore_from_rack")
    }

    /// Drop the copy on `rack`. The other destination of the same save stays.
    ///
    /// [`CkptError::NotFound`] if `rack` holds nothing. After both
    /// destinations are lost, both restores fail and
    /// [`Self::cross_rack_destinations`] still returns the last pair.
    pub fn lose_rack(&mut self, rack: &str) -> Result<()> {
        let _ = rack;
        unimplemented!("lose_rack")
    }

    /// Flip one byte of the copy on `rack` only.
    ///
    /// The other destination still restores to the original prepared
    /// checkpoint. Restoring `rack` then returns [`CkptError::HashMismatch`].
    /// [`CkptError::NotFound`] if `rack` holds nothing. [`CkptError::Rack`]
    /// if the stored checkpoint has no shard bytes to flip.
    pub fn corrupt_rack(&mut self, rack: &str) -> Result<()> {
        let _ = rack;
        unimplemented!("corrupt_rack")
    }

    /// Last successful destination pair for `source`, in walk order.
    ///
    /// Still returns that pair after [`Self::lose_rack`] or
    /// [`Self::corrupt_rack`]. [`CkptError::NotFound`] if `source` has never
    /// saved. Not a recompute from a fleet list.
    pub fn cross_rack_destinations(&self, source: &str) -> Result<(String, String)> {
        let _ = source;
        unimplemented!("cross_rack_destinations")
    }
}

/// Next two distinct racks after `source`, wrapping.
///
/// Walk starts at the index after the first occurrence of `source`. Skip
/// ids equal to `source` and ids already chosen. Return those two, in walk
/// order, not sorted.
///
/// [`CkptError::Rack`] when `source` is empty, any rack id is empty,
/// `source` is absent, or fewer than three distinct ids exist.
///
/// Examples: `["a","b","c","d"]` with source `"b"` is `("c","d")`; source
/// `"d"` is `("a","b")`; source `"c"` is `("d","a")`. `["a","b","a","c"]`
/// with source `"b"` is `("a","c")`.
pub fn plan_cross_rack(racks: &[String], source: &str) -> Result<(String, String)> {
    let _ = (racks, source);
    unimplemented!("plan_cross_rack")
}
