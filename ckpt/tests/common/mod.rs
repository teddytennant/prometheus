//! Shared builders and assertions for A5 `prometheus-ckpt` oracle tests.
//!
//! Documented rules the implementer must match (also see `tests/reference`):
//!
//! - `MEMORY_REPLICAS = 2`. `replica_len(i)` for `i in 0..2` after
//!   `save_in_memory` is equal and at least the sum of shard payload bytes.
//!   `replica_len(i)` for `i >= MEMORY_REPLICAS` is an error.
//! - `sha256_hex` is FIPS 180-4 SHA-256, lowercase 64-char hex. A shard's
//!   `content_hash` is `sha256_hex` of that shard's raw bytes.
//! - `MemoryStore`: `put` overwrites; `get` of a missing key is
//!   `CkptError::NotFound(key)`; `contains` is `Ok(true/false)`.
//! - `DirStore::open` creates the directory if needed. `put` then drop then
//!   `open` the same path still `get`s the bytes (crash-durability analog).
//!   Tests use keys matching `[A-Za-z0-9._-]+` (no slashes).
//! - `HostRamOffload`: named blobs in host RAM. `get` / `remove` of a missing
//!   name is `NotFound(name)`. `put` overwrites.
//! - `Checkpointer::save_in_memory` / `save_persistent` return the caller's
//!   `manifest.checkpoint_id`. They do **not** invent `created_at`.
//! - Restore is bitwise-equal on weight and optimizer blob bytes. Manifest
//!   fields (`step`, `rung`, `mesh`, `loader_state`, `rng`, `precision`,
//!   `parent_checkpoint_id`, hashes, `git_commit`) are preserved.
//! - A `WeightMeta` / `OptimizerMeta` with no matching `ShardBlob`
//!   (`name`, `shard_rank`) is `CkptError::MissingShard` on save or restore.
//! - After `save_persistent`, flipping one stored blob byte then `restore`
//!   is `CkptError::HashMismatch`.
//! - In-memory replicas are host-RAM stand-ins for Grace / two other racks.
//!   Persistent bytes go through `Store` (`DirStore` is the object-store
//!   stand-in). No elastic DP, SDC, or spike rollback (those are A6).
//! - Only one DP replica's shards are written. The iface has no DP-rank
//!   field on shards, so `MultipleReplicas` is not exercised.
//! - `ReplicaLost` cannot be triggered through the frozen iface (no API
//!   drops a replica); tests do not require it.
//! - No GPU coverage in A5 CPU tests (no `gpu` marker). V4 GPU jobs are
//!   out of scope for this oracle.
//!
//! Production must never import this module.

#![allow(dead_code)]

use crate::reference;
use prometheus_ckpt::{
    Checkpoint, CkptError, Dtype, Manifest, Mesh, OptimizerKind, OptimizerMeta, Precision,
    RngState, ShardBlob, Store, WeightMeta,
};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::rc::Rc;

pub const GOLDEN_JSON: &str = include_str!("../../../contracts/goldens/v1/checkpoint.default.json");

pub fn golden_manifest() -> Manifest {
    let value: Value = serde_json::from_str(GOLDEN_JSON).expect("golden json");
    Manifest::from_json(&value).expect("golden Manifest::from_json")
}

pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

pub fn blobs_by_key(blobs: &[ShardBlob]) -> BTreeMap<(String, u64), Vec<u8>> {
    let mut m = BTreeMap::new();
    for b in blobs {
        m.insert((b.name.clone(), b.shard_rank), b.bytes.clone());
    }
    m
}

pub fn payload_len(ckpt: &Checkpoint) -> usize {
    reference::payload_len(ckpt)
}

pub fn tiny_checkpoint() -> Checkpoint {
    tiny_checkpoint_with_id("ckpt-tiny")
}

pub fn tiny_checkpoint_with_id(id: &str) -> Checkpoint {
    let mut weight_bytes = b"\x00WGT\xff".to_vec();
    weight_bytes.extend_from_slice(id.as_bytes());
    let mut opt_bytes = b"\x00OPT\xff".to_vec();
    opt_bytes.extend_from_slice(id.as_bytes());
    checkpoint_from_shards(
        id,
        vec![(
            "embed",
            0,
            vec![2, 4],
            Dtype::Fp32,
            weight_bytes,
            vec!["fsdp".to_string()],
        )],
        vec![("embed.m", 0, OptimizerKind::Adamw, opt_bytes)],
    )
}

pub fn checkpoint_from_shards(
    id: &str,
    weights: Vec<(&str, u64, Vec<u64>, Dtype, Vec<u8>, Vec<String>)>,
    optimizer: Vec<(&str, u64, OptimizerKind, Vec<u8>)>,
) -> Checkpoint {
    let weight_meta: Vec<WeightMeta> = weights
        .iter()
        .map(|(name, rank, shape, dtype, bytes, axes)| WeightMeta {
            name: (*name).to_string(),
            shard_rank: *rank,
            shape: shape.clone(),
            dtype: *dtype,
            content_hash: reference::sha256_hex(bytes),
            mesh_axes: axes.clone(),
        })
        .collect();
    let weight_blobs: Vec<ShardBlob> = weights
        .iter()
        .map(|(name, rank, _, _, bytes, _)| ShardBlob {
            name: (*name).to_string(),
            shard_rank: *rank,
            bytes: bytes.clone(),
        })
        .collect();
    let opt_meta: Vec<OptimizerMeta> = optimizer
        .iter()
        .map(|(name, rank, kind, bytes)| OptimizerMeta {
            name: (*name).to_string(),
            shard_rank: *rank,
            kind: *kind,
            content_hash: reference::sha256_hex(bytes),
        })
        .collect();
    let opt_blobs: Vec<ShardBlob> = optimizer
        .iter()
        .map(|(name, rank, _, bytes)| ShardBlob {
            name: (*name).to_string(),
            shard_rank: *rank,
            bytes: bytes.clone(),
        })
        .collect();
    Checkpoint {
        manifest: Manifest {
            checkpoint_id: id.to_string(),
            parent_checkpoint_id: None,
            step: 7,
            rung: 0,
            model_config_hash: reference::sha256_hex(b"model-config"),
            data_mix_hash: reference::sha256_hex(b"data-mix"),
            tokenizer_id: "byte-bpe-128k-r0".to_string(),
            precision: Some(Precision {
                param_dtype: Dtype::Fp32,
                remat: "full".to_string(),
            }),
            mesh: Mesh {
                dp: 1,
                fsdp: 1,
                ep: 1,
                pp: 1,
                cp: 1,
            },
            weights: weight_meta,
            optimizer: opt_meta,
            loader_state: json!({
                "epoch": 0,
                "step": 7,
                "shuffle_seed": 3,
                "shard_index": 1,
                "shard_offset": 0,
            }),
            rng: vec![RngState {
                scope: "dropout".to_string(),
                rank: 0,
                state_hash: reference::sha256_hex(b"rng-dropout-0"),
            }],
            created_at: "2020-01-02T03:04:05Z".to_string(),
            git_commit: Some("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef".to_string()),
        },
        weights: weight_blobs,
        optimizer: opt_blobs,
    }
}

pub fn assert_not_found(err: &CkptError, key: &str) {
    match err {
        CkptError::NotFound(k) => assert_eq!(k, key, "NotFound payload"),
        other => panic!("expected NotFound({key}), got {other:?}"),
    }
}

pub fn assert_hash_mismatch(err: &CkptError, expected_hash: Option<&str>) {
    match err {
        CkptError::HashMismatch { expected, got } => {
            assert!(is_sha256_hex(expected), "expected hash {expected}");
            assert!(is_sha256_hex(got), "got hash {got}");
            assert_ne!(expected, got, "HashMismatch hashes must differ");
            if let Some(exp) = expected_hash {
                assert_eq!(expected, exp, "HashMismatch.expected");
            }
        }
        other => panic!("expected HashMismatch, got {other:?}"),
    }
}

pub fn assert_missing_shard(err: &CkptError, name: &str, shard_rank: u64) {
    match err {
        CkptError::MissingShard {
            name: n,
            shard_rank: r,
        } => {
            assert_eq!(n, name);
            assert_eq!(*r, shard_rank);
        }
        other => panic!("expected MissingShard {{ {name}, {shard_rank} }}, got {other:?}"),
    }
}

pub fn assert_blobs_equal(got: &[ShardBlob], expected: &[ShardBlob]) {
    assert_eq!(blobs_by_key(got), blobs_by_key(expected));
}

pub fn assert_checkpoint_roundtrip(got: &Checkpoint, src: &Checkpoint) {
    assert_eq!(got.manifest.checkpoint_id, src.manifest.checkpoint_id);
    assert_eq!(
        got.manifest.parent_checkpoint_id,
        src.manifest.parent_checkpoint_id
    );
    assert_eq!(got.manifest.step, src.manifest.step);
    assert_eq!(got.manifest.rung, src.manifest.rung);
    assert_eq!(
        got.manifest.model_config_hash,
        src.manifest.model_config_hash
    );
    assert_eq!(got.manifest.data_mix_hash, src.manifest.data_mix_hash);
    assert_eq!(got.manifest.tokenizer_id, src.manifest.tokenizer_id);
    assert_eq!(got.manifest.precision, src.manifest.precision);
    assert_eq!(got.manifest.mesh, src.manifest.mesh);
    assert_eq!(got.manifest.loader_state, src.manifest.loader_state);
    assert_eq!(got.manifest.rng, src.manifest.rng);
    assert_eq!(
        got.manifest.created_at, src.manifest.created_at,
        "save must not invent wall-clock time"
    );
    assert_eq!(got.manifest.git_commit, src.manifest.git_commit);
    assert_eq!(got.manifest.weights.len(), src.manifest.weights.len());
    for (g, s) in got.manifest.weights.iter().zip(src.manifest.weights.iter()) {
        assert_eq!(g.name, s.name);
        assert_eq!(g.shard_rank, s.shard_rank);
        assert_eq!(g.shape, s.shape);
        assert_eq!(g.dtype, s.dtype);
        assert_eq!(g.mesh_axes, s.mesh_axes);
        assert_eq!(
            g.content_hash,
            reference::sha256_hex(&blob_bytes(got, true, &s.name, s.shard_rank))
        );
        assert_eq!(g.content_hash, s.content_hash);
    }
    assert_eq!(got.manifest.optimizer.len(), src.manifest.optimizer.len());
    for (g, s) in got
        .manifest
        .optimizer
        .iter()
        .zip(src.manifest.optimizer.iter())
    {
        assert_eq!(g.name, s.name);
        assert_eq!(g.shard_rank, s.shard_rank);
        assert_eq!(g.kind, s.kind);
        assert_eq!(
            g.content_hash,
            reference::sha256_hex(&blob_bytes(got, false, &s.name, s.shard_rank))
        );
        assert_eq!(g.content_hash, s.content_hash);
    }
    assert_blobs_equal(&got.weights, &src.weights);
    assert_blobs_equal(&got.optimizer, &src.optimizer);
}

fn blob_bytes(ckpt: &Checkpoint, weights: bool, name: &str, shard_rank: u64) -> Vec<u8> {
    let blobs = if weights {
        &ckpt.weights
    } else {
        &ckpt.optimizer
    };
    blobs
        .iter()
        .find(|b| b.name == name && b.shard_rank == shard_rank)
        .map(|b| b.bytes.clone())
        .unwrap_or_default()
}

/// Shared `Store` the test keeps a handle on, for HashMismatch / MissingShard
/// injection without changing `lib.rs`.
#[derive(Clone)]
pub struct MapStore {
    pub map: Rc<RefCell<HashMap<String, Vec<u8>>>>,
}

impl MapStore {
    pub fn new() -> Self {
        Self {
            map: Rc::new(RefCell::new(HashMap::new())),
        }
    }
}

impl Store for MapStore {
    fn put(&mut self, key: &str, bytes: &[u8]) -> prometheus_ckpt::Result<()> {
        self.map
            .borrow_mut()
            .insert(key.to_string(), bytes.to_vec());
        Ok(())
    }

    fn get(&self, key: &str) -> prometheus_ckpt::Result<Vec<u8>> {
        self.map
            .borrow()
            .get(key)
            .cloned()
            .ok_or_else(|| CkptError::NotFound(key.to_string()))
    }

    fn contains(&self, key: &str) -> prometheus_ckpt::Result<bool> {
        Ok(self.map.borrow().contains_key(key))
    }
}

/// Store that panics if the persistent path is touched. `save_in_memory` must
/// not call it.
pub struct PanicStore;

impl Store for PanicStore {
    fn put(&mut self, key: &str, _bytes: &[u8]) -> prometheus_ckpt::Result<()> {
        panic!("persistent Store::put({key}) must not run during in-memory save");
    }
    fn get(&self, key: &str) -> prometheus_ckpt::Result<Vec<u8>> {
        panic!("persistent Store::get({key}) must not run during in-memory restore");
    }
    fn contains(&self, key: &str) -> prometheus_ckpt::Result<bool> {
        panic!("persistent Store::contains({key}) must not run during in-memory ops");
    }
}

pub fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

pub fn corrupt_first_blob(map: &mut HashMap<String, Vec<u8>>, blob: &[u8]) -> bool {
    for v in map.values_mut() {
        if let Some(pos) = find_subslice(v, blob) {
            v[pos] ^= 0xff;
            return true;
        }
    }
    false
}

pub fn walk_files(root: &Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    fn rec(p: &Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(rd) = std::fs::read_dir(p) else {
            return;
        };
        for ent in rd.flatten() {
            let path = ent.path();
            if path.is_dir() {
                rec(&path, out);
            } else if path.is_file() {
                out.push(path);
            }
        }
    }
    rec(root, &mut out);
    out
}

pub fn corrupt_blob_in_dir(root: &Path, blob: &[u8]) -> bool {
    for path in walk_files(root) {
        let Ok(mut bytes) = std::fs::read(&path) else {
            continue;
        };
        if let Some(pos) = find_subslice(&bytes, blob) {
            bytes[pos] ^= 0xff;
            let _ = std::fs::write(&path, bytes);
            return true;
        }
    }
    false
}

pub fn delete_blob_in_dir(root: &Path, blob: &[u8]) -> bool {
    for path in walk_files(root) {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        if find_subslice(&bytes, blob).is_some() && bytes == blob {
            let _ = std::fs::remove_file(&path);
            return true;
        }
    }
    false
}
