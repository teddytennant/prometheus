//! Independent A5 reference: FIPS 180-4 SHA-256 and slow in-memory stand-ins.
//!
//! Production (`prometheus_ckpt`) must never import this module. Tests compare
//! production save/restore bytes and `content_hash` values against these
//! functions. Hashing is written out as the FIPS compression function, not
//! `sha2`, so the oracle does not share an implementation with the crate.

#![allow(dead_code)]

use prometheus_ckpt::{Checkpoint, CkptError, Manifest, Result, ShardBlob, MEMORY_REPLICAS};
use std::collections::HashMap;
use std::path::PathBuf;

/// NIST CAVP SHA-256, empty message.
pub const NIST_SHA256_EMPTY: &str =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// NIST CAVP SHA-256, message `abc`.
pub const NIST_SHA256_ABC: &str =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

const SHA256_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn ch(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (!x & z)
}

fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}

fn bsig0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}

fn bsig1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

fn ssig0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}

fn ssig1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        let o = i * 4;
        w[i] = u32::from_be_bytes([block[o], block[o + 1], block[o + 2], block[o + 3]]);
    }
    for i in 16..64 {
        w[i] = ssig1(w[i - 2])
            .wrapping_add(w[i - 7])
            .wrapping_add(ssig0(w[i - 15]))
            .wrapping_add(w[i - 16]);
    }
    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];
    for i in 0..64 {
        let t1 = h
            .wrapping_add(bsig1(e))
            .wrapping_add(ch(e, f, g))
            .wrapping_add(SHA256_K[i])
            .wrapping_add(w[i]);
        let t2 = bsig0(a).wrapping_add(maj(a, b, c));
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

/// Raw SHA-256 digest (FIPS 180-4). Slow, obvious, no `sha2`.
pub fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let mut state = SHA256_IV;
    let bit_len = (data.len() as u64).saturating_mul(8);
    let mut padded = data.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut block = [0u8; 64];
        block.copy_from_slice(chunk);
        sha256_compress(&mut state, &block);
    }
    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Lowercase hex SHA-256 of `data`. Source of expected `content_hash` values.
pub fn sha256_hex(data: &[u8]) -> String {
    sha256_bytes(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub fn assert_nist_goldens() {
    assert_eq!(sha256_hex(b""), NIST_SHA256_EMPTY);
    assert_eq!(sha256_hex(b"abc"), NIST_SHA256_ABC);
}

pub fn payload_len(ckpt: &Checkpoint) -> usize {
    ckpt.weights.iter().map(|b| b.bytes.len()).sum::<usize>()
        + ckpt.optimizer.iter().map(|b| b.bytes.len()).sum::<usize>()
}

fn find_blob<'a>(blobs: &'a [ShardBlob], name: &str, shard_rank: u64) -> Option<&'a ShardBlob> {
    blobs
        .iter()
        .find(|b| b.name == name && b.shard_rank == shard_rank)
}

fn fill_and_check_hashes(ckpt: &Checkpoint) -> Result<Checkpoint> {
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

fn verify_hashes(ckpt: &Checkpoint) -> Result<()> {
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

/// In-memory map with the same put/get/contains contract as `MemoryStore`.
#[derive(Default)]
pub struct RefMemoryStore {
    map: HashMap<String, Vec<u8>>,
}

impl RefMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn put(&mut self, key: &str, bytes: &[u8]) -> Result<()> {
        self.map.insert(key.to_string(), bytes.to_vec());
        Ok(())
    }

    pub fn get(&self, key: &str) -> Result<Vec<u8>> {
        self.map
            .get(key)
            .cloned()
            .ok_or_else(|| CkptError::NotFound(key.to_string()))
    }

    pub fn contains(&self, key: &str) -> Result<bool> {
        Ok(self.map.contains_key(key))
    }
}

/// Directory-backed map. Bytes are also written under `root` so a second
/// `RefDirStore::open` of the same path still `get`s them (crash analog).
pub struct RefDirStore {
    root: PathBuf,
    map: HashMap<String, Vec<u8>>,
}

impl RefDirStore {
    pub fn open(root: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&root)
            .map_err(|e| CkptError::Message(format!("mkdir {}: {e}", root.display())))?;
        let mut map = HashMap::new();
        let entries = std::fs::read_dir(&root)
            .map_err(|e| CkptError::Message(format!("readdir {}: {e}", root.display())))?;
        for ent in entries {
            let ent =
                ent.map_err(|e| CkptError::Message(format!("readdir {}: {e}", root.display())))?;
            let path = ent.path();
            if !path.is_file() {
                continue;
            }
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            let bytes = std::fs::read(&path)
                .map_err(|e| CkptError::Message(format!("read {}: {e}", path.display())))?;
            map.insert(name, bytes);
        }
        Ok(Self { root, map })
    }

    pub fn put(&mut self, key: &str, bytes: &[u8]) -> Result<()> {
        let path = self.root.join(key);
        std::fs::write(&path, bytes)
            .map_err(|e| CkptError::Message(format!("write {}: {e}", path.display())))?;
        self.map.insert(key.to_string(), bytes.to_vec());
        Ok(())
    }

    pub fn get(&self, key: &str) -> Result<Vec<u8>> {
        self.map
            .get(key)
            .cloned()
            .ok_or_else(|| CkptError::NotFound(key.to_string()))
    }

    pub fn contains(&self, key: &str) -> Result<bool> {
        Ok(self.map.contains_key(key))
    }
}

/// Host-RAM named blobs. Same contract as `HostRamOffload`.
#[derive(Default)]
pub struct RefHostRamOffload {
    map: HashMap<String, Vec<u8>>,
}

impl RefHostRamOffload {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn put(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        self.map.insert(name.to_string(), bytes.to_vec());
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

/// Slow checkpointer: `MEMORY_REPLICAS` cloned copies in RAM plus a HashMap
/// persistent stand-in. Does not invent `created_at` or `checkpoint_id`.
pub struct RefCheckpointer {
    persistent: HashMap<String, Checkpoint>,
    replicas: Vec<Option<Checkpoint>>,
}

impl Default for RefCheckpointer {
    fn default() -> Self {
        Self::new()
    }
}

impl RefCheckpointer {
    pub fn new() -> Self {
        Self {
            persistent: HashMap::new(),
            replicas: vec![None; MEMORY_REPLICAS],
        }
    }

    pub fn save_in_memory(&mut self, ckpt: &Checkpoint) -> Result<String> {
        let prepared = fill_and_check_hashes(ckpt)?;
        let id = prepared.manifest.checkpoint_id.clone();
        for slot in &mut self.replicas {
            *slot = Some(prepared.clone());
        }
        Ok(id)
    }

    pub fn restore_latest_memory(&self) -> Result<Checkpoint> {
        let ckpt = self.replicas[0]
            .clone()
            .ok_or_else(|| CkptError::NotFound("latest".to_string()))?;
        verify_hashes(&ckpt)?;
        Ok(ckpt)
    }

    pub fn replica_len(&self, index: usize) -> Result<usize> {
        if index >= MEMORY_REPLICAS {
            return Err(CkptError::Message(format!(
                "replica index {index} out of range (MEMORY_REPLICAS={MEMORY_REPLICAS})"
            )));
        }
        match &self.replicas[index] {
            Some(c) => Ok(payload_len(c)),
            None => Err(CkptError::ReplicaLost(index)),
        }
    }

    pub fn save_persistent(&mut self, ckpt: &Checkpoint) -> Result<String> {
        let prepared = fill_and_check_hashes(ckpt)?;
        let id = prepared.manifest.checkpoint_id.clone();
        self.persistent.insert(id.clone(), prepared);
        Ok(id)
    }

    pub fn restore(&self, checkpoint_id: &str) -> Result<Checkpoint> {
        let ckpt = self
            .persistent
            .get(checkpoint_id)
            .cloned()
            .ok_or_else(|| CkptError::NotFound(checkpoint_id.to_string()))?;
        verify_hashes(&ckpt)?;
        Ok(ckpt)
    }
}

/// Expected `content_hash` for a shard: SHA-256 of the raw bytes.
pub fn expected_content_hash(bytes: &[u8]) -> String {
    sha256_hex(bytes)
}

pub fn expected_manifest_hashes(manifest: &Manifest, ckpt: &Checkpoint) -> Manifest {
    let mut m = manifest.clone();
    for meta in &mut m.weights {
        if let Some(blob) = find_blob(&ckpt.weights, &meta.name, meta.shard_rank) {
            meta.content_hash = sha256_hex(&blob.bytes);
        }
    }
    for meta in &mut m.optimizer {
        if let Some(blob) = find_blob(&ckpt.optimizer, &meta.name, meta.shard_rank) {
            meta.content_hash = sha256_hex(&blob.bytes);
        }
    }
    m
}
