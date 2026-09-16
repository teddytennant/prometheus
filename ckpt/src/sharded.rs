//! Sharded in-memory checkpoints with A/B replica copies (spec 5.5).

use crate::sha;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CkptError {
    #[error("replica quarantined")]
    Quarantined,
    #[error("shard hash mismatch")]
    HashMismatch,
    #[error("no live replica")]
    NoLiveReplica,
    #[error("io: {0}")]
    Io(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReplicaId {
    A,
    B,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shard {
    pub index: usize,
    pub data: Vec<u8>,
    pub hash: String,
}

impl Shard {
    pub fn new(index: usize, data: Vec<u8>) -> Self {
        let hash = sha(&data);
        Self { index, data, hash }
    }

    pub fn verify(&self) -> bool {
        sha(&self.data) == self.hash
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Replica {
    pub id: ReplicaId,
    pub shards: Vec<Shard>,
    pub quarantined: bool,
}

impl Replica {
    pub fn payload(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for s in &self.shards {
            out.extend_from_slice(&s.data);
        }
        out
    }

    fn matches_canonical(&self, canonical: &[String]) -> bool {
        if self.shards.len() != canonical.len() {
            return false;
        }
        self.shards
            .iter()
            .enumerate()
            .all(|(i, s)| s.verify() && s.index == i && s.hash == canonical[i])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistMeta {
    step: u64,
    n_shards: usize,
    canonical: Vec<String>,
    quarantined_a: bool,
    quarantined_b: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedCheckpoint {
    pub step: u64,
    pub n_shards: usize,
    pub canonical: Vec<String>,
    pub replica_a: Replica,
    pub replica_b: Replica,
}

fn split_payload(payload: &[u8], n: usize) -> Vec<Vec<u8>> {
    let n = n.max(1);
    let len = payload.len();
    let base = len / n;
    let rem = len % n;
    let mut shards = Vec::with_capacity(n);
    let mut off = 0usize;
    for i in 0..n {
        let extra = usize::from(i < rem);
        let end = (off + base + extra).min(len);
        shards.push(payload[off..end].to_vec());
        off = end;
    }
    shards
}

impl ShardedCheckpoint {
    pub fn new(step: u64, payload: &[u8], n_shards: usize) -> Self {
        let n_shards = n_shards.max(1);
        let parts = split_payload(payload, n_shards);
        let shards: Vec<Shard> = parts
            .into_iter()
            .enumerate()
            .map(|(i, d)| Shard::new(i, d))
            .collect();
        let canonical: Vec<String> = shards.iter().map(|s| s.hash.clone()).collect();
        Self {
            step,
            n_shards,
            canonical,
            replica_a: Replica {
                id: ReplicaId::A,
                shards: shards.clone(),
                quarantined: false,
            },
            replica_b: Replica {
                id: ReplicaId::B,
                shards,
                quarantined: false,
            },
        }
    }

    pub fn replica(&self, id: ReplicaId) -> &Replica {
        match id {
            ReplicaId::A => &self.replica_a,
            ReplicaId::B => &self.replica_b,
        }
    }

    pub fn replica_mut(&mut self, id: ReplicaId) -> &mut Replica {
        match id {
            ReplicaId::A => &mut self.replica_a,
            ReplicaId::B => &mut self.replica_b,
        }
    }

    pub fn restore_from_replica(&self, id: ReplicaId) -> Result<Vec<u8>, CkptError> {
        let r = self.replica(id);
        if r.quarantined {
            return Err(CkptError::Quarantined);
        }
        if !r.matches_canonical(&self.canonical) {
            return Err(CkptError::HashMismatch);
        }
        Ok(r.payload())
    }

    /// Compare each replica against canonical per-shard hashes. A mismatch
    /// quarantines that replica and returns the ids that failed.
    pub fn detect_sdc(&mut self) -> Vec<ReplicaId> {
        let mut hit = Vec::new();
        for id in [ReplicaId::A, ReplicaId::B] {
            let bad = !self.replica(id).matches_canonical(&self.canonical);
            if bad {
                self.replica_mut(id).quarantined = true;
                hit.push(id);
            }
        }
        hit
    }

    /// Step-boundary join: copy shards from a live replica onto the other.
    pub fn join_healed_replica(&mut self, live: ReplicaId) -> Result<(), CkptError> {
        let src = self.replica(live).clone();
        if src.quarantined || !src.matches_canonical(&self.canonical) {
            return Err(CkptError::Quarantined);
        }
        let dead = match live {
            ReplicaId::A => ReplicaId::B,
            ReplicaId::B => ReplicaId::A,
        };
        let dst = self.replica_mut(dead);
        dst.shards = src.shards;
        dst.quarantined = false;
        Ok(())
    }

    pub fn live_replica(&self) -> Result<ReplicaId, CkptError> {
        if !self.replica_a.quarantined && self.replica_a.matches_canonical(&self.canonical) {
            Ok(ReplicaId::A)
        } else if !self.replica_b.quarantined && self.replica_b.matches_canonical(&self.canonical) {
            Ok(ReplicaId::B)
        } else {
            Err(CkptError::NoLiveReplica)
        }
    }

    /// Durable persist. Writes are synchronous; the API is `persist()`.
    pub fn persist(&self, dir: &Path) -> io::Result<()> {
        fs::create_dir_all(dir)?;
        let meta = PersistMeta {
            step: self.step,
            n_shards: self.n_shards,
            canonical: self.canonical.clone(),
            quarantined_a: self.replica_a.quarantined,
            quarantined_b: self.replica_b.quarantined,
        };
        let bytes = serde_json::to_vec(&meta).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        atomic_write(&dir.join("meta.json"), &bytes)?;
        for (name, r) in [("A", &self.replica_a), ("B", &self.replica_b)] {
            let rd = dir.join(name);
            fs::create_dir_all(&rd)?;
            for s in &r.shards {
                atomic_write(&rd.join(format!("{}.bin", s.index)), &s.data)?;
                atomic_write(rd.join(format!("{}.sha", s.index)).as_path(), s.hash.as_bytes())?;
            }
        }
        Ok(())
    }

    pub fn load_persisted(dir: &Path) -> io::Result<Self> {
        let meta: PersistMeta =
            serde_json::from_slice(&fs::read(dir.join("meta.json"))?)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let load_rep = |id: ReplicaId, name: &str, quarantined: bool| -> io::Result<Replica> {
            let rd = dir.join(name);
            let mut shards = Vec::with_capacity(meta.n_shards);
            for i in 0..meta.n_shards {
                let data = fs::read(rd.join(format!("{i}.bin")))?;
                let stored = fs::read_to_string(rd.join(format!("{i}.sha"))).unwrap_or_default();
                let hash = sha(&data);
                if !stored.is_empty() && stored != hash {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "persisted shard hash mismatch",
                    ));
                }
                shards.push(Shard {
                    index: i,
                    data,
                    hash,
                });
            }
            Ok(Replica {
                id,
                shards,
                quarantined,
            })
        };
        Ok(Self {
            step: meta.step,
            n_shards: meta.n_shards,
            canonical: meta.canonical,
            replica_a: load_rep(ReplicaId::A, "A", meta.quarantined_a)?,
            replica_b: load_rep(ReplicaId::B, "B", meta.quarantined_b)?,
        })
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_file_name(format!(
        "{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy()
    ));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn roundtrip_shards_bitwise() {
        let payload: Vec<u8> = (0..100u8).collect();
        let ck = ShardedCheckpoint::new(7, &payload, 4);
        assert_eq!(ck.n_shards, 4);
        assert_eq!(ck.restore_from_replica(ReplicaId::A).unwrap(), payload);
        assert_eq!(ck.restore_from_replica(ReplicaId::B).unwrap(), payload);
        let dir = tempdir().unwrap();
        ck.persist(dir.path()).unwrap();
        let loaded = ShardedCheckpoint::load_persisted(dir.path()).unwrap();
        assert_eq!(loaded.step, 7);
        assert_eq!(loaded.restore_from_replica(ReplicaId::A).unwrap(), payload);
    }

    #[test]
    fn replica_mismatch_quarantines() {
        let mut ck = ShardedCheckpoint::new(1, b"hello-world-payload", 3);
        ck.replica_b.shards[1].data[0] ^= 0xff;
        ck.replica_b.shards[1].hash = sha(&ck.replica_b.shards[1].data);
        let hit = ck.detect_sdc();
        assert_eq!(hit, vec![ReplicaId::B]);
        assert!(ck.replica_b.quarantined);
        assert!(!ck.replica_a.quarantined);
        assert!(ck.restore_from_replica(ReplicaId::B).is_err());
        assert_eq!(
            ck.restore_from_replica(ReplicaId::A).unwrap(),
            b"hello-world-payload"
        );
    }

    #[test]
    fn join_copies_live_shards() {
        let mut ck = ShardedCheckpoint::new(2, b"abcdefghijklmnop", 4);
        ck.replica_a.shards[0].data[0] ^= 1;
        ck.replica_a.shards[0].hash = sha(&ck.replica_a.shards[0].data);
        ck.detect_sdc();
        assert!(ck.replica_a.quarantined);
        ck.join_healed_replica(ReplicaId::B).unwrap();
        assert!(!ck.replica_a.quarantined);
        assert_eq!(
            ck.restore_from_replica(ReplicaId::A).unwrap(),
            b"abcdefghijklmnop"
        );
    }

    #[test]
    fn persist_detects_bit_rot() {
        let ck = ShardedCheckpoint::new(3, b"rot-me-please-xx", 2);
        let dir = tempdir().unwrap();
        ck.persist(dir.path()).unwrap();
        let p = dir.path().join("A").join("0.bin");
        let mut b = fs::read(&p).unwrap();
        b[0] ^= 0x0f;
        fs::write(&p, b).unwrap();
        assert!(ShardedCheckpoint::load_persisted(dir.path()).is_err());
    }
}
