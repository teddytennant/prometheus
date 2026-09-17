//! Independent in-memory H8 artifact store (map + replica vec).
//!
//! Slow and obvious. Production (`harness/cas/src`) must never import this.
//! SHA-256 is computed here with `sha2` as a test-only oracle.

#![allow(dead_code)]

use prometheus_cas::{
    Digest, Error, GcReport, PinName, Result, StoreConfig, EVENT_GC, EVENT_PIN, EVENT_PUT,
    EVENT_UNPIN,
};
use sha2::{Digest as _, Sha256};
use std::collections::{HashMap, HashSet};

/// NIST SHA-256 of empty bytes.
pub const GOLDEN_EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// NIST SHA-256 of `b"abc"`.
pub const GOLDEN_ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

/// In-memory store mirroring the locked H8 replica / pin / GC rules.
pub struct RefStore {
    config: StoreConfig,
    /// One map per replica: digest hex -> raw bytes.
    replicas: Vec<HashMap<String, Vec<u8>>>,
    /// Pin name -> digest hex.
    pins: HashMap<String, String>,
    events: Vec<String>,
}

impl RefStore {
    pub fn new(n_replicas: usize, config: StoreConfig) -> Self {
        Self {
            config,
            replicas: vec![HashMap::new(); n_replicas],
            pins: HashMap::new(),
            events: Vec::new(),
        }
    }

    pub fn with_default_replicas() -> Self {
        Self::new(2, StoreConfig::default())
    }

    pub fn config(&self) -> &StoreConfig {
        &self.config
    }

    pub fn events(&self) -> &[String] {
        &self.events
    }

    pub fn event_types(&self) -> Vec<String> {
        self.events.clone()
    }

    pub fn put(&mut self, bytes: &[u8]) -> Result<Digest> {
        let hex = digest_hex(bytes);
        let mut wrote = 0usize;
        for replica in &mut self.replicas {
            replica.insert(hex.clone(), bytes.to_vec());
            wrote += 1;
        }
        if wrote < self.config.min_replicas {
            return Err(Error::UnderReplicated {
                wrote,
                need: self.config.min_replicas,
            });
        }
        self.events.push(EVENT_PUT.to_string());
        Ok(Digest(hex))
    }

    pub fn get(&self, digest: &Digest) -> Result<Vec<u8>> {
        for replica in &self.replicas {
            if let Some(bytes) = replica.get(&digest.0) {
                let got = digest_hex(bytes);
                if got != digest.0 {
                    return Err(Error::DigestMismatch {
                        expected: digest.0.clone(),
                        got,
                    });
                }
                return Ok(bytes.clone());
            }
        }
        Err(Error::NotFound(digest.0.clone()))
    }

    pub fn pin(&mut self, digest: &Digest, name: PinName) -> Result<()> {
        let exists = self.replicas.iter().any(|r| r.contains_key(&digest.0));
        if !exists {
            return Err(Error::NotFound(digest.0.clone()));
        }
        if self.pins.contains_key(&name.0) {
            return Err(Error::DuplicatePin(name.0));
        }
        self.pins.insert(name.0, digest.0.clone());
        self.events.push(EVENT_PIN.to_string());
        Ok(())
    }

    pub fn unpin(&mut self, name: &PinName) -> Result<()> {
        if self.pins.remove(&name.0).is_none() {
            return Err(Error::NotFound(name.0.clone()));
        }
        self.events.push(EVENT_UNPIN.to_string());
        Ok(())
    }

    pub fn gc(&mut self) -> Result<GcReport> {
        let pinned: HashSet<&str> = self.pins.values().map(String::as_str).collect();
        let mut present = HashSet::new();
        for replica in &self.replicas {
            for k in replica.keys() {
                present.insert(k.clone());
            }
        }
        let mut blobs_removed = 0u64;
        let mut bytes_freed = 0u64;
        for digest in present {
            if pinned.contains(digest.as_str()) {
                continue;
            }
            let mut removed_this = false;
            for replica in &mut self.replicas {
                if let Some(bytes) = replica.remove(&digest) {
                    bytes_freed += bytes.len() as u64;
                    removed_this = true;
                }
            }
            if removed_this {
                blobs_removed += 1;
            }
        }
        self.events.push(EVENT_GC.to_string());
        Ok(GcReport {
            blobs_removed,
            bytes_freed,
        })
    }

    pub fn replica_count(&self, digest: &Digest) -> Result<usize> {
        Ok(self
            .replicas
            .iter()
            .filter(|r| r.contains_key(&digest.0))
            .count())
    }

    /// Drop one in-memory replica's copy (replica-loss oracle).
    pub fn drop_replica(&mut self, index: usize, digest: &Digest) {
        if let Some(replica) = self.replicas.get_mut(index) {
            replica.remove(&digest.0);
        }
    }
}

/// SHA-256 of raw bytes, lowercase hex. Independent of EventLog hashing.
pub fn digest_hex(bytes: &[u8]) -> String {
    to_hex(&Sha256::digest(bytes))
}

fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

pub fn assert_self_consistent_goldens() {
    assert_eq!(digest_hex(b""), GOLDEN_EMPTY);
    assert_eq!(digest_hex(b"abc"), GOLDEN_ABC);
    assert_eq!(GOLDEN_EMPTY.len(), 64);
    assert!(GOLDEN_EMPTY
        .bytes()
        .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')));
}
