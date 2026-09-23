//! Shared builders and assertions for H8 `prometheus-cas` oracle tests.
//!
//! Documented rules the implementer must match (also see `tests/reference`):
//! - `MIN_REPLICAS = 2`. `create` / `open` error `TooFewBackends` if
//!   `backends.len() < config.min_replicas` (default 2).
//! - `create` fails if `dir` exists. Creates H2 EventLog at `dir/log/`.
//!   No events on create. A `TooFewBackends` create must not mkdir `dir`.
//! - Digest is SHA-256 of the raw bytes, lowercase hex (64 chars `0-9a-f`).
//! - `put` writes every backend and succeeds if at least `min_replicas`
//!   writes succeed. Emit `cas.put`. A failed put may leave orphans; GC
//!   of unpinned blobs removes them.
//! - `get` reads any replica, verifies SHA-256, `DigestMismatch` on mismatch,
//!   `NotFound` if none have it.
//! - Named pins. Unknown digest `NotFound`. Duplicate pin name `DuplicatePin`.
//!   Unpin missing name `NotFound`.
//! - `gc` deletes blobs with zero pins from every backend that has them.
//!   Pinned blobs stay. `blobs_removed` is unique digests deleted.
//!   `bytes_freed` is the sum of deleted replica payload sizes (physical).
//! - Event types: `cas.put`, `cas.pin`, `cas.unpin`, `cas.gc`.
//! - `open` replays pins from the log. Backend paths are caller-provided
//!   (a lost replica is a missing directory you still pass, or a replacement).
//! - CPU tests use two local directories as backends. No GPU coverage in H8
//!   (no `gpu` marker).
//!
//! Production must never import this module.

#![allow(dead_code)]

use prometheus_cas::{Digest, Error, PinName, Result, Store, StoreConfig, MIN_REPLICAS};
use std::fs;
use std::path::{Path, PathBuf};

/// Parent temp dir, a not-yet-created `store/` child, and N backend dirs.
pub struct Harness {
    pub parent: tempfile::TempDir,
    pub dir: PathBuf,
    pub backends: Vec<PathBuf>,
}

pub fn cfg() -> StoreConfig {
    StoreConfig::default()
}

pub fn cfg_min(min_replicas: usize) -> StoreConfig {
    StoreConfig { min_replicas }
}

/// Two backends, default config. Backend dirs exist; store dir does not.
pub fn fresh_store() -> Harness {
    fresh_store_n(MIN_REPLICAS)
}

pub fn fresh_store_n(n: usize) -> Harness {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("store");
    let mut backends = Vec::with_capacity(n);
    for i in 0..n {
        let b = parent.path().join(format!("b{i}"));
        fs::create_dir(&b).expect("mkdir backend");
        backends.push(b);
    }
    Harness {
        parent,
        dir,
        backends,
    }
}

pub fn log_dir(store_dir: &Path) -> PathBuf {
    store_dir.join("log")
}

pub fn events_jsonl(store_dir: &Path) -> PathBuf {
    log_dir(store_dir).join("events.jsonl")
}

pub fn pin(name: &str) -> PinName {
    PinName(name.to_string())
}

pub fn digest(hex: &str) -> Digest {
    Digest(hex.to_string())
}

pub fn event_types(store: &Store) -> Vec<String> {
    store.log().iter().map(|e| e.event_type.clone()).collect()
}

pub fn assert_err<T>(r: Result<T>, what: &str) {
    if r.is_ok() {
        panic!("expected error ({what}), got Ok");
    }
}

pub fn unwrap_err<T>(r: Result<T>, what: &str) -> Error {
    match r {
        Ok(_) => panic!("expected error ({what}), got Ok"),
        Err(e) => e,
    }
}

pub fn assert_too_few(err: &Error, have: usize) {
    match err {
        Error::TooFewBackends(n) => assert_eq!(*n, have, "TooFewBackends count"),
        other => panic!("expected TooFewBackends({have}), got {other:?}"),
    }
}

pub fn assert_not_found(err: &Error, key: &str) {
    match err {
        Error::NotFound(s) => assert_eq!(s, key, "NotFound key"),
        other => panic!("expected NotFound({key}), got {other:?}"),
    }
}

pub fn assert_duplicate_pin(err: &Error, name: &str) {
    match err {
        Error::DuplicatePin(s) => assert_eq!(s, name, "DuplicatePin name"),
        other => panic!("expected DuplicatePin({name}), got {other:?}"),
    }
}

pub fn assert_digest_mismatch(err: &Error, expected: &str) {
    match err {
        Error::DigestMismatch {
            expected: got_exp,
            got,
        } => {
            assert_eq!(got_exp, expected, "DigestMismatch.expected");
            assert_ne!(
                got, expected,
                "DigestMismatch.got must differ from expected"
            );
            assert!(
                is_sha256_hex(got),
                "DigestMismatch.got must be lowercase sha256 hex, got {got}"
            );
        }
        other => panic!("expected DigestMismatch expected={expected}, got {other:?}"),
    }
}

pub fn assert_under_replicated(err: &Error, wrote: usize, need: usize) {
    match err {
        Error::UnderReplicated { wrote: w, need: n } => {
            assert_eq!(*w, wrote, "UnderReplicated.wrote");
            assert_eq!(*n, need, "UnderReplicated.need");
        }
        other => {
            panic!("expected UnderReplicated {{ wrote: {wrote}, need: {need} }}, got {other:?}")
        }
    }
}

pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

pub fn walk_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if !dir.exists() {
        return out;
    }
    fn rec(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for e in entries {
            let p = e.expect("read_dir entry").path();
            if p.is_dir() {
                rec(&p, out);
            } else if p.is_file() {
                out.push(p);
            }
        }
    }
    rec(dir, &mut out);
    out
}

/// Remove files and subdirectories inside `dir`, leaving `dir` itself.
pub fn wipe_contents(dir: &Path) {
    if !dir.exists() {
        return;
    }
    for e in fs::read_dir(dir).expect("read_dir wipe") {
        let p = e.expect("entry").path();
        if p.is_dir() {
            fs::remove_dir_all(&p).expect("remove_dir_all");
        } else {
            fs::remove_file(&p).expect("remove_file");
        }
    }
}

/// Overwrite every regular file under `dir` so SHA-256 no longer matches.
pub fn corrupt_all_files(dir: &Path) {
    for p in walk_files(dir) {
        fs::write(&p, b"not-the-original-bytes").expect("corrupt write");
    }
}

/// Make `path` readonly and restore owner-write on drop so TempDir can clean up.
pub struct RestoreReadonly {
    path: PathBuf,
}

impl RestoreReadonly {
    pub fn make(path: &Path) -> Self {
        let meta =
            fs::metadata(path).unwrap_or_else(|e| panic!("metadata {}: {e}", path.display()));
        let mut perms = meta.permissions();
        perms.set_readonly(true);
        fs::set_permissions(path, perms)
            .unwrap_or_else(|e| panic!("chmod readonly {}: {e}", path.display()));
        Self {
            path: path.to_path_buf(),
        }
    }
}

impl Drop for RestoreReadonly {
    // Undoes a test's chmod so the tempdir can be removed.
    #[allow(clippy::permissions_set_readonly_false)]
    fn drop(&mut self) {
        if let Ok(meta) = fs::metadata(&self.path) {
            let mut perms = meta.permissions();
            perms.set_readonly(false);
            let _ = fs::set_permissions(&self.path, perms);
        }
    }
}
