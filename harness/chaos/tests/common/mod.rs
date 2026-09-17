//! Shared helpers for H11 chaos oracle tests.
//!
//! H11 is CPU-only. There are no `#[cfg(feature = "gpu")]` tests in this crate;
//! CPU runs execute the whole suite and GPU CI can skip nothing here.
//!
//! # Naming (iface is silent; this is the contract the reference implements)
//!
//! - Replica ids are decimal strings `"0"`, `"1"`, … `n_replicas-1`.
//! - Process ids: `"coordinator"` then `"worker-0"` … `"worker-{n_processes-2}"`.
//!   Default `n_processes = 4` ⇒ coordinator + worker-0 + worker-1 + worker-2.
//! - Hosted providers the world knows (D5): [`HOSTED_PROVIDERS`].
//!
//! Production `src/` must never import this module.

#![allow(dead_code)]

use prometheus_chaos::{
    Error, ProcessId, ReplicaId, Result, TaskId, WorldConfig, DEFAULT_PROCESSES, MIN_REPLICAS,
};
use prometheus_log::EventLog;
use std::path::{Path, PathBuf};

/// Hosted providers D5 must outage. Last-resort open-weights SGLang is not hosted.
pub const HOSTED_PROVIDERS: [&str; 2] = ["xai", "openai"];

/// D1 "worker partitioned past lease expiry". H3 default TTL is 5s × 2 = 10s.
pub const D1_LEASE_EXPIRY_MS: u64 = 10_000;

pub fn fresh_world_dir() -> (tempfile::TempDir, PathBuf) {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("world");
    (parent, dir)
}

pub fn default_config() -> WorldConfig {
    WorldConfig::default()
}

pub fn cfg(n_replicas: usize, n_processes: u32, seed: u64) -> WorldConfig {
    WorldConfig {
        n_replicas,
        n_processes,
        seed,
    }
}

pub fn task(id: &str) -> TaskId {
    TaskId(id.into())
}

pub fn replica(id: &str) -> ReplicaId {
    ReplicaId(id.into())
}

pub fn process(id: &str) -> ProcessId {
    ProcessId(id.into())
}

pub fn replica_ids(n: usize) -> Vec<ReplicaId> {
    (0..n).map(|i| ReplicaId(i.to_string())).collect()
}

pub fn process_ids(n: u32) -> Vec<ProcessId> {
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![ProcessId("coordinator".into())];
    for i in 0..n.saturating_sub(1) {
        out.push(ProcessId(format!("worker-{i}")));
    }
    out
}

pub fn replica_path(world_dir: &Path, id: &ReplicaId) -> PathBuf {
    world_dir.join("replicas").join(&id.0)
}

pub fn events_jsonl(world_dir: &Path, id: &ReplicaId) -> PathBuf {
    replica_path(world_dir, id).join("events.jsonl")
}

pub fn event_types(log: &EventLog) -> Vec<String> {
    log.iter().map(|e| e.event_type.clone()).collect()
}

pub fn assert_err<T>(r: Result<T>, what: &str) {
    if r.is_ok() {
        panic!("{what}: expected Err, got Ok");
    }
}

pub fn assert_too_few(err: &Error, n: usize) {
    match err {
        Error::TooFewReplicas(got) => assert_eq!(*got, n, "TooFewReplicas n"),
        other => panic!("expected TooFewReplicas({n}), got {other:?}"),
    }
}

pub fn assert_no_replica(err: &Error, id: &str) {
    match err {
        Error::NoReplica(got) => assert_eq!(got, id, "NoReplica id"),
        other => panic!("expected NoReplica({id}), got {other:?}"),
    }
}

pub fn assert_no_process(err: &Error, id: &str) {
    match err {
        Error::NoProcess(got) => assert_eq!(got, id, "NoProcess id"),
        other => panic!("expected NoProcess({id}), got {other:?}"),
    }
}

pub fn assert_clock_skew(err: &Error, delta: i64) {
    match err {
        Error::ClockSkewBound(got) => assert_eq!(*got, delta, "ClockSkewBound delta"),
        other => panic!("expected ClockSkewBound({delta}), got {other:?}"),
    }
}

pub fn assert_invariant_err(err: &Error) {
    match err {
        Error::Invariant { .. } => {}
        other => panic!("expected Error::Invariant, got {other:?}"),
    }
}

pub fn err_kind_eq(a: &Error, b: &Error) -> bool {
    use Error::*;
    match (a, b) {
        (Invariant { .. }, Invariant { .. }) => true,
        (TooFewReplicas(x), TooFewReplicas(y)) => x == y,
        (ClockSkewBound(x), ClockSkewBound(y)) => x == y,
        (NoReplica(x), NoReplica(y)) => x == y,
        (NoProcess(x), NoProcess(y)) => x == y,
        (Disk(_), Disk(_)) => true,
        (Log(_), Log(_)) => true,
        (Other(_), Other(_)) => true,
        _ => false,
    }
}

pub fn assert_default_shape(config: &WorldConfig) {
    assert_eq!(config.n_replicas, MIN_REPLICAS);
    assert_eq!(config.n_processes, DEFAULT_PROCESSES);
}

pub fn coordinator() -> ProcessId {
    ProcessId("coordinator".into())
}

pub fn worker(i: u32) -> ProcessId {
    ProcessId(format!("worker-{i}"))
}
