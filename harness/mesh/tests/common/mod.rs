//! Shared builders and assertions for H5 `prometheus-mesh` oracle tests.
//!
//! Documented rules the implementer must match (also see `tests/reference`):
//! - `MIN_RING = 3`, `MIN_WATCHERS = 2`. `LocalMesh::start(n)` is
//!   `TooFewNodes(n)` if `n < 3`.
//! - Slot `i`: id `"{i}"` (decimal, no leading zeros), dir `base_dir/<i>/`,
//!   trusted, key `"key-{i}"`. Tests never use real keys.
//! - `Mesh::create` fails if `dir` exists (`AlreadyExists`). Untrusted `pull`
//!   is `Untrusted("pull")`. Frozen `pull` / `advertise` is `Frozen`.
//! - Fake freeze scheme: `signature == payload.as_bytes()`. Anything else is
//!   `BadSignature`. Freeze is durable across `Mesh::open`. A partitioned node
//!   applies a gossiped freeze on `tick` after `heal`, not on `heal` itself.
//! - `pull` claims via H3 `Queue::claim` with `WorkerId` equal to the mesh
//!   node id string. First claim is attempt 1. `NowMs` is injected; no sleep.
//! - Skip (do not claim) a queued task whose payload has a `"requires"` object
//!   the node's last advert cannot satisfy. Missing `requires` means any node
//!   can run it. Fields: `gpus` (u32, node >=), `cpus` (u32, node >=),
//!   `kvm` (bool, required true ⇒ node.kvm), `providers` (array of strings;
//!   node must include all listed). Unknown extra fields are ignored.
//! - Cap mismatch of the *only* queued task: `Ok(None)` without claiming
//!   (empty and mismatch look the same). A later fitting task in the same
//!   `pull` can still be claimed (scan; first mismatch is not terminal).
//! - No last advert: treat caps as `Capabilities::default()`.
//! - Watcher ring: node `i` is watched by `(i+1)%n` and `(i+2)%n`.
//!   `watches()` returns `2n` pairs, order watchee 0's two (watcher i+1 then
//!   i+2), then watchee 1, .... Isolated nodes stay in the ring (no crash API).
//! - `LocalMesh::tick` gossips adverts and freeze among non-isolated nodes.
//!
//! No GPU coverage in H5; these tests are CPU-only (no `gpu` marker).
//!
//! Production must never import this module.

#![allow(dead_code)]

use prometheus_leases::{Lease, Queue, QueueConfig, Task, TaskId, TaskState, WorkItem, WorkerId};
use prometheus_mesh::{
    Advert, Capabilities, Error, Freeze, LocalMesh, MeshConfig, NodeId, NowMs, PublicKey, Result,
    Watch,
};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

pub const SHORT_PERIOD_MS: u64 = 1_000;
pub const SHORT_MISSED: u32 = 2;
pub const T0: NowMs = 10_000;

/// Parent temp dir plus a not-yet-created `mesh/` child for create / start.
pub fn fresh_base() -> (tempfile::TempDir, PathBuf) {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("mesh");
    (parent, dir)
}

/// Parent plus a not-yet-created `node/` child for standalone `Mesh::create`.
pub fn fresh_node_dir() -> (tempfile::TempDir, PathBuf) {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("node");
    (parent, dir)
}

/// Parent plus a not-yet-created `queue/` child for H3 `Queue::create`.
pub fn fresh_queue_dir() -> (tempfile::TempDir, PathBuf) {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("queue");
    (parent, dir)
}

pub fn short_queue_config() -> QueueConfig {
    QueueConfig {
        heartbeat_period_ms: SHORT_PERIOD_MS,
        missed_heartbeats: SHORT_MISSED,
    }
}

pub fn short_ttl() -> u64 {
    short_queue_config().lease_ttl_ms()
}

pub fn slot_id(i: usize) -> NodeId {
    NodeId(i.to_string())
}

pub fn slot_key(i: usize) -> PublicKey {
    PublicKey(format!("key-{i}"))
}

pub fn slot_config(i: usize) -> MeshConfig {
    MeshConfig {
        this_id: slot_id(i),
        this_key: slot_key(i),
        trusted: true,
    }
}

pub fn untrusted_config(id: &str, key: &str) -> MeshConfig {
    MeshConfig {
        this_id: NodeId(id.to_string()),
        this_key: PublicKey(key.to_string()),
        trusted: false,
    }
}

pub fn trusted_config(id: &str, key: &str) -> MeshConfig {
    MeshConfig {
        this_id: NodeId(id.to_string()),
        this_key: PublicKey(key.to_string()),
        trusted: true,
    }
}

pub fn caps_cpu() -> Capabilities {
    Capabilities {
        gpus: 0,
        cpus: 4,
        kvm: false,
        providers: vec![],
    }
}

pub fn caps_gpu(n: u32) -> Capabilities {
    Capabilities {
        gpus: n,
        cpus: 4,
        kvm: false,
        providers: vec!["cuda".into()],
    }
}

pub fn caps_kvm() -> Capabilities {
    Capabilities {
        gpus: 0,
        cpus: 2,
        kvm: true,
        providers: vec![],
    }
}

pub fn caps_providers(providers: &[&str]) -> Capabilities {
    Capabilities {
        gpus: 0,
        cpus: 2,
        kvm: false,
        providers: providers.iter().map(|s| (*s).to_string()).collect(),
    }
}

pub fn item(id: &str, payload: Value) -> WorkItem {
    WorkItem {
        task_id: TaskId(id.to_string()),
        payload,
    }
}

pub fn item_requires_gpus(id: &str, gpus: u32) -> WorkItem {
    item(id, json!({"requires": {"gpus": gpus}}))
}

pub fn item_requires_cpus(id: &str, cpus: u32) -> WorkItem {
    item(id, json!({"requires": {"cpus": cpus}}))
}

pub fn item_requires_kvm(id: &str, kvm: bool) -> WorkItem {
    item(id, json!({"requires": {"kvm": kvm}}))
}

pub fn item_requires_providers(id: &str, providers: &[&str]) -> WorkItem {
    item(id, json!({"requires": {"providers": providers}}))
}

pub fn item_plain(id: &str) -> WorkItem {
    item(id, json!({"job": id}))
}

pub fn good_freeze(payload: &str, signer: &str) -> Freeze {
    Freeze {
        payload: payload.to_string(),
        signature: payload.as_bytes().to_vec(),
        signer: PublicKey(signer.to_string()),
    }
}

pub fn bad_freeze(payload: &str, signer: &str) -> Freeze {
    Freeze {
        payload: payload.to_string(),
        signature: b"not-the-payload".to_vec(),
        signer: PublicKey(signer.to_string()),
    }
}

pub fn start_n(n: usize, base: &Path) -> LocalMesh {
    LocalMesh::start(n, base).unwrap_or_else(|e| panic!("LocalMesh::start({n}): {e}"))
}

pub fn start3(base: &Path) -> LocalMesh {
    start_n(3, base)
}

pub fn open_queue() -> (tempfile::TempDir, Queue) {
    let (parent, dir) = fresh_queue_dir();
    let q = Queue::create(&dir, short_queue_config()).expect("queue create");
    (parent, q)
}

pub fn assert_err<T>(r: Result<T>, what: &str) {
    if r.is_ok() {
        panic!("expected error ({what}), got Ok");
    }
}

/// `Result::expect_err` needs `T: Debug`. Production `Mesh` / `LocalMesh` do not.
pub fn unwrap_err<T>(r: Result<T>, what: &str) -> Error {
    match r {
        Ok(_) => panic!("expected error ({what}), got Ok"),
        Err(e) => e,
    }
}

pub fn assert_too_few_nodes(err: &Error, n: usize) {
    match err {
        Error::TooFewNodes(got) => assert_eq!(*got, n, "TooFewNodes n"),
        other => panic!("expected TooFewNodes({n}), got {other:?}"),
    }
}

pub fn assert_untrusted(err: &Error, action: &str) {
    match err {
        Error::Untrusted(s) => assert_eq!(s, action, "Untrusted action"),
        other => panic!("expected Untrusted({action:?}), got {other:?}"),
    }
}

pub fn assert_frozen(err: &Error, what: &str) {
    match err {
        Error::Frozen => {}
        other => panic!("{what}: expected Frozen, got {other:?}"),
    }
}

pub fn assert_bad_signature(err: &Error) {
    match err {
        Error::BadSignature => {}
        other => panic!("expected BadSignature, got {other:?}"),
    }
}

pub fn assert_already_exists(err: &Error, what: &str) {
    match err {
        Error::AlreadyExists(_) => {}
        other => panic!("{what}: expected AlreadyExists, got {other:?}"),
    }
}

pub fn assert_lease(got: &Lease, task_id: &str, worker_id: &str, attempt: u64, expires_at: NowMs) {
    assert_eq!(got.task_id.0, task_id, "lease.task_id");
    assert_eq!(got.worker_id.0, worker_id, "lease.worker_id");
    assert_eq!(got.attempt, attempt, "lease.attempt");
    assert_eq!(got.expires_at, expires_at, "lease.expires_at");
}

pub fn assert_queued(task: &Task, id: &str) {
    assert_eq!(task.id.0, id);
    assert_eq!(task.state, TaskState::Queued, "expected Queued {id}");
    assert!(task.worker_id.is_none(), "queued worker_id {id}");
}

pub fn assert_leased(task: &Task, id: &str, worker_id: &str, attempt: u64) {
    assert_eq!(task.id.0, id);
    assert_eq!(task.state, TaskState::Leased, "expected Leased {id}");
    assert_eq!(task.attempt, attempt, "leased attempt {id}");
    assert_eq!(
        task.worker_id.as_ref().map(|w| w.0.as_str()),
        Some(worker_id),
        "leased worker {id}"
    );
}

pub fn worker(id: &str) -> WorkerId {
    WorkerId(id.to_string())
}

pub fn task_id(id: &str) -> TaskId {
    TaskId(id.to_string())
}

/// Compare adverts as a set keyed by node id (order is not locked).
pub fn advert_set(ads: &[Advert]) -> BTreeSet<(String, String, u32, u32, bool, Vec<String>, u64)> {
    ads.iter()
        .map(|a| {
            (
                a.node.0.clone(),
                a.key.0.clone(),
                a.caps.gpus,
                a.caps.cpus,
                a.caps.kvm,
                a.caps.providers.clone(),
                a.at,
            )
        })
        .collect()
}

pub fn watch_pairs(watches: &[Watch]) -> Vec<(String, String)> {
    watches
        .iter()
        .map(|w| (w.watcher.0.clone(), w.watchee.0.clone()))
        .collect()
}

pub fn this_id_of(mesh: &mut LocalMesh, i: usize) -> NodeId {
    mesh.get(i)
        .unwrap_or_else(|e| panic!("get({i}): {e}"))
        .config()
        .this_id
        .clone()
}
