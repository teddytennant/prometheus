//! Shared builders and assertions for H3 `prometheus-leases` oracle tests.
//!
//! Documented rules the implementer must match (also see `tests/reference`):
//! - First claim is attempt 1. Reclaim after expiry bumps by 1 at claim time.
//! - `NowMs` is caller-supplied; tests never sleep for lease TTL.
//! - `MISSED_HEARTBEATS = 2`. TTL = `heartbeat_period_ms * missed_heartbeats`.
//! - A lease is due when `expires_at <= now`.
//! - Failed is terminal: only expiry of a `Leased` task requeues. Failed and
//!   completed ids stay known (`Duplicate` on re-enqueue) and are not claimed.
//! - `create` fails if `dir` exists. On-disk: `<dir>/log/` is an H2 EventLog;
//!   `<dir>/outputs/<task_id>/<attempt>` is the raw output bytes (decimal
//!   attempt, no leading zeros). `outputs/` may be created lazily on complete.
//! - Event types: `task.enqueued`, `task.claimed`, `task.heartbeat`,
//!   `task.expired`, `task.completed`, `task.failed`, `task.output`.
//! - `expire_due` may either apply expiry itself or be a no-op if `claim`
//!   folds expiry. Tests call `expire_due` then `claim` and only assert the
//!   reclaim (attempt bump, old worker cannot complete).
//! - After reclaim, old-worker `complete` / `heartbeat` is `StaleAttempt` or
//!   `NotHolder`. Refused ops must not write output bytes.
//! - `claim` FIFO: enqueue order. Tasks that become claimable via expiry are
//!   appended behind already-queued ids (enqueue order among those expired).
//! - `get` returns `None` if the id was never enqueued.
//! - Output hash is SHA-256 of the raw output bytes, lowercase hex.
//! - Checkpoint branch is `task/{id}/attempt/{n}` (format helper on the iface).
//!
//! No GPU coverage in H3; these tests are CPU-only (no `gpu` marker).
//!
//! Production must never import this module.

#![allow(dead_code)]

use prometheus_leases::{
    Error, Lease, NowMs, Queue, QueueConfig, Result, Task, TaskId, TaskState, WorkerId, WorkItem,
    DEFAULT_HEARTBEAT_PERIOD_MS, MISSED_HEARTBEATS,
};
use serde_json::Value;
use std::path::{Path, PathBuf};

pub const SHORT_PERIOD_MS: u64 = 1_000;

pub fn short_config() -> QueueConfig {
    QueueConfig {
        heartbeat_period_ms: SHORT_PERIOD_MS,
        missed_heartbeats: MISSED_HEARTBEATS,
    }
}

pub fn default_config() -> QueueConfig {
    QueueConfig::default()
}

pub fn short_ttl() -> u64 {
    short_config().lease_ttl_ms()
}

/// Parent temp dir plus a not-yet-created `queue/` child for `Queue::create`.
pub fn fresh_queue_dir() -> (tempfile::TempDir, PathBuf) {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("queue");
    (parent, dir)
}

pub fn log_dir(queue_dir: &Path) -> PathBuf {
    queue_dir.join("log")
}

pub fn events_jsonl(queue_dir: &Path) -> PathBuf {
    log_dir(queue_dir).join("events.jsonl")
}

pub fn output_path(queue_dir: &Path, task_id: &TaskId, attempt: u64) -> PathBuf {
    queue_dir
        .join("outputs")
        .join(&task_id.0)
        .join(attempt.to_string())
}

pub fn worker(id: &str) -> WorkerId {
    WorkerId(id.to_string())
}

pub fn item(id: &str, payload: Value) -> WorkItem {
    WorkItem {
        task_id: TaskId(id.to_string()),
        payload,
    }
}

pub fn item_n(n: u32) -> WorkItem {
    item(&format!("t{n}"), serde_json::json!({"n": n}))
}

pub fn assert_err<T>(r: Result<T>, what: &str) {
    if r.is_ok() {
        panic!("expected error ({what}), got Ok");
    }
}

pub fn assert_duplicate(err: &Error, task_id: &str) {
    match err {
        Error::Duplicate(id) => assert_eq!(id, task_id, "Duplicate id"),
        other => panic!("expected Duplicate({task_id}), got {other:?}"),
    }
}

pub fn assert_not_found(err: &Error, task_id: &str) {
    match err {
        Error::NotFound(id) => assert_eq!(id, task_id, "NotFound id"),
        other => panic!("expected NotFound({task_id}), got {other:?}"),
    }
}

/// After reclaim, the old worker is refused with either variant.
pub fn assert_stale_or_not_holder(err: &Error, what: &str) {
    match err {
        Error::StaleAttempt { .. } | Error::NotHolder(..) => {}
        other => panic!("{what}: expected StaleAttempt or NotHolder, got {other:?}"),
    }
}

pub fn assert_lease_eq(got: &Lease, task_id: &str, worker_id: &str, attempt: u64, expires_at: NowMs) {
    assert_eq!(got.task_id.0, task_id, "lease.task_id");
    assert_eq!(got.worker_id.0, worker_id, "lease.worker_id");
    assert_eq!(got.attempt, attempt, "lease.attempt");
    assert_eq!(got.expires_at, expires_at, "lease.expires_at");
}

pub fn assert_task_eq(got: &Task, expect: &Task) {
    assert_eq!(got.id.0, expect.id.0, "task.id");
    assert_eq!(got.payload, expect.payload, "task.payload");
    assert_eq!(got.state, expect.state, "task.state");
    assert_eq!(got.attempt, expect.attempt, "task.attempt");
    assert_eq!(
        got.worker_id.as_ref().map(|w| &w.0),
        expect.worker_id.as_ref().map(|w| &w.0),
        "task.worker_id"
    );
    assert_eq!(got.expires_at, expect.expires_at, "task.expires_at");
    assert_eq!(got.output_hash, expect.output_hash, "task.output_hash");
}

pub fn assert_queued(task: &Task, id: &str, attempt: u64) {
    assert_eq!(task.id.0, id);
    assert_eq!(task.state, TaskState::Queued, "expected Queued");
    assert_eq!(task.attempt, attempt, "queued attempt");
    assert!(task.worker_id.is_none(), "queued worker_id");
    assert!(task.expires_at.is_none(), "queued expires_at");
}

pub fn assert_leased(
    task: &Task,
    id: &str,
    worker_id: &str,
    attempt: u64,
    expires_at: NowMs,
) {
    assert_eq!(task.id.0, id);
    assert_eq!(task.state, TaskState::Leased, "expected Leased");
    assert_eq!(task.attempt, attempt);
    assert_eq!(
        task.worker_id.as_ref().map(|w| w.0.as_str()),
        Some(worker_id)
    );
    assert_eq!(task.expires_at, Some(expires_at));
}

pub fn assert_completed(task: &Task, id: &str, attempt: u64, output_hash: &str) {
    assert_eq!(task.id.0, id);
    assert_eq!(task.state, TaskState::Completed, "expected Completed");
    assert_eq!(task.attempt, attempt);
    assert_eq!(task.output_hash.as_deref(), Some(output_hash));
    assert!(task.expires_at.is_none(), "completed expires_at");
}

pub fn assert_failed(task: &Task, id: &str, attempt: u64) {
    assert_eq!(task.id.0, id);
    assert_eq!(task.state, TaskState::Failed, "expected Failed");
    assert_eq!(task.attempt, attempt);
    assert!(task.output_hash.is_none(), "failed output_hash");
    assert!(task.expires_at.is_none(), "failed expires_at");
}

pub fn event_types(queue: &Queue) -> Vec<String> {
    queue
        .log()
        .iter()
        .map(|e| e.event_type.clone())
        .collect()
}

pub fn default_ttl() -> u64 {
    DEFAULT_HEARTBEAT_PERIOD_MS * u64::from(MISSED_HEARTBEATS)
}
