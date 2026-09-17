//! Task queue, work leases, heartbeats, and attempt-keyed outputs (spec 15.2, 15.5 H3).
//!
//! Durable state lives in an H2 `EventLog`. Startup is replay. A lease expires
//! after two missed heartbeats. Outputs and checkpoint branches are keyed by
//! `(task_id, attempt)` so a zombie worker cannot overwrite a newer attempt.
//!
//! This crate does not elect a coordinator (H5) and does not run Raft (H4).

use prometheus_log::EventLog;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Milliseconds since Unix epoch. Callers pass this so tests control expiry
/// without sleeping. Clock skew across nodes is an H4/D2 concern, not H3.
pub type NowMs = u64;

/// Default heartbeat period in milliseconds. Lease TTL is this times
/// [`MISSED_HEARTBEATS`].
pub const DEFAULT_HEARTBEAT_PERIOD_MS: u64 = 15_000;

/// A lease expires after this many heartbeat periods with no heartbeat.
pub const MISSED_HEARTBEATS: u32 = 2;

/// Event types written to the H2 log. Implementers must use these strings.
pub const EVENT_ENQUEUED: &str = "task.enqueued";
pub const EVENT_CLAIMED: &str = "task.claimed";
pub const EVENT_HEARTBEAT: &str = "task.heartbeat";
pub const EVENT_EXPIRED: &str = "task.expired";
pub const EVENT_COMPLETED: &str = "task.completed";
pub const EVENT_FAILED: &str = "task.failed";
pub const EVENT_OUTPUT: &str = "task.output";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("task {0} not found")]
    NotFound(String),
    #[error("task {0} is not leased to {1} attempt {2}")]
    NotHolder(String, String, u64),
    #[error("stale attempt {attempt} for task {task_id} (current {current})")]
    StaleAttempt {
        task_id: String,
        attempt: u64,
        current: u64,
    },
    #[error("task {0} is not claimable")]
    NotClaimable(String),
    #[error("duplicate task_id {0}")]
    Duplicate(String),
    #[error("{0}")]
    Log(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<prometheus_log::Error> for Error {
    fn from(err: prometheus_log::Error) -> Self {
        Error::Log(err.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerId(pub String);

/// First claim of a task is attempt 1. Reclaim after expiry bumps this by 1.
pub type Attempt = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Queued,
    Leased,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfig {
    pub heartbeat_period_ms: u64,
    pub missed_heartbeats: u32,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            heartbeat_period_ms: DEFAULT_HEARTBEAT_PERIOD_MS,
            missed_heartbeats: MISSED_HEARTBEATS,
        }
    }
}

impl QueueConfig {
    /// `heartbeat_period_ms * missed_heartbeats`.
    pub fn lease_ttl_ms(&self) -> u64 {
        self.heartbeat_period_ms
            .saturating_mul(u64::from(self.missed_heartbeats.max(1)))
    }
}

/// Work item the harness queues. This is not F1 `prometheus.task_spec` (that
/// schema is RL env tasks). Payload is opaque JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkItem {
    pub task_id: TaskId,
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lease {
    pub task_id: TaskId,
    pub worker_id: WorkerId,
    pub attempt: Attempt,
    pub expires_at: NowMs,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    pub payload: Value,
    pub state: TaskState,
    pub attempt: Attempt,
    pub worker_id: Option<WorkerId>,
    pub expires_at: Option<NowMs>,
    pub output_hash: Option<String>,
}

/// Durable task queue. On-disk layout: `<dir>/log/` is an H2 `EventLog`;
/// `<dir>/outputs/<task_id>/<attempt>` holds attempt-keyed bytes.
pub struct Queue {
    dir: PathBuf,
    log: EventLog,
    config: QueueConfig,
}

impl Queue {
    /// Create a new queue directory. Fails if it already exists.
    pub fn create(dir: impl AsRef<Path>, config: QueueConfig) -> Result<Self> {
        let _ = (dir, config);
        unimplemented!("H3: Queue::create")
    }

    /// Replay the H2 log and rebuild leases. Fail-closed on a corrupt log.
    pub fn open(dir: impl AsRef<Path>, config: QueueConfig) -> Result<Self> {
        let _ = (dir, config);
        unimplemented!("H3: Queue::open")
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config(&self) -> &QueueConfig {
        &self.config
    }

    pub fn log(&self) -> &EventLog {
        &self.log
    }

    /// Enqueue a work item. If `task_id` is already known, fail with `Duplicate`.
    pub fn enqueue(&mut self, item: WorkItem, now: NowMs) -> Result<TaskId> {
        let _ = (item, now);
        unimplemented!("H3: Queue::enqueue")
    }

    /// Claim the next queued task, or a task whose lease has expired as of `now`.
    /// Assigns attempt 1 on first claim and bumps attempt on reclaim.
    pub fn claim(&mut self, worker: &WorkerId, now: NowMs) -> Result<Option<Lease>> {
        let _ = (worker, now);
        unimplemented!("H3: Queue::claim")
    }

    /// Extend the lease if `worker` holds `attempt`. Expiry is `now + lease_ttl`.
    pub fn heartbeat(
        &mut self,
        task_id: &TaskId,
        worker: &WorkerId,
        attempt: Attempt,
        now: NowMs,
    ) -> Result<Lease> {
        let _ = (task_id, worker, attempt, now);
        unimplemented!("H3: Queue::heartbeat")
    }

    /// Reclaim every lease with `expires_at <= now`. Returns the expired task ids.
    /// After this they are claimable again (next `claim` bumps attempt).
    pub fn expire_due(&mut self, now: NowMs) -> Result<Vec<TaskId>> {
        let _ = now;
        unimplemented!("H3: Queue::expire_due")
    }

    /// Store bytes under `(task_id, attempt)` and mark completed. Refuses a
    /// stale attempt. A zombie holding an old lease cannot overwrite.
    pub fn complete(
        &mut self,
        task_id: &TaskId,
        worker: &WorkerId,
        attempt: Attempt,
        output: &[u8],
        now: NowMs,
    ) -> Result<()> {
        let _ = (task_id, worker, attempt, output, now);
        unimplemented!("H3: Queue::complete")
    }

    /// Mark failed for this attempt. Same holder checks as `complete`.
    pub fn fail(
        &mut self,
        task_id: &TaskId,
        worker: &WorkerId,
        attempt: Attempt,
        reason: &str,
        now: NowMs,
    ) -> Result<()> {
        let _ = (task_id, worker, attempt, reason, now);
        unimplemented!("H3: Queue::fail")
    }

    pub fn get(&self, task_id: &TaskId) -> Option<&Task> {
        let _ = task_id;
        unimplemented!("H3: Queue::get")
    }

    /// Bytes written by `complete` for this attempt, if any.
    pub fn get_output(&self, task_id: &TaskId, attempt: Attempt) -> Result<Option<Vec<u8>>> {
        let _ = (task_id, attempt);
        unimplemented!("H3: Queue::get_output")
    }

    /// Git branch the worker for this attempt commits onto. Reclaim bumps
    /// attempt; the new worker resumes from this name, not from `main`.
    pub fn checkpoint_branch(task_id: &TaskId, attempt: Attempt) -> String {
        format!("task/{}/attempt/{}", task_id.0, attempt)
    }
}
