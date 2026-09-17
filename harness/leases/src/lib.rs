//! Task queue, work leases, heartbeats, attempt-keyed outputs (spec 15.2, 15.5 H3).
//!
//! Durable source of truth is the H2 `EventLog` at `<dir>/log/`. In-memory
//! tasks and FIFO are a cache rebuilt by replaying that log.

use prometheus_log::{Append, Event, EventLog};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const EVENT_ENQUEUED: &str = "task.enqueued";
pub const EVENT_CLAIMED: &str = "task.claimed";
pub const EVENT_HEARTBEAT: &str = "task.heartbeat";
pub const EVENT_EXPIRED: &str = "task.expired";
pub const EVENT_COMPLETED: &str = "task.completed";
pub const EVENT_FAILED: &str = "task.failed";
pub const EVENT_OUTPUT: &str = "task.output";

pub const DEFAULT_HEARTBEAT_PERIOD_MS: u64 = 15_000;
pub const MISSED_HEARTBEATS: u32 = 2;

pub type Attempt = u64;
pub type NowMs = u64;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerId(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkItem {
    pub task_id: TaskId,
    pub payload: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskState {
    Queued,
    Leased,
    Completed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    pub payload: Value,
    pub state: TaskState,
    pub attempt: Attempt,
    pub worker_id: Option<WorkerId>,
    pub expires_at: Option<NowMs>,
    pub output_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lease {
    pub task_id: TaskId,
    pub worker_id: WorkerId,
    pub attempt: Attempt,
    pub expires_at: NowMs,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    pub fn lease_ttl_ms(&self) -> u64 {
        self.heartbeat_period_ms
            .saturating_mul(u64::from(self.missed_heartbeats.max(1)))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("not holder: task={0} worker={1} attempt={2}")]
    NotHolder(String, String, u64),
    #[error("stale attempt: task={task_id} attempt={attempt} current={current}")]
    StaleAttempt {
        task_id: String,
        attempt: u64,
        current: u64,
    },
    #[error("not claimable: {0}")]
    NotClaimable(String),
    #[error("duplicate: {0}")]
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

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Other(err.to_string())
    }
}

pub struct Queue {
    dir: PathBuf,
    log: EventLog,
    config: QueueConfig,
    tasks: HashMap<String, Task>,
    /// Enqueue order of every known id (including terminal). Expire scans this.
    all_ids: Vec<String>,
    /// FIFO of claimable (Queued) task ids.
    queued: VecDeque<String>,
}

impl Queue {
    pub fn create(dir: impl AsRef<Path>, config: QueueConfig) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir(&dir)?;
        let log = EventLog::create(dir.join("log"))?;
        Ok(Self {
            dir,
            log,
            config,
            tasks: HashMap::new(),
            all_ids: Vec::new(),
            queued: VecDeque::new(),
        })
    }

    pub fn open(dir: impl AsRef<Path>, config: QueueConfig) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let log = EventLog::open(dir.join("log"))?;
        let events: Vec<Event> = log.iter().cloned().collect();
        let mut queue = Self {
            dir,
            log,
            config,
            tasks: HashMap::new(),
            all_ids: Vec::new(),
            queued: VecDeque::new(),
        };
        for event in events {
            queue.apply(&event)?;
        }
        Ok(queue)
    }

    pub fn enqueue(&mut self, item: WorkItem, now: NowMs) -> Result<TaskId> {
        let id = item.task_id.0.clone();
        if self.tasks.contains_key(&id) {
            return Err(Error::Duplicate(id));
        }
        self.append(
            EVENT_ENQUEUED,
            Some(id.as_str()),
            None,
            item.payload,
            None,
            now,
        )?;
        Ok(item.task_id)
    }

    pub fn claim(&mut self, worker: &WorkerId, now: NowMs) -> Result<Option<Lease>> {
        self.expire_due(now)?;
        let Some(id) = self.queued.front().cloned() else {
            return Ok(None);
        };
        let attempt = self
            .tasks
            .get(&id)
            .map(|t| t.attempt.saturating_add(1))
            .expect("queued id must exist");
        let expires_at = now.saturating_add(self.config.lease_ttl_ms());
        let payload = json!({
            "worker_id": worker.0,
            "expires_at": expires_at,
        });
        self.append(
            EVENT_CLAIMED,
            Some(id.as_str()),
            Some(attempt),
            payload,
            Some(worker.0.as_str()),
            now,
        )?;
        Ok(Some(Lease {
            task_id: TaskId(id),
            worker_id: worker.clone(),
            attempt,
            expires_at,
        }))
    }

    pub fn heartbeat(
        &mut self,
        task_id: &TaskId,
        worker: &WorkerId,
        attempt: Attempt,
        now: NowMs,
    ) -> Result<Lease> {
        self.require_holder(task_id, worker, attempt)?;
        let expires_at = now.saturating_add(self.config.lease_ttl_ms());
        let payload = json!({ "expires_at": expires_at });
        self.append(
            EVENT_HEARTBEAT,
            Some(task_id.0.as_str()),
            Some(attempt),
            payload,
            Some(worker.0.as_str()),
            now,
        )?;
        Ok(Lease {
            task_id: task_id.clone(),
            worker_id: worker.clone(),
            attempt,
            expires_at,
        })
    }

    pub fn expire_due(&mut self, now: NowMs) -> Result<Vec<TaskId>> {
        let due: Vec<(String, Attempt)> = self
            .all_ids
            .iter()
            .filter_map(|id| {
                let t = self.tasks.get(id)?;
                if t.state == TaskState::Leased && t.expires_at.is_some_and(|exp| exp <= now) {
                    Some((id.clone(), t.attempt))
                } else {
                    None
                }
            })
            .collect();
        let mut expired = Vec::with_capacity(due.len());
        for (id, attempt) in due {
            self.append(
                EVENT_EXPIRED,
                Some(id.as_str()),
                Some(attempt),
                json!({}),
                None,
                now,
            )?;
            expired.push(TaskId(id));
        }
        Ok(expired)
    }

    pub fn complete(
        &mut self,
        task_id: &TaskId,
        worker: &WorkerId,
        attempt: Attempt,
        output: &[u8],
        now: NowMs,
    ) -> Result<()> {
        self.require_holder(task_id, worker, attempt)?;
        self.persist_output(task_id, attempt, output)?;
        let output_hash = sha256_hex(output);
        let payload = json!({ "output_hash": output_hash });
        self.append(
            EVENT_OUTPUT,
            Some(task_id.0.as_str()),
            Some(attempt),
            payload.clone(),
            Some(worker.0.as_str()),
            now,
        )?;
        self.append(
            EVENT_COMPLETED,
            Some(task_id.0.as_str()),
            Some(attempt),
            payload,
            Some(worker.0.as_str()),
            now,
        )?;
        Ok(())
    }

    pub fn fail(
        &mut self,
        task_id: &TaskId,
        worker: &WorkerId,
        attempt: Attempt,
        reason: &str,
        now: NowMs,
    ) -> Result<()> {
        self.require_holder(task_id, worker, attempt)?;
        let payload = json!({ "reason": reason });
        self.append(
            EVENT_FAILED,
            Some(task_id.0.as_str()),
            Some(attempt),
            payload,
            Some(worker.0.as_str()),
            now,
        )?;
        Ok(())
    }

    pub fn get(&self, task_id: &TaskId) -> Option<&Task> {
        self.tasks.get(&task_id.0)
    }

    pub fn get_output(&self, task_id: &TaskId, attempt: Attempt) -> Result<Option<Vec<u8>>> {
        if !self.tasks.contains_key(&task_id.0) {
            return Err(Error::NotFound(task_id.0.clone()));
        }
        match std::fs::read(self.output_path(task_id, attempt)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    pub fn checkpoint_branch(task_id: &TaskId, attempt: Attempt) -> String {
        format!("task/{}/attempt/{attempt}", task_id.0)
    }

    pub fn log(&self) -> &EventLog {
        &self.log
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config(&self) -> &QueueConfig {
        &self.config
    }

    fn require_holder(&self, task_id: &TaskId, worker: &WorkerId, attempt: Attempt) -> Result<()> {
        let Some(t) = self.tasks.get(&task_id.0) else {
            return Err(Error::NotFound(task_id.0.clone()));
        };
        if t.attempt != attempt {
            return Err(Error::StaleAttempt {
                task_id: task_id.0.clone(),
                attempt,
                current: t.attempt,
            });
        }
        if t.state != TaskState::Leased || t.worker_id.as_ref() != Some(worker) {
            return Err(Error::NotHolder(
                task_id.0.clone(),
                worker.0.clone(),
                attempt,
            ));
        }
        Ok(())
    }

    fn append(
        &mut self,
        event_type: &str,
        task_id: Option<&str>,
        attempt: Option<Attempt>,
        payload: Value,
        node_id: Option<&str>,
        now: NowMs,
    ) -> Result<()> {
        let event = self.log.append(Append {
            event_type: event_type.to_string(),
            payload,
            timestamp: now.to_string(),
            task_id: task_id.map(str::to_string),
            attempt,
            node_id: node_id.map(str::to_string),
        })?;
        self.apply(&event)?;
        Ok(())
    }

    fn apply(&mut self, event: &Event) -> Result<()> {
        let Some(id) = event.task_id.clone() else {
            return Err(Error::Log(format!(
                "event {} seq {} missing task_id",
                event.event_type, event.seq
            )));
        };
        match event.event_type.as_str() {
            EVENT_ENQUEUED => {
                if self.tasks.contains_key(&id) {
                    return Err(Error::Duplicate(id));
                }
                self.tasks.insert(
                    id.clone(),
                    Task {
                        id: TaskId(id.clone()),
                        payload: event.payload.clone(),
                        state: TaskState::Queued,
                        attempt: event.attempt.unwrap_or(0),
                        worker_id: None,
                        expires_at: None,
                        output_hash: None,
                    },
                );
                self.all_ids.push(id.clone());
                self.queued.push_back(id);
            }
            EVENT_CLAIMED => {
                {
                    let t = self.task_mut(&id)?;
                    t.attempt = event.attempt.unwrap_or_else(|| t.attempt.saturating_add(1));
                    t.state = TaskState::Leased;
                    t.worker_id = worker_from_event(event).map(WorkerId);
                    t.expires_at = event.payload.get("expires_at").and_then(Value::as_u64);
                }
                self.queued.retain(|q| q != &id);
            }
            EVENT_HEARTBEAT => {
                let t = self.task_mut(&id)?;
                t.expires_at = event.payload.get("expires_at").and_then(Value::as_u64);
            }
            EVENT_EXPIRED => {
                {
                    let t = self.task_mut(&id)?;
                    t.state = TaskState::Queued;
                    t.worker_id = None;
                    t.expires_at = None;
                }
                if !self.queued.iter().any(|q| q == &id) {
                    self.queued.push_back(id);
                }
            }
            EVENT_OUTPUT => {
                let t = self.task_mut(&id)?;
                t.output_hash = event
                    .payload
                    .get("output_hash")
                    .and_then(Value::as_str)
                    .map(str::to_string);
            }
            EVENT_COMPLETED => {
                {
                    let t = self.task_mut(&id)?;
                    t.state = TaskState::Completed;
                    t.expires_at = None;
                    if let Some(hash) = event.payload.get("output_hash").and_then(Value::as_str) {
                        t.output_hash = Some(hash.to_string());
                    }
                }
                self.queued.retain(|q| q != &id);
            }
            EVENT_FAILED => {
                {
                    let t = self.task_mut(&id)?;
                    t.state = TaskState::Failed;
                    t.expires_at = None;
                }
                self.queued.retain(|q| q != &id);
            }
            _ => {
                return Err(Error::Log(format!(
                    "unknown event type {} at seq {}",
                    event.event_type, event.seq
                )));
            }
        }
        Ok(())
    }

    fn task_mut(&mut self, id: &str) -> Result<&mut Task> {
        self.tasks
            .get_mut(id)
            .ok_or_else(|| Error::Log(format!("replay of unknown task {id}")))
    }

    fn persist_output(&self, task_id: &TaskId, attempt: Attempt, output: &[u8]) -> Result<()> {
        let task_dir = self.dir.join("outputs").join(&task_id.0);
        std::fs::create_dir_all(&task_dir)?;
        let path = task_dir.join(attempt.to_string());
        let mut file = File::create(&path)?;
        file.write_all(output)?;
        file.sync_all()?;
        fsync_dir(&task_dir)?;
        if let Some(outputs) = task_dir.parent() {
            fsync_dir(outputs)?;
        }
        Ok(())
    }

    fn output_path(&self, task_id: &TaskId, attempt: Attempt) -> PathBuf {
        self.dir
            .join("outputs")
            .join(&task_id.0)
            .join(attempt.to_string())
    }
}

fn worker_from_event(event: &Event) -> Option<String> {
    event.node_id.clone().or_else(|| {
        event
            .payload
            .get("worker_id")
            .and_then(Value::as_str)
            .map(str::to_string)
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn fsync_dir(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}
