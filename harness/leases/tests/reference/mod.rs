//! Independent in-memory H3 queue + hashed outputs.
//!
//! Slow and obvious. Production (`harness/leases/src`) must never import this.
//! SHA-256 is computed here with `sha2` as a test-only oracle; the on-disk
//! EventLog hash chain is H2's job and is not reimplemented.

#![allow(dead_code)]

use prometheus_leases::{
    Error, Lease, NowMs, QueueConfig, Result, Task, TaskId, TaskState, WorkerId, WorkItem,
    EVENT_CLAIMED, EVENT_COMPLETED, EVENT_ENQUEUED, EVENT_EXPIRED, EVENT_FAILED, EVENT_HEARTBEAT,
    EVENT_OUTPUT,
};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};

/// SHA-256 of empty bytes (complete with `[]`).
pub const GOLDEN_EMPTY_OUTPUT_HASH: &str =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefEvent {
    pub event_type: String,
    pub task_id: Option<String>,
    pub attempt: Option<u64>,
}

/// In-memory queue mirroring the locked H3 state machine.
pub struct RefQueue {
    config: QueueConfig,
    tasks: HashMap<String, Task>,
    /// Enqueue order of every known id (for deterministic expire scans).
    all_ids: Vec<String>,
    /// Claimable FIFO (`Queued` ids).
    queued: VecDeque<String>,
    outputs: HashMap<(String, u64), Vec<u8>>,
    events: Vec<RefEvent>,
}

impl RefQueue {
    pub fn new(config: QueueConfig) -> Self {
        Self {
            config,
            tasks: HashMap::new(),
            all_ids: Vec::new(),
            queued: VecDeque::new(),
            outputs: HashMap::new(),
            events: Vec::new(),
        }
    }

    pub fn config(&self) -> &QueueConfig {
        &self.config
    }

    pub fn events(&self) -> &[RefEvent] {
        &self.events
    }

    pub fn event_types(&self) -> Vec<String> {
        self.events.iter().map(|e| e.event_type.clone()).collect()
    }

    fn ttl(&self) -> u64 {
        self.config.lease_ttl_ms()
    }

    fn emit(&mut self, event_type: &str, task_id: Option<&str>, attempt: Option<u64>) {
        self.events.push(RefEvent {
            event_type: event_type.to_string(),
            task_id: task_id.map(str::to_string),
            attempt,
        });
    }

    pub fn enqueue(&mut self, item: WorkItem, _now: NowMs) -> Result<TaskId> {
        let id = item.task_id.0;
        if self.tasks.contains_key(&id) {
            return Err(Error::Duplicate(id));
        }
        let task = Task {
            id: TaskId(id.clone()),
            payload: item.payload,
            state: TaskState::Queued,
            attempt: 0,
            worker_id: None,
            expires_at: None,
            output_hash: None,
        };
        self.tasks.insert(id.clone(), task);
        self.all_ids.push(id.clone());
        self.queued.push_back(id.clone());
        self.emit(EVENT_ENQUEUED, Some(&id), None);
        Ok(TaskId(id))
    }

    /// Apply due leases. A lease is due when `expires_at <= now`.
    ///
    /// Expired tasks go to the back of the claimable FIFO in original enqueue
    /// order. Attempt is not bumped here; the next `claim` bumps it.
    /// Holder is cleared so a later reclaim cannot be confused with this lease.
    pub fn expire_due(&mut self, now: NowMs) -> Result<Vec<TaskId>> {
        let mut expired = Vec::new();
        let ids = self.all_ids.clone();
        for id in ids {
            let (due, attempt) = {
                let t = self.tasks.get(&id).expect("known id");
                let due = t.state == TaskState::Leased
                    && t.expires_at.map(|e| e <= now).unwrap_or(false);
                (due, t.attempt)
            };
            if !due {
                continue;
            }
            {
                let t = self.tasks.get_mut(&id).expect("known id");
                t.state = TaskState::Queued;
                t.worker_id = None;
                t.expires_at = None;
            }
            if !self.queued.iter().any(|q| q == &id) {
                self.queued.push_back(id.clone());
            }
            self.emit(EVENT_EXPIRED, Some(&id), Some(attempt));
            expired.push(TaskId(id));
        }
        Ok(expired)
    }

    /// Folds `expire_due` then pops the claimable FIFO head.
    pub fn claim(&mut self, worker: &WorkerId, now: NowMs) -> Result<Option<Lease>> {
        self.expire_due(now)?;
        let id = match self.queued.pop_front() {
            Some(id) => id,
            None => return Ok(None),
        };
        let expires_at = now.saturating_add(self.ttl());
        let attempt = {
            let t = self.tasks.get_mut(&id).expect("queued id");
            t.attempt = t.attempt.saturating_add(1);
            t.state = TaskState::Leased;
            t.worker_id = Some(worker.clone());
            t.expires_at = Some(expires_at);
            t.attempt
        };
        self.emit(EVENT_CLAIMED, Some(&id), Some(attempt));
        Ok(Some(Lease {
            task_id: TaskId(id),
            worker_id: worker.clone(),
            attempt,
            expires_at,
        }))
    }

    /// Independent `claim_if`: expire_due first, then walk FIFO of currently
    /// `Queued` tasks and lease the first for which `pred` is true. Non-matches
    /// stay queued with the same attempt and no CLAIMED/EXPIRED events.
    pub fn claim_if<F>(
        &mut self,
        worker: &WorkerId,
        now: NowMs,
        pred: F,
    ) -> Result<Option<Lease>>
    where
        F: Fn(&Task) -> bool,
    {
        self.expire_due(now)?;
        let mut match_id: Option<String> = None;
        for id in self.queued.iter() {
            let task = self
                .tasks
                .get(id)
                .expect("queued id missing from tasks");
            if pred(task) {
                match_id = Some(id.clone());
                break;
            }
        }
        let Some(id) = match_id else {
            return Ok(None);
        };
        self.queued.retain(|q| q != &id);
        let expires_at = now.saturating_add(self.ttl());
        let attempt = {
            let t = self.tasks.get_mut(&id).expect("queued id");
            t.attempt = t.attempt.saturating_add(1);
            t.state = TaskState::Leased;
            t.worker_id = Some(worker.clone());
            t.expires_at = Some(expires_at);
            t.attempt
        };
        self.emit(EVENT_CLAIMED, Some(&id), Some(attempt));
        Ok(Some(Lease {
            task_id: TaskId(id),
            worker_id: worker.clone(),
            attempt,
            expires_at,
        }))
    }

    fn require_holder(
        &self,
        task_id: &TaskId,
        worker: &WorkerId,
        attempt: u64,
    ) -> Result<()> {
        let t = match self.tasks.get(&task_id.0) {
            Some(t) => t,
            None => return Err(Error::NotFound(task_id.0.clone())),
        };
        if t.attempt != attempt {
            return Err(Error::StaleAttempt {
                task_id: task_id.0.clone(),
                attempt,
                current: t.attempt,
            });
        }
        let holder = t.state == TaskState::Leased
            && t.worker_id.as_ref().map(|w| w.0.as_str()) == Some(worker.0.as_str());
        if !holder {
            return Err(Error::NotHolder(
                task_id.0.clone(),
                worker.0.clone(),
                attempt,
            ));
        }
        Ok(())
    }

    pub fn heartbeat(
        &mut self,
        task_id: &TaskId,
        worker: &WorkerId,
        attempt: u64,
        now: NowMs,
    ) -> Result<Lease> {
        self.require_holder(task_id, worker, attempt)?;
        let expires_at = now.saturating_add(self.ttl());
        let t = self.tasks.get_mut(&task_id.0).expect("holder");
        t.expires_at = Some(expires_at);
        self.emit(EVENT_HEARTBEAT, Some(&task_id.0), Some(attempt));
        Ok(Lease {
            task_id: task_id.clone(),
            worker_id: worker.clone(),
            attempt,
            expires_at,
        })
    }

    pub fn complete(
        &mut self,
        task_id: &TaskId,
        worker: &WorkerId,
        attempt: u64,
        output: &[u8],
        _now: NowMs,
    ) -> Result<()> {
        self.require_holder(task_id, worker, attempt)?;
        let hash = output_hash(output);
        self.outputs
            .insert((task_id.0.clone(), attempt), output.to_vec());
        let t = self.tasks.get_mut(&task_id.0).expect("holder");
        t.state = TaskState::Completed;
        t.expires_at = None;
        t.output_hash = Some(hash);
        self.emit(EVENT_OUTPUT, Some(&task_id.0), Some(attempt));
        self.emit(EVENT_COMPLETED, Some(&task_id.0), Some(attempt));
        Ok(())
    }

    pub fn fail(
        &mut self,
        task_id: &TaskId,
        worker: &WorkerId,
        attempt: u64,
        _reason: &str,
        _now: NowMs,
    ) -> Result<()> {
        self.require_holder(task_id, worker, attempt)?;
        let t = self.tasks.get_mut(&task_id.0).expect("holder");
        t.state = TaskState::Failed;
        t.expires_at = None;
        self.emit(EVENT_FAILED, Some(&task_id.0), Some(attempt));
        Ok(())
    }

    pub fn get(&self, task_id: &TaskId) -> Option<&Task> {
        self.tasks.get(&task_id.0)
    }

    pub fn get_output(&self, task_id: &TaskId, attempt: u64) -> Result<Option<Vec<u8>>> {
        if !self.tasks.contains_key(&task_id.0) {
            return Err(Error::NotFound(task_id.0.clone()));
        }
        Ok(self.outputs.get(&(task_id.0.clone(), attempt)).cloned())
    }
}

/// SHA-256 of raw output bytes, lowercase hex. Independent of EventLog hashing.
pub fn output_hash(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    to_hex(&digest)
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
    assert_eq!(output_hash(b""), GOLDEN_EMPTY_OUTPUT_HASH);
    assert_eq!(GOLDEN_EMPTY_OUTPUT_HASH.len(), 64);
    assert!(GOLDEN_EMPTY_OUTPUT_HASH
        .bytes()
        .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')));
}
