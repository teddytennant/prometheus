//! Slow, obvious reference for `prometheus-chaos`.
//!
//! Independent of `harness/chaos/src/`. Production code must never import this
//! module. Event logs are real H2 [`EventLog`]s; fault/reachability math is
//! in-memory and deliberate.
//!
//! # Semantics (the contract tests encode)
//!
//! - `NowMs` is injected. `advance` adds to `now`. Nothing reads the wall clock.
//! - Replica ids: `"0"`..`n-1`. Processes: `coordinator`, `worker-0`, …
//! - `enqueue_task` writes [`EVENT_ENQUEUED`] to every replica reachable from
//!   origin replica 0 (falling back to the first writeable replica).
//! - `complete_task` writes [`EVENT_COMPLETED`] from origin replica 1 when it
//!   is writeable (so an asymmetric `0→1` partition still delivers the reverse).
//! - A replica is writeable iff it has an open handle, is not disk-full, not
//!   corrupt, and not node-lost.
//! - Directed partition edges drop replication. Symmetric partition drops both
//!   directions; `asymmetric: true` drops only `from→to`.
//! - `Kill` marks the process dead and drops **all** in-memory EventLog handles
//!   (crash-only). Disk files remain. `recover` reopens non-corrupt replicas
//!   (startup is replay), clears process kills and node-loss, appends
//!   [`EVENT_RECOVER`]. DiskCorrupt is not healed.
//! - `lost_tasks`: API-enqueued task ids with no `EVENT_COMPLETED` on any
//!   replica that still counts (not corrupt, not node-lost).
//! - `duplicated_outputs`: extra `complete_task` calls for the same
//!   `(task, attempt)`.
//! - Clock skew: `|delta_ms| > CLOCK_SKEW_MS` ⇒ `Error::ClockSkewBound`.
//! - D0 RNG: Knuth LCG `state = state.wrapping_mul(6364136223846793005).wrapping_add(1)`
//!   starting at `seed`; `pick(n) = (next() as usize) % n`.
//!
//! Hosted providers: [`crate::common::HOSTED_PROVIDERS`].

#![allow(dead_code)]

use crate::common::{
    process_ids, replica_ids, replica_path, D1_LEASE_EXPIRY_MS, HOSTED_PROVIDERS,
};
use prometheus_chaos::{
    Error, Fault, Gate, Invariants, JobId, NowMs, ProcessId, ReplicaId, Result, RunReport,
    SoakConfig, TaskId, WorldConfig, CLOCK_SKEW_MS, D0_KILLS, EVENT_COMPLETED, EVENT_ENQUEUED,
    EVENT_FAULT, EVENT_RECOVER, MIN_REPLICAS, SOAK_MS,
};
use prometheus_log::{Append, EventLog};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Deterministic Knuth LCG used by D0 / soak.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }

    pub fn pick(&mut self, n: usize) -> usize {
        assert!(n > 0, "pick empty");
        (self.next() as usize) % n
    }
}

#[derive(Clone, Debug)]
struct StoredEvent {
    event_type: String,
    task_id: Option<String>,
    attempt: Option<u64>,
}

struct Replica {
    id: ReplicaId,
    log: Option<EventLog>,
    events: Vec<StoredEvent>,
    disk_full: bool,
    corrupt: bool,
    lost: bool,
    clock_delta: i64,
}

pub struct RefWorld {
    dir: PathBuf,
    config: WorldConfig,
    now: NowMs,
    replicas: Vec<Replica>,
    processes: Vec<ProcessId>,
    dropped_edges: HashSet<(String, String)>,
    outages: HashSet<String>,
    hung_squeue: bool,
    broker_dead: bool,
    token_expired: bool,
    api_enqueued: Vec<String>,
    complete_counts: HashMap<(String, u64), u64>,
}

impl RefWorld {
    pub fn create(dir: impl AsRef<Path>, config: WorldConfig) -> Result<Self> {
        if config.n_replicas < MIN_REPLICAS {
            return Err(Error::TooFewReplicas(config.n_replicas));
        }
        let dir = dir.as_ref().to_path_buf();
        if dir.exists() {
            return Err(Error::Other(format!(
                "directory exists: {}",
                dir.display()
            )));
        }
        std::fs::create_dir(&dir).map_err(|e| Error::Other(e.to_string()))?;
        std::fs::create_dir(dir.join("replicas")).map_err(|e| Error::Other(e.to_string()))?;
        let mut replicas = Vec::new();
        for id in replica_ids(config.n_replicas) {
            let path = replica_path(&dir, &id);
            let log = EventLog::create(&path)?;
            replicas.push(Replica {
                id,
                log: Some(log),
                events: Vec::new(),
                disk_full: false,
                corrupt: false,
                lost: false,
                clock_delta: 0,
            });
        }
        let processes = process_ids(config.n_processes);
        Ok(Self {
            dir,
            config,
            now: 0,
            replicas,
            processes,
            dropped_edges: HashSet::new(),
            outages: HashSet::new(),
            hung_squeue: false,
            broker_dead: false,
            token_expired: false,
            api_enqueued: Vec::new(),
            complete_counts: HashMap::new(),
        })
    }

    pub fn open(dir: impl AsRef<Path>, config: WorldConfig) -> Result<Self> {
        if config.n_replicas < MIN_REPLICAS {
            return Err(Error::TooFewReplicas(config.n_replicas));
        }
        let dir = dir.as_ref().to_path_buf();
        if !dir.is_dir() {
            return Err(Error::Other(format!(
                "missing world dir: {}",
                dir.display()
            )));
        }
        let mut replicas = Vec::new();
        for id in replica_ids(config.n_replicas) {
            let path = replica_path(&dir, &id);
            let log = EventLog::open(&path)?;
            let events = log
                .iter()
                .map(|e| StoredEvent {
                    event_type: e.event_type.clone(),
                    task_id: e.task_id.clone(),
                    attempt: e.attempt,
                })
                .collect();
            replicas.push(Replica {
                id,
                log: Some(log),
                events,
                disk_full: false,
                corrupt: false,
                lost: false,
                clock_delta: 0,
            });
        }
        let mut world = Self {
            processes: process_ids(config.n_processes),
            dir,
            config,
            now: 0,
            replicas,
            dropped_edges: HashSet::new(),
            outages: HashSet::new(),
            hung_squeue: false,
            broker_dead: false,
            token_expired: false,
            api_enqueued: Vec::new(),
            complete_counts: HashMap::new(),
        };
        world.rebuild_api_from_logs();
        Ok(world)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config(&self) -> &WorldConfig {
        &self.config
    }

    pub fn now(&self) -> NowMs {
        self.now
    }

    pub fn advance(&mut self, ms: u64) {
        self.now = self.now.saturating_add(ms);
    }

    pub fn replicas(&self) -> Vec<ReplicaId> {
        self.replicas.iter().map(|r| r.id.clone()).collect()
    }

    pub fn processes(&self) -> Vec<ProcessId> {
        self.processes.clone()
    }

    pub fn replica_dir(&self, id: &ReplicaId) -> PathBuf {
        replica_path(&self.dir, id)
    }

    pub fn enqueue_task(&mut self, task: &TaskId, now: NowMs) -> Result<()> {
        let origin = self.enqueue_origin().ok_or_else(|| {
            Error::Other("no writeable replica (killed: recover required)".into())
        })?;
        let dests = self.destinations(origin);
        let payload = json!({"task_id": task.0, "now": now});
        for i in dests {
            self.append_event(i, EVENT_ENQUEUED, Some(&task.0), None, payload.clone(), now)?;
        }
        self.api_enqueued.push(task.0.clone());
        Ok(())
    }

    pub fn complete_task(
        &mut self,
        task: &TaskId,
        attempt: u64,
        output: &[u8],
        now: NowMs,
    ) -> Result<()> {
        let origin = self.complete_origin().ok_or_else(|| {
            Error::Other("no writeable replica (killed: recover required)".into())
        })?;
        let dests = self.destinations(origin);
        let payload = json!({
            "task_id": task.0,
            "attempt": attempt,
            "output": String::from_utf8_lossy(output),
            "now": now,
        });
        for i in dests {
            self.append_event(
                i,
                EVENT_COMPLETED,
                Some(&task.0),
                Some(attempt),
                payload.clone(),
                now,
            )?;
        }
        let key = (task.0.clone(), attempt);
        *self.complete_counts.entry(key).or_insert(0) += 1;
        Ok(())
    }

    pub fn inject(&mut self, fault: &Fault) -> Result<()> {
        match fault {
            Fault::Kill { process } => self.require_process(process)?,
            Fault::Partition { from, to, .. } | Fault::HealPartition { from, to } => {
                self.require_replica(from)?;
                self.require_replica(to)?;
            }
            Fault::DiskFull { replica }
            | Fault::DiskCorrupt { replica }
            | Fault::ClockSkew { replica, .. }
            | Fault::NodeLoss { replica } => self.require_replica(replica)?,
            Fault::NodeReplace { old, .. } => self.require_replica(old)?,
            _ => {}
        }
        if let Fault::ClockSkew { delta_ms, .. } = fault {
            if delta_ms.unsigned_abs() > CLOCK_SKEW_MS.unsigned_abs() {
                return Err(Error::ClockSkewBound(*delta_ms));
            }
        }

        // Record the fault on every currently writeable replica, ignoring
        // partitions (injection is local to the simulator).
        let payload = serde_json::to_value(fault).unwrap_or_else(|_| json!({}));
        let dests: Vec<usize> = (0..self.replicas.len())
            .filter(|&i| self.can_write(i))
            .collect();
        let now = self.now;
        for i in dests {
            self.append_event(i, EVENT_FAULT, None, None, payload.clone(), now)?;
        }

        match fault {
            Fault::Kill { process: _ } => {
                for r in &mut self.replicas {
                    r.log = None;
                }
            }
            Fault::Partition {
                from,
                to,
                asymmetric,
            } => {
                self.dropped_edges.insert((from.0.clone(), to.0.clone()));
                if !asymmetric {
                    self.dropped_edges.insert((to.0.clone(), from.0.clone()));
                }
            }
            Fault::HealPartition { from, to } => {
                self.dropped_edges.remove(&(from.0.clone(), to.0.clone()));
                self.dropped_edges.remove(&(to.0.clone(), from.0.clone()));
            }
            Fault::Outage { provider } => {
                self.outages.insert(provider.clone());
            }
            Fault::HealOutage { provider } => {
                self.outages.remove(provider);
            }
            Fault::DiskFull { replica } => {
                let i = self.index_of(replica)?;
                self.replicas[i].disk_full = true;
            }
            Fault::DiskCorrupt { replica } => {
                let i = self.index_of(replica)?;
                self.replicas[i].log = None;
                self.replicas[i].corrupt = true;
                self.replicas[i].events.clear();
                let path = replica_path(&self.dir, replica).join("events.jsonl");
                std::fs::write(&path, b"{broken hash chain\n")
                    .map_err(|e| Error::Disk(e.to_string()))?;
            }
            Fault::ClockSkew { replica, delta_ms } => {
                let i = self.index_of(replica)?;
                self.replicas[i].clock_delta = *delta_ms;
            }
            Fault::TokenExpiry => self.token_expired = true,
            Fault::TokenRotate => self.token_expired = false,
            Fault::NodeLoss { replica } => {
                let i = self.index_of(replica)?;
                self.replicas[i].lost = true;
                self.replicas[i].log = None;
            }
            Fault::NodeReplace { old, new } => {
                let i = self.index_of(old)?;
                self.replicas[i].lost = true;
                self.replicas[i].log = None;
                if self.index_of(new).is_err() {
                    let path = replica_path(&self.dir, new);
                    let log = EventLog::create(&path)?;
                    self.replicas.push(Replica {
                        id: new.clone(),
                        log: Some(log),
                        events: Vec::new(),
                        disk_full: false,
                        corrupt: false,
                        lost: false,
                        clock_delta: 0,
                    });
                }
            }
            Fault::JobPreempt { .. } | Fault::WalltimeKill { .. } => {}
            Fault::HungSqueue => self.hung_squeue = true,
            Fault::UnhangSqueue => self.hung_squeue = false,
            Fault::BrokerDeath => self.broker_dead = true,
            Fault::RateLimit { provider } => {
                self.outages.insert(provider.clone());
            }
        }
        Ok(())
    }

    pub fn recover(&mut self) -> Result<()> {
        let now = self.now;
        let n = self.replicas.len();
        for i in 0..n {
            if self.replicas[i].corrupt {
                continue;
            }
            if self.replicas[i].log.is_none() {
                let path = replica_path(&self.dir, &self.replicas[i].id);
                match EventLog::open(&path) {
                    Ok(log) => {
                        self.replicas[i].log = Some(log);
                        self.replicas[i].lost = false;
                    }
                    Err(e) => {
                        // Disk still broken.
                        return Err(e.into());
                    }
                }
            } else {
                self.replicas[i].lost = false;
            }
            self.append_event(i, EVENT_RECOVER, None, None, json!({}), now)?;
        }
        Ok(())
    }

    pub fn invariants(&self) -> Result<Invariants> {
        Ok(self.compute_invariants())
    }

    pub fn log_len(&self, id: &ReplicaId) -> Result<usize> {
        let r = self.find(id)?;
        if r.corrupt {
            return Err(Error::Disk(format!("corrupt replica {}", id.0)));
        }
        Ok(r.events.len())
    }

    pub fn replica_log(&self, id: &ReplicaId) -> Result<&EventLog> {
        let r = self.find(id)?;
        r.log
            .as_ref()
            .ok_or_else(|| Error::Other(format!("replica {} unavailable", id.0)))
    }

    fn compute_invariants(&self) -> Invariants {
        let mut completed: HashSet<String> = HashSet::new();
        for r in &self.replicas {
            if r.corrupt || r.lost {
                continue;
            }
            for e in &r.events {
                if e.event_type == EVENT_COMPLETED {
                    if let Some(id) = &e.task_id {
                        completed.insert(id.clone());
                    }
                }
            }
        }
        let mut seen = HashSet::new();
        let mut lost_tasks = 0u64;
        for id in &self.api_enqueued {
            if !seen.insert(id.clone()) {
                continue;
            }
            if !completed.contains(id) {
                lost_tasks += 1;
            }
        }
        let duplicated_outputs = self
            .complete_counts
            .values()
            .map(|c| c.saturating_sub(1))
            .sum();
        Invariants {
            lost_tasks,
            duplicated_outputs,
        }
    }

    fn rebuild_api_from_logs(&mut self) {
        self.api_enqueued.clear();
        self.complete_counts.clear();
        let mut seen_enq = HashSet::new();
        for r in &self.replicas {
            for e in &r.events {
                if e.event_type == EVENT_ENQUEUED {
                    if let Some(id) = &e.task_id {
                        if seen_enq.insert(id.clone()) {
                            self.api_enqueued.push(id.clone());
                        }
                    }
                }
                if e.event_type == EVENT_COMPLETED {
                    if let (Some(id), Some(a)) = (&e.task_id, e.attempt) {
                        let key = (id.clone(), a);
                        let slot = self.complete_counts.entry(key).or_insert(0);
                        if *slot == 0 {
                            *slot = 1;
                        }
                    }
                }
            }
        }
        // Duplicates: max per-replica extra completes.
        let mut per_key_max: HashMap<(String, u64), u64> = HashMap::new();
        for r in &self.replicas {
            let mut local: HashMap<(String, u64), u64> = HashMap::new();
            for e in &r.events {
                if e.event_type == EVENT_COMPLETED {
                    if let (Some(id), Some(a)) = (&e.task_id, e.attempt) {
                        *local.entry((id.clone(), a)).or_insert(0) += 1;
                    }
                }
            }
            for (k, c) in local {
                let m = per_key_max.entry(k).or_insert(0);
                if c > *m {
                    *m = c;
                }
            }
        }
        self.complete_counts = per_key_max;
    }

    fn find(&self, id: &ReplicaId) -> Result<&Replica> {
        self.replicas
            .iter()
            .find(|r| r.id == *id)
            .ok_or_else(|| Error::NoReplica(id.0.clone()))
    }

    fn index_of(&self, id: &ReplicaId) -> Result<usize> {
        self.replicas
            .iter()
            .position(|r| r.id == *id)
            .ok_or_else(|| Error::NoReplica(id.0.clone()))
    }

    fn require_replica(&self, id: &ReplicaId) -> Result<()> {
        self.index_of(id).map(|_| ())
    }

    fn require_process(&self, id: &ProcessId) -> Result<()> {
        if self.processes.iter().any(|p| p == id) {
            Ok(())
        } else {
            Err(Error::NoProcess(id.0.clone()))
        }
    }

    fn can_write(&self, i: usize) -> bool {
        let r = &self.replicas[i];
        r.log.is_some() && !r.disk_full && !r.corrupt && !r.lost
    }

    fn enqueue_origin(&self) -> Option<usize> {
        if !self.replicas.is_empty() && self.can_write(0) {
            Some(0)
        } else {
            (0..self.replicas.len()).find(|&i| self.can_write(i))
        }
    }

    fn complete_origin(&self) -> Option<usize> {
        if self.replicas.len() > 1 && self.can_write(1) {
            Some(1)
        } else {
            self.enqueue_origin()
        }
    }

    fn delivers(&self, from: usize, to: usize) -> bool {
        if !self.can_write(to) {
            return false;
        }
        if from == to {
            return true;
        }
        let a = &self.replicas[from].id.0;
        let b = &self.replicas[to].id.0;
        !self.dropped_edges.contains(&(a.clone(), b.clone()))
    }

    fn destinations(&self, origin: usize) -> Vec<usize> {
        (0..self.replicas.len())
            .filter(|&i| self.delivers(origin, i))
            .collect()
    }

    fn append_event(
        &mut self,
        idx: usize,
        event_type: &str,
        task_id: Option<&str>,
        attempt: Option<u64>,
        payload: Value,
        _now: NowMs,
    ) -> Result<()> {
        let node = self.replicas[idx].id.0.clone();
        if let Some(log) = self.replicas[idx].log.as_mut() {
            log.append(Append {
                event_type: event_type.into(),
                payload,
                timestamp: "1970-01-01T00:00:00Z".into(),
                task_id: task_id.map(|s| s.to_string()),
                attempt,
                node_id: Some(node),
            })?;
            self.replicas[idx].events.push(StoredEvent {
                event_type: event_type.into(),
                task_id: task_id.map(|s| s.to_string()),
                attempt,
            });
        }
        Ok(())
    }
}

fn bind_process(world: &RefWorld, p: &ProcessId) -> ProcessId {
    if world.processes.iter().any(|x| x == p) {
        p.clone()
    } else {
        world
            .processes
            .first()
            .cloned()
            .unwrap_or_else(|| p.clone())
    }
}

fn bind_replica(world: &RefWorld, r: &ReplicaId) -> ReplicaId {
    if world.replicas.iter().any(|x| x.id == *r) {
        r.clone()
    } else {
        world
            .replicas
            .first()
            .map(|x| x.id.clone())
            .unwrap_or_else(|| r.clone())
    }
}

fn bind_fault(world: &RefWorld, fault: Fault) -> Fault {
    match fault {
        Fault::Kill { process } => Fault::Kill {
            process: bind_process(world, &process),
        },
        Fault::Partition {
            from,
            to,
            asymmetric,
        } => Fault::Partition {
            from: bind_replica(world, &from),
            to: bind_replica(world, &to),
            asymmetric,
        },
        Fault::HealPartition { from, to } => Fault::HealPartition {
            from: bind_replica(world, &from),
            to: bind_replica(world, &to),
        },
        Fault::DiskFull { replica } => Fault::DiskFull {
            replica: bind_replica(world, &replica),
        },
        Fault::DiskCorrupt { replica } => Fault::DiskCorrupt {
            replica: bind_replica(world, &replica),
        },
        Fault::ClockSkew { replica, delta_ms } => Fault::ClockSkew {
            replica: bind_replica(world, &replica),
            delta_ms,
        },
        Fault::NodeLoss { replica } => Fault::NodeLoss {
            replica: bind_replica(world, &replica),
        },
        Fault::NodeReplace { old, new } => Fault::NodeReplace {
            old: bind_replica(world, &old),
            new,
        },
        other => other,
    }
}

fn bind_all(world: &RefWorld, faults: Vec<Fault>) -> Vec<Fault> {
    faults.into_iter().map(|f| bind_fault(world, f)).collect()
}

const WORK_N: u32 = 4;

fn work_tasks() -> Vec<TaskId> {
    (0..WORK_N)
        .map(|i| TaskId(format!("t{i}")))
        .collect()
}

fn enqueue_work(world: &mut RefWorld) -> Result<Vec<TaskId>> {
    let tasks = work_tasks();
    let now = world.now();
    for t in &tasks {
        world.enqueue_task(t, now)?;
    }
    Ok(tasks)
}

fn complete_work(world: &mut RefWorld, tasks: &[TaskId]) -> Result<()> {
    let now = world.now();
    for t in tasks {
        world.complete_task(t, 1, b"ok", now)?;
    }
    Ok(())
}

/// Fault kinds each gate must survive. Ids are placeholders.
pub fn ref_faults_for(gate: Gate) -> Vec<Fault> {
    match gate {
        Gate::D0 => vec![Fault::Kill {
            process: ProcessId("coordinator".into()),
        }],
        Gate::D1 => vec![
            Fault::Kill {
                process: ProcessId("coordinator".into()),
            },
            Fault::Partition {
                from: ReplicaId("0".into()),
                to: ReplicaId("1".into()),
                asymmetric: false,
            },
        ],
        Gate::D2 => vec![
            Fault::Partition {
                from: ReplicaId("0".into()),
                to: ReplicaId("1".into()),
                asymmetric: true,
            },
            Fault::ClockSkew {
                replica: ReplicaId("0".into()),
                delta_ms: CLOCK_SKEW_MS,
            },
            Fault::ClockSkew {
                replica: ReplicaId("1".into()),
                delta_ms: -CLOCK_SKEW_MS,
            },
            Fault::NodeLoss {
                replica: ReplicaId("0".into()),
            },
        ],
        Gate::D3 => vec![
            Fault::JobPreempt {
                job: JobId("job-0".into()),
            },
            Fault::WalltimeKill {
                job: JobId("job-0".into()),
            },
            Fault::HungSqueue,
        ],
        Gate::D4 => vec![
            Fault::BrokerDeath,
            Fault::TokenExpiry,
            Fault::Outage {
                provider: "xai".into(),
            },
        ],
        Gate::D5 => HOSTED_PROVIDERS
            .iter()
            .map(|p| Fault::Outage {
                provider: (*p).into(),
            })
            .collect(),
        Gate::Soak => {
            let mut all = Vec::new();
            for g in [Gate::D0, Gate::D1, Gate::D2, Gate::D3, Gate::D4, Gate::D5] {
                all.extend(ref_faults_for(g));
            }
            all
        }
    }
}

pub fn ref_invariants(world: &RefWorld) -> Result<Invariants> {
    world.invariants()
}

pub fn ref_run_gate(world: &mut RefWorld, gate: Gate) -> Result<RunReport> {
    if gate == Gate::Soak {
        return ref_run_soak(
            world,
            SoakConfig {
                duration_ms: SOAK_MS,
                seed: world.config().seed,
            },
        );
    }
    let tasks = enqueue_work(world)?;
    let mut faults_applied = 0u32;
    match gate {
        Gate::D0 => {
            let procs = world.processes();
            let mut rng = Lcg::new(world.config().seed);
            for _ in 0..D0_KILLS {
                let p = procs[rng.pick(procs.len())].clone();
                world.inject(&Fault::Kill { process: p })?;
                faults_applied += 1;
                world.recover()?;
            }
        }
        Gate::Soak => unreachable!(),
        other => {
            for f in bind_all(world, ref_faults_for(other)) {
                let is_kill = matches!(f, Fault::Kill { .. });
                world.inject(&f)?;
                faults_applied += 1;
                if is_kill {
                    world.recover()?;
                }
            }
            if other == Gate::D1 {
                world.advance(D1_LEASE_EXPIRY_MS);
            }
        }
    }
    world.recover()?;
    complete_work(world, &tasks)?;
    let inv = world.invariants()?;
    if !inv.hold() {
        return Err(Error::Invariant {
            lost_tasks: inv.lost_tasks,
            duplicated_outputs: inv.duplicated_outputs,
        });
    }
    Ok(RunReport {
        gate,
        faults_applied,
        invariants: inv,
        recovered: true,
    })
}

pub fn ref_run_soak(world: &mut RefWorld, cfg: SoakConfig) -> Result<RunReport> {
    let start = world.now();
    let duration = cfg.duration_ms;
    let tasks = enqueue_work(world)?;
    let mut faults_applied = 0u32;
    let gates = [Gate::D0, Gate::D1, Gate::D2, Gate::D3, Gate::D4, Gate::D5];
    let slice = duration / gates.len() as u64;
    for g in gates {
        let faults = if g == Gate::D0 {
            let procs = world.processes();
            let mut rng = Lcg::new(cfg.seed);
            let p = if procs.is_empty() {
                ProcessId("coordinator".into())
            } else {
                procs[rng.pick(procs.len())].clone()
            };
            vec![Fault::Kill { process: p }]
        } else {
            bind_all(world, ref_faults_for(g))
        };
        for f in faults {
            let is_kill = matches!(f, Fault::Kill { .. });
            if world.inject(&f).is_ok() {
                faults_applied += 1;
            }
            if is_kill {
                let _ = world.recover();
            }
        }
        world.advance(slice);
        let _ = world.recover();
    }
    if let Some(last) = world.replicas().last().cloned() {
        if world.inject(&Fault::DiskFull { replica: last }).is_ok() {
            faults_applied += 1;
        }
    }
    let elapsed = world.now().saturating_sub(start);
    if elapsed < duration {
        world.advance(duration - elapsed);
    }
    world.recover()?;
    complete_work(world, &tasks)?;
    let inv = world.invariants()?;
    if !inv.hold() {
        return Err(Error::Invariant {
            lost_tasks: inv.lost_tasks,
            duplicated_outputs: inv.duplicated_outputs,
        });
    }
    Ok(RunReport {
        gate: Gate::Soak,
        faults_applied,
        invariants: inv,
        recovered: true,
    })
}
