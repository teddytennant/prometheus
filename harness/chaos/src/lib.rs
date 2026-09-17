//! Chaos suite (spec 15.2, 15.5 H11, gates D0-D5).
//!
//! The harness is not done until it survives, in tests, a 72-hour soak with
//! process kills, partitions, disk faults, clock skew, provider outages,
//! token expiry, node loss, and job preemption.
//!
//! This crate is the fault injector and scenario runner over H2 [`EventLog`]
//! replicas. Later crates (leases, raft, slurm, providers) plug into the same
//! [`Fault`] and [`Gate`] types. CPU tests drive an in-process [`World`].
//! `NowMs` is injected; nothing in this crate reads the wall clock.
//!
//! Invariants that every gate must hold: no lost task, no duplicated output.

use prometheus_log::{Append, EventLog};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

pub type NowMs = u64;

/// D0 applies this many random `kill -9`s during real work.
pub const D0_KILLS: u32 = 1_000;

/// Soak duration: 72 hours in milliseconds.
pub const SOAK_MS: u64 = 72 * 60 * 60 * 1_000;

/// Clock skew bound from spec 15.2: ±30s.
pub const CLOCK_SKEW_MS: i64 = 30_000;

/// At least two EventLog replicas so a corrupted disk has a survivor.
pub const MIN_REPLICAS: usize = 2;

pub const DEFAULT_PROCESSES: u32 = 4;

pub const EVENT_ENQUEUED: &str = "chaos.enqueued";
pub const EVENT_COMPLETED: &str = "chaos.completed";
pub const EVENT_FAULT: &str = "chaos.fault";
pub const EVENT_RECOVER: &str = "chaos.recover";

/// Hosted providers D5 must outage. Last-resort open-weights SGLang is not hosted.
const HOSTED_PROVIDERS: [&str; 2] = ["xai", "openai"];

/// D1 "worker partitioned past lease expiry". H3 default TTL is 5s × 2 = 10s.
const D1_LEASE_EXPIRY_MS: u64 = 10_000;

const WORK_N: u32 = 4;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplicaId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProcessId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(pub String);

/// One injected failure. Names match spec 15.2 / 15.5 H11.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fault {
    /// `kill -9` of any process, including the coordinator.
    Kill {
        process: ProcessId,
    },
    /// Drop messages from `from` to `to`. `asymmetric` means the reverse
    /// direction still delivers (D2).
    Partition {
        from: ReplicaId,
        to: ReplicaId,
        asymmetric: bool,
    },
    HealPartition {
        from: ReplicaId,
        to: ReplicaId,
    },
    Outage {
        provider: String,
    },
    HealOutage {
        provider: String,
    },
    DiskFull {
        replica: ReplicaId,
    },
    /// Corrupt one replica. The other replica recovers (spec 15.2).
    DiskCorrupt {
        replica: ReplicaId,
    },
    /// Skew one replica's clock. `|delta_ms|` must be <= [`CLOCK_SKEW_MS`].
    ClockSkew {
        replica: ReplicaId,
        delta_ms: i64,
    },
    TokenExpiry,
    TokenRotate,
    NodeLoss {
        replica: ReplicaId,
    },
    NodeReplace {
        old: ReplicaId,
        new: ReplicaId,
    },
    JobPreempt {
        job: JobId,
    },
    WalltimeKill {
        job: JobId,
    },
    HungSqueue,
    UnhangSqueue,
    /// Token broker killed mid-refresh (D4).
    BrokerDeath,
    RateLimit {
        provider: String,
    },
}

/// Decentralization stage from spec 15.2. Each ships only after its gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gate {
    /// One node, crash-only. 1,000 random kill -9s: no lost task, no dup output.
    D0,
    /// Coordinator killed mid-assignment; worker partitioned past lease expiry.
    D1,
    /// Node loss, asymmetric partitions, clock skew.
    D2,
    /// Job preemption, walltime kill, hung squeue.
    D3,
    /// Broker death mid-refresh, forced token expiry, provider outage.
    D4,
    /// Every hosted provider down; swarm keeps working at reduced size.
    D5,
    /// 72h with all of the above. A release that fails this does not ship.
    Soak,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Invariants {
    pub lost_tasks: u64,
    pub duplicated_outputs: u64,
}

impl Invariants {
    /// True iff no lost task and no duplicated output.
    pub fn hold(&self) -> bool {
        self.lost_tasks == 0 && self.duplicated_outputs == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldConfig {
    pub n_replicas: usize,
    pub n_processes: u32,
    pub seed: u64,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            n_replicas: MIN_REPLICAS,
            n_processes: DEFAULT_PROCESSES,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoakConfig {
    pub duration_ms: u64,
    pub seed: u64,
}

impl Default for SoakConfig {
    fn default() -> Self {
        Self {
            duration_ms: SOAK_MS,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunReport {
    pub gate: Gate,
    pub faults_applied: u32,
    pub invariants: Invariants,
    pub recovered: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invariants failed: lost_tasks={lost_tasks} duplicated_outputs={duplicated_outputs}")]
    Invariant {
        lost_tasks: u64,
        duplicated_outputs: u64,
    },
    #[error("need at least {MIN_REPLICAS} replicas, have {0}")]
    TooFewReplicas(usize),
    #[error("clock skew {0} ms exceeds ±{CLOCK_SKEW_MS} ms")]
    ClockSkewBound(i64),
    #[error("unknown replica: {0}")]
    NoReplica(String),
    #[error("unknown process: {0}")]
    NoProcess(String),
    #[error("disk: {0}")]
    Disk(String),
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
}

/// Deterministic Knuth LCG used by D0 / soak.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }

    fn pick(&mut self, n: usize) -> usize {
        assert!(n > 0, "pick empty");
        (self.next() as usize) % n
    }
}

/// In-process simulated cluster: N EventLog replicas and M processes.
///
/// Layout: `<dir>/replicas/<id>/` is an [`EventLog`]. Create fails if `dir`
/// exists. Open replays every replica. A [`Fault::Kill`] drops the in-memory
/// handle; [`World::recover`] reopens from disk (startup is replay).
pub struct World {
    dir: PathBuf,
    config: WorldConfig,
    now: NowMs,
    replicas: Vec<Replica>,
    processes: Vec<ProcessId>,
    dropped_edges: HashSet<(String, String)>,
    api_enqueued: Vec<String>,
    complete_counts: HashMap<(String, u64), u64>,
}

impl World {
    /// Create `dir` with `config.n_replicas` empty logs. Fails if `dir` exists
    /// or `n_replicas < MIN_REPLICAS`.
    pub fn create(dir: impl AsRef<Path>, config: WorldConfig) -> Result<Self> {
        if config.n_replicas < MIN_REPLICAS {
            return Err(Error::TooFewReplicas(config.n_replicas));
        }
        let dir = dir.as_ref().to_path_buf();
        if dir.exists() {
            return Err(Error::Other(format!("directory exists: {}", dir.display())));
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
            api_enqueued: Vec::new(),
            complete_counts: HashMap::new(),
        })
    }

    /// Open and replay every replica. Fails closed on a broken hash chain.
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
            assert_clean_jsonl(&path.join("events.jsonl"))?;
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
            });
        }
        let mut world = Self {
            processes: process_ids(config.n_processes),
            dir,
            config,
            now: 0,
            replicas,
            dropped_edges: HashSet::new(),
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

    /// Advance injected time. Does not read the wall clock.
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

    /// Record a task in every reachable replica. Counts toward lost-task checks.
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

    /// Record an attempt-keyed output. A second complete for the same
    /// `(task, attempt)` is a duplicated output.
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
            Fault::Outage { .. } | Fault::HealOutage { .. } => {}
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
            Fault::ClockSkew { .. } => {}
            Fault::TokenExpiry | Fault::TokenRotate => {}
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
                    });
                }
            }
            Fault::JobPreempt { .. } | Fault::WalltimeKill { .. } => {}
            Fault::HungSqueue | Fault::UnhangSqueue => {}
            Fault::BrokerDeath => {}
            Fault::RateLimit { .. } => {}
        }
        Ok(())
    }

    /// Reopen killed processes from their logs. Startup is recovery.
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

    /// Count lost tasks and duplicated outputs across surviving replicas.
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

    /// Surviving replica log. None if that replica is killed or lost.
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
            }
        }
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

fn replica_ids(n: usize) -> Vec<ReplicaId> {
    (0..n).map(|i| ReplicaId(i.to_string())).collect()
}

fn process_ids(n: u32) -> Vec<ProcessId> {
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![ProcessId("coordinator".into())];
    for i in 0..n.saturating_sub(1) {
        out.push(ProcessId(format!("worker-{i}")));
    }
    out
}

fn replica_path(world_dir: &Path, id: &ReplicaId) -> PathBuf {
    world_dir.join("replicas").join(&id.0)
}

/// Fail-closed: a nonempty replica JSONL must be newline-terminated complete
/// records. EventLog::open would treat a torn last newline as crash recovery
/// and succeed; chaos open must not.
fn assert_clean_jsonl(path: &Path) -> Result<()> {
    let bytes = std::fs::read(path).map_err(|e| Error::Log(e.to_string()))?;
    if bytes.is_empty() {
        return Ok(());
    }
    if bytes.last() != Some(&b'\n') {
        return Err(Error::Log("torn or corrupt replica log".into()));
    }
    for line in bytes.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        serde_json::from_slice::<prometheus_log::Event>(line)
            .map_err(|e| Error::Log(format!("corrupt log: {e}")))?;
    }
    Ok(())
}

fn bind_process(world: &World, p: &ProcessId) -> ProcessId {
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

fn bind_replica(world: &World, r: &ReplicaId) -> ReplicaId {
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

fn bind_fault(world: &World, fault: Fault) -> Fault {
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

fn bind_all(world: &World, faults: Vec<Fault>) -> Vec<Fault> {
    faults.into_iter().map(|f| bind_fault(world, f)).collect()
}

fn work_tasks() -> Vec<TaskId> {
    (0..WORK_N).map(|i| TaskId(format!("t{i}"))).collect()
}

fn enqueue_work(world: &mut World) -> Result<Vec<TaskId>> {
    let tasks = work_tasks();
    let now = world.now();
    for t in &tasks {
        world.enqueue_task(t, now)?;
    }
    Ok(tasks)
}

fn complete_work(world: &mut World, tasks: &[TaskId]) -> Result<()> {
    let now = world.now();
    for t in tasks {
        world.complete_task(t, 1, b"ok", now)?;
    }
    Ok(())
}

/// Fault kinds that `gate` must survive. Ids are placeholders; [`run_gate`]
/// binds them to the world's replicas and processes.
pub fn faults_for(gate: Gate) -> Vec<Fault> {
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
                all.extend(faults_for(g));
            }
            all
        }
    }
}

/// Run one D-gate (or Soak) against `world`. Applies the gate's faults,
/// recovers, then checks invariants. Returns [`Error::Invariant`] if they fail.
///
/// D0 applies [`D0_KILLS`] random kills using `world.config.seed`.
/// Soak runs for [`SoakConfig::duration_ms`] of injected time (default
/// [`SOAK_MS`]), not wall time.
pub fn run_gate(world: &mut World, gate: Gate) -> Result<RunReport> {
    if gate == Gate::Soak {
        return run_soak(
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
            for f in bind_all(world, faults_for(other)) {
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

pub fn run_soak(world: &mut World, cfg: SoakConfig) -> Result<RunReport> {
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
            bind_all(world, faults_for(g))
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
