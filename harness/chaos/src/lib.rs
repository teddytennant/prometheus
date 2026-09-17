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

use prometheus_log::EventLog;
use serde::{Deserialize, Serialize};
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
    Kill { process: ProcessId },
    /// Drop messages from `from` to `to`. `asymmetric` means the reverse
    /// direction still delivers (D2).
    Partition {
        from: ReplicaId,
        to: ReplicaId,
        asymmetric: bool,
    },
    HealPartition { from: ReplicaId, to: ReplicaId },
    Outage { provider: String },
    HealOutage { provider: String },
    DiskFull { replica: ReplicaId },
    /// Corrupt one replica. The other replica recovers (spec 15.2).
    DiskCorrupt { replica: ReplicaId },
    /// Skew one replica's clock. `|delta_ms|` must be <= [`CLOCK_SKEW_MS`].
    ClockSkew { replica: ReplicaId, delta_ms: i64 },
    TokenExpiry,
    TokenRotate,
    NodeLoss { replica: ReplicaId },
    NodeReplace { old: ReplicaId, new: ReplicaId },
    JobPreempt { job: JobId },
    WalltimeKill { job: JobId },
    HungSqueue,
    UnhangSqueue,
    /// Token broker killed mid-refresh (D4).
    BrokerDeath,
    RateLimit { provider: String },
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

/// In-process simulated cluster: N EventLog replicas and M processes.
///
/// Layout: `<dir>/replicas/<id>/` is an [`EventLog`]. Create fails if `dir`
/// exists. Open replays every replica. A [`Fault::Kill`] drops the in-memory
/// handle; [`World::recover`] reopens from disk (startup is replay).
pub struct World {
    dir: PathBuf,
    config: WorldConfig,
    now: NowMs,
}

impl World {
    /// Create `dir` with `config.n_replicas` empty logs. Fails if `dir` exists
    /// or `n_replicas < MIN_REPLICAS`.
    pub fn create(_dir: impl AsRef<Path>, _config: WorldConfig) -> Result<Self> {
        unimplemented!("H11: World::create")
    }

    /// Open and replay every replica. Fails closed on a broken hash chain.
    pub fn open(_dir: impl AsRef<Path>, _config: WorldConfig) -> Result<Self> {
        unimplemented!("H11: World::open")
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
    pub fn advance(&mut self, _ms: u64) {
        unimplemented!("H11: World::advance")
    }

    pub fn replicas(&self) -> Vec<ReplicaId> {
        unimplemented!("H11: World::replicas")
    }

    pub fn processes(&self) -> Vec<ProcessId> {
        unimplemented!("H11: World::processes")
    }

    pub fn replica_dir(&self, _id: &ReplicaId) -> PathBuf {
        unimplemented!("H11: World::replica_dir")
    }

    /// Record a task in every reachable replica. Counts toward lost-task checks.
    pub fn enqueue_task(&mut self, _task: &TaskId, _now: NowMs) -> Result<()> {
        unimplemented!("H11: World::enqueue_task")
    }

    /// Record an attempt-keyed output. A second complete for the same
    /// `(task, attempt)` is a duplicated output.
    pub fn complete_task(
        &mut self,
        _task: &TaskId,
        _attempt: u64,
        _output: &[u8],
        _now: NowMs,
    ) -> Result<()> {
        unimplemented!("H11: World::complete_task")
    }

    pub fn inject(&mut self, _fault: &Fault) -> Result<()> {
        unimplemented!("H11: World::inject")
    }

    /// Reopen killed processes from their logs. Startup is recovery.
    pub fn recover(&mut self) -> Result<()> {
        unimplemented!("H11: World::recover")
    }

    /// Count lost tasks and duplicated outputs across surviving replicas.
    pub fn invariants(&self) -> Result<Invariants> {
        unimplemented!("H11: World::invariants")
    }

    pub fn log_len(&self, _id: &ReplicaId) -> Result<usize> {
        unimplemented!("H11: World::log_len")
    }

    /// Surviving replica log. None if that replica is killed or lost.
    pub fn replica_log(&self, _id: &ReplicaId) -> Result<&EventLog> {
        unimplemented!("H11: World::replica_log")
    }
}

/// Fault kinds that `gate` must survive. Ids are placeholders; [`run_gate`]
/// binds them to the world's replicas and processes.
pub fn faults_for(_gate: Gate) -> Vec<Fault> {
    unimplemented!("H11: faults_for")
}

/// Run one D-gate (or Soak) against `world`. Applies the gate's faults,
/// recovers, then checks invariants. Returns [`Error::Invariant`] if they fail.
///
/// D0 applies [`D0_KILLS`] random kills using `world.config.seed`.
/// Soak runs for [`SoakConfig::duration_ms`] of injected time (default
/// [`SOAK_MS`]), not wall time.
pub fn run_gate(_world: &mut World, _gate: Gate) -> Result<RunReport> {
    unimplemented!("H11: run_gate")
}

pub fn run_soak(_world: &mut World, _cfg: SoakConfig) -> Result<RunReport> {
    unimplemented!("H11: run_soak")
}
