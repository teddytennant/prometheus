//! Elastic scheduler, health, stragglers, SDC, spike rollback (spec 5.5, 15.5 A6).
//!
//! CPU analog of V4: simulated DP replicas, no GPUs, no wall clock (`NowMs` is
//! injected). Elastic control at 220k GPUs, 90 TB checkpoints, EP=72, and real
//! IB weight copies are S3 (spec 15.6) and out of scope.
//!
//! A failed replica drops out. Survivors keep going. [`Controller::grad_accumulation`]
//! rises so [`ElasticConfig::tokens_per_step`] stays held. A spare heals in and
//! rejoins at the next step boundary.
//!
//! Silent Data Corruption: replicas report per-shard hashes every
//! [`ElasticConfig::sdc_period_steps`]. A mismatch quarantines the rack and
//! treats it as a drop.
//!
//! Stragglers (step duration above [`ElasticConfig::straggler_timeout_ms`]) drain
//! the same way as dead replicas. Health events (Xid, ECC, NVLink, NIC flap,
//! thermal, collective watchdog) feed the scheduler before a hang.
//!
//! Loss-spike *execution* lives here. Detection lives in `obs/` (I10). On
//! [`Controller::on_loss_spike`] the caller gets the in-memory checkpoint id to
//! restore (via `prometheus-ckpt`, not this crate), the data shard to skip, and
//! whether to page. Page on the second spike within [`PAGE_WINDOW_STEPS`].
//!
//! Membership here is the DP replica set, not H4 Raft voters.

use serde::{Deserialize, Serialize};

/// Injected clock. Milliseconds since an arbitrary origin. Never wall time.
pub type NowMs = u64;

/// Optimizer step index.
pub type Step = u64;

/// Spec 5.5: page a human on the second spike within this many steps.
pub const PAGE_WINDOW_STEPS: u64 = 10_000;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplicaId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RackId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RankId(pub String);

/// Data-shard id (loader / skip list), not a weight-shard name.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShardId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplicaState {
    Live,
    Draining,
    Dead,
    Quarantined,
    Spare,
    Healing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthKind {
    Xid,
    Ecc,
    Nvlink,
    NicFlap,
    Thermal,
    WatchdogTimeout,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthEvent {
    pub replica: ReplicaId,
    pub rank: RankId,
    pub kind: HealthKind,
    pub now_ms: NowMs,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicaSpec {
    pub id: ReplicaId,
    pub rack: RackId,
    /// `true`: spare rack, not in the live DP set until [`Controller::heal_spare`].
    pub spare: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElasticConfig {
    /// Target tokens per optimizer step. Held fixed when a replica drops by
    /// raising [`Controller::grad_accumulation`].
    pub tokens_per_step: u64,
    /// Tokens one live replica contributes per microbatch.
    pub microbatch_tokens: u64,
    /// Hash-and-compare weight shards every N steps (spec 5.5 SDC).
    pub sdc_period_steps: u64,
    /// A rank slower than this (injected ms) is a straggler and is drained.
    pub straggler_timeout_ms: u64,
    /// Every collective must finish within this injected-clock window.
    pub collective_watchdog_ms: u64,
    /// Spec 5.5 page window. Production is [`PAGE_WINDOW_STEPS`].
    pub page_window_steps: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpikeReport {
    /// Last in-memory checkpoint id the caller passed in. Restore via ckpt.
    pub rollback_checkpoint_id: String,
    pub skipped_shard: ShardId,
    /// True on the second spike inside [`ElasticConfig::page_window_steps`].
    pub page: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ControlError {
    #[error("replica not found: {0:?}")]
    ReplicaNotFound(ReplicaId),
    #[error("no live replicas remain")]
    NoLiveReplicas,
    #[error("replica is not a spare: {0:?}")]
    NotSpare(ReplicaId),
    #[error("rejoin is only allowed at a step boundary")]
    NotStepBoundary,
    #[error("SDC hash mismatch replica={replica:?} shard={shard} expected={expected} got={got}")]
    HashMismatch {
        replica: ReplicaId,
        shard: String,
        expected: String,
        got: String,
    },
    #[error("replica quarantined: {0:?}")]
    Quarantined(ReplicaId),
    #[error("replica is not live: {0:?}")]
    NotLive(ReplicaId),
    #[error("missing SDC hash replica={replica:?} shard={shard}")]
    MissingHash { replica: ReplicaId, shard: String },
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, ControlError>;

/// Elastic DP controller. CPU tests drive a handful of simulated replicas.
pub struct Controller {
    _private: (),
}

impl Controller {
    pub fn new(_config: ElasticConfig, _replicas: Vec<ReplicaSpec>) -> Result<Self> {
        unimplemented!("A6: Controller::new")
    }

    pub fn live_replicas(&self) -> Result<Vec<ReplicaId>> {
        unimplemented!("A6: Controller::live_replicas")
    }

    pub fn replica_state(&self, _id: &ReplicaId) -> Result<ReplicaState> {
        unimplemented!("A6: Controller::replica_state")
    }

    pub fn n_live(&self) -> Result<usize> {
        unimplemented!("A6: Controller::n_live")
    }

    /// Microbatches per optimizer step. Rises when a replica drops so
    /// `n_live * microbatch_tokens * grad_accumulation >= tokens_per_step`.
    pub fn grad_accumulation(&self) -> Result<u64> {
        unimplemented!("A6: Controller::grad_accumulation")
    }

    /// `n_live * microbatch_tokens * grad_accumulation`.
    pub fn effective_tokens_per_step(&self) -> Result<u64> {
        unimplemented!("A6: Controller::effective_tokens_per_step")
    }

    /// Drop a live replica immediately. Raises grad accumulation. Errors if
    /// this would leave zero live replicas.
    pub fn fail_replica(&mut self, _id: &ReplicaId) -> Result<()> {
        unimplemented!("A6: Controller::fail_replica")
    }

    /// Record one rank's step duration. Above `straggler_timeout_ms` drains
    /// the replica the same way as [`Self::fail_replica`].
    pub fn observe_step_time(
        &mut self,
        _id: &ReplicaId,
        _step: Step,
        _duration_ms: u64,
    ) -> Result<()> {
        unimplemented!("A6: Controller::observe_step_time")
    }

    /// Start copying weights onto a spare from a live source. Completes at
    /// [`Self::rejoin`].
    pub fn heal_spare(&mut self, _spare: &ReplicaId, _source: &ReplicaId) -> Result<()> {
        unimplemented!("A6: Controller::heal_spare")
    }

    /// Finish a heal at a step boundary. The spare becomes live and
    /// accumulation falls to hold `tokens_per_step` without dropping below it.
    pub fn rejoin(&mut self, _id: &ReplicaId, _step: Step) -> Result<()> {
        unimplemented!("A6: Controller::rejoin")
    }

    /// Report a weight-shard content hash from one replica at `step`.
    pub fn report_shard_hash(
        &mut self,
        _replica: &ReplicaId,
        _shard: &str,
        _hex: &str,
        _step: Step,
    ) -> Result<()> {
        unimplemented!("A6: Controller::report_shard_hash")
    }

    /// Compare hashes reported at `step`. A mismatch quarantines the offending
    /// replica's rack (treated as a drop). No-op when `step` is not a multiple
    /// of `sdc_period_steps` (including period 0 meaning "never").
    pub fn check_sdc(&mut self, _step: Step) -> Result<()> {
        unimplemented!("A6: Controller::check_sdc")
    }

    /// Feed a device/fabric/thermal/watchdog event. Fatal kinds drain the replica.
    pub fn report_health(&mut self, _event: HealthEvent) -> Result<()> {
        unimplemented!("A6: Controller::report_health")
    }

    pub fn begin_collective(
        &mut self,
        _name: &str,
        _replica: &ReplicaId,
        _now_ms: NowMs,
    ) -> Result<()> {
        unimplemented!("A6: Controller::begin_collective")
    }

    pub fn end_collective(
        &mut self,
        _name: &str,
        _replica: &ReplicaId,
        _now_ms: NowMs,
    ) -> Result<()> {
        unimplemented!("A6: Controller::end_collective")
    }

    /// Advance the injected clock. Collectives past `collective_watchdog_ms`
    /// drain their replica with [`HealthKind::WatchdogTimeout`].
    pub fn tick(&mut self, _now_ms: NowMs) -> Result<()> {
        unimplemented!("A6: Controller::tick")
    }

    /// Execute the spec 5.5 spike policy. Does not talk to ckpt.
    pub fn on_loss_spike(
        &mut self,
        _step: Step,
        _shard: ShardId,
        _last_memory_checkpoint_id: String,
    ) -> Result<SpikeReport> {
        unimplemented!("A6: Controller::on_loss_spike")
    }

    pub fn skipped_shards(&self) -> Result<Vec<ShardId>> {
        unimplemented!("A6: Controller::skipped_shards")
    }

    /// How many times [`SpikeReport::page`] was true.
    pub fn pages(&self) -> Result<u64> {
        unimplemented!("A6: Controller::pages")
    }
}
