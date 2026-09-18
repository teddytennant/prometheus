//! Async rollout/trainer orchestration, weight sync, parity halt (spec 9.2, 15.5 I2).
//!
//! CPU analog of V7's system pieces: simulated racks, integer policy versions,
//! an in-memory batch queue, and injected log-probs. No GPUs, no JAX, no
//! SGLang, no RDMA. Weight-sync time at 7.4T is S3 (spec 15.6) and out of
//! scope. The GSPO/DAPO math lives in Python `rl.loss`; this crate does not
//! import it. Reward hacking controls live in `rl/rewards`. Elastic DP
//! membership lives in `prometheus-control` and is not a crate dependency.
//!
//! Spec 9.2, CPU contract:
//! - Rollout engines produce groups; the trainer consumes batches up to
//!   [`MAX_STALENESS`] policy versions stale. Older batches are dropped.
//! - After a trainer update, [`Coordinator::publish_weights`] bumps the
//!   version and [`Coordinator::sync_rack`] pushes it onto a rollout rack
//!   (instant here).
//! - A parity job runs every weight update. Log-prob drift past
//!   [`CoordinatorConfig::parity_threshold`] HALTS RL.
//! - ~65% of RL GPUs on rollouts, ~35% on the trainer, tunable online by
//!   watching queue depths ([`Coordinator::rebalance`]).
//! - A group shares one prompt ([`GROUP_SIZE`] is the spec size; enqueue
//!   does not require exactly that many samples).
//! - Routing ids are recorded on the batch so a trainer can force the same
//!   experts. This crate does not replay them.

use serde::{Deserialize, Serialize};

/// Optimizer / policy version. Monotonic. Starts at 0.
pub type PolicyVersion = u64;

/// Spec 9.2: trainer consumes batches up to k=4 policy versions stale.
pub const MAX_STALENESS: u64 = 4;

/// Spec 9.2: a GRPO group of 16 shares one prompt.
pub const GROUP_SIZE: u32 = 16;

/// Spec 9.2 default split: ~65% of RL GPUs on rollouts.
pub const DEFAULT_ROLLOUT_NUMER: u64 = 65;

/// Denominator for [`DEFAULT_ROLLOUT_NUMER`].
pub const DEFAULT_ROLLOUT_DENOM: u64 = 100;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RackId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BatchId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PromptId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RackRole {
    Rollout,
    Trainer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinatorState {
    Running,
    Halted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RackSpec {
    pub id: RackId,
    /// GPUs on this rack. Split math is by GPU count, not rack count.
    pub gpus: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    /// Spec 9.2 k<=4. Production is [`MAX_STALENESS`].
    pub max_staleness: u64,
    /// Target rollout share. Production is 65/100.
    pub rollout_numer: u64,
    pub rollout_denom: u64,
    /// Halt if max |trainer_logp - engine_logp| exceeds this.
    pub parity_threshold: f64,
    /// Queue depth at or above this moves a rack toward trainer.
    /// `0` means queue depth never triggers a trainer shift.
    pub queue_high: usize,
    /// Never drop below this many rollout racks while rebalancing.
    pub min_rollout_racks: usize,
    /// Never drop below this many trainer racks while rebalancing.
    pub min_trainer_racks: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutBatch {
    pub id: BatchId,
    /// Members of the group share this prompt (spec 9.2 prefix sharing).
    pub prompt_id: PromptId,
    pub policy_version: PolicyVersion,
    pub n_samples: u32,
    /// SGLang recorded expert ids (9.2 routing replay). Pass-through only.
    pub routing_present: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParitySample {
    pub trainer_logp: f64,
    pub engine_logp: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityReport {
    pub policy_version: PolicyVersion,
    pub max_abs_err: f64,
    pub n_samples: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum CoordError {
    #[error("RL halted after parity drift")]
    Halted,
    #[error("parity drift version={version} max_abs_err={max_abs_err} threshold={threshold}")]
    ParityHalt {
        version: PolicyVersion,
        max_abs_err: f64,
        threshold: f64,
    },
    #[error("rack not found: {0:?}")]
    RackNotFound(RackId),
    #[error("batch not found: {0:?}")]
    BatchNotFound(BatchId),
    #[error("duplicate batch id: {0:?}")]
    DuplicateBatch(BatchId),
    #[error("duplicate rack id: {0:?}")]
    DuplicateRack(RackId),
    #[error("no racks in the fleet")]
    EmptyFleet,
    #[error("not enough racks for min_rollout_racks + min_trainer_racks")]
    NotEnoughRacks,
    #[error("rollout_denom must be greater than 0")]
    BadSplit,
    #[error("rack is not a rollout rack: {0:?}")]
    NotRollout(RackId),
    #[error("rack is not a trainer rack: {0:?}")]
    NotTrainer(RackId),
    #[error("policy version {0} was not published")]
    Unpublished(PolicyVersion),
    #[error("policy version must be trainer_version + 1")]
    NotNextVersion,
    #[error("batch policy version is ahead of the trainer")]
    FromTheFuture,
    #[error("invalid log-prob")]
    InvalidLogprob,
    #[error("n_samples must be greater than 0")]
    EmptyGroup,
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, CoordError>;

/// `trainer_version - batch_version`. Errors if the batch is from the future.
pub fn staleness(_trainer_version: PolicyVersion, _batch_version: PolicyVersion) -> Result<u64> {
    unimplemented!("I2: staleness")
}

/// Async RL coordinator. CPU tests drive a handful of simulated racks.
pub struct Coordinator {
    _private: (),
}

impl Coordinator {
    /// Assigns racks toward the 65/35 GPU split. Requires at least
    /// `min_rollout_racks + min_trainer_racks` racks, each with `gpus > 0`.
    pub fn new(_config: CoordinatorConfig, _racks: Vec<RackSpec>) -> Result<Self> {
        unimplemented!("I2: Coordinator::new")
    }

    pub fn state(&self) -> Result<CoordinatorState> {
        unimplemented!("I2: Coordinator::state")
    }

    pub fn halted(&self) -> Result<bool> {
        unimplemented!("I2: Coordinator::halted")
    }

    pub fn trainer_version(&self) -> Result<PolicyVersion> {
        unimplemented!("I2: Coordinator::trainer_version")
    }

    /// Last version pushed onto this rollout rack. Trainer racks error
    /// [`CoordError::NotRollout`].
    pub fn rack_version(&self, _id: &RackId) -> Result<PolicyVersion> {
        unimplemented!("I2: Coordinator::rack_version")
    }

    pub fn rack_role(&self, _id: &RackId) -> Result<RackRole> {
        unimplemented!("I2: Coordinator::rack_role")
    }

    /// `(rollout_gpus, trainer_gpus)`.
    pub fn split_gpus(&self) -> Result<(u64, u64)> {
        unimplemented!("I2: Coordinator::split_gpus")
    }

    pub fn n_rollout_racks(&self) -> Result<usize> {
        unimplemented!("I2: Coordinator::n_rollout_racks")
    }

    pub fn n_trainer_racks(&self) -> Result<usize> {
        unimplemented!("I2: Coordinator::n_trainer_racks")
    }

    pub fn queue_depth(&self) -> Result<usize> {
        unimplemented!("I2: Coordinator::queue_depth")
    }

    /// Move at most one rack. Queue empty prefers more rollout. Depth at or
    /// above `queue_high` (when `queue_high > 0`) prefers more trainer.
    /// Otherwise move toward `rollout_numer / rollout_denom`. Never breaks
    /// the min-rack floors. `None` if already at target or a floor blocks.
    pub fn rebalance(&mut self) -> Result<Option<(RackId, RackRole)>> {
        unimplemented!("I2: Coordinator::rebalance")
    }

    /// Rollout engines produce a group. Duplicate ids error. Halted errors.
    pub fn enqueue_batch(&mut self, _batch: RolloutBatch) -> Result<()> {
        unimplemented!("I2: Coordinator::enqueue_batch")
    }

    /// Drop batches with staleness `> max_staleness`, then pop the oldest
    /// remaining. `Ok(None)` if the queue is empty after drops.
    pub fn consume_batch(&mut self) -> Result<Option<RolloutBatch>> {
        unimplemented!("I2: Coordinator::consume_batch")
    }

    /// Trainer finished an update. `version` must be `trainer_version + 1`.
    pub fn publish_weights(&mut self, _version: PolicyVersion) -> Result<()> {
        unimplemented!("I2: Coordinator::publish_weights")
    }

    /// Instant CPU stand-in for the RDMA weight push. `version` must equal
    /// the published trainer version. Rollout racks only.
    pub fn sync_rack(&mut self, _id: &RackId, _version: PolicyVersion) -> Result<()> {
        unimplemented!("I2: Coordinator::sync_rack")
    }

    /// Parity job after a weight update. Drift past `parity_threshold` HALTS.
    /// Non-finite log-probs error [`CoordError::InvalidLogprob`]. Empty
    /// `samples` is a pass (`max_abs_err = 0.0`).
    pub fn run_parity(
        &mut self,
        _version: PolicyVersion,
        _samples: &[ParitySample],
    ) -> Result<ParityReport> {
        unimplemented!("I2: Coordinator::run_parity")
    }
}
