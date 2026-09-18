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
use std::collections::{HashMap, HashSet, VecDeque};

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
pub fn staleness(trainer_version: PolicyVersion, batch_version: PolicyVersion) -> Result<u64> {
    if batch_version > trainer_version {
        return Err(CoordError::FromTheFuture);
    }
    Ok(trainer_version - batch_version)
}

fn target_rollout_gpus(total_gpus: u64, numer: u64, denom: u64) -> u64 {
    ((total_gpus as u128) * (numer as u128) / (denom as u128)) as u64
}

struct Rack {
    gpus: u64,
    role: RackRole,
    /// `Some` iff `role == Rollout`.
    version: Option<PolicyVersion>,
}

/// Async RL coordinator. CPU tests drive a handful of simulated racks.
pub struct Coordinator {
    config: CoordinatorConfig,
    /// Lexicographically sorted rack ids.
    order: Vec<RackId>,
    racks: HashMap<RackId, Rack>,
    trainer_version: PolicyVersion,
    halted: bool,
    queue: VecDeque<RolloutBatch>,
    total_gpus: u64,
}

impl Coordinator {
    /// Assigns racks toward the 65/35 GPU split. Requires at least
    /// `min_rollout_racks + min_trainer_racks` racks, each with `gpus > 0`.
    pub fn new(config: CoordinatorConfig, racks: Vec<RackSpec>) -> Result<Self> {
        if racks.is_empty() {
            return Err(CoordError::EmptyFleet);
        }
        let mut seen: HashSet<RackId> = HashSet::new();
        for r in &racks {
            if !seen.insert(r.id.clone()) {
                return Err(CoordError::DuplicateRack(r.id.clone()));
            }
        }
        for r in &racks {
            if r.gpus == 0 {
                return Err(CoordError::Message(format!(
                    "rack {:?} has 0 gpus; each rack must have gpus > 0",
                    r.id
                )));
            }
        }
        if config.rollout_denom == 0 || config.rollout_numer > config.rollout_denom {
            return Err(CoordError::BadSplit);
        }
        if racks.len()
            < config
                .min_rollout_racks
                .saturating_add(config.min_trainer_racks)
        {
            return Err(CoordError::NotEnoughRacks);
        }

        let mut sorted = racks;
        sorted.sort_by(|a, b| a.id.0.cmp(&b.id.0));
        let n = sorted.len();
        let mut total_gpus: u64 = 0;
        for r in &sorted {
            total_gpus = total_gpus.saturating_add(r.gpus);
        }
        let target = target_rollout_gpus(total_gpus, config.rollout_numer, config.rollout_denom);

        // Longest lex prefix with sum <= target and leftover racks for min_trainer.
        let mut n_rollout = 0usize;
        let mut running: u128 = 0;
        for (i, r) in sorted.iter().enumerate() {
            let remaining_after = n - i - 1;
            let next = running + r.gpus as u128;
            if next <= target as u128 && remaining_after >= config.min_trainer_racks {
                running = next;
                n_rollout += 1;
            } else {
                break;
            }
        }
        // Honor min_rollout even if it overshoots the GPU target.
        while n_rollout < config.min_rollout_racks {
            let n_trainer = n - n_rollout;
            if n_trainer <= config.min_trainer_racks {
                break;
            }
            n_rollout += 1;
        }

        let mut map = HashMap::new();
        let mut order = Vec::with_capacity(n);
        for (i, spec) in sorted.into_iter().enumerate() {
            let role = if i < n_rollout {
                RackRole::Rollout
            } else {
                RackRole::Trainer
            };
            let version = match role {
                RackRole::Rollout => Some(0),
                RackRole::Trainer => None,
            };
            order.push(spec.id.clone());
            map.insert(
                spec.id,
                Rack {
                    gpus: spec.gpus,
                    role,
                    version,
                },
            );
        }

        Ok(Self {
            config,
            order,
            racks: map,
            trainer_version: 0,
            halted: false,
            queue: VecDeque::new(),
            total_gpus,
        })
    }

    pub fn state(&self) -> Result<CoordinatorState> {
        if self.halted {
            Ok(CoordinatorState::Halted)
        } else {
            Ok(CoordinatorState::Running)
        }
    }

    pub fn halted(&self) -> Result<bool> {
        Ok(self.halted)
    }

    pub fn trainer_version(&self) -> Result<PolicyVersion> {
        Ok(self.trainer_version)
    }

    /// Last version pushed onto this rollout rack. Trainer racks error
    /// [`CoordError::NotRollout`].
    pub fn rack_version(&self, id: &RackId) -> Result<PolicyVersion> {
        let r = self.lookup(id)?;
        match r.version {
            Some(v) => Ok(v),
            None => Err(CoordError::NotRollout(id.clone())),
        }
    }

    pub fn rack_role(&self, id: &RackId) -> Result<RackRole> {
        Ok(self.lookup(id)?.role)
    }

    /// `(rollout_gpus, trainer_gpus)`.
    pub fn split_gpus(&self) -> Result<(u64, u64)> {
        let ro = self.gpu_sum(RackRole::Rollout);
        let tr = self.gpu_sum(RackRole::Trainer);
        Ok((ro, tr))
    }

    pub fn n_rollout_racks(&self) -> Result<usize> {
        Ok(self.n_role(RackRole::Rollout))
    }

    pub fn n_trainer_racks(&self) -> Result<usize> {
        Ok(self.n_role(RackRole::Trainer))
    }

    pub fn queue_depth(&self) -> Result<usize> {
        Ok(self.queue.len())
    }

    /// Move at most one rack. Queue empty prefers more rollout. Depth at or
    /// above `queue_high` (when `queue_high > 0`) prefers more trainer.
    /// Otherwise move toward `rollout_numer / rollout_denom`. Never breaks
    /// the min-rack floors. `None` if already at target or a floor blocks.
    pub fn rebalance(&mut self) -> Result<Option<(RackId, RackRole)>> {
        self.halt_guard()?;
        let depth = self.queue.len();
        // Rule 1: empty queue prefers rollout (GPU target ignored).
        if depth == 0 {
            if let Some(id) = self.move_trainer_to_rollout() {
                return Ok(Some((id, RackRole::Rollout)));
            }
        }
        // Rule 2: queue_high pressure prefers trainer. queue_high == 0 never
        // uses this rule.
        if self.config.queue_high > 0 && depth >= self.config.queue_high {
            if let Some(id) = self.move_rollout_to_trainer() {
                return Ok(Some((id, RackRole::Trainer)));
            }
        }
        let ro = self.gpu_sum(RackRole::Rollout);
        let target = self.target();
        // Rule 3.
        if ro < target {
            if let Some(id) = self.move_trainer_to_rollout() {
                return Ok(Some((id, RackRole::Rollout)));
            }
        } else if ro > target {
            // Rule 4.
            if let Some(id) = self.move_rollout_to_trainer() {
                return Ok(Some((id, RackRole::Trainer)));
            }
        }
        Ok(None)
    }

    /// Rollout engines produce a group. Duplicate ids error. Halted errors.
    pub fn enqueue_batch(&mut self, batch: RolloutBatch) -> Result<()> {
        self.halt_guard()?;
        if batch.n_samples == 0 {
            return Err(CoordError::EmptyGroup);
        }
        if batch.id.0.is_empty() {
            return Err(CoordError::Message("batch id must be non-empty".into()));
        }
        if batch.prompt_id.0.is_empty() {
            return Err(CoordError::Message("prompt id must be non-empty".into()));
        }
        if self.queue.iter().any(|b| b.id == batch.id) {
            return Err(CoordError::DuplicateBatch(batch.id));
        }
        self.queue.push_back(batch);
        Ok(())
    }

    /// Drop batches with staleness `> max_staleness`, then pop the oldest
    /// remaining. `Ok(None)` if the queue is empty after drops.
    pub fn consume_batch(&mut self) -> Result<Option<RolloutBatch>> {
        self.halt_guard()?;
        // Scan first so FromTheFuture leaves the queue unchanged.
        let mut drop_count = 0usize;
        for batch in &self.queue {
            match staleness(self.trainer_version, batch.policy_version) {
                Err(CoordError::FromTheFuture) => return Err(CoordError::FromTheFuture),
                Err(e) => return Err(e),
                Ok(s) if s > self.config.max_staleness => {
                    drop_count += 1;
                }
                Ok(_) => break,
            }
        }
        for _ in 0..drop_count {
            let _ = self.queue.pop_front();
        }
        Ok(self.queue.pop_front())
    }

    /// Trainer finished an update. `version` must be `trainer_version + 1`.
    pub fn publish_weights(&mut self, version: PolicyVersion) -> Result<()> {
        self.halt_guard()?;
        if version != self.trainer_version.saturating_add(1) {
            return Err(CoordError::NotNextVersion);
        }
        self.trainer_version = version;
        Ok(())
    }

    /// Instant CPU stand-in for the RDMA weight push. `version` must equal
    /// the published trainer version. Rollout racks only.
    pub fn sync_rack(&mut self, id: &RackId, version: PolicyVersion) -> Result<()> {
        self.halt_guard()?;
        let trainer = self.trainer_version;
        let r = self.lookup_mut(id)?;
        if r.role != RackRole::Rollout {
            return Err(CoordError::NotRollout(id.clone()));
        }
        if version != trainer {
            return Err(CoordError::Unpublished(version));
        }
        r.version = Some(version);
        Ok(())
    }

    /// Parity job after a weight update. Drift past `parity_threshold` HALTS.
    /// Non-finite log-probs error [`CoordError::InvalidLogprob`]. Empty
    /// `samples` is a pass (`max_abs_err = 0.0`).
    pub fn run_parity(
        &mut self,
        version: PolicyVersion,
        samples: &[ParitySample],
    ) -> Result<ParityReport> {
        self.halt_guard()?;
        if version != self.trainer_version {
            return Err(CoordError::Unpublished(version));
        }
        for s in samples {
            if !s.trainer_logp.is_finite() || !s.engine_logp.is_finite() {
                return Err(CoordError::InvalidLogprob);
            }
        }
        if samples.is_empty() {
            return Ok(ParityReport {
                policy_version: version,
                max_abs_err: 0.0,
                n_samples: 0,
            });
        }
        let mut max_abs_err = 0.0_f64;
        for s in samples {
            let err = (s.trainer_logp - s.engine_logp).abs();
            if err > max_abs_err {
                max_abs_err = err;
            }
        }
        if max_abs_err > self.config.parity_threshold {
            self.halted = true;
            return Err(CoordError::ParityHalt {
                version,
                max_abs_err,
                threshold: self.config.parity_threshold,
            });
        }
        Ok(ParityReport {
            policy_version: version,
            max_abs_err,
            n_samples: samples.len(),
        })
    }
}

impl Coordinator {
    fn halt_guard(&self) -> Result<()> {
        if self.halted {
            Err(CoordError::Halted)
        } else {
            Ok(())
        }
    }

    fn lookup(&self, id: &RackId) -> Result<&Rack> {
        self.racks
            .get(id)
            .ok_or_else(|| CoordError::RackNotFound(id.clone()))
    }

    fn lookup_mut(&mut self, id: &RackId) -> Result<&mut Rack> {
        self.racks
            .get_mut(id)
            .ok_or_else(|| CoordError::RackNotFound(id.clone()))
    }

    fn n_role(&self, role: RackRole) -> usize {
        self.racks.values().filter(|r| r.role == role).count()
    }

    fn gpu_sum(&self, role: RackRole) -> u64 {
        self.racks
            .values()
            .filter(|r| r.role == role)
            .map(|r| r.gpus)
            .fold(0u64, |a, b| a.saturating_add(b))
    }

    fn target(&self) -> u64 {
        target_rollout_gpus(
            self.total_gpus,
            self.config.rollout_numer,
            self.config.rollout_denom,
        )
    }

    fn lex_smallest(&self, role: RackRole) -> Option<RackId> {
        self.order
            .iter()
            .find(|id| self.racks.get(id).map(|r| r.role) == Some(role))
            .cloned()
    }

    fn move_trainer_to_rollout(&mut self) -> Option<RackId> {
        if self.n_role(RackRole::Trainer) <= self.config.min_trainer_racks {
            return None;
        }
        let id = self.lex_smallest(RackRole::Trainer)?;
        if let Some(r) = self.racks.get_mut(&id) {
            r.role = RackRole::Rollout;
            r.version = Some(0);
        }
        Some(id)
    }

    fn move_rollout_to_trainer(&mut self) -> Option<RackId> {
        if self.n_role(RackRole::Rollout) <= self.config.min_rollout_racks {
            return None;
        }
        let id = self.lex_smallest(RackRole::Rollout)?;
        if let Some(r) = self.racks.get_mut(&id) {
            r.role = RackRole::Trainer;
            r.version = None;
        }
        Some(id)
    }
}
