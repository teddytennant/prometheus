//! Independent I2 coordinator reference: slow, obvious CPU state machine.
//!
//! Production (`prometheus_coordinator`) must never import this module. Tests
//! drive `Coordinator` and `RefCoordinator` on every behavior assertion.
//!
//! There is no tensor math in this crate (no NumPy/PyTorch). The "reference"
//! is this state machine. Integer GPU splits use floor division; parity uses
//! a straightforward max-|diff| loop.
//!
//! # Locked rules (implementers must match; also see `tests/common`)
//!
//! ## Construction
//! - Empty `racks`: `EmptyFleet`.
//! - Duplicate `RackId` (first repeat in input order): `DuplicateRack`.
//! - Any rack with `gpus == 0` (first in input order): `Message` containing
//!   `"gpu"` (case-insensitive).
//! - `rollout_denom == 0` or `rollout_numer > rollout_denom`: `BadSplit`.
//! - `racks.len() < min_rollout_racks + min_trainer_racks`: `NotEnoughRacks`.
//! - Error order is the list above. `max_staleness` may be 0.
//!
//! ## Initial assignment (deterministic)
//! - Sort racks by `id.0` lexicographically (byte-wise `String` order).
//! - `total_gpus` = sum of `gpus`.
//! - `target_rollout = total_gpus * rollout_numer / rollout_denom` (u128
//!   multiply then divide; integer floor).
//! - Walk the sorted list, assigning `Rollout` while the running rollout GPU
//!   sum stays `<= target_rollout` **and** the remaining *unassigned after
//!   this rack* are enough to satisfy `min_trainer_racks`. Rest are `Trainer`.
//!   This yields a lex prefix of rollout racks.
//! - If that prefix is shorter than `min_rollout_racks`, convert the next
//!   lex racks from Trainer → Rollout until the floor holds, even if that
//!   overshoots the GPU target. Construction guarantees this cannot break
//!   `min_trainer_racks`.
//! - After `new`: `trainer_version == 0`. Every Rollout rack's
//!   `rack_version == 0`. Trainer racks: `rack_version` is `NotRollout`.
//! - `state == Running`, `halted == false`, `queue_depth == 0`.
//!
//! ## `staleness(trainer, batch)`
//! - `batch > trainer` → `FromTheFuture`.
//! - Else `Ok(trainer - batch)`.
//!
//! ## Enqueue
//! - Halted → `Halted`.
//! - `n_samples == 0` → `EmptyGroup` (even if ids are empty).
//! - Empty `BatchId` → `Message` containing `"id"` (case-insensitive).
//! - Empty `PromptId` → `Message` containing `"prompt"` (case-insensitive).
//! - Duplicate batch id still in the queue → `DuplicateBatch`.
//! - Ids of already-consumed (or stale-dropped) batches MAY be reused.
//! - FIFO. `routing_present` and `prompt_id` stored and returned unchanged.
//! - Do not require `n_samples == GROUP_SIZE`.
//!
//! ## Consume
//! - Halted → `Halted`.
//! - Scan from the front. A batch with `policy_version > trainer_version` is
//!   a bug: return `FromTheFuture` and leave the queue **unchanged** (do not
//!   drop a stale prefix if a future batch is encountered during the scan).
//! - Otherwise drop a stale *prefix* while
//!   `staleness(trainer, batch) > max_staleness`. Do **not** drop stale
//!   items behind a fresh one. `[stale, fresh, stale]` returns `fresh` and
//!   leaves the last stale. If every remaining batch is stale, drop them
//!   all and return `Ok(None)`.
//! - `k == max_staleness` is kept; `k == max_staleness + 1` is dropped.
//! - `Ok(None)` if empty after drops.
//!
//! ## Publish
//! - Halted → `Halted`.
//! - `version` must be exactly `trainer_version + 1` else `NotNextVersion`.
//! - First publish from 0 is 1. Version-0 batches are valid before any
//!   publish. Publish does **not** update per-rack versions (that is
//!   `sync_rack`).
//!
//! ## Sync
//! - Halted → `Halted`.
//! - Unknown rack → `RackNotFound`. Trainer rack → `NotRollout`.
//! - `version` must equal current `trainer_version` else `Unpublished`
//!   (payload is the requested version). Instant. Sets `rack_version`.
//! - Sync of version 0 while `trainer_version == 0` is allowed.
//!
//! ## Rebalance (at most one rack per call)
//! - Halted → `Halted`.
//! - Preference order:
//!   1. If `queue_depth == 0` and a trainer rack can move to rollout without
//!      breaking `min_trainer_racks`, move the lex-smallest such trainer
//!      rack to Rollout. Its `rack_version` becomes 0.
//!   2. Else if `queue_high > 0` and `queue_depth >= queue_high` and a
//!      rollout rack can move to trainer without breaking `min_rollout_racks`,
//!      move the lex-smallest such rollout rack to Trainer.
//!   3. Else if rollout GPU sum < `target_rollout`, same as (1).
//!   4. Else if rollout GPU sum > `target_rollout`, same as (2).
//!   5. Else `Ok(None)`.
//! - Rule 1 does **not** consult the GPU target: an empty queue prefers
//!   more rollout even when already at/over target, until `min_trainer`.
//! - `queue_high == 0` never uses rule 2 (even if the queue is deep).
//! - Because rule 1 ignores the GPU target, an empty queue that is already
//!   over target can oscillate across calls once `min_trainer` starts
//!   blocking (rule 1 then rule 4). That is required, not a bug.
//! - Rollout → Trainer: subsequent `rack_version` is `NotRollout`.
//!   Trainer → Rollout: version resets to 0 until `sync_rack`.
//! - In-flight queue batches are untouched.
//!
//! ## Parity (`run_parity`)
//! - Halted → `Halted`.
//! - `version` must equal `trainer_version` else `Unpublished`.
//! - Any non-finite log-prob (NaN or ±inf on either side) →
//!   `InvalidLogprob`, do **not** halt.
//! - Empty samples → `Ok(ParityReport { policy_version, max_abs_err: 0.0,
//!   n_samples: 0 })`, do not halt.
//! - `max_abs_err = max_i |trainer_logp_i - engine_logp_i|`.
//! - If `max_abs_err > parity_threshold`: set halted, return
//!   `ParityHalt { version, max_abs_err, threshold }`. Equal to threshold
//!   does **not** halt.
//! - After `ParityHalt`, `state == Halted`, `halted == true`. Later mutators
//!   (`enqueue_batch`, `consume_batch`, `rebalance`, `publish_weights`,
//!   `sync_rack`, `run_parity`) return `Halted`. Read methods still work.
//!
//! Read methods never return `Halted`.
//! `split_gpus` → `(rollout_gpus, trainer_gpus)` summing to total.

#![allow(dead_code)]

use prometheus_coordinator::{
    CoordError, CoordinatorConfig, CoordinatorState, ParityReport, ParitySample, PolicyVersion,
    RackId, RackRole, RackSpec, Result, RolloutBatch,
};
use std::collections::{HashMap, HashSet, VecDeque};

/// Independent copy of the free function. Tests must also call the
/// production `prometheus_coordinator::staleness`.
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

/// Slow reference coordinator. Same public method names as `Coordinator`.
pub struct RefCoordinator {
    config: CoordinatorConfig,
    /// Lexicographically sorted rack ids.
    order: Vec<RackId>,
    racks: HashMap<RackId, Rack>,
    trainer_version: PolicyVersion,
    halted: bool,
    queue: VecDeque<RolloutBatch>,
    total_gpus: u64,
}

impl RefCoordinator {
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

        // First pass: longest lex prefix with sum <= target and enough
        // leftover racks for min_trainer.
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
        // Second pass: honor min_rollout even if it overshoots the GPU target.
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

    pub fn ids(&self) -> Vec<RackId> {
        self.order.clone()
    }

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

    pub fn publish_weights(&mut self, version: PolicyVersion) -> Result<()> {
        self.halt_guard()?;
        if version != self.trainer_version.saturating_add(1) {
            return Err(CoordError::NotNextVersion);
        }
        self.trainer_version = version;
        Ok(())
    }

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
