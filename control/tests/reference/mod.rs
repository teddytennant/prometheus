//! Independent A6 reference: slow, obvious elastic-DP membership machine.
//!
//! Production (`prometheus_control`) must never import this module. Tests
//! compare `Controller` membership, accumulation, SDC, health, collectives,
//! spike bookkeeping, and heal weight-copy against these functions. Integer
//! policy is written out as loops and ceil-division, not clever bit tricks.
//!
//! There is no tensor math in A6 (no NumPy/PyTorch). The "reference" is this
//! state machine plus [`min_grad_accumulation`].
//!
//! # Heal weight copy
//! CPU analog only (real IB is S3). `heal_spare` records the source and does
//! not copy bytes. `offer_weight_copy` appends an owned copy of the shards
//! into a staging vec. `rejoin` installs that vec (empty if nothing was
//! staged) and records `step` without comparing it to a clock. A later heal
//! replaces the installed copy only at that later rejoin, not at offer.
//! Membership errors of `heal_spare` / `rejoin` are unchanged.

#![allow(dead_code)]

use prometheus_control::{
    ControlError, ElasticConfig, HealthEvent, ReplicaId, ReplicaSpec, ReplicaState, Result,
    ShardId, SpikeReport, Step, WeightShard,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Smallest `u64` accum such that
/// `n_live * microbatch_tokens * accum >= tokens_per_step`.
///
/// `tokens_per_step == 0` yields `0`. Callers must not pass `n_live == 0`
/// or `microbatch_tokens == 0` (those are rejected at `Controller::new`).
pub fn min_grad_accumulation(n_live: u64, microbatch_tokens: u64, tokens_per_step: u64) -> u64 {
    if tokens_per_step == 0 {
        return 0;
    }
    if n_live == 0 || microbatch_tokens == 0 {
        return 0;
    }
    let per = n_live.saturating_mul(microbatch_tokens);
    if per == 0 {
        return 0;
    }
    let q = tokens_per_step / per;
    let r = tokens_per_step % per;
    if r == 0 {
        q
    } else {
        q + 1
    }
}

#[derive(Clone)]
struct Rep {
    rack: prometheus_control::RackId,
    state: ReplicaState,
}

struct OpenCol {
    name: String,
    replica: ReplicaId,
    begin_ms: u64,
}

/// Slow reference controller. Same public method names as `Controller`.
pub struct RefController {
    config: ElasticConfig,
    order: Vec<ReplicaId>,
    replicas: HashMap<ReplicaId, Rep>,
    /// `(step, replica, shard) -> hex`. BTreeMap so iteration is stable.
    hashes: BTreeMap<(Step, String, String), String>,
    open: Vec<OpenCol>,
    skipped: Vec<ShardId>,
    pages: u64,
    last_spike_step: Option<Step>,
    /// Source recorded by the in-progress heal. Present while state is Healing.
    heal_source: HashMap<ReplicaId, ReplicaId>,
    /// Shards staged by `offer_weight_copy`, not yet installed.
    staged: HashMap<ReplicaId, Vec<WeightShard>>,
    /// Copy installed by a successful rejoin. Absent means never completed a heal.
    installed: HashMap<ReplicaId, Vec<WeightShard>>,
    /// `step` argument of the rejoin that installed `installed`. Not a clock.
    installed_step: HashMap<ReplicaId, Step>,
}

impl RefController {
    pub fn new(config: ElasticConfig, replicas: Vec<ReplicaSpec>) -> Result<Self> {
        if config.microbatch_tokens == 0 {
            return Err(ControlError::Message(
                "microbatch_tokens must be greater than 0".into(),
            ));
        }
        let mut seen: HashSet<ReplicaId> = HashSet::new();
        let mut order = Vec::new();
        let mut map = HashMap::new();
        let mut n_live = 0usize;
        for spec in replicas {
            if !seen.insert(spec.id.clone()) {
                return Err(ControlError::Message(format!(
                    "duplicate replica id: {:?}",
                    spec.id
                )));
            }
            let state = if spec.spare {
                ReplicaState::Spare
            } else {
                ReplicaState::Live
            };
            if state == ReplicaState::Live {
                n_live += 1;
            }
            order.push(spec.id.clone());
            map.insert(
                spec.id,
                Rep {
                    rack: spec.rack,
                    state,
                },
            );
        }
        if n_live == 0 {
            return Err(ControlError::NoLiveReplicas);
        }
        Ok(Self {
            config,
            order,
            replicas: map,
            hashes: BTreeMap::new(),
            open: Vec::new(),
            skipped: Vec::new(),
            pages: 0,
            last_spike_step: None,
            heal_source: HashMap::new(),
            staged: HashMap::new(),
            installed: HashMap::new(),
            installed_step: HashMap::new(),
        })
    }

    pub fn ids(&self) -> Vec<ReplicaId> {
        self.order.clone()
    }

    pub fn live_replicas(&self) -> Result<Vec<ReplicaId>> {
        Ok(self
            .order
            .iter()
            .filter(|id| self.replicas.get(id).map(|r| r.state) == Some(ReplicaState::Live))
            .cloned()
            .collect())
    }

    pub fn replica_state(&self, id: &ReplicaId) -> Result<ReplicaState> {
        Ok(self.lookup(id)?.state)
    }

    pub fn n_live(&self) -> Result<usize> {
        Ok(self.n_live_count())
    }

    pub fn grad_accumulation(&self) -> Result<u64> {
        Ok(self.accum())
    }

    pub fn effective_tokens_per_step(&self) -> Result<u64> {
        let n = self.n_live_count() as u64;
        let a = self.accum();
        match n
            .checked_mul(self.config.microbatch_tokens)
            .and_then(|x| x.checked_mul(a))
        {
            Some(v) => Ok(v),
            None => Err(ControlError::Message(
                "effective_tokens_per_step overflow".into(),
            )),
        }
    }

    pub fn fail_replica(&mut self, id: &ReplicaId) -> Result<()> {
        self.require_live(id)?;
        self.drop_live(id)
    }

    pub fn observe_step_time(
        &mut self,
        id: &ReplicaId,
        _step: Step,
        duration_ms: u64,
    ) -> Result<()> {
        self.require_live(id)?;
        if duration_ms > self.config.straggler_timeout_ms {
            self.drop_live(id)
        } else {
            Ok(())
        }
    }

    pub fn heal_spare(&mut self, spare: &ReplicaId, source: &ReplicaId) -> Result<()> {
        let spare_state = self.lookup(spare)?.state;
        if spare_state != ReplicaState::Spare {
            return Err(ControlError::NotSpare(spare.clone()));
        }
        self.require_live(source)?;
        self.replicas.get_mut(spare).unwrap().state = ReplicaState::Healing;
        self.heal_source.insert(spare.clone(), source.clone());
        // Fresh staging area. A previous install (only reachable if this id
        // is Spare again) stays until the next successful rejoin.
        self.staged.insert(spare.clone(), Vec::new());
        Ok(())
    }

    pub fn rejoin(&mut self, id: &ReplicaId, step: Step) -> Result<()> {
        let state = self.lookup(id)?.state;
        if state != ReplicaState::Healing {
            return Err(ControlError::NotSpare(id.clone()));
        }
        // No staged shards (offer never called, or only empty slices) installs
        // an empty vec. That is Ok, not WeightCopyNotInstalled.
        let staged = self.staged.remove(id).unwrap_or_default();
        self.installed.insert(id.clone(), staged);
        self.installed_step.insert(id.clone(), step);
        self.heal_source.remove(id);
        self.replicas.get_mut(id).unwrap().state = ReplicaState::Live;
        Ok(())
    }

    /// Source recorded by the last successful `heal_spare` while `spare` is Healing.
    pub fn heal_source(&self, spare: &ReplicaId) -> Result<ReplicaId> {
        let state = self.lookup(spare)?.state;
        if state != ReplicaState::Healing {
            return Err(ControlError::NoHealInProgress(spare.clone()));
        }
        Ok(self
            .heal_source
            .get(spare)
            .cloned()
            .expect("Healing replica has a recorded source"))
    }

    /// Stage an owned copy of `shards`. See module docs for check order.
    ///
    /// Check order (first failure wins, and a failure stages nothing):
    /// 1. Spare unknown: `ReplicaNotFound`.
    /// 2. Spare not `Healing`: `NoHealInProgress` (before source or shard checks).
    /// 3. Source liveness, same function as `heal_spare`: unknown
    ///    `ReplicaNotFound`, `Quarantined`, otherwise `NotLive`. A non-live
    ///    source yields that error even if it also mismatches the recorded source.
    /// 4. Source != recorded source: `HealSourceMismatch { spare, expected, got }`.
    /// 5. Shards in order: empty name, then duplicate against already-staged
    ///    names and earlier names in this call.
    ///
    /// Empty `shards` still runs steps 1–4, then stages nothing. Bytes are
    /// cloned. Does not change replica state, `n_live`, or accumulation, and
    /// does not write the installed copy.
    pub fn offer_weight_copy(
        &mut self,
        spare: &ReplicaId,
        source: &ReplicaId,
        shards: &[WeightShard],
    ) -> Result<()> {
        let spare_state = self.lookup(spare)?.state;
        if spare_state != ReplicaState::Healing {
            return Err(ControlError::NoHealInProgress(spare.clone()));
        }
        self.require_live(source)?;
        let expected = self
            .heal_source
            .get(spare)
            .cloned()
            .expect("Healing replica has a recorded source");
        if source != &expected {
            return Err(ControlError::HealSourceMismatch {
                spare: spare.clone(),
                expected,
                got: source.clone(),
            });
        }
        let mut names: HashSet<String> = self
            .staged
            .get(spare)
            .map(|v| v.iter().map(|s| s.name.clone()).collect())
            .unwrap_or_default();
        let mut accepted = Vec::with_capacity(shards.len());
        for shard in shards {
            if shard.name.is_empty() {
                return Err(ControlError::EmptyWeightShardName);
            }
            if !names.insert(shard.name.clone()) {
                return Err(ControlError::DuplicateWeightShard(shard.name.clone()));
            }
            accepted.push(WeightShard {
                name: shard.name.clone(),
                bytes: shard.bytes.clone(),
            });
        }
        self.staged
            .entry(spare.clone())
            .or_default()
            .extend(accepted);
        Ok(())
    }

    /// Owned clone of the copy installed by the last successful rejoin of `id`.
    pub fn healed_weight_shards(&self, id: &ReplicaId) -> Result<Vec<WeightShard>> {
        self.lookup(id)?;
        match self.installed.get(id) {
            Some(shards) => Ok(shards.clone()),
            None => Err(ControlError::WeightCopyNotInstalled(id.clone())),
        }
    }

    /// Step recorded by the rejoin that installed `id`'s copy. Reference-only
    /// (production has no getter). `None` if `id` has never completed a heal.
    pub fn recorded_rejoin_step(&self, id: &ReplicaId) -> Option<Step> {
        self.installed_step.get(id).copied()
    }

    pub fn report_shard_hash(
        &mut self,
        replica: &ReplicaId,
        shard: &str,
        hex: &str,
        step: Step,
    ) -> Result<()> {
        self.require_live(replica)?;
        self.hashes.insert(
            (step, replica.0.clone(), shard.to_string()),
            hex.to_string(),
        );
        Ok(())
    }

    pub fn check_sdc(&mut self, step: Step) -> Result<()> {
        let period = self.config.sdc_period_steps;
        if period == 0 {
            return Ok(());
        }
        if !step.is_multiple_of(period) {
            return Ok(());
        }
        let live = self.live_replicas()?;
        if live.is_empty() {
            return Ok(());
        }
        let mut shards: BTreeSet<String> = BTreeSet::new();
        for id in &live {
            let prefix = (step, id.0.clone());
            for (s, rid, shard) in self.hashes.keys() {
                if *s == prefix.0 && *rid == prefix.1 {
                    shards.insert(shard.clone());
                }
            }
        }
        if shards.is_empty() {
            return Err(ControlError::MissingHash {
                replica: live[0].clone(),
                shard: String::new(),
            });
        }
        for shard in shards {
            let mut reports: Vec<(ReplicaId, String)> = Vec::new();
            for id in &live {
                match self.hashes.get(&(step, id.0.clone(), shard.clone())) {
                    Some(hex) => reports.push((id.clone(), hex.clone())),
                    None => {
                        return Err(ControlError::MissingHash {
                            replica: id.clone(),
                            shard,
                        });
                    }
                }
            }
            let n = reports.len();
            let mut counts: BTreeMap<String, usize> = BTreeMap::new();
            for (_, hex) in &reports {
                *counts.entry(hex.clone()).or_insert(0) += 1;
            }
            let majority = counts
                .iter()
                .find(|(_, c)| **c > n / 2)
                .map(|(h, _)| h.clone());
            let expected = majority.unwrap_or_else(|| reports[0].1.clone());
            for (id, hex) in &reports {
                if hex != &expected {
                    return self.quarantine_rack_of(id, shard, expected.clone(), hex.clone());
                }
            }
        }
        Ok(())
    }

    pub fn report_health(&mut self, event: HealthEvent) -> Result<()> {
        self.require_live(&event.replica)?;
        self.drop_live(&event.replica)
    }

    pub fn begin_collective(&mut self, name: &str, replica: &ReplicaId, now_ms: u64) -> Result<()> {
        self.require_live(replica)?;
        if self
            .open
            .iter()
            .any(|c| c.name == name && c.replica == *replica)
        {
            return Err(ControlError::Message(format!(
                "collective already open: {name}"
            )));
        }
        self.open.push(OpenCol {
            name: name.to_string(),
            replica: replica.clone(),
            begin_ms: now_ms,
        });
        Ok(())
    }

    pub fn end_collective(&mut self, name: &str, replica: &ReplicaId, _now_ms: u64) -> Result<()> {
        self.require_live(replica)?;
        self.open
            .retain(|c| !(c.name == name && c.replica == *replica));
        Ok(())
    }

    pub fn tick(&mut self, now_ms: u64) -> Result<()> {
        let watchdog = self.config.collective_watchdog_ms;
        let ids = self.order.clone();
        for id in ids {
            if self.replicas.get(&id).map(|r| r.state) != Some(ReplicaState::Live) {
                continue;
            }
            let overdue = self
                .open
                .iter()
                .any(|c| c.replica == id && now_ms.saturating_sub(c.begin_ms) > watchdog);
            if overdue {
                self.drop_live(&id)?;
            }
        }
        Ok(())
    }

    pub fn on_loss_spike(
        &mut self,
        step: Step,
        shard: ShardId,
        last_memory_checkpoint_id: String,
    ) -> Result<SpikeReport> {
        let page = match self.last_spike_step {
            Some(prev) => step.saturating_sub(prev) < self.config.page_window_steps,
            None => false,
        };
        if page {
            self.pages = self.pages.saturating_add(1);
        }
        self.last_spike_step = Some(step);
        self.skipped.push(shard.clone());
        Ok(SpikeReport {
            rollback_checkpoint_id: last_memory_checkpoint_id,
            skipped_shard: shard,
            page,
        })
    }

    pub fn skipped_shards(&self) -> Result<Vec<ShardId>> {
        Ok(self.skipped.clone())
    }

    pub fn pages(&self) -> Result<u64> {
        Ok(self.pages)
    }

    fn n_live_count(&self) -> usize {
        self.order
            .iter()
            .filter(|id| self.replicas.get(id).map(|r| r.state) == Some(ReplicaState::Live))
            .count()
    }

    fn accum(&self) -> u64 {
        min_grad_accumulation(
            self.n_live_count() as u64,
            self.config.microbatch_tokens,
            self.config.tokens_per_step,
        )
    }

    fn lookup(&self, id: &ReplicaId) -> Result<&Rep> {
        self.replicas
            .get(id)
            .ok_or_else(|| ControlError::ReplicaNotFound(id.clone()))
    }

    fn require_live(&self, id: &ReplicaId) -> Result<()> {
        match self.lookup(id)?.state {
            ReplicaState::Live => Ok(()),
            ReplicaState::Quarantined => Err(ControlError::Quarantined(id.clone())),
            _ => Err(ControlError::NotLive(id.clone())),
        }
    }

    fn drop_live(&mut self, id: &ReplicaId) -> Result<()> {
        if self.n_live_count() <= 1 {
            return Err(ControlError::NoLiveReplicas);
        }
        self.replicas.get_mut(id).unwrap().state = ReplicaState::Dead;
        self.open.retain(|c| c.replica != *id);
        Ok(())
    }

    fn quarantine_rack_of(
        &mut self,
        offender: &ReplicaId,
        shard: String,
        expected: String,
        got: String,
    ) -> Result<()> {
        let rack = self.lookup(offender)?.rack.clone();
        let on_rack: Vec<ReplicaId> = self
            .order
            .iter()
            .filter(|id| self.replicas.get(id).map(|r| r.rack.clone()) == Some(rack.clone()))
            .cloned()
            .collect();
        let live_on_rack = on_rack
            .iter()
            .filter(|id| self.replicas.get(id).map(|r| r.state) == Some(ReplicaState::Live))
            .count();
        if self.n_live_count().saturating_sub(live_on_rack) == 0 {
            return Err(ControlError::HashMismatch {
                replica: offender.clone(),
                shard,
                expected,
                got,
            });
        }
        for id in &on_rack {
            if let Some(r) = self.replicas.get_mut(id) {
                r.state = ReplicaState::Quarantined;
            }
            self.open.retain(|c| c.replica != *id);
        }
        Err(ControlError::HashMismatch {
            replica: offender.clone(),
            shard,
            expected,
            got,
        })
    }
}

pub mod rung;
pub mod scale;
