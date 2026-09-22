//! Elastic scheduler, health, stragglers, SDC, spike rollback (spec 5.5, 15.5 A6).
//!
//! CPU analog of V4: simulated DP replicas, no GPUs, no wall clock (`NowMs` is
//! injected). Elastic control at 220k GPUs, 90 TB checkpoints, and EP=72 are S3
//! (spec 15.6) and out of scope. Real IB is also S3. The in-memory copy in
//! [`Controller::offer_weight_copy`] is the S1 analog of spec 5.5 "weights copied
//! from a live replica".
//!
//! A failed replica drops out. Survivors keep going. [`Controller::grad_accumulation`]
//! rises so [`ElasticConfig::tokens_per_step`] stays held. A spare heals in and
//! rejoins at the next step boundary, holding a copy of the source weights.
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
//!
//! Rung 0 end-to-end bookkeeping is [`rung`] (spec 15.5 A7, CPU analog of V5).
//! Scaling-law fit on rungs 1 to 3 is [`scale`] (spec 15.5 I1).

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub mod rung;
pub mod scale;

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

/// One named weight shard copied from a live replica onto a healing spare.
///
/// CPU analog of spec 5.5 "weights copied from a live replica over IB". `bytes`
/// are opaque. Real IB is S3 (spec 15.6) and is not this type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightShard {
    pub name: String,
    pub bytes: Vec<u8>,
}

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
    #[error("heal source mismatch spare={spare:?} expected={expected:?} got={got:?}")]
    HealSourceMismatch {
        spare: ReplicaId,
        expected: ReplicaId,
        got: ReplicaId,
    },
    #[error("no heal in progress: {0:?}")]
    NoHealInProgress(ReplicaId),
    #[error("weight copy not installed: {0:?}")]
    WeightCopyNotInstalled(ReplicaId),
    #[error("duplicate weight shard name: {0}")]
    DuplicateWeightShard(String),
    #[error("empty weight shard name")]
    EmptyWeightShardName,
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, ControlError>;

struct ReplicaRec {
    rack: RackId,
    state: ReplicaState,
}

struct OpenCollective {
    name: String,
    replica: ReplicaId,
    begin_ms: NowMs,
}

/// Elastic DP controller. CPU tests drive a handful of simulated replicas.
pub struct Controller {
    config: ElasticConfig,
    order: Vec<ReplicaId>,
    replicas: HashMap<ReplicaId, ReplicaRec>,
    hashes: BTreeMap<(Step, String, String), String>,
    open: Vec<OpenCollective>,
    skipped: Vec<ShardId>,
    pages: u64,
    last_spike_step: Option<Step>,
}

fn min_grad_accumulation(n_live: u64, microbatch_tokens: u64, tokens_per_step: u64) -> u64 {
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

impl Controller {
    pub fn new(config: ElasticConfig, replicas: Vec<ReplicaSpec>) -> Result<Self> {
        if config.microbatch_tokens == 0 {
            return Err(ControlError::Message(
                "microbatch_tokens must be greater than 0".into(),
            ));
        }
        let mut order = Vec::new();
        let mut map = HashMap::new();
        let mut n_live = 0usize;
        for spec in replicas {
            if map.contains_key(&spec.id) {
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
                ReplicaRec {
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
        })
    }

    pub fn live_replicas(&self) -> Result<Vec<ReplicaId>> {
        Ok(self.live_ids())
    }

    pub fn replica_state(&self, id: &ReplicaId) -> Result<ReplicaState> {
        Ok(self.lookup(id)?.state)
    }

    pub fn n_live(&self) -> Result<usize> {
        Ok(self.n_live_count())
    }

    /// Microbatches per optimizer step. Rises when a replica drops so
    /// `n_live * microbatch_tokens * grad_accumulation >= tokens_per_step`.
    pub fn grad_accumulation(&self) -> Result<u64> {
        Ok(self.accum())
    }

    /// `n_live * microbatch_tokens * grad_accumulation`.
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

    /// Drop a live replica immediately. Raises grad accumulation. Errors if
    /// this would leave zero live replicas.
    pub fn fail_replica(&mut self, id: &ReplicaId) -> Result<()> {
        self.require_live(id)?;
        self.drop_live(id)
    }

    /// Record one rank's step duration. Above `straggler_timeout_ms` drains
    /// the replica the same way as [`Self::fail_replica`].
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

    /// Start a heal of `spare` from `source`. Records `source` for
    /// [`Self::offer_weight_copy`]. Does not copy bytes. Completes at
    /// [`Self::rejoin`].
    pub fn heal_spare(&mut self, spare: &ReplicaId, source: &ReplicaId) -> Result<()> {
        let spare_state = self.lookup(spare)?.state;
        if spare_state != ReplicaState::Spare {
            return Err(ControlError::NotSpare(spare.clone()));
        }
        self.require_live(source)?;
        match self.replicas.get_mut(spare) {
            Some(rec) => rec.state = ReplicaState::Healing,
            None => return Err(ControlError::ReplicaNotFound(spare.clone())),
        }
        Ok(())
    }

    /// Finish a heal at a step boundary. The spare becomes live and
    /// accumulation falls to hold `tokens_per_step` without dropping below it.
    ///
    /// Installs the shards staged by [`Self::offer_weight_copy`] for this heal.
    /// No staged shards installs an empty copy, not an error. `step` is recorded
    /// on that copy and is not compared to a clock (`rejoin` itself is the
    /// step-boundary API).
    pub fn rejoin(&mut self, id: &ReplicaId, _step: Step) -> Result<()> {
        let state = self.lookup(id)?.state;
        if state != ReplicaState::Healing {
            return Err(ControlError::NotSpare(id.clone()));
        }
        match self.replicas.get_mut(id) {
            Some(rec) => rec.state = ReplicaState::Live,
            None => return Err(ControlError::ReplicaNotFound(id.clone())),
        }
        Ok(())
    }

    /// Source recorded by the last successful [`Self::heal_spare`] for `spare`.
    ///
    /// `ReplicaNotFound` if unknown. `NoHealInProgress` if `spare` is not
    /// `Healing`, including before heal and after rejoin.
    pub fn heal_source(&self, spare: &ReplicaId) -> Result<ReplicaId> {
        let _ = spare;
        unimplemented!("a6 heal weight copy")
    }

    /// Stage an in-memory copy of `shards` from `source` onto a spare that is
    /// `Healing`. Completes at [`Self::rejoin`].
    ///
    /// - Spare unknown: `ReplicaNotFound`. Spare not `Healing`: `NoHealInProgress`.
    /// - `source` must equal the source passed to `heal_spare`, else
    ///   `HealSourceMismatch`. Source must still be `Live`, else `NotLive`,
    ///   `Quarantined`, or `ReplicaNotFound` (same rules as `heal_spare`).
    /// - Shard `name` must be non-empty, else `EmptyWeightShardName`. A duplicate
    ///   name in this call, or already staged for this heal, is
    ///   `DuplicateWeightShard`. The first error in shard order wins. A failed
    ///   call stages nothing.
    /// - Bytes are copied. Mutating the caller's buffers after return must not
    ///   change what [`Self::healed_weight_shards`] returns after rejoin.
    /// - A later offer of a new name appends. Order is offer order, then shard
    ///   order within each offer.
    /// - Does not change replica state, `n_live`, or grad accumulation.
    /// - Empty `shards` is `Ok` and stages nothing.
    pub fn offer_weight_copy(
        &mut self,
        spare: &ReplicaId,
        source: &ReplicaId,
        shards: &[WeightShard],
    ) -> Result<()> {
        let _ = (spare, source, shards);
        unimplemented!("a6 heal weight copy")
    }

    /// Weight shards installed on `id` by the rejoin that finished its heal.
    ///
    /// `ReplicaNotFound` if unknown. `WeightCopyNotInstalled` if `id` has never
    /// completed a heal (still `Spare` or `Healing`, or was never a spare). A
    /// rejoin with no staged shards installs an empty vec: `Ok` of an empty
    /// vec, not `WeightCopyNotInstalled`.
    ///
    /// The returned vec is an owned copy. A later heal must not mutate a vec
    /// the caller already received. A second heal replaces the installed copy
    /// only at that second rejoin.
    pub fn healed_weight_shards(&self, id: &ReplicaId) -> Result<Vec<WeightShard>> {
        let _ = id;
        unimplemented!("a6 heal weight copy")
    }

    /// Report a weight-shard content hash from one replica at `step`.
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

    /// Compare hashes reported at `step`. A mismatch quarantines the offending
    /// replica's rack (treated as a drop). No-op when `step` is not a multiple
    /// of `sdc_period_steps` (including period 0 meaning "never").
    pub fn check_sdc(&mut self, step: Step) -> Result<()> {
        let period = self.config.sdc_period_steps;
        if period == 0 {
            return Ok(());
        }
        if !step.is_multiple_of(period) {
            return Ok(());
        }
        let live = self.live_ids();
        if live.is_empty() {
            return Ok(());
        }
        let mut shards: BTreeSet<String> = BTreeSet::new();
        for id in &live {
            for (s, rid, shard) in self.hashes.keys() {
                if *s == step && *rid == id.0 {
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
            let expected = match majority {
                Some(h) => h,
                None => match reports.first() {
                    Some((_, hex)) => hex.clone(),
                    None => {
                        return Err(ControlError::MissingHash {
                            replica: live[0].clone(),
                            shard,
                        });
                    }
                },
            };
            for (id, hex) in &reports {
                if hex != &expected {
                    return self.quarantine_rack_of(id, shard, expected, hex.clone());
                }
            }
        }
        Ok(())
    }

    /// Feed a device/fabric/thermal/watchdog event. Fatal kinds drain the replica.
    pub fn report_health(&mut self, event: HealthEvent) -> Result<()> {
        self.require_live(&event.replica)?;
        self.drop_live(&event.replica)
    }

    pub fn begin_collective(
        &mut self,
        name: &str,
        replica: &ReplicaId,
        now_ms: NowMs,
    ) -> Result<()> {
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
        self.open.push(OpenCollective {
            name: name.to_string(),
            replica: replica.clone(),
            begin_ms: now_ms,
        });
        Ok(())
    }

    pub fn end_collective(
        &mut self,
        name: &str,
        replica: &ReplicaId,
        _now_ms: NowMs,
    ) -> Result<()> {
        self.require_live(replica)?;
        self.open
            .retain(|c| !(c.name == name && c.replica == *replica));
        Ok(())
    }

    /// Advance the injected clock. Collectives past `collective_watchdog_ms`
    /// drain their replica with [`HealthKind::WatchdogTimeout`].
    pub fn tick(&mut self, now_ms: NowMs) -> Result<()> {
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

    /// Execute the spec 5.5 spike policy. Does not talk to ckpt.
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

    /// How many times [`SpikeReport::page`] was true.
    pub fn pages(&self) -> Result<u64> {
        Ok(self.pages)
    }

    fn live_ids(&self) -> Vec<ReplicaId> {
        self.order
            .iter()
            .filter(|id| self.replicas.get(id).map(|r| r.state) == Some(ReplicaState::Live))
            .cloned()
            .collect()
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

    fn lookup(&self, id: &ReplicaId) -> Result<&ReplicaRec> {
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
        match self.replicas.get_mut(id) {
            Some(rec) => rec.state = ReplicaState::Dead,
            None => return Err(ControlError::ReplicaNotFound(id.clone())),
        }
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
            .filter(|id| self.replicas.get(id).map(|r| &r.rack) == Some(&rack))
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
