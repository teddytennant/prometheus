//! Shared builders and assertions for A6 `prometheus-control` oracle tests.
//!
//! Documented rules the implementer must match (also see `tests/reference`):
//!
//! # Construction
//! - `Controller::new` with zero Live replicas (empty vec, or every spec
//!   `spare: true`) is `ControlError::NoLiveReplicas` or `Message`.
//! - Duplicate `ReplicaId` (regardless of spare flag) is `Message` whose
//!   text contains "duplicate" (case-insensitive).
//! - `microbatch_tokens == 0` is `Message` containing "microbatch".
//! - `tokens_per_step == 0` is allowed; `grad_accumulation` is then 0 and
//!   `effective_tokens_per_step` is 0.
//!
//! # Initial membership
//! - Non-spare specs start `ReplicaState::Live` in the order they appear in
//!   the spec vec. Spares start `ReplicaState::Spare`.
//! - `live_replicas()` is the spec-order filter of `Live` (not insertion at
//!   the end on rejoin). `n_live()` equals that length.
//! - `grad_accumulation` is the smallest `u64` such that
//!   `n_live * microbatch_tokens * grad_accumulation >= tokens_per_step`
//!   (ceil division). `effective_tokens_per_step` equals that product.
//! - After every successful public call with `n_live >= 1`, the tokens
//!   invariant holds with that *minimum* accum (not a larger one).
//!
//! # fail / drain
//! - `fail_replica` of a Live replica sets `Dead`, drops it from the live
//!   set, and raises accum to the new minimum. Open collectives on it close.
//! - `fail_replica` of the last Live replica is `NoLiveReplicas` and does
//!   not change any replica's state or accum.
//! - Unknown id: `ReplicaNotFound`. Spare / Healing / Dead / Draining:
//!   `NotLive`. Quarantined: `Quarantined`.
//! - After any public method returns, no replica is `Draining`. Drain and
//!   fail are immediately `Dead`. The `Draining` variant is unused on this
//!   CPU analog (iface gap; tests do not invent an API to reach it).
//!
//! # Stragglers
//! - `observe_step_time`: `duration_ms > straggler_timeout_ms` drains like
//!   `fail_replica` (Dead, accum, last-live protection). `duration_ms ==
//!   timeout` does not drain. Unknown: `ReplicaNotFound`. Non-live as above.
//!
//! # Heal / rejoin
//! - `heal_spare(spare, source)` requires spare in `Spare` else `NotSpare`,
//!   source in `Live` else `NotLive` / `Quarantined` / `ReplicaNotFound`.
//!   Success: spare becomes `Healing`. `n_live` and accum unchanged.
//! - `rejoin` of `Healing` becomes `Live` at that replica's original spec
//!   position. accum falls to the new minimum; effective tokens stay
//!   `>= tokens_per_step`. The `step` argument is not compared to anything
//!   (`rejoin` itself is the step-boundary API).
//! - `rejoin` before `heal_spare` (state still `Spare`) is `NotSpare` or
//!   `NotStepBoundary`. `rejoin` of any other non-Healing known replica is
//!   `NotSpare`. Unknown: `ReplicaNotFound`.
//! - `NotStepBoundary` is otherwise unused (iface gap: there is no mid-step
//!   flag). Tests do not invent a clock/step API to trigger it.
//!
//! # SDC
//! - Hashes are opaque case-sensitive strings, keyed by `(replica, shard,
//!   step)`. `report_shard_hash` requires Live.
//! - `sdc_period_steps == 0`: `check_sdc` is always `Ok` (never inspects).
//! - `step % sdc_period_steps != 0`: `Ok` no-op. Step 0 *is* a check when
//!   period > 0 because `0 % N == 0`.
//! - At a check step the shard set is the union of shard names reported by
//!   Live replicas at that step. Empty set: `MissingHash` for the first
//!   Live replica in spec order, `shard = ""`.
//! - For each shard in lexicographic order, every Live replica must have a
//!   hash; first missing in spec order is `MissingHash` (no state change).
//! - Matching hashes: `Ok`. Mismatch: strict majority (`count > n_live/2`)
//!   is `expected`; else `expected` is the first Live replica's hash. The
//!   first subsequent Live replica whose hash differs is the offender.
//!   Return `HashMismatch { replica: offender, shard, expected, got }`.
//! - On mismatch, every replica on the offender's rack becomes
//!   `Quarantined` (Live ones drop out of n_live, accum rises) *unless*
//!   that would leave zero Live replicas, in which case state is unchanged
//!   and the same `HashMismatch` is still returned (last-live protection).
//! - Only one rack is quarantined per `check_sdc` call (first mismatch).
//!
//! # Health
//! - Every `HealthKind` (Xid, Ecc, Nvlink, NicFlap, Thermal,
//!   WatchdogTimeout) drains like `fail_replica`. Rank/detail/now_ms are
//!   ignored for membership. Unknown replica: `ReplicaNotFound`.
//!
//! # Collectives
//! - `begin_collective` requires Live. A second begin of the same
//!   `(name, replica)` before end is `Message` containing "collective" or
//!   "open". `end_collective` of a Live replica with no matching begin is
//!   `Ok` (no-op).
//! - `tick(now)` with no open collective is `Ok`. A replica with any open
//!   collective where `now_ms.saturating_sub(begin_ms) > collective_watchdog_ms`
//!   is drained like `fail_replica` (strictly greater; equal does not drain).
//!   Replicas are considered in spec order. Last-live protection applies.
//!
//! # Spike
//! - `on_loss_spike` does not call ckpt. `rollback_checkpoint_id` equals
//!   the string the caller passed. `skipped_shard` equals the shard.
//! - First spike: `page = false`. Later: `page = true` iff
//!   `step.saturating_sub(last_spike_step) < config.page_window_steps`
//!   (the config field, not only the `PAGE_WINDOW_STEPS` constant).
//! - `pages()` counts how many times `page` was true. `skipped_shards()`
//!   lists every skipped id in call order (duplicates kept).
//!
//! # Clocks
//! - Never wall clock. `NowMs` is injected. Tests pass literal milliseconds.
//!
//! Production must never import this module.

#![allow(dead_code)]

use crate::reference::{self, RefController};
use prometheus_control::{
    ControlError, Controller, ElasticConfig, HealthEvent, HealthKind, ReplicaId, ReplicaSpec,
    ReplicaState, Result, ShardId, PAGE_WINDOW_STEPS,
};

pub fn rid(s: &str) -> ReplicaId {
    ReplicaId(s.to_string())
}

pub fn spec(id: &str, rack: &str, spare: bool) -> ReplicaSpec {
    ReplicaSpec {
        id: rid(id),
        rack: prometheus_control::RackId(rack.to_string()),
        spare,
    }
}

pub fn default_config() -> ElasticConfig {
    ElasticConfig {
        tokens_per_step: 1024,
        microbatch_tokens: 128,
        sdc_period_steps: 10,
        straggler_timeout_ms: 1000,
        collective_watchdog_ms: 500,
        page_window_steps: PAGE_WINDOW_STEPS,
    }
}

pub fn n_live_m_spare(n_live: usize, n_spare: usize) -> Vec<ReplicaSpec> {
    let mut v = Vec::with_capacity(n_live + n_spare);
    for i in 0..n_live {
        v.push(spec(&format!("live-{i}"), &format!("rack-{i}"), false));
    }
    for i in 0..n_spare {
        v.push(spec(
            &format!("spare-{i}"),
            &format!("spare-rack-{i}"),
            true,
        ));
    }
    v
}

pub fn spec_ids(specs: &[ReplicaSpec]) -> Vec<ReplicaId> {
    specs.iter().map(|s| s.id.clone()).collect()
}

pub fn health(id: &str, kind: HealthKind, now_ms: u64) -> HealthEvent {
    HealthEvent {
        replica: rid(id),
        rank: prometheus_control::RankId(format!("rank-{id}")),
        kind,
        now_ms,
        detail: format!("{kind:?}"),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Membership {
    pub live: Vec<ReplicaId>,
    pub n_live: usize,
    pub accum: u64,
    pub effective: u64,
    pub states: Vec<(ReplicaId, ReplicaState)>,
    pub skipped: Vec<ShardId>,
    pub pages: u64,
}

pub fn membership(ctrl: &Controller, ids: &[ReplicaId]) -> Membership {
    Membership {
        live: ctrl.live_replicas().expect("live_replicas"),
        n_live: ctrl.n_live().expect("n_live"),
        accum: ctrl.grad_accumulation().expect("grad_accumulation"),
        effective: ctrl
            .effective_tokens_per_step()
            .expect("effective_tokens_per_step"),
        states: ids
            .iter()
            .map(|id| (id.clone(), ctrl.replica_state(id).expect("replica_state")))
            .collect(),
        skipped: ctrl.skipped_shards().expect("skipped_shards"),
        pages: ctrl.pages().expect("pages"),
    }
}

pub fn assert_tokens(n_live: usize, accum: u64, effective: u64, cfg: &ElasticConfig) {
    if n_live == 0 {
        return;
    }
    let n = n_live as u64;
    let expect_acc =
        reference::min_grad_accumulation(n, cfg.microbatch_tokens, cfg.tokens_per_step);
    assert_eq!(accum, expect_acc, "grad_accumulation must be the minimum");
    let expect_eff = n
        .checked_mul(cfg.microbatch_tokens)
        .and_then(|x| x.checked_mul(accum))
        .expect("effective product fits u64 in tests");
    assert_eq!(effective, expect_eff, "effective_tokens_per_step");
    assert!(
        expect_eff >= cfg.tokens_per_step,
        "effective {expect_eff} < tokens_per_step {}",
        cfg.tokens_per_step
    );
}

pub fn assert_ctrl_tokens(ctrl: &Controller, cfg: &ElasticConfig) {
    let n = ctrl.n_live().expect("n_live");
    assert_tokens(
        n,
        ctrl.grad_accumulation().expect("accum"),
        ctrl.effective_tokens_per_step().expect("effective"),
        cfg,
    );
    assert_eq!(ctrl.live_replicas().expect("live").len(), n);
    assert!(
        !ctrl
            .live_replicas()
            .expect("live")
            .iter()
            .any(|id| ctrl.replica_state(id).expect("st") != ReplicaState::Live),
        "live_replicas must all be Live"
    );
}

pub fn assert_prod_matches_ref(prod: &Controller, refer: &RefController) {
    assert_eq!(
        prod.n_live().expect("prod n_live"),
        refer.n_live().expect("ref n_live"),
        "n_live"
    );
    assert_eq!(
        prod.live_replicas().expect("prod live"),
        refer.live_replicas().expect("ref live"),
        "live_replicas"
    );
    assert_eq!(
        prod.grad_accumulation().expect("prod accum"),
        refer.grad_accumulation().expect("ref accum"),
        "grad_accumulation"
    );
    assert_eq!(
        prod.effective_tokens_per_step().expect("prod eff"),
        refer.effective_tokens_per_step().expect("ref eff"),
        "effective_tokens_per_step"
    );
    assert_eq!(
        prod.skipped_shards().expect("prod skipped"),
        refer.skipped_shards().expect("ref skipped"),
        "skipped_shards"
    );
    assert_eq!(
        prod.pages().expect("prod pages"),
        refer.pages().expect("ref pages"),
        "pages"
    );
    for id in refer.ids() {
        assert_eq!(
            prod.replica_state(&id).expect("prod state"),
            refer.replica_state(&id).expect("ref state"),
            "state {id:?}"
        );
        assert_ne!(
            prod.replica_state(&id).unwrap(),
            ReplicaState::Draining,
            "Draining is not an observed A6 state"
        );
    }
}

pub fn pair(cfg: ElasticConfig, specs: Vec<ReplicaSpec>) -> (Controller, RefController) {
    let prod = Controller::new(cfg.clone(), specs.clone()).expect("Controller::new");
    let refer = RefController::new(cfg, specs).expect("RefController::new");
    (prod, refer)
}

/// `Controller` is not `Debug`, so `Result::expect_err` cannot be used on `new`.
pub fn new_err(cfg: ElasticConfig, specs: Vec<ReplicaSpec>) -> ControlError {
    match Controller::new(cfg, specs) {
        Ok(_) => panic!("Controller::new succeeded, expected error"),
        Err(e) => e,
    }
}

pub fn assert_replica_not_found(err: &ControlError, id: &ReplicaId) {
    match err {
        ControlError::ReplicaNotFound(got) => assert_eq!(got, id),
        other => panic!("expected ReplicaNotFound({id:?}), got {other:?}"),
    }
}

pub fn assert_no_live_replicas(err: &ControlError) {
    match err {
        ControlError::NoLiveReplicas => {}
        other => panic!("expected NoLiveReplicas, got {other:?}"),
    }
}

pub fn assert_no_live_replicas_or_message(err: &ControlError) {
    match err {
        ControlError::NoLiveReplicas => {}
        ControlError::Message(s) => {
            let l = s.to_lowercase();
            assert!(
                l.contains("live") || l.contains("replica"),
                "Message should mention live/replica, got {s:?}"
            );
        }
        other => panic!("expected NoLiveReplicas or Message, got {other:?}"),
    }
}

pub fn assert_not_live(err: &ControlError, id: &ReplicaId) {
    match err {
        ControlError::NotLive(got) => assert_eq!(got, id),
        other => panic!("expected NotLive({id:?}), got {other:?}"),
    }
}

pub fn assert_not_spare(err: &ControlError, id: &ReplicaId) {
    match err {
        ControlError::NotSpare(got) => assert_eq!(got, id),
        other => panic!("expected NotSpare({id:?}), got {other:?}"),
    }
}

pub fn assert_rejoin_before_heal(err: &ControlError, id: &ReplicaId) {
    match err {
        ControlError::NotSpare(got) => assert_eq!(got, id),
        ControlError::NotStepBoundary => {}
        other => panic!("expected NotSpare({id:?}) or NotStepBoundary, got {other:?}"),
    }
}

pub fn assert_quarantined(err: &ControlError, id: &ReplicaId) {
    match err {
        ControlError::Quarantined(got) => assert_eq!(got, id),
        other => panic!("expected Quarantined({id:?}), got {other:?}"),
    }
}

pub fn assert_hash_mismatch(
    err: &ControlError,
    replica: &ReplicaId,
    shard: &str,
    expected: &str,
    got: &str,
) {
    match err {
        ControlError::HashMismatch {
            replica: r,
            shard: s,
            expected: e,
            got: g,
        } => {
            assert_eq!(r, replica, "HashMismatch.replica");
            assert_eq!(s, shard, "HashMismatch.shard");
            assert_eq!(e, expected, "HashMismatch.expected");
            assert_eq!(g, got, "HashMismatch.got");
        }
        other => panic!("expected HashMismatch, got {other:?}"),
    }
}

pub fn assert_missing_hash(err: &ControlError, replica: &ReplicaId, shard: &str) {
    match err {
        ControlError::MissingHash {
            replica: r,
            shard: s,
        } => {
            assert_eq!(r, replica, "MissingHash.replica");
            assert_eq!(s, shard, "MissingHash.shard");
        }
        other => panic!("expected MissingHash, got {other:?}"),
    }
}

pub fn assert_duplicate_message(err: &ControlError) {
    match err {
        ControlError::Message(s) => {
            assert!(
                s.to_lowercase().contains("duplicate"),
                "duplicate Message must contain 'duplicate', got {s:?}"
            );
        }
        other => panic!("expected Message containing duplicate, got {other:?}"),
    }
}

pub fn err_tag(err: &ControlError) -> &'static str {
    match err {
        ControlError::ReplicaNotFound(_) => "ReplicaNotFound",
        ControlError::NoLiveReplicas => "NoLiveReplicas",
        ControlError::NotSpare(_) => "NotSpare",
        ControlError::NotStepBoundary => "NotStepBoundary",
        ControlError::HashMismatch { .. } => "HashMismatch",
        ControlError::Quarantined(_) => "Quarantined",
        ControlError::NotLive(_) => "NotLive",
        ControlError::MissingHash { .. } => "MissingHash",
        ControlError::HealSourceMismatch { .. } => "HealSourceMismatch",
        ControlError::NoHealInProgress(_) => "NoHealInProgress",
        ControlError::WeightCopyNotInstalled(_) => "WeightCopyNotInstalled",
        ControlError::DuplicateWeightShard(_) => "DuplicateWeightShard",
        ControlError::EmptyWeightShardName => "EmptyWeightShardName",
        ControlError::Message(_) => "Message",
    }
}

pub fn assert_err_eq(prod: &ControlError, refer: &ControlError) {
    match (prod, refer) {
        (ControlError::ReplicaNotFound(a), ControlError::ReplicaNotFound(b)) => {
            assert_eq!(a, b)
        }
        (ControlError::NoLiveReplicas, ControlError::NoLiveReplicas) => {}
        (ControlError::NotSpare(a), ControlError::NotSpare(b)) => assert_eq!(a, b),
        (ControlError::NotSpare(_), ControlError::NotStepBoundary)
        | (ControlError::NotStepBoundary, ControlError::NotSpare(_))
        | (ControlError::NotStepBoundary, ControlError::NotStepBoundary) => {}
        (
            ControlError::HashMismatch {
                replica: r1,
                shard: s1,
                expected: e1,
                got: g1,
            },
            ControlError::HashMismatch {
                replica: r2,
                shard: s2,
                expected: e2,
                got: g2,
            },
        ) => {
            assert_eq!(r1, r2);
            assert_eq!(s1, s2);
            assert_eq!(e1, e2);
            assert_eq!(g1, g2);
        }
        (ControlError::Quarantined(a), ControlError::Quarantined(b)) => assert_eq!(a, b),
        (ControlError::NotLive(a), ControlError::NotLive(b)) => assert_eq!(a, b),
        (
            ControlError::MissingHash {
                replica: r1,
                shard: s1,
            },
            ControlError::MissingHash {
                replica: r2,
                shard: s2,
            },
        ) => {
            assert_eq!(r1, r2);
            assert_eq!(s1, s2);
        }
        (ControlError::Message(_), ControlError::Message(_)) => {}
        (
            ControlError::HealSourceMismatch {
                spare: s1,
                expected: e1,
                got: g1,
            },
            ControlError::HealSourceMismatch {
                spare: s2,
                expected: e2,
                got: g2,
            },
        ) => {
            assert_eq!(s1, s2);
            assert_eq!(e1, e2);
            assert_eq!(g1, g2);
        }
        (ControlError::NoHealInProgress(a), ControlError::NoHealInProgress(b)) => {
            assert_eq!(a, b)
        }
        (ControlError::WeightCopyNotInstalled(a), ControlError::WeightCopyNotInstalled(b)) => {
            assert_eq!(a, b)
        }
        (ControlError::DuplicateWeightShard(a), ControlError::DuplicateWeightShard(b)) => {
            assert_eq!(a, b)
        }
        (ControlError::EmptyWeightShardName, ControlError::EmptyWeightShardName) => {}
        (a, b) => panic!(
            "prod error {a:?} != ref error {b:?} ({} vs {})",
            err_tag(a),
            err_tag(b)
        ),
    }
}

pub fn assert_both_ok<T, U>(prod: Result<T>, refer: Result<U>) {
    match (prod, refer) {
        (Ok(_), Ok(_)) => {}
        (Err(e), Err(r)) => panic!("both erred: prod {e:?} ref {r:?}"),
        (Ok(_), Err(r)) => panic!("prod Ok, ref Err {r:?}"),
        (Err(e), Ok(_)) => panic!("prod Err {e:?}, ref Ok"),
    }
}

pub fn assert_both_err<T: std::fmt::Debug, U: std::fmt::Debug>(
    prod: Result<T>,
    refer: Result<U>,
) -> ControlError {
    match (prod, refer) {
        (Err(e), Err(r)) => {
            assert_err_eq(&e, &r);
            e
        }
        (Ok(v), Ok(w)) => panic!("both Ok: prod {v:?} ref {w:?}"),
        (Ok(v), Err(r)) => panic!("prod Ok {v:?}, ref Err {r:?}"),
        (Err(e), Ok(w)) => panic!("prod Err {e:?}, ref Ok {w:?}"),
    }
}
