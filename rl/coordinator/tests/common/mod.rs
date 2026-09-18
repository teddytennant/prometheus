//! Shared builders and assertions for I2 `prometheus-coordinator` oracle tests.
//!
//! Documented rules the implementer must match (also see `tests/reference`):
//!
//! # Construction
//! - Empty racks vec: `CoordError::EmptyFleet`.
//! - Duplicate `RackId` (first repeat in input order): `DuplicateRack`.
//! - Any rack with `gpus == 0` (first in input order): `Message` whose text
//!   contains `"gpu"` (case-insensitive).
//! - `rollout_denom == 0` or `rollout_numer > rollout_denom`: `BadSplit`.
//! - `racks.len() < min_rollout_racks + min_trainer_racks`: `NotEnoughRacks`.
//! - Check order is the list above. `max_staleness` may be 0 (then only
//!   matching versions are consumable).
//!
//! # Initial assignment (deterministic)
//! - Sort racks by `id.0` lexicographically (`String` / byte order, not
//!   numeric). `"r10" < "r2"` because `'1' < '2'`. Tests use zero-padded
//!   ids (`r00`, `r01`, …) or single letters when order matters.
//! - `total_gpus` = sum of `gpus`.
//! - `target_rollout = total_gpus * rollout_numer / rollout_denom` (integer
//!   floor, computed in u128).
//! - Walk the sorted list, assigning `Rollout` while the running rollout GPU
//!   sum stays `<= target_rollout` AND the remaining unassigned racks *after
//!   taking this one* are enough to satisfy `min_trainer_racks`. Rest are
//!   `Trainer`. The rollout set is always a lex prefix of the sorted racks.
//! - If that would leave fewer than `min_rollout_racks` rollout racks,
//!   assign more `Rollout` (still lex order, next racks in the suffix) until
//!   the floor holds, even if it overshoots the GPU target.
//! - After `new`: `trainer_version == 0`. Every Rollout rack's
//!   `rack_version == 0`. Trainer racks: `rack_version` is `NotRollout`.
//! - `state == Running`, `halted == false`, `queue_depth == 0`.
//!
//! # `staleness(trainer, batch)`
//! - If `batch > trainer`: `FromTheFuture`.
//! - Else `Ok(trainer - batch)`.
//!
//! # Enqueue
//! - Halted: `Halted`.
//! - `n_samples == 0`: `EmptyGroup` (checked before empty-id Messages).
//! - Empty `BatchId` (empty string): `Message` containing `"id"`
//!   (case-insensitive).
//! - Empty `PromptId`: `Message` containing `"prompt"` (case-insensitive).
//! - If both ids are empty, BatchId is checked first → `"id"`.
//! - Duplicate batch id still in the queue: `DuplicateBatch`. Ids of
//!   already-consumed or stale-dropped batches MAY be reused.
//! - FIFO queue. `routing_present` and `prompt_id` are stored and returned
//!   unchanged on consume.
//! - Do not require `n_samples == GROUP_SIZE`.
//!
//! # Consume
//! - Halted: `Halted`.
//! - Drop from the front while
//!   `staleness(trainer_version, batch.policy_version) > max_staleness`.
//! - A future batch (`policy_version > trainer_version`) is a bug: if the
//!   scan hits one, `consume_batch` returns `FromTheFuture` and leaves the
//!   queue **unchanged** (stale prefix is not dropped either).
//! - Then pop the oldest remaining. `Ok(None)` if empty after drops.
//! - Dropped stale batches are gone (not returned).
//! - "Drop from the front while stale" means only a stale **prefix**, not
//!   stale items behind a fresh one. `[stale, fresh, stale]` returns `fresh`
//!   and leaves the last stale. If ALL remaining are stale, drop them and
//!   return `None`.
//! - `k == max_staleness` is kept; `k == max_staleness + 1` is dropped.
//!
//! # Publish
//! - Halted: `Halted`.
//! - `version` must be exactly `trainer_version + 1` else `NotNextVersion`.
//! - First publish from 0 is 1. Batches produced at version 0 are valid
//!   before any publish. Publish does not change per-rack versions.
//!
//! # Sync
//! - Halted: `Halted`.
//! - Unknown rack: `RackNotFound`. Trainer rack: `NotRollout`.
//! - `version` must equal current `trainer_version` else `Unpublished`
//!   (payload = the requested version). Instant. Sets `rack_version`.
//! - Sync of `0` at `trainer_version == 0` is allowed.
//!
//! # Rebalance (at most one rack per call)
//! - Halted: `Halted`.
//! - Preference order:
//!   1. If `queue_depth == 0` and a trainer rack can move to rollout without
//!      breaking `min_trainer_racks`, move the lex-smallest such trainer
//!      rack to Rollout. Its `rack_version` becomes 0.
//!   2. Else if `queue_high > 0` and `queue_depth >= queue_high` and a
//!      rollout rack can move to trainer without breaking `min_rollout_racks`,
//!      move the lex-smallest such rollout rack to Trainer.
//!   3. Else if rollout GPU sum < target_rollout, same as (1).
//!   4. Else if rollout GPU sum > target_rollout, same as (2).
//!   5. Else `Ok(None)`.
//! - Return `Ok(Some((id, new_role)))` on a move.
//! - Rule 1 ignores the GPU target. `queue_high == 0` never uses rule 2.
//! - Empty queue already over the GPU target can therefore oscillate:
//!   rule 1 adds rollout until `min_trainer` binds, then rule 4 peels one
//!   back. Implementers must match this, not "stabilize at target".
//! - Moving Rollout → Trainer: subsequent `rack_version` is `NotRollout`.
//!   Moving back to Rollout resets version to 0 until `sync_rack`.
//! - In-flight queue batches are untouched by rebalance.
//!
//! # Parity (`run_parity`)
//! - Halted: `Halted`.
//! - `version` must equal `trainer_version` else `Unpublished`.
//! - Any non-finite log-prob (NaN or ±inf on either side): `InvalidLogprob`,
//!   do not halt.
//! - Empty samples: `Ok(ParityReport { policy_version, max_abs_err: 0.0,
//!   n_samples: 0 })`, do not halt.
//! - `max_abs_err = max_i |trainer_logp_i - engine_logp_i|`.
//! - If `max_abs_err > parity_threshold`: set halted, return
//!   `ParityHalt { version, max_abs_err, threshold }`. Equal to threshold
//!   does NOT halt.
//! - After a ParityHalt, `state == Halted`, `halted == true`. Later mutators
//!   (`enqueue_batch`, `consume_batch`, `rebalance`, `publish_weights`,
//!   `sync_rack`, `run_parity`) return `Halted`. Read methods still work
//!   (`state`, `halted`, versions, split, queue_depth, rack_role, n_*).
//!
//! Read methods never return `Halted`.
//! `split_gpus` → `(rollout_gpus, trainer_gpus)` summing to total.
//!
//! Production must never import this module.

#![allow(dead_code)]

use crate::reference::RefCoordinator;
use prometheus_coordinator::{
    BatchId, CoordError, Coordinator, CoordinatorConfig, CoordinatorState, ParitySample,
    PolicyVersion, PromptId, RackId, RackRole, RackSpec, Result, RolloutBatch,
    DEFAULT_ROLLOUT_DENOM, DEFAULT_ROLLOUT_NUMER, MAX_STALENESS,
};

pub fn rid(s: &str) -> RackId {
    RackId(s.to_string())
}

pub fn bid(s: &str) -> BatchId {
    BatchId(s.to_string())
}

pub fn pid(s: &str) -> PromptId {
    PromptId(s.to_string())
}

pub fn rack(id: &str, gpus: u64) -> RackSpec {
    RackSpec { id: rid(id), gpus }
}

/// Zero-padded ids `r00`, `r01`, … so lex order matches numeric order.
pub fn n_equal_racks(n: usize, gpus: u64) -> Vec<RackSpec> {
    (0..n).map(|i| rack(&format!("r{i:02}"), gpus)).collect()
}

pub fn default_config() -> CoordinatorConfig {
    CoordinatorConfig {
        max_staleness: MAX_STALENESS,
        rollout_numer: DEFAULT_ROLLOUT_NUMER,
        rollout_denom: DEFAULT_ROLLOUT_DENOM,
        parity_threshold: 0.25,
        queue_high: 8,
        min_rollout_racks: 1,
        min_trainer_racks: 1,
    }
}

/// Four racks × 8 GPUs, 65/35, mins 1+1. Target = 32*65/100 = 20 → 2+2 split.
pub fn default_racks() -> Vec<RackSpec> {
    n_equal_racks(4, 8)
}

pub fn batch(
    id: &str,
    prompt: &str,
    policy_version: PolicyVersion,
    n_samples: u32,
    routing_present: bool,
) -> RolloutBatch {
    RolloutBatch {
        id: bid(id),
        prompt_id: pid(prompt),
        policy_version,
        n_samples,
        routing_present,
    }
}

pub fn sample(trainer_logp: f64, engine_logp: f64) -> ParitySample {
    ParitySample {
        trainer_logp,
        engine_logp,
    }
}

pub fn total_gpus(racks: &[RackSpec]) -> u64 {
    racks
        .iter()
        .map(|r| r.gpus)
        .fold(0u64, |a, b| a.saturating_add(b))
}

/// `Coordinator` is not `Debug`, so `Result::expect_err` cannot be used on `new`.
pub fn new_err(cfg: CoordinatorConfig, racks: Vec<RackSpec>) -> CoordError {
    match Coordinator::new(cfg, racks) {
        Ok(_) => panic!("Coordinator::new succeeded, expected error"),
        Err(e) => e,
    }
}

pub fn new_err_both(cfg: CoordinatorConfig, racks: Vec<RackSpec>) -> CoordError {
    let refer = match RefCoordinator::new(cfg.clone(), racks.clone()) {
        Ok(_) => panic!("RefCoordinator::new succeeded, expected error"),
        Err(e) => e,
    };
    let prod = match Coordinator::new(cfg, racks) {
        Ok(_) => panic!("Coordinator::new succeeded, expected error"),
        Err(e) => e,
    };
    assert_err_eq(&prod, &refer);
    prod
}

pub fn pair(cfg: CoordinatorConfig, racks: Vec<RackSpec>) -> (Coordinator, RefCoordinator) {
    let prod = match Coordinator::new(cfg.clone(), racks.clone()) {
        Ok(c) => c,
        Err(e) => panic!("Coordinator::new failed: {e:?}"),
    };
    let refer = match RefCoordinator::new(cfg, racks) {
        Ok(c) => c,
        Err(e) => panic!("RefCoordinator::new failed: {e:?}"),
    };
    (prod, refer)
}

pub fn err_tag(err: &CoordError) -> &'static str {
    match err {
        CoordError::EmptyFleet => "EmptyFleet",
        CoordError::DuplicateRack(_) => "DuplicateRack",
        CoordError::BadSplit => "BadSplit",
        CoordError::NotEnoughRacks => "NotEnoughRacks",
        CoordError::Halted => "Halted",
        CoordError::EmptyGroup => "EmptyGroup",
        CoordError::DuplicateBatch(_) => "DuplicateBatch",
        CoordError::FromTheFuture => "FromTheFuture",
        CoordError::NotNextVersion => "NotNextVersion",
        CoordError::Unpublished(_) => "Unpublished",
        CoordError::RackNotFound(_) => "RackNotFound",
        CoordError::NotRollout(_) => "NotRollout",
        CoordError::NotTrainer(_) => "NotTrainer",
        CoordError::BatchNotFound(_) => "BatchNotFound",
        CoordError::InvalidLogprob => "InvalidLogprob",
        CoordError::ParityHalt { .. } => "ParityHalt",
        CoordError::Message(_) => "Message",
    }
}

pub fn assert_err_eq(prod: &CoordError, refer: &CoordError) {
    match (prod, refer) {
        (CoordError::EmptyFleet, CoordError::EmptyFleet) => {}
        (CoordError::DuplicateRack(a), CoordError::DuplicateRack(b)) => assert_eq!(a, b),
        (CoordError::BadSplit, CoordError::BadSplit) => {}
        (CoordError::NotEnoughRacks, CoordError::NotEnoughRacks) => {}
        (CoordError::Halted, CoordError::Halted) => {}
        (CoordError::EmptyGroup, CoordError::EmptyGroup) => {}
        (CoordError::DuplicateBatch(a), CoordError::DuplicateBatch(b)) => assert_eq!(a, b),
        (CoordError::FromTheFuture, CoordError::FromTheFuture) => {}
        (CoordError::NotNextVersion, CoordError::NotNextVersion) => {}
        (CoordError::Unpublished(a), CoordError::Unpublished(b)) => assert_eq!(a, b),
        (CoordError::RackNotFound(a), CoordError::RackNotFound(b)) => assert_eq!(a, b),
        (CoordError::NotRollout(a), CoordError::NotRollout(b)) => assert_eq!(a, b),
        (CoordError::NotTrainer(a), CoordError::NotTrainer(b)) => assert_eq!(a, b),
        (CoordError::BatchNotFound(a), CoordError::BatchNotFound(b)) => assert_eq!(a, b),
        (CoordError::InvalidLogprob, CoordError::InvalidLogprob) => {}
        (
            CoordError::ParityHalt {
                version: v1,
                max_abs_err: e1,
                threshold: t1,
            },
            CoordError::ParityHalt {
                version: v2,
                max_abs_err: e2,
                threshold: t2,
            },
        ) => {
            assert_eq!(v1, v2, "ParityHalt.version");
            assert_eq!(e1, e2, "ParityHalt.max_abs_err");
            assert_eq!(t1, t2, "ParityHalt.threshold");
        }
        (CoordError::Message(_), CoordError::Message(_)) => {}
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
) -> CoordError {
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

pub fn assert_message_contains(err: &CoordError, needle: &str) {
    match err {
        CoordError::Message(s) => {
            assert!(
                s.to_lowercase().contains(&needle.to_lowercase()),
                "Message {s:?} does not contain {needle:?}"
            );
        }
        other => panic!("expected Message containing {needle:?}, got {other:?}"),
    }
}

pub fn assert_halted(err: &CoordError) {
    match err {
        CoordError::Halted => {}
        other => panic!("expected Halted, got {other:?}"),
    }
}

pub fn assert_prod_matches_ref(prod: &Coordinator, refer: &RefCoordinator, racks: &[RackSpec]) {
    assert_eq!(
        prod.state().expect("prod state"),
        refer.state().expect("ref state"),
        "state"
    );
    assert_eq!(
        prod.halted().expect("prod halted"),
        refer.halted().expect("ref halted"),
        "halted"
    );
    assert_eq!(
        prod.trainer_version().expect("prod tv"),
        refer.trainer_version().expect("ref tv"),
        "trainer_version"
    );
    assert_eq!(
        prod.split_gpus().expect("prod split"),
        refer.split_gpus().expect("ref split"),
        "split_gpus"
    );
    assert_eq!(
        prod.n_rollout_racks().expect("prod n_ro"),
        refer.n_rollout_racks().expect("ref n_ro"),
        "n_rollout_racks"
    );
    assert_eq!(
        prod.n_trainer_racks().expect("prod n_tr"),
        refer.n_trainer_racks().expect("ref n_tr"),
        "n_trainer_racks"
    );
    assert_eq!(
        prod.queue_depth().expect("prod depth"),
        refer.queue_depth().expect("ref depth"),
        "queue_depth"
    );
    let (ro, tr) = prod.split_gpus().expect("split");
    assert_eq!(
        ro.saturating_add(tr),
        total_gpus(racks),
        "split sums to total"
    );
    assert_eq!(
        prod.n_rollout_racks().unwrap() + prod.n_trainer_racks().unwrap(),
        racks.len(),
        "role counts cover the fleet"
    );
    for spec in racks {
        let id = &spec.id;
        assert_eq!(
            prod.rack_role(id).expect("prod role"),
            refer.rack_role(id).expect("ref role"),
            "role {id:?}"
        );
        match (prod.rack_version(id), refer.rack_version(id)) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "rack_version {id:?}"),
            (Err(e), Err(r)) => assert_err_eq(&e, &r),
            (Ok(v), Err(r)) => panic!("prod rack_version {id:?} = {v}, ref Err {r:?}"),
            (Err(e), Ok(v)) => panic!("prod rack_version {id:?} Err {e:?}, ref {v}"),
        }
    }
}

pub fn assert_floors_while_running(prod: &Coordinator, cfg: &CoordinatorConfig, n_racks: usize) {
    if prod.state().expect("state") != CoordinatorState::Running {
        return;
    }
    let nr = prod.n_rollout_racks().expect("n_rollout");
    let nt = prod.n_trainer_racks().expect("n_trainer");
    assert_eq!(nr + nt, n_racks, "roles partition the fleet");
    if n_racks >= cfg.min_rollout_racks.saturating_add(cfg.min_trainer_racks) {
        assert!(
            nr >= cfg.min_rollout_racks,
            "n_rollout_racks {nr} < min_rollout {}",
            cfg.min_rollout_racks
        );
        assert!(
            nt >= cfg.min_trainer_racks,
            "n_trainer_racks {nt} < min_trainer {}",
            cfg.min_trainer_racks
        );
    }
}

pub fn assert_initial_running(prod: &Coordinator, racks: &[RackSpec]) {
    assert_eq!(prod.state().unwrap(), CoordinatorState::Running);
    assert!(!prod.halted().unwrap());
    assert_eq!(prod.trainer_version().unwrap(), 0);
    assert_eq!(prod.queue_depth().unwrap(), 0);
    for spec in racks {
        match prod.rack_role(&spec.id).unwrap() {
            RackRole::Rollout => {
                assert_eq!(prod.rack_version(&spec.id).unwrap(), 0);
            }
            RackRole::Trainer => match prod.rack_version(&spec.id) {
                Err(CoordError::NotRollout(id)) => assert_eq!(id, spec.id),
                other => panic!("trainer rack_version expected NotRollout, got {other:?}"),
            },
        }
    }
}
