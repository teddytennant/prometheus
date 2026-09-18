//! Group 8: property tests. After every successful public call while
//! `Running`: n_rollout >= min_rollout, n_trainer >= min_trainer,
//! split_gpus sums to total, queue_depth matches remaining batches.
//! Production and reference are driven in lockstep.

mod common;
mod reference;

use common::*;
use prometheus_coordinator::{
    Coordinator, CoordinatorState, PolicyVersion, RackRole, RackSpec, RolloutBatch,
};
use reference::RefCoordinator;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn bounded(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next() % n
        }
    }
}

struct Shadow {
    batches: Vec<RolloutBatch>,
}

impl Shadow {
    fn new() -> Self {
        Self {
            batches: Vec::new(),
        }
    }
    fn depth(&self) -> usize {
        self.batches.len()
    }
}

fn cfg_from_rng(rng: &mut Lcg, n_racks: usize) -> prometheus_coordinator::CoordinatorConfig {
    let mut cfg = default_config();
    cfg.max_staleness = rng.bounded(5); // 0..=4
    cfg.rollout_denom = 1 + rng.bounded(20);
    cfg.rollout_numer = rng.bounded(cfg.rollout_denom + 1); // 0..=denom
    cfg.parity_threshold = 1.0;
    cfg.queue_high = rng.bounded(6) as usize; // 0..=5
    let min_sum = 1 + rng.bounded(n_racks as u64) as usize; // 1..=n
    cfg.min_rollout_racks = rng.bounded((min_sum + 1) as u64) as usize;
    cfg.min_trainer_racks = min_sum - cfg.min_rollout_racks;
    // Guarantee construction: mins sum <= n_racks. min_sum <= n already.
    if cfg.min_rollout_racks + cfg.min_trainer_racks > n_racks {
        cfg.min_trainer_racks = 0;
        cfg.min_rollout_racks = 0;
    }
    cfg
}

fn racks_from_rng(rng: &mut Lcg, n: usize) -> Vec<RackSpec> {
    (0..n)
        .map(|i| rack(&format!("r{i:02}"), 1 + rng.bounded(8)))
        .collect()
}

fn assert_invariants(
    prod: &Coordinator,
    refer: &RefCoordinator,
    cfg: &prometheus_coordinator::CoordinatorConfig,
    racks: &[RackSpec],
    shadow: &Shadow,
) {
    assert_prod_matches_ref(prod, refer, racks);
    if prod.state().unwrap() != CoordinatorState::Running {
        return;
    }
    assert_floors_while_running(prod, cfg, racks.len());
    let (ro, tr) = prod.split_gpus().unwrap();
    assert_eq!(ro + tr, total_gpus(racks));
    assert_eq!(prod.queue_depth().unwrap(), shadow.depth());
    assert_eq!(refer.queue_depth().unwrap(), shadow.depth());
}

fn apply_op(
    rng: &mut Lcg,
    prod: &mut Coordinator,
    refer: &mut RefCoordinator,
    racks: &[RackSpec],
    shadow: &mut Shadow,
    next_id: &mut u64,
    trainer_hint: PolicyVersion,
) {
    match rng.bounded(8) {
        0 => {
            // Enqueue a unique batch at a non-future version.
            let id = format!("b{}", *next_id);
            *next_id += 1;
            let stale = rng.bounded(trainer_hint.saturating_add(1).max(1));
            let ver = trainer_hint.saturating_sub(stale);
            let n = 1 + rng.bounded(20) as u32;
            let routing = rng.bounded(2) == 0;
            let b = batch(&id, "prompt", ver, n, routing);
            match (
                prod.enqueue_batch(b.clone()),
                refer.enqueue_batch(b.clone()),
            ) {
                (Ok(()), Ok(())) => shadow.batches.push(b),
                (Err(e), Err(r)) => assert_err_eq(&e, &r),
                other => panic!("enqueue mismatch {other:?}"),
            }
        }
        1 => match (prod.consume_batch(), refer.consume_batch()) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a, b, "consume");
                match a {
                    None => shadow.batches.clear(),
                    Some(got) => {
                        // Drop prefix until we find `got` (stale prefix).
                        let pos = shadow
                            .batches
                            .iter()
                            .position(|x| x.id == got.id)
                            .expect("consumed id must have been queued");
                        shadow.batches.drain(0..=pos);
                    }
                }
            }
            (Err(e), Err(r)) => assert_err_eq(&e, &r),
            other => panic!("consume mismatch {other:?}"),
        },
        2 => {
            let next = prod.trainer_version().unwrap().saturating_add(1);
            match (prod.publish_weights(next), refer.publish_weights(next)) {
                (Ok(()), Ok(())) => {}
                (Err(e), Err(r)) => assert_err_eq(&e, &r),
                other => panic!("publish mismatch {other:?}"),
            }
        }
        3 => {
            // Sometimes publish a wrong version.
            let wrong = rng.bounded(8);
            match (prod.publish_weights(wrong), refer.publish_weights(wrong)) {
                (Ok(()), Ok(())) => {}
                (Err(e), Err(r)) => assert_err_eq(&e, &r),
                other => panic!("publish-wrong mismatch {other:?}"),
            }
        }
        4 => {
            let tv = prod.trainer_version().unwrap();
            let i = rng.bounded(racks.len() as u64) as usize;
            let id = &racks[i].id;
            match (prod.sync_rack(id, tv), refer.sync_rack(id, tv)) {
                (Ok(()), Ok(())) => {}
                (Err(e), Err(r)) => assert_err_eq(&e, &r),
                other => panic!("sync mismatch {other:?}"),
            }
        }
        5 => match (prod.rebalance(), refer.rebalance()) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "rebalance"),
            (Err(e), Err(r)) => assert_err_eq(&e, &r),
            other => panic!("rebalance mismatch {other:?}"),
        },
        6 => {
            // Matching logps — must not halt.
            let samples = [sample(-1.25, -1.25), sample(0.0, 0.0)];
            let v = prod.trainer_version().unwrap();
            match (prod.run_parity(v, &samples), refer.run_parity(v, &samples)) {
                (Ok(a), Ok(b)) => assert_eq!(a, b),
                (Err(e), Err(r)) => assert_err_eq(&e, &r),
                other => panic!("parity mismatch {other:?}"),
            }
        }
        _ => {
            // Empty group / duplicate enqueue — error paths.
            let b = batch("dup-empty", "p", 0, 0, false);
            match (prod.enqueue_batch(b.clone()), refer.enqueue_batch(b)) {
                (Ok(()), Ok(())) => panic!("EmptyGroup must err"),
                (Err(e), Err(r)) => assert_err_eq(&e, &r),
                other => panic!("empty-group mismatch {other:?}"),
            }
        }
    }
}

#[test]
fn random_ops_hold_floors_split_and_depth_and_match_reference() {
    for trial in 0..16u64 {
        let mut rng = Lcg(0x12C0_0D00 + trial * 19);
        let n = 2 + rng.bounded(5) as usize; // 2..=6
        let cfg = cfg_from_rng(&mut rng, n);
        let racks = racks_from_rng(&mut rng, n);
        let (mut prod, mut refer) = pair(cfg.clone(), racks.clone());
        let mut shadow = Shadow::new();
        assert_invariants(&prod, &refer, &cfg, &racks, &shadow);
        let mut next_id = 0u64;
        for _ in 0..32 {
            let tv = prod.trainer_version().unwrap();
            apply_op(
                &mut rng,
                &mut prod,
                &mut refer,
                &racks,
                &mut shadow,
                &mut next_id,
                tv,
            );
            assert_invariants(&prod, &refer, &cfg, &racks, &shadow);
            assert_eq!(prod.state().unwrap(), CoordinatorState::Running);
        }
    }
}

#[test]
fn floors_hold_after_each_empty_queue_rebalance_until_blocked() {
    let mut cfg = default_config();
    cfg.rollout_numer = 50;
    cfg.rollout_denom = 100;
    cfg.min_rollout_racks = 1;
    cfg.min_trainer_racks = 1;
    let racks = n_equal_racks(6, 4);
    let (mut prod, mut refer) = pair(cfg.clone(), racks.clone());
    for _ in 0..8 {
        let p = prod.rebalance();
        let r = refer.rebalance();
        match (p, r) {
            (Ok(a), Ok(b)) => assert_eq!(a, b),
            (Err(e), Err(re)) => assert_err_eq(&e, &re),
            other => panic!("{other:?}"),
        }
        assert_floors_while_running(&prod, &cfg, racks.len());
        let (ro, tr) = prod.split_gpus().unwrap();
        assert_eq!(ro + tr, total_gpus(&racks));
        assert_eq!(prod.queue_depth().unwrap(), 0);
        assert_prod_matches_ref(&prod, &refer, &racks);
    }
}

#[test]
fn split_always_covers_every_rack_gpu() {
    for n in 2..=5 {
        for gpus in 1..=4u64 {
            let cfg = default_config();
            let racks = n_equal_racks(n, gpus);
            let (prod, refer) = pair(cfg, racks.clone());
            let (ro, tr) = prod.split_gpus().unwrap();
            assert_eq!(ro + tr, n as u64 * gpus);
            assert_eq!(
                prod.n_rollout_racks().unwrap() + prod.n_trainer_racks().unwrap(),
                n
            );
            assert_prod_matches_ref(&prod, &refer, &racks);
            for spec in &racks {
                let role = prod.rack_role(&spec.id).unwrap();
                match role {
                    RackRole::Rollout => {
                        assert_eq!(prod.rack_version(&spec.id).unwrap(), 0);
                    }
                    RackRole::Trainer => {
                        assert!(prod.rack_version(&spec.id).is_err());
                    }
                }
            }
        }
    }
}
