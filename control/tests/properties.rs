//! Group: property tests. Random small replica counts, fail/heal/rejoin
//! sequences, tokens invariant, match the independent reference.

mod common;
mod reference;

use common::*;
use prometheus_control::{Controller, HealthKind, ReplicaState, ShardId};
use reference::RefController;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn bounded(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        self.next() % n
    }
}

fn cfg_from_rng(rng: &mut Lcg) -> prometheus_control::ElasticConfig {
    let tokens = [64u64, 100, 256, 1024][rng.bounded(4) as usize];
    let micro = [8u64, 16, 32, 64][rng.bounded(4) as usize];
    prometheus_control::ElasticConfig {
        tokens_per_step: tokens,
        microbatch_tokens: micro,
        sdc_period_steps: [0u64, 5, 10][rng.bounded(3) as usize],
        straggler_timeout_ms: 100,
        collective_watchdog_ms: 50,
        page_window_steps: [3u64, 10, 10_000][rng.bounded(3) as usize],
    }
}

fn apply_op(
    rng: &mut Lcg,
    prod: &mut Controller,
    refer: &mut RefController,
    cfg: &prometheus_control::ElasticConfig,
    step: &mut u64,
) {
    let live = prod.live_replicas().expect("live");
    let n_live = live.len();
    let ids = refer.ids();
    let op = rng.bounded(8);
    match op {
        0 if n_live > 1 => {
            let i = rng.bounded(n_live as u64) as usize;
            let id = live[i].clone();
            match (prod.fail_replica(&id), refer.fail_replica(&id)) {
                (Ok(()), Ok(())) => {}
                (Err(e), Err(r)) => assert_err_eq(&e, &r),
                other => panic!("fail mismatch {other:?}"),
            }
        }
        1 => {
            // fail a random known id (may be spare / dead / live)
            let i = rng.bounded(ids.len() as u64) as usize;
            let id = ids[i].clone();
            match (prod.fail_replica(&id), refer.fail_replica(&id)) {
                (Ok(()), Ok(())) => {}
                (Err(e), Err(r)) => assert_err_eq(&e, &r),
                other => panic!("fail-any mismatch {other:?}"),
            }
        }
        2 => {
            let spare = ids
                .iter()
                .find(|id| prod.replica_state(id).ok() == Some(ReplicaState::Spare));
            if let (Some(sp), Some(src)) = (spare, live.first()) {
                match (prod.heal_spare(sp, src), refer.heal_spare(sp, src)) {
                    (Ok(()), Ok(())) => {}
                    (Err(e), Err(r)) => assert_err_eq(&e, &r),
                    other => panic!("heal mismatch {other:?}"),
                }
            }
        }
        3 => {
            let healing = ids
                .iter()
                .find(|id| prod.replica_state(id).ok() == Some(ReplicaState::Healing));
            if let Some(h) = healing {
                *step += 1;
                match (prod.rejoin(h, *step), refer.rejoin(h, *step)) {
                    (Ok(()), Ok(())) => {}
                    (Err(e), Err(r)) => assert_err_eq(&e, &r),
                    other => panic!("rejoin mismatch {other:?}"),
                }
            }
        }
        4 if n_live > 1 => {
            let i = rng.bounded(n_live as u64) as usize;
            let id = live[i].clone();
            let over = rng.bounded(2) == 0;
            let dur = if over {
                cfg.straggler_timeout_ms + 1
            } else {
                cfg.straggler_timeout_ms
            };
            *step += 1;
            match (
                prod.observe_step_time(&id, *step, dur),
                refer.observe_step_time(&id, *step, dur),
            ) {
                (Ok(()), Ok(())) => {}
                (Err(e), Err(r)) => assert_err_eq(&e, &r),
                other => panic!("straggler mismatch {other:?}"),
            }
        }
        5 if n_live >= 1 => {
            let kinds = [
                HealthKind::Xid,
                HealthKind::Ecc,
                HealthKind::Nvlink,
                HealthKind::NicFlap,
                HealthKind::Thermal,
                HealthKind::WatchdogTimeout,
            ];
            let kind = kinds[rng.bounded(6) as usize];
            let i = rng.bounded(n_live as u64) as usize;
            let ev = health(&live[i].0, kind, 7);
            match (prod.report_health(ev.clone()), refer.report_health(ev)) {
                (Ok(()), Ok(())) => {}
                (Err(e), Err(r)) => assert_err_eq(&e, &r),
                other => panic!("health mismatch {other:?}"),
            }
        }
        6 => {
            *step += 1;
            let sh = ShardId(format!("d{}", rng.bounded(4)));
            let ck = format!("ckpt-{}", rng.bounded(9));
            match (
                prod.on_loss_spike(*step, sh.clone(), ck.clone()),
                refer.on_loss_spike(*step, sh, ck),
            ) {
                (Ok(a), Ok(b)) => assert_eq!(a, b, "spike report"),
                (Err(e), Err(r)) => assert_err_eq(&e, &r),
                other => panic!("spike mismatch {other:?}"),
            }
        }
        _ if n_live >= 2 && cfg.sdc_period_steps > 0 => {
            let period = cfg.sdc_period_steps;
            let check_step = period; // always a check
            let mismatch = rng.bounded(3) == 0;
            for (k, id) in live.iter().enumerate() {
                let hex = if mismatch && k + 1 == live.len() {
                    "bad"
                } else {
                    "ok"
                };
                match (
                    prod.report_shard_hash(id, "w", hex, check_step),
                    refer.report_shard_hash(id, "w", hex, check_step),
                ) {
                    (Ok(()), Ok(())) => {}
                    (Err(e), Err(r)) => {
                        assert_err_eq(&e, &r);
                        return;
                    }
                    other => panic!("report hash mismatch {other:?}"),
                }
            }
            match (prod.check_sdc(check_step), refer.check_sdc(check_step)) {
                (Ok(()), Ok(())) => {}
                (Err(e), Err(r)) => assert_err_eq(&e, &r),
                other => panic!("check_sdc mismatch {other:?}"),
            }
        }
        _ => {
            // collective watchdog on a live replica when we can spare it
            if n_live > 1 {
                let id = live[0].clone();
                let _ = prod.begin_collective("p", &id, 0);
                let _ = refer.begin_collective("p", &id, 0);
                match (
                    prod.tick(cfg.collective_watchdog_ms + 1),
                    refer.tick(cfg.collective_watchdog_ms + 1),
                ) {
                    (Ok(()), Ok(())) => {}
                    (Err(e), Err(r)) => assert_err_eq(&e, &r),
                    other => panic!("tick mismatch {other:?}"),
                }
            }
        }
    }
}

#[test]
fn random_fail_heal_rejoin_tokens_invariant_matches_reference() {
    for trial in 0..16u64 {
        let mut rng = Lcg(0xA6A6_0000 + trial * 17);
        let n_live = 2 + rng.bounded(7) as usize; // 2..=8
        let n_spare = rng.bounded(3) as usize; // 0..=2
        let cfg = cfg_from_rng(&mut rng);
        let specs = n_live_m_spare(n_live, n_spare);
        let mut prod = Controller::new(cfg.clone(), specs.clone()).expect("new");
        let mut refer = RefController::new(cfg.clone(), specs).expect("ref new");
        assert_prod_matches_ref(&prod, &refer);
        assert_ctrl_tokens(&prod, &cfg);
        let mut step = 0u64;
        for _ in 0..24 {
            apply_op(&mut rng, &mut prod, &mut refer, &cfg, &mut step);
            assert_prod_matches_ref(&prod, &refer);
            let n = prod.n_live().expect("n_live");
            assert!(n >= 1, "last-live protection: n_live never 0");
            assert_ctrl_tokens(&prod, &cfg);
        }
    }
}

#[test]
fn tokens_invariant_after_each_fail_until_one_left() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg.clone(), n_live_m_spare(8, 2));
    for i in 0..7 {
        let id = rid(&format!("live-{i}"));
        assert_both_ok(prod.fail_replica(&id), refer.fail_replica(&id));
        assert_ctrl_tokens(&prod, &cfg);
        assert_prod_matches_ref(&prod, &refer);
    }
    let err = assert_both_err(
        prod.fail_replica(&rid("live-7")),
        refer.fail_replica(&rid("live-7")),
    );
    assert_no_live_replicas(&err);
    assert_eq!(prod.n_live().unwrap(), 1);
    assert_ctrl_tokens(&prod, &cfg);
}

#[test]
fn heal_rejoin_restores_minimum_accum() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg.clone(), n_live_m_spare(2, 2));
    let start_acc = prod.grad_accumulation().unwrap();
    assert_both_ok(
        prod.fail_replica(&rid("live-1")),
        refer.fail_replica(&rid("live-1")),
    );
    let mid = prod.grad_accumulation().unwrap();
    assert!(mid > start_acc);
    assert_both_ok(
        prod.heal_spare(&rid("spare-0"), &rid("live-0")),
        refer.heal_spare(&rid("spare-0"), &rid("live-0")),
    );
    assert_eq!(prod.grad_accumulation().unwrap(), mid);
    assert_both_ok(
        prod.rejoin(&rid("spare-0"), 3),
        refer.rejoin(&rid("spare-0"), 3),
    );
    assert_eq!(prod.grad_accumulation().unwrap(), start_acc);
    assert_ctrl_tokens(&prod, &cfg);
    assert_prod_matches_ref(&prod, &refer);
}
