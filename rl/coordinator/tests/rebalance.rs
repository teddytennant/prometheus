//! Group 6: rebalance — empty queue prefers rollout; queue_high pressure
//! prefers trainer; queue_high==0 never uses rule 2; min floors block;
//! lex-smallest rack chosen; version reset on role flip.

mod common;
mod reference;

use common::*;
use prometheus_coordinator::{CoordError, RackRole};

fn reb(
    prod: &mut prometheus_coordinator::Coordinator,
    refer: &mut reference::RefCoordinator,
) -> prometheus_coordinator::Result<Option<(prometheus_coordinator::RackId, RackRole)>> {
    let p = prod.rebalance();
    let r = refer.rebalance();
    match (p, r) {
        (Ok(a), Ok(b)) => {
            assert_eq!(a, b, "rebalance result");
            Ok(a)
        }
        (Err(e), Err(re)) => {
            assert_err_eq(&e, &re);
            Err(e)
        }
        other => panic!("rebalance mismatch {other:?}"),
    }
}

#[test]
fn empty_queue_prefers_rollout_even_over_gpu_target() {
    // Default 4×8, 65/35 → 2 rollout (16 GPUs) vs target 20, actually UNDER
    // target. Use a 100% target with min_trainer=1 so initial is 3+1 and
    // already at/over target; empty queue still peels the last trainer if
    // min_trainer is 0.
    let mut cfg = default_config();
    cfg.rollout_numer = 50;
    cfg.rollout_denom = 100;
    cfg.min_rollout_racks = 1;
    cfg.min_trainer_racks = 1;
    // 4×8=32, target=16. Initial prefix: 8<=16, 16<=16, 24>16 → 2+2, at target.
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    assert_eq!(prod.split_gpus().unwrap(), (16, 16));
    assert_eq!(prod.queue_depth().unwrap(), 0);
    let moved = reb(&mut prod, &mut refer).unwrap().expect("rule 1 move");
    assert_eq!(moved.0, rid("r02"));
    assert_eq!(moved.1, RackRole::Rollout);
    assert_eq!(prod.rack_role(&rid("r02")).unwrap(), RackRole::Rollout);
    assert_eq!(prod.rack_version(&rid("r02")).unwrap(), 0);
    assert_eq!(prod.n_rollout_racks().unwrap(), 3);
    assert_eq!(prod.n_trainer_racks().unwrap(), 1);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn empty_queue_moves_lex_smallest_trainer() {
    let mut cfg = default_config();
    cfg.rollout_numer = 50;
    cfg.rollout_denom = 100;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    // trainers r02, r03 — lex-smallest is r02, not r03.
    let moved = reb(&mut prod, &mut refer).unwrap().unwrap();
    assert_eq!(moved.0, rid("r02"));
    assert_ne!(moved.0, rid("r03"));
    assert_eq!(prod.rack_role(&rid("r03")).unwrap(), RackRole::Trainer);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn queue_high_pressure_prefers_trainer_lex_smallest_rollout() {
    let mut cfg = default_config();
    cfg.queue_high = 2;
    cfg.rollout_numer = 50;
    cfg.rollout_denom = 100;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    assert_eq!(prod.n_rollout_racks().unwrap(), 2);
    assert_both_ok(
        prod.enqueue_batch(batch("a", "p", 0, 1, false)),
        refer.enqueue_batch(batch("a", "p", 0, 1, false)),
    );
    assert_both_ok(
        prod.enqueue_batch(batch("b", "p", 0, 1, false)),
        refer.enqueue_batch(batch("b", "p", 0, 1, false)),
    );
    assert_eq!(prod.queue_depth().unwrap(), 2);
    let moved = reb(&mut prod, &mut refer).unwrap().expect("rule 2");
    assert_eq!(moved.0, rid("r00"));
    assert_eq!(moved.1, RackRole::Trainer);
    assert_eq!(prod.rack_role(&rid("r00")).unwrap(), RackRole::Trainer);
    match prod.rack_version(&rid("r00")) {
        Err(CoordError::NotRollout(id)) => assert_eq!(id, rid("r00")),
        other => panic!("expected NotRollout after flip, got {other:?}"),
    }
    // Queue untouched.
    assert_eq!(prod.queue_depth().unwrap(), 2);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn queue_high_zero_never_uses_rule_two() {
    let mut cfg = default_config();
    cfg.queue_high = 0;
    cfg.rollout_numer = 50;
    cfg.rollout_denom = 100;
    // Initial 2+2 at target 16. Deep queue, rule 1 no (depth!=0), rule 2
    // skipped because queue_high==0, rule 3/4 no (equal target) → None.
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    for i in 0..5 {
        let id = format!("b{i}");
        assert_both_ok(
            prod.enqueue_batch(batch(&id, "p", 0, 1, false)),
            refer.enqueue_batch(batch(&id, "p", 0, 1, false)),
        );
    }
    match reb(&mut prod, &mut refer) {
        Ok(None) => {}
        other => panic!("queue_high==0 must not fire rule 2, got {other:?}"),
    }
    assert_eq!(prod.n_rollout_racks().unwrap(), 2);
    assert_eq!(prod.queue_depth().unwrap(), 5);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn min_floors_block_both_directions() {
    let mut cfg = default_config();
    cfg.min_rollout_racks = 1;
    cfg.min_trainer_racks = 1;
    let racks = n_equal_racks(2, 8);
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    assert_eq!(prod.n_rollout_racks().unwrap(), 1);
    assert_eq!(prod.n_trainer_racks().unwrap(), 1);
    match reb(&mut prod, &mut refer) {
        Ok(None) => {}
        other => panic!("floors should block, got {other:?}"),
    }
    // Even with empty queue (rule 1 wants rollout) min_trainer blocks.
    assert_eq!(prod.queue_depth().unwrap(), 0);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn queue_below_high_uses_gpu_target_rule_three() {
    // 4×8, 65/35, target 20, initial 16 rollout < 20. queue_depth 1 < high 8
    // → rule 3 moves trainer → rollout.
    let cfg = default_config();
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    assert_eq!(prod.split_gpus().unwrap(), (16, 16));
    assert_both_ok(
        prod.enqueue_batch(batch("only", "p", 0, 1, false)),
        refer.enqueue_batch(batch("only", "p", 0, 1, false)),
    );
    let moved = reb(&mut prod, &mut refer).unwrap().expect("rule 3");
    assert_eq!(moved.1, RackRole::Rollout);
    assert_eq!(moved.0, rid("r02"));
    assert_eq!(prod.split_gpus().unwrap(), (24, 8));
    assert_eq!(prod.queue_depth().unwrap(), 1);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn rule_four_when_over_target_and_queue_nonempty_below_high() {
    // Start 100% with min_trainer=1 → 3 rollout (24) vs target 32, still under.
    // Use numer=25/100, 4×8=32, target=8. Initial: 8<=8, 16>8 → 1 rollout
    // (8 GPUs) == target. Empty queue would rule-1 add more. So enqueue one
    // (below high) to skip rule 1, equal target → None. Then force overshoot
    // via min_rollout.
    let mut cfg = default_config();
    cfg.rollout_numer = 25;
    cfg.rollout_denom = 100;
    cfg.min_rollout_racks = 2; // first pass k=1, floor forces 2 → 16 GPUs > 8
    cfg.min_trainer_racks = 1;
    cfg.queue_high = 8;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    assert_eq!(prod.n_rollout_racks().unwrap(), 2);
    assert_eq!(prod.split_gpus().unwrap(), (16, 16)); // 16 > target 8
    assert_both_ok(
        prod.enqueue_batch(batch("q", "p", 0, 1, false)),
        refer.enqueue_batch(batch("q", "p", 0, 1, false)),
    );
    let moved = reb(&mut prod, &mut refer).unwrap().expect("rule 4");
    assert_eq!(moved.1, RackRole::Trainer);
    assert_eq!(moved.0, rid("r00")); // lex-smallest rollout
    assert_eq!(prod.n_rollout_racks().unwrap(), 1);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn version_resets_to_zero_on_flip_back_to_rollout() {
    let mut cfg = default_config();
    cfg.queue_high = 1;
    cfg.rollout_numer = 50;
    cfg.rollout_denom = 100;
    cfg.min_rollout_racks = 1;
    cfg.min_trainer_racks = 1;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    assert_both_ok(prod.publish_weights(1), refer.publish_weights(1));
    assert_both_ok(
        prod.sync_rack(&rid("r00"), 1),
        refer.sync_rack(&rid("r00"), 1),
    );
    assert_eq!(prod.rack_version(&rid("r00")).unwrap(), 1);

    // Pressure: move lex-smallest rollout (r00) to trainer.
    assert_both_ok(
        prod.enqueue_batch(batch("q", "p", 1, 1, false)),
        refer.enqueue_batch(batch("q", "p", 1, 1, false)),
    );
    let moved = reb(&mut prod, &mut refer).unwrap().unwrap();
    assert_eq!(moved.0, rid("r00"));
    assert_eq!(moved.1, RackRole::Trainer);
    match prod.rack_version(&rid("r00")) {
        Err(CoordError::NotRollout(_)) => {}
        other => panic!("{other:?}"),
    }

    // Drain queue so rule 1 can pull a trainer (lex-smallest trainer is now
    // r00, which we just moved, plus original r02,r03 — wait: original
    // trainers r02,r03 plus r00. Lex-smallest trainer is r00.
    let _ = prod.consume_batch().unwrap();
    let _ = refer.consume_batch().unwrap();
    assert_eq!(prod.queue_depth().unwrap(), 0);
    let moved = reb(&mut prod, &mut refer).unwrap().unwrap();
    assert_eq!(moved.0, rid("r00"));
    assert_eq!(moved.1, RackRole::Rollout);
    assert_eq!(prod.rack_version(&rid("r00")).unwrap(), 0);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn in_flight_queue_untouched_by_rebalance() {
    let mut cfg = default_config();
    cfg.rollout_numer = 50;
    cfg.rollout_denom = 100;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    assert_both_ok(
        prod.enqueue_batch(batch("keep", "prompt-x", 0, 4, true)),
        refer.enqueue_batch(batch("keep", "prompt-x", 0, 4, true)),
    );
    // Non-empty, at target, below high → None, still check consume.
    let _ = reb(&mut prod, &mut refer).unwrap();
    let got = prod.consume_batch().unwrap().unwrap();
    let got_r = refer.consume_batch().unwrap().unwrap();
    assert_eq!(got, got_r);
    assert_eq!(got.id, bid("keep"));
    assert_eq!(got.prompt_id, pid("prompt-x"));
    assert!(got.routing_present);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn empty_queue_over_target_oscillates_once_min_trainer_binds() {
    // Documented oscillation: rule 1 ignores GPU target.
    let mut cfg = default_config();
    cfg.rollout_numer = 50;
    cfg.rollout_denom = 100;
    cfg.min_rollout_racks = 1;
    cfg.min_trainer_racks = 1;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    // 2+2 at target 16. Rule 1 → 3+1 (24 > 16).
    let m1 = reb(&mut prod, &mut refer).unwrap().unwrap();
    assert_eq!(m1.1, RackRole::Rollout);
    assert_eq!(prod.n_trainer_racks().unwrap(), 1);
    // Rule 1 blocked (min_trainer). Rule 4: over target → rollout→trainer.
    let m2 = reb(&mut prod, &mut refer).unwrap().unwrap();
    assert_eq!(m2.1, RackRole::Trainer);
    assert_eq!(prod.n_rollout_racks().unwrap(), 2);
    // Back to 2+2; next call rule 1 again.
    let m3 = reb(&mut prod, &mut refer).unwrap().unwrap();
    assert_eq!(m3.1, RackRole::Rollout);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn rebalance_none_when_at_target_with_nonempty_queue_below_high() {
    let mut cfg = default_config();
    cfg.rollout_numer = 50;
    cfg.rollout_denom = 100;
    cfg.queue_high = 8;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    assert_eq!(prod.split_gpus().unwrap(), (16, 16));
    assert_both_ok(
        prod.enqueue_batch(batch("q", "p", 0, 1, false)),
        refer.enqueue_batch(batch("q", "p", 0, 1, false)),
    );
    match reb(&mut prod, &mut refer) {
        Ok(None) => {}
        other => panic!("expected None at target, got {other:?}"),
    }
    assert_prod_matches_ref(&prod, &refer, &racks);
}
