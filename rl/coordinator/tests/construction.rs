//! Group 1: construction errors and the initial 65/35 split
//! (floors that overshoot the GPU target, unequal GPU counts, 100% numer==denom).

mod common;
mod reference;

use common::*;
use prometheus_coordinator::{CoordError, RackRole};

#[test]
fn new_rejects_empty_fleet() {
    let err = new_err_both(default_config(), vec![]);
    match err {
        CoordError::EmptyFleet => {}
        other => panic!("expected EmptyFleet, got {other:?}"),
    }
}

#[test]
fn new_rejects_duplicate_rack_id() {
    let racks = vec![rack("a", 8), rack("b", 8), rack("a", 4)];
    let err = new_err_both(default_config(), racks);
    match err {
        CoordError::DuplicateRack(id) => assert_eq!(id, rid("a")),
        other => panic!("expected DuplicateRack(a), got {other:?}"),
    }
}

#[test]
fn new_rejects_zero_gpu_rack() {
    let racks = vec![rack("a", 8), rack("b", 0), rack("c", 8)];
    let err = new_err_both(default_config(), racks);
    assert_message_contains(&err, "gpu");
}

#[test]
fn new_rejects_zero_gpu_before_not_enough_racks() {
    // One zero-gpu rack would also fail NotEnoughRacks (1 < 1+1), but gpus==0
    // is checked first.
    let err = new_err_both(default_config(), vec![rack("solo", 0)]);
    assert_message_contains(&err, "gpu");
}

#[test]
fn new_rejects_duplicate_before_zero_gpu() {
    let racks = vec![rack("x", 0), rack("x", 8)];
    let err = new_err_both(default_config(), racks);
    match err {
        CoordError::DuplicateRack(id) => assert_eq!(id, rid("x")),
        other => panic!("expected DuplicateRack, got {other:?}"),
    }
}

#[test]
fn new_rejects_zero_denom() {
    let mut cfg = default_config();
    cfg.rollout_denom = 0;
    let err = new_err_both(cfg, default_racks());
    match err {
        CoordError::BadSplit => {}
        other => panic!("expected BadSplit, got {other:?}"),
    }
}

#[test]
fn new_rejects_numer_greater_than_denom() {
    let mut cfg = default_config();
    cfg.rollout_numer = 101;
    cfg.rollout_denom = 100;
    let err = new_err_both(cfg, default_racks());
    match err {
        CoordError::BadSplit => {}
        other => panic!("expected BadSplit, got {other:?}"),
    }
}

#[test]
fn new_rejects_not_enough_racks() {
    let mut cfg = default_config();
    cfg.min_rollout_racks = 2;
    cfg.min_trainer_racks = 2;
    // 3 < 2+2
    let err = new_err_both(cfg, n_equal_racks(3, 8));
    match err {
        CoordError::NotEnoughRacks => {}
        other => panic!("expected NotEnoughRacks, got {other:?}"),
    }
}

#[test]
fn new_empty_fleet_not_bad_split() {
    let mut cfg = default_config();
    cfg.rollout_denom = 0;
    let err = new_err_both(cfg, vec![]);
    match err {
        CoordError::EmptyFleet => {}
        other => panic!("expected EmptyFleet before BadSplit, got {other:?}"),
    }
}

#[test]
fn initial_65_35_four_equal_racks_is_two_and_two() {
    // 4×8 = 32 GPUs, target = 32*65/100 = 20. Prefix: 8,16 <=20; 24>20 → 2+2.
    let cfg = default_config();
    let racks = default_racks();
    let (prod, refer) = pair(cfg, racks.clone());
    assert_initial_running(&prod, &racks);
    assert_eq!(prod.n_rollout_racks().unwrap(), 2);
    assert_eq!(prod.n_trainer_racks().unwrap(), 2);
    assert_eq!(prod.split_gpus().unwrap(), (16, 16));
    assert_eq!(prod.rack_role(&rid("r00")).unwrap(), RackRole::Rollout);
    assert_eq!(prod.rack_role(&rid("r01")).unwrap(), RackRole::Rollout);
    assert_eq!(prod.rack_role(&rid("r02")).unwrap(), RackRole::Trainer);
    assert_eq!(prod.rack_role(&rid("r03")).unwrap(), RackRole::Trainer);
    assert_eq!(prod.rack_version(&rid("r00")).unwrap(), 0);
    match prod.rack_version(&rid("r02")) {
        Err(CoordError::NotRollout(id)) => assert_eq!(id, rid("r02")),
        other => panic!("expected NotRollout, got {other:?}"),
    }
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn initial_assignment_sorts_by_lex_id_not_input_order() {
    let cfg = default_config();
    let racks = vec![
        rack("r03", 8),
        rack("r01", 8),
        rack("r02", 8),
        rack("r00", 8),
    ];
    let (prod, refer) = pair(cfg, racks.clone());
    assert_eq!(prod.rack_role(&rid("r00")).unwrap(), RackRole::Rollout);
    assert_eq!(prod.rack_role(&rid("r01")).unwrap(), RackRole::Rollout);
    assert_eq!(prod.rack_role(&rid("r02")).unwrap(), RackRole::Trainer);
    assert_eq!(prod.rack_role(&rid("r03")).unwrap(), RackRole::Trainer);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn min_rollout_floor_overshoots_gpu_target() {
    // 3×10 = 30, target = 30*65/100 = 19. First pass: 10<=19, 20>19 → 1 rollout.
    // min_rollout=2 forces a second rack even though 20 > 19.
    let mut cfg = default_config();
    cfg.min_rollout_racks = 2;
    cfg.min_trainer_racks = 1;
    let racks = n_equal_racks(3, 10);
    let (prod, refer) = pair(cfg, racks.clone());
    assert_eq!(prod.n_rollout_racks().unwrap(), 2);
    assert_eq!(prod.n_trainer_racks().unwrap(), 1);
    assert_eq!(prod.split_gpus().unwrap(), (20, 10));
    assert_eq!(prod.rack_role(&rid("r00")).unwrap(), RackRole::Rollout);
    assert_eq!(prod.rack_role(&rid("r01")).unwrap(), RackRole::Rollout);
    assert_eq!(prod.rack_role(&rid("r02")).unwrap(), RackRole::Trainer);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn unequal_gpu_counts_take_lex_prefix_under_target() {
    // sorted a:1, b:10, c:10, d:1. total=22, target=22*65/100=14.
    // 1<=14, 11<=14, 21>14 → rollout a+b = 11, trainer c+d = 11.
    let cfg = default_config();
    let racks = vec![rack("d", 1), rack("c", 10), rack("b", 10), rack("a", 1)];
    let (prod, refer) = pair(cfg, racks.clone());
    assert_eq!(prod.split_gpus().unwrap(), (11, 11));
    assert_eq!(prod.rack_role(&rid("a")).unwrap(), RackRole::Rollout);
    assert_eq!(prod.rack_role(&rid("b")).unwrap(), RackRole::Rollout);
    assert_eq!(prod.rack_role(&rid("c")).unwrap(), RackRole::Trainer);
    assert_eq!(prod.rack_role(&rid("d")).unwrap(), RackRole::Trainer);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn single_fat_lex_first_rack_exceeds_target_then_min_rollout_overshoots() {
    // a:100, b:1. total=101, target=65. 100<=65? no. min_rollout=1 converts a.
    let cfg = default_config();
    let racks = vec![rack("a", 100), rack("b", 1)];
    let (prod, refer) = pair(cfg, racks.clone());
    assert_eq!(prod.n_rollout_racks().unwrap(), 1);
    assert_eq!(prod.split_gpus().unwrap(), (100, 1));
    assert_eq!(prod.rack_role(&rid("a")).unwrap(), RackRole::Rollout);
    assert_eq!(prod.rack_role(&rid("b")).unwrap(), RackRole::Trainer);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn hundred_percent_min_trainer_zero_assigns_all_rollout() {
    let mut cfg = default_config();
    cfg.rollout_numer = 100;
    cfg.rollout_denom = 100;
    cfg.min_rollout_racks = 0;
    cfg.min_trainer_racks = 0;
    let racks = n_equal_racks(4, 8);
    let (prod, refer) = pair(cfg, racks.clone());
    assert_eq!(prod.n_rollout_racks().unwrap(), 4);
    assert_eq!(prod.n_trainer_racks().unwrap(), 0);
    assert_eq!(prod.split_gpus().unwrap(), (32, 0));
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn hundred_percent_min_trainer_one_keeps_last_lex_as_trainer() {
    // target = total, but remaining_after last = 0 < min_trainer=1.
    let mut cfg = default_config();
    cfg.rollout_numer = 100;
    cfg.rollout_denom = 100;
    cfg.min_rollout_racks = 1;
    cfg.min_trainer_racks = 1;
    let racks = n_equal_racks(4, 8);
    let (prod, refer) = pair(cfg, racks.clone());
    assert_eq!(prod.n_rollout_racks().unwrap(), 3);
    assert_eq!(prod.n_trainer_racks().unwrap(), 1);
    assert_eq!(prod.split_gpus().unwrap(), (24, 8));
    assert_eq!(prod.rack_role(&rid("r03")).unwrap(), RackRole::Trainer);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn zero_percent_min_rollout_one_overshoots_to_first_lex() {
    let mut cfg = default_config();
    cfg.rollout_numer = 0;
    cfg.rollout_denom = 100;
    cfg.min_rollout_racks = 1;
    cfg.min_trainer_racks = 1;
    let racks = n_equal_racks(3, 8);
    let (prod, refer) = pair(cfg, racks.clone());
    assert_eq!(prod.n_rollout_racks().unwrap(), 1);
    assert_eq!(prod.split_gpus().unwrap(), (8, 16));
    assert_eq!(prod.rack_role(&rid("r00")).unwrap(), RackRole::Rollout);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn min_trainer_caps_prefix_even_when_gpus_under_target() {
    // 4×1, 100%, min_trainer=2. target=4. Walk stops when remaining_after < 2
    // → 2 rollout / 2 trainer, not 4/0.
    let mut cfg = default_config();
    cfg.rollout_numer = 100;
    cfg.rollout_denom = 100;
    cfg.min_rollout_racks = 1;
    cfg.min_trainer_racks = 2;
    let racks = n_equal_racks(4, 1);
    let (prod, refer) = pair(cfg, racks.clone());
    assert_eq!(prod.n_rollout_racks().unwrap(), 2);
    assert_eq!(prod.n_trainer_racks().unwrap(), 2);
    assert_eq!(prod.split_gpus().unwrap(), (2, 2));
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn max_staleness_zero_is_allowed() {
    let mut cfg = default_config();
    cfg.max_staleness = 0;
    let racks = default_racks();
    let (prod, refer) = pair(cfg, racks.clone());
    assert_eq!(prod.trainer_version().unwrap(), 0);
    assert_initial_running(&prod, &racks);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn unknown_rack_role_is_not_found() {
    let racks = default_racks();
    let (prod, refer) = pair(default_config(), racks.clone());
    let id = rid("nope");
    let err = assert_both_err(prod.rack_role(&id), refer.rack_role(&id));
    match err {
        CoordError::RackNotFound(got) => assert_eq!(got, id),
        other => panic!("expected RackNotFound, got {other:?}"),
    }
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn five_unit_racks_65_35_is_three_rollout() {
    // 5×1 = 5, target = 5*65/100 = 3. Prefix of 3, remaining 2 >= min_trainer 1.
    let racks = n_equal_racks(5, 1);
    let (prod, refer) = pair(default_config(), racks.clone());
    assert_eq!(prod.split_gpus().unwrap(), (3, 2));
    assert_eq!(prod.n_rollout_racks().unwrap(), 3);
    assert_prod_matches_ref(&prod, &refer, &racks);
}
