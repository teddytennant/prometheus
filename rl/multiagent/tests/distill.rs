//! Group 9: distill_back: win, tie, loss, NaN/inf InvalidReward.

mod common;
mod reference;

use common::*;
use prometheus_multiagent::{DistillPair, DistillTarget, Topology};

fn pair_of(task: &str, budget: u64, multi: f64, single: f64) -> DistillPair {
    DistillPair {
        task: tid(task),
        token_budget: budget,
        multi_reward: multi,
        single_reward: single,
    }
}

#[test]
fn distill_win() {
    let (prod, refer) = pair_default();
    let d = pair_of("task-w", 128, 1.0, 0.5);
    let p = prod.distill_back(d.clone()).unwrap();
    let r = refer.distill_back(d).unwrap();
    assert_eq!(p, r);
    assert_eq!(
        p,
        Some(DistillTarget {
            task: tid("task-w"),
            token_budget: 128,
        })
    );
}

#[test]
fn distill_tie_is_none() {
    let (prod, refer) = pair_default();
    let d = pair_of("task-t", 64, 1.0, 1.0);
    assert_eq!(prod.distill_back(d.clone()).unwrap(), None);
    assert_eq!(refer.distill_back(d).unwrap(), None);
}

#[test]
fn distill_loss_is_none() {
    let (prod, refer) = pair_default();
    let d = pair_of("task-l", 64, 0.5, 1.0);
    assert_eq!(prod.distill_back(d.clone()).unwrap(), None);
    assert_eq!(refer.distill_back(d).unwrap(), None);
}

#[test]
fn distill_negative_rewards_strict_greater() {
    let (prod, refer) = pair_default();
    let win = pair_of("t", 1, -1.0, -2.0);
    assert_eq!(
        prod.distill_back(win.clone()).unwrap(),
        Some(DistillTarget {
            task: tid("t"),
            token_budget: 1,
        })
    );
    assert_eq!(refer.distill_back(win).unwrap().unwrap().token_budget, 1);
    let loss = pair_of("t", 1, -2.0, -1.0);
    assert_eq!(prod.distill_back(loss.clone()).unwrap(), None);
    assert_eq!(refer.distill_back(loss).unwrap(), None);
}

#[test]
fn distill_signed_zero_tie() {
    let (prod, refer) = pair_default();
    let d = pair_of("t", 4, -0.0, 0.0);
    assert_eq!(prod.distill_back(d.clone()).unwrap(), None);
    assert_eq!(refer.distill_back(d).unwrap(), None);
}

#[test]
fn distill_nan_multi_is_invalid_reward() {
    let (prod, refer) = pair_default();
    let d = pair_of("t", 8, f64::NAN, 1.0);
    match assert_both_err(prod.distill_back(d.clone()), refer.distill_back(d)) {
        prometheus_multiagent::MultiError::InvalidReward => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn distill_nan_single_is_invalid_reward() {
    let (prod, refer) = pair_default();
    let d = pair_of("t", 8, 1.0, f64::NAN);
    match assert_both_err(prod.distill_back(d.clone()), refer.distill_back(d)) {
        prometheus_multiagent::MultiError::InvalidReward => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn distill_inf_multi_is_invalid_reward() {
    let (prod, refer) = pair_default();
    for multi in [f64::INFINITY, f64::NEG_INFINITY] {
        let d = pair_of("t", 8, multi, 0.0);
        match assert_both_err(prod.distill_back(d.clone()), refer.distill_back(d)) {
            prometheus_multiagent::MultiError::InvalidReward => {}
            other => panic!("multi={multi}: {other:?}"),
        }
    }
}

#[test]
fn distill_inf_single_is_invalid_reward() {
    let (prod, refer) = pair_default();
    for single in [f64::INFINITY, f64::NEG_INFINITY] {
        let d = pair_of("t", 8, 0.0, single);
        match assert_both_err(prod.distill_back(d.clone()), refer.distill_back(d)) {
            prometheus_multiagent::MultiError::InvalidReward => {}
            other => panic!("single={single}: {other:?}"),
        }
    }
}

#[test]
fn distill_both_nan_is_invalid_reward_multi_checked_first() {
    let (prod, refer) = pair_default();
    let d = pair_of("t", 8, f64::NAN, f64::NAN);
    match assert_both_err(prod.distill_back(d.clone()), refer.distill_back(d)) {
        prometheus_multiagent::MultiError::InvalidReward => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn distill_does_not_require_live_episode_and_does_not_mutate() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "live", Topology::Single, 0, 8),
    );
    prod.scratch_append(&eid("e"), &aid("solo"), "pad").unwrap();
    refer
        .scratch_append(&eid("e"), &aid("solo"), "pad")
        .unwrap();
    let d = pair_of("other-task", 999, 2.0, 1.0);
    let p = prod.distill_back(d.clone()).unwrap();
    let r = refer.distill_back(d).unwrap();
    assert_eq!(
        p,
        Some(DistillTarget {
            task: tid("other-task"),
            token_budget: 999,
        })
    );
    assert_eq!(p, r);
    assert_eq!(prod.scratch_read(&eid("e")).unwrap(), "pad");
    assert_eq!(refer.scratch_read(&eid("e")).unwrap(), "pad");
    assert_eq!(prod.tokens_used(&eid("e")).unwrap(), 0);
    assert_prod_matches_ref(&prod, &refer, &eid("e"));
}

#[test]
fn distill_does_not_look_up_episodes() {
    let (prod, refer) = pair_default();
    // No spawn at all.
    let d = pair_of("never-spawned", 1, 0.25, 0.0);
    assert_eq!(
        prod.distill_back(d.clone()).unwrap(),
        Some(DistillTarget {
            task: tid("never-spawned"),
            token_budget: 1,
        })
    );
    assert_eq!(
        refer.distill_back(d).unwrap().unwrap().task,
        tid("never-spawned")
    );
}

#[test]
fn distill_strictly_greater_tiny_delta() {
    let (prod, refer) = pair_default();
    let d = pair_of("t", 3, 1.0 + f64::EPSILON, 1.0);
    assert!(prod.distill_back(d.clone()).unwrap().is_some());
    assert!(refer.distill_back(d).unwrap().is_some());
}
