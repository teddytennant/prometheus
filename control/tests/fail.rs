//! Group: `fail_replica` membership and last-live protection.

mod common;
mod reference;

use common::*;
use prometheus_control::ReplicaState;

#[test]
fn fail_live_sets_dead_drops_n_live_raises_accum() {
    let cfg = default_config();
    let specs = n_live_m_spare(4, 1);
    let ids = spec_ids(&specs);
    let (mut prod, mut refer) = pair(cfg.clone(), specs);
    assert_both_ok(
        prod.fail_replica(&rid("live-1")),
        refer.fail_replica(&rid("live-1")),
    );
    assert_eq!(
        prod.replica_state(&rid("live-1")).unwrap(),
        ReplicaState::Dead
    );
    assert_eq!(prod.n_live().unwrap(), 3);
    assert_eq!(
        prod.live_replicas().unwrap(),
        vec![rid("live-0"), rid("live-2"), rid("live-3")]
    );
    assert_eq!(prod.grad_accumulation().unwrap(), 3);
    assert_eq!(prod.effective_tokens_per_step().unwrap(), 1152);
    assert_eq!(
        prod.replica_state(&rid("spare-0")).unwrap(),
        ReplicaState::Spare
    );
    assert_ctrl_tokens(&prod, &cfg);
    assert_prod_matches_ref(&prod, &refer);
    let snap = membership(&prod, &ids);
    assert_eq!(snap.pages, 0);
    assert!(snap.skipped.is_empty());
}

#[test]
fn fail_last_live_is_no_live_replicas_and_does_not_change_state() {
    let cfg = default_config();
    let specs = n_live_m_spare(1, 1);
    let ids = spec_ids(&specs);
    let (mut prod, mut refer) = pair(cfg, specs);
    let before = membership(&prod, &ids);
    let err = assert_both_err(
        prod.fail_replica(&rid("live-0")),
        refer.fail_replica(&rid("live-0")),
    );
    assert_no_live_replicas(&err);
    assert_eq!(membership(&prod, &ids), before);
    assert_eq!(
        prod.replica_state(&rid("live-0")).unwrap(),
        ReplicaState::Live
    );
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn fail_unknown_is_replica_not_found() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    let id = rid("ghost");
    let err = assert_both_err(prod.fail_replica(&id), refer.fail_replica(&id));
    assert_replica_not_found(&err, &id);
}

#[test]
fn fail_spare_is_not_live() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 1));
    let id = rid("spare-0");
    let err = assert_both_err(prod.fail_replica(&id), refer.fail_replica(&id));
    assert_not_live(&err, &id);
    assert_eq!(prod.replica_state(&id).unwrap(), ReplicaState::Spare);
    assert_eq!(prod.n_live().unwrap(), 2);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn fail_already_dead_is_not_live() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(3, 0));
    assert_both_ok(
        prod.fail_replica(&rid("live-0")),
        refer.fail_replica(&rid("live-0")),
    );
    let err = assert_both_err(
        prod.fail_replica(&rid("live-0")),
        refer.fail_replica(&rid("live-0")),
    );
    assert_not_live(&err, &rid("live-0"));
    assert_eq!(prod.n_live().unwrap(), 2);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn two_fails_raise_accum_to_hold_tokens() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg.clone(), n_live_m_spare(4, 0));
    assert_both_ok(
        prod.fail_replica(&rid("live-0")),
        refer.fail_replica(&rid("live-0")),
    );
    assert_eq!(prod.n_live().unwrap(), 3);
    assert_eq!(prod.grad_accumulation().unwrap(), 3);
    assert_both_ok(
        prod.fail_replica(&rid("live-1")),
        refer.fail_replica(&rid("live-1")),
    );
    assert_eq!(prod.n_live().unwrap(), 2);
    assert_eq!(prod.grad_accumulation().unwrap(), 4);
    assert_eq!(prod.effective_tokens_per_step().unwrap(), 1024);
    assert_ctrl_tokens(&prod, &cfg);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn fail_healing_is_not_live() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 1));
    assert_both_ok(
        prod.heal_spare(&rid("spare-0"), &rid("live-0")),
        refer.heal_spare(&rid("spare-0"), &rid("live-0")),
    );
    let err = assert_both_err(
        prod.fail_replica(&rid("spare-0")),
        refer.fail_replica(&rid("spare-0")),
    );
    assert_not_live(&err, &rid("spare-0"));
    assert_eq!(
        prod.replica_state(&rid("spare-0")).unwrap(),
        ReplicaState::Healing
    );
    assert_prod_matches_ref(&prod, &refer);
}
