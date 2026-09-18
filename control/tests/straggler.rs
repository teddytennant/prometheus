//! Group: `observe_step_time` straggler drain.

mod common;
mod reference;

use common::*;
use prometheus_control::ReplicaState;

#[test]
fn duration_above_timeout_drains_like_fail_replica() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg.clone(), n_live_m_spare(3, 0));
    assert_both_ok(
        prod.observe_step_time(&rid("live-1"), 7, cfg.straggler_timeout_ms + 1),
        refer.observe_step_time(&rid("live-1"), 7, cfg.straggler_timeout_ms + 1),
    );
    assert_eq!(
        prod.replica_state(&rid("live-1")).unwrap(),
        ReplicaState::Dead
    );
    assert_eq!(prod.n_live().unwrap(), 2);
    assert_eq!(prod.grad_accumulation().unwrap(), 4);
    assert_ctrl_tokens(&prod, &cfg);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn duration_equal_timeout_does_not_drain() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg.clone(), n_live_m_spare(3, 0));
    assert_both_ok(
        prod.observe_step_time(&rid("live-1"), 7, cfg.straggler_timeout_ms),
        refer.observe_step_time(&rid("live-1"), 7, cfg.straggler_timeout_ms),
    );
    assert_eq!(
        prod.replica_state(&rid("live-1")).unwrap(),
        ReplicaState::Live
    );
    assert_eq!(prod.n_live().unwrap(), 3);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn duration_below_timeout_does_not_drain() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg, n_live_m_spare(3, 0));
    assert_both_ok(
        prod.observe_step_time(&rid("live-0"), 1, 0),
        refer.observe_step_time(&rid("live-0"), 1, 0),
    );
    assert_eq!(prod.n_live().unwrap(), 3);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn observe_unknown_is_replica_not_found() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    let id = rid("missing");
    let err = assert_both_err(
        prod.observe_step_time(&id, 1, 9_999),
        refer.observe_step_time(&id, 1, 9_999),
    );
    assert_replica_not_found(&err, &id);
}

#[test]
fn observe_spare_is_not_live() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg.clone(), n_live_m_spare(2, 1));
    let err = assert_both_err(
        prod.observe_step_time(&rid("spare-0"), 1, cfg.straggler_timeout_ms + 1),
        refer.observe_step_time(&rid("spare-0"), 1, cfg.straggler_timeout_ms + 1),
    );
    assert_not_live(&err, &rid("spare-0"));
    assert_eq!(
        prod.replica_state(&rid("spare-0")).unwrap(),
        ReplicaState::Spare
    );
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn straggler_of_last_live_is_no_live_replicas_no_change() {
    let cfg = default_config();
    let specs = n_live_m_spare(1, 0);
    let ids = spec_ids(&specs);
    let (mut prod, mut refer) = pair(cfg.clone(), specs);
    let before = membership(&prod, &ids);
    let err = assert_both_err(
        prod.observe_step_time(&rid("live-0"), 3, cfg.straggler_timeout_ms + 1),
        refer.observe_step_time(&rid("live-0"), 3, cfg.straggler_timeout_ms + 1),
    );
    assert_no_live_replicas(&err);
    assert_eq!(membership(&prod, &ids), before);
    assert_prod_matches_ref(&prod, &refer);
}
