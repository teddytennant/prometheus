//! Group: `Controller::new` and the initial live set / accum.

mod common;
mod reference;

use common::*;
use prometheus_control::ReplicaState;

#[test]
fn new_rejects_empty_replicas() {
    let err = new_err(default_config(), vec![]);
    assert_no_live_replicas_or_message(&err);
}

#[test]
fn new_rejects_all_spares() {
    let specs = vec![spec("s0", "r0", true), spec("s1", "r1", true)];
    let err = new_err(default_config(), specs);
    assert_no_live_replicas_or_message(&err);
}

#[test]
fn new_rejects_duplicate_replica_id() {
    let specs = vec![spec("a", "r0", false), spec("a", "r1", false)];
    let err = new_err(default_config(), specs);
    assert_duplicate_message(&err);
}

#[test]
fn new_rejects_duplicate_id_live_and_spare() {
    let specs = vec![spec("a", "r0", false), spec("a", "r1", true)];
    let err = new_err(default_config(), specs);
    assert_duplicate_message(&err);
}

#[test]
fn new_rejects_zero_microbatch_tokens() {
    let mut cfg = default_config();
    cfg.microbatch_tokens = 0;
    let err = new_err(cfg, n_live_m_spare(2, 0));
    match err {
        prometheus_control::ControlError::Message(s) => {
            assert!(s.to_lowercase().contains("microbatch"), "got {s:?}");
        }
        other => panic!("expected Message containing microbatch, got {other:?}"),
    }
}

#[test]
fn new_accepts_one_live_zero_spare() {
    let cfg = default_config();
    let specs = n_live_m_spare(1, 0);
    let (prod, refer) = pair(cfg.clone(), specs);
    assert_eq!(prod.n_live().unwrap(), 1);
    assert_eq!(prod.live_replicas().unwrap(), vec![rid("live-0")]);
    assert_eq!(
        prod.replica_state(&rid("live-0")).unwrap(),
        ReplicaState::Live
    );
    assert_eq!(prod.grad_accumulation().unwrap(), 8);
    assert_eq!(prod.effective_tokens_per_step().unwrap(), 1024);
    assert_ctrl_tokens(&prod, &cfg);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn initial_live_set_is_non_spares_in_spec_order() {
    let cfg = default_config();
    let specs = vec![
        spec("s0", "sr0", true),
        spec("a", "r0", false),
        spec("s1", "sr1", true),
        spec("b", "r1", false),
        spec("c", "r2", false),
    ];
    let ids = spec_ids(&specs);
    let (prod, refer) = pair(cfg.clone(), specs);
    assert_eq!(
        prod.live_replicas().unwrap(),
        vec![rid("a"), rid("b"), rid("c")]
    );
    assert_eq!(prod.n_live().unwrap(), 3);
    assert_eq!(prod.replica_state(&rid("s0")).unwrap(), ReplicaState::Spare);
    assert_eq!(prod.replica_state(&rid("s1")).unwrap(), ReplicaState::Spare);
    assert_eq!(prod.replica_state(&rid("a")).unwrap(), ReplicaState::Live);
    assert_eq!(prod.grad_accumulation().unwrap(), 3);
    assert_eq!(prod.effective_tokens_per_step().unwrap(), 1152);
    assert_ctrl_tokens(&prod, &cfg);
    let snap = membership(&prod, &ids);
    assert_eq!(snap.states[0], (rid("s0"), ReplicaState::Spare));
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn four_live_accum_is_two_and_effective_equals_product() {
    let cfg = default_config();
    let specs = n_live_m_spare(4, 2);
    let (prod, refer) = pair(cfg.clone(), specs);
    assert_eq!(prod.n_live().unwrap(), 4);
    assert_eq!(prod.grad_accumulation().unwrap(), 2);
    assert_eq!(prod.effective_tokens_per_step().unwrap(), 4 * 128 * 2);
    assert_eq!(
        prod.replica_state(&rid("spare-0")).unwrap(),
        ReplicaState::Spare
    );
    assert_eq!(
        prod.replica_state(&rid("spare-1")).unwrap(),
        ReplicaState::Spare
    );
    assert_ctrl_tokens(&prod, &cfg);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn replica_state_unknown_is_not_found() {
    let (prod, _) = pair(default_config(), n_live_m_spare(2, 0));
    let id = rid("nope");
    let err = prod.replica_state(&id).expect_err("unknown");
    assert_replica_not_found(&err, &id);
}

#[test]
fn tokens_per_step_zero_accum_zero() {
    let mut cfg = default_config();
    cfg.tokens_per_step = 0;
    let (prod, refer) = pair(cfg.clone(), n_live_m_spare(2, 1));
    assert_eq!(prod.grad_accumulation().unwrap(), 0);
    assert_eq!(prod.effective_tokens_per_step().unwrap(), 0);
    assert_ctrl_tokens(&prod, &cfg);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn uneven_tokens_uses_ceil_accum() {
    let mut cfg = default_config();
    cfg.tokens_per_step = 1000;
    cfg.microbatch_tokens = 128;
    let (prod, refer) = pair(cfg.clone(), n_live_m_spare(3, 0));
    // 3*128=384, ceil(1000/384)=3, effective=1152.
    assert_eq!(prod.grad_accumulation().unwrap(), 3);
    assert_eq!(prod.effective_tokens_per_step().unwrap(), 1152);
    assert_ctrl_tokens(&prod, &cfg);
    assert_prod_matches_ref(&prod, &refer);
}
