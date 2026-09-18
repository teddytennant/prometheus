//! Group: collective begin / end / watchdog `tick`.

mod common;
mod reference;

use common::*;
use prometheus_control::ReplicaState;

#[test]
fn tick_past_watchdog_without_end_drains() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg.clone(), n_live_m_spare(3, 0));
    let begin = 100u64;
    assert_both_ok(
        prod.begin_collective("allreduce", &rid("live-1"), begin),
        refer.begin_collective("allreduce", &rid("live-1"), begin),
    );
    let past = begin + cfg.collective_watchdog_ms + 1;
    assert_both_ok(prod.tick(past), refer.tick(past));
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
fn end_before_deadline_leaves_live() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg.clone(), n_live_m_spare(3, 0));
    let begin = 100u64;
    assert_both_ok(
        prod.begin_collective("allreduce", &rid("live-1"), begin),
        refer.begin_collective("allreduce", &rid("live-1"), begin),
    );
    let end = begin + cfg.collective_watchdog_ms;
    assert_both_ok(
        prod.end_collective("allreduce", &rid("live-1"), end),
        refer.end_collective("allreduce", &rid("live-1"), end),
    );
    let past = begin + cfg.collective_watchdog_ms + 50;
    assert_both_ok(prod.tick(past), refer.tick(past));
    assert_eq!(
        prod.replica_state(&rid("live-1")).unwrap(),
        ReplicaState::Live
    );
    assert_eq!(prod.n_live().unwrap(), 3);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn tick_with_no_open_collective_is_ok() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    assert_both_ok(prod.tick(1_000_000), refer.tick(1_000_000));
    assert_eq!(prod.n_live().unwrap(), 2);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn elapsed_equal_watchdog_does_not_drain() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg.clone(), n_live_m_spare(3, 0));
    let begin = 10u64;
    assert_both_ok(
        prod.begin_collective("b", &rid("live-0"), begin),
        refer.begin_collective("b", &rid("live-0"), begin),
    );
    let eq = begin + cfg.collective_watchdog_ms;
    assert_both_ok(prod.tick(eq), refer.tick(eq));
    assert_eq!(
        prod.replica_state(&rid("live-0")).unwrap(),
        ReplicaState::Live
    );
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn begin_unknown_is_not_found() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    let id = rid("ghost");
    let err = assert_both_err(
        prod.begin_collective("x", &id, 0),
        refer.begin_collective("x", &id, 0),
    );
    assert_replica_not_found(&err, &id);
}

#[test]
fn begin_spare_is_not_live() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 1));
    let err = assert_both_err(
        prod.begin_collective("x", &rid("spare-0"), 0),
        refer.begin_collective("x", &rid("spare-0"), 0),
    );
    assert_not_live(&err, &rid("spare-0"));
}

#[test]
fn watchdog_of_last_live_is_no_live_replicas_no_change() {
    let cfg = default_config();
    let specs = n_live_m_spare(1, 0);
    let ids = spec_ids(&specs);
    let (mut prod, mut refer) = pair(cfg.clone(), specs);
    assert_both_ok(
        prod.begin_collective("x", &rid("live-0"), 0),
        refer.begin_collective("x", &rid("live-0"), 0),
    );
    let before = membership(&prod, &ids);
    let err = assert_both_err(
        prod.tick(cfg.collective_watchdog_ms + 1),
        refer.tick(cfg.collective_watchdog_ms + 1),
    );
    assert_no_live_replicas(&err);
    assert_eq!(membership(&prod, &ids), before);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn unmatched_end_collective_is_ok() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    assert_both_ok(
        prod.end_collective("never-began", &rid("live-0"), 5),
        refer.end_collective("never-began", &rid("live-0"), 5),
    );
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn duplicate_begin_is_message() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    assert_both_ok(
        prod.begin_collective("x", &rid("live-0"), 0),
        refer.begin_collective("x", &rid("live-0"), 0),
    );
    let err = assert_both_err(
        prod.begin_collective("x", &rid("live-0"), 1),
        refer.begin_collective("x", &rid("live-0"), 1),
    );
    match err {
        prometheus_control::ControlError::Message(s) => {
            let l = s.to_lowercase();
            assert!(l.contains("collective") || l.contains("open"), "got {s:?}");
        }
        other => panic!("expected Message, got {other:?}"),
    }
}

#[test]
fn one_replica_times_out_other_stays_live() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg.clone(), n_live_m_spare(2, 0));
    assert_both_ok(
        prod.begin_collective("x", &rid("live-0"), 0),
        refer.begin_collective("x", &rid("live-0"), 0),
    );
    assert_both_ok(
        prod.begin_collective("x", &rid("live-1"), 0),
        refer.begin_collective("x", &rid("live-1"), 0),
    );
    assert_both_ok(
        prod.end_collective("x", &rid("live-0"), 10),
        refer.end_collective("x", &rid("live-0"), 10),
    );
    assert_both_ok(
        prod.tick(cfg.collective_watchdog_ms + 1),
        refer.tick(cfg.collective_watchdog_ms + 1),
    );
    assert_eq!(
        prod.replica_state(&rid("live-0")).unwrap(),
        ReplicaState::Live
    );
    assert_eq!(
        prod.replica_state(&rid("live-1")).unwrap(),
        ReplicaState::Dead
    );
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn now_ms_before_begin_does_not_drain() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg, n_live_m_spare(2, 0));
    assert_both_ok(
        prod.begin_collective("x", &rid("live-0"), 1_000),
        refer.begin_collective("x", &rid("live-0"), 1_000),
    );
    assert_both_ok(prod.tick(0), refer.tick(0));
    assert_eq!(
        prod.replica_state(&rid("live-0")).unwrap(),
        ReplicaState::Live
    );
    assert_prod_matches_ref(&prod, &refer);
}
