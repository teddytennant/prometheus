//! Group: `report_health` drains.

mod common;
mod reference;

use common::*;
use prometheus_control::{HealthKind, ReplicaState};

fn all_kinds() -> [HealthKind; 6] {
    [
        HealthKind::Xid,
        HealthKind::Ecc,
        HealthKind::Nvlink,
        HealthKind::NicFlap,
        HealthKind::Thermal,
        HealthKind::WatchdogTimeout,
    ]
}

#[test]
fn every_health_kind_drains_like_fail_replica() {
    for kind in all_kinds() {
        let cfg = default_config();
        let (mut prod, mut refer) = pair(cfg.clone(), n_live_m_spare(3, 0));
        assert_both_ok(
            prod.report_health(health("live-1", kind, 42)),
            refer.report_health(health("live-1", kind, 42)),
        );
        assert_eq!(
            prod.replica_state(&rid("live-1")).unwrap(),
            ReplicaState::Dead,
            "{kind:?}"
        );
        assert_eq!(prod.n_live().unwrap(), 2, "{kind:?}");
        assert_eq!(prod.grad_accumulation().unwrap(), 4, "{kind:?}");
        assert_ctrl_tokens(&prod, &cfg);
        assert_prod_matches_ref(&prod, &refer);
    }
}

#[test]
fn health_unknown_replica_is_not_found() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    let ev = health("ghost", HealthKind::Xid, 1);
    let err = assert_both_err(prod.report_health(ev.clone()), refer.report_health(ev));
    assert_replica_not_found(&err, &rid("ghost"));
}

#[test]
fn health_of_last_live_is_no_live_replicas_no_change() {
    let specs = n_live_m_spare(1, 0);
    let ids = spec_ids(&specs);
    let (mut prod, mut refer) = pair(default_config(), specs);
    let before = membership(&prod, &ids);
    let ev = health("live-0", HealthKind::Ecc, 9);
    let err = assert_both_err(prod.report_health(ev.clone()), refer.report_health(ev));
    assert_no_live_replicas(&err);
    assert_eq!(membership(&prod, &ids), before);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn health_of_spare_is_not_live() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 1));
    let ev = health("spare-0", HealthKind::Thermal, 3);
    let err = assert_both_err(prod.report_health(ev.clone()), refer.report_health(ev));
    assert_not_live(&err, &rid("spare-0"));
    assert_eq!(
        prod.replica_state(&rid("spare-0")).unwrap(),
        ReplicaState::Spare
    );
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn nic_flap_and_thermal_drain() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg.clone(), n_live_m_spare(4, 0));
    assert_both_ok(
        prod.report_health(health("live-0", HealthKind::NicFlap, 1)),
        refer.report_health(health("live-0", HealthKind::NicFlap, 1)),
    );
    assert_both_ok(
        prod.report_health(health("live-1", HealthKind::Thermal, 2)),
        refer.report_health(health("live-1", HealthKind::Thermal, 2)),
    );
    assert_eq!(
        prod.replica_state(&rid("live-0")).unwrap(),
        ReplicaState::Dead
    );
    assert_eq!(
        prod.replica_state(&rid("live-1")).unwrap(),
        ReplicaState::Dead
    );
    assert_eq!(prod.n_live().unwrap(), 2);
    assert_ctrl_tokens(&prod, &cfg);
    assert_prod_matches_ref(&prod, &refer);
}
