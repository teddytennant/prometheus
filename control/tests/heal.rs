//! Group: `heal_spare` / `rejoin`.

mod common;
mod reference;

use common::*;
use prometheus_control::ReplicaState;

#[test]
fn heal_spare_sets_healing_n_live_unchanged() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg.clone(), n_live_m_spare(2, 1));
    assert_eq!(prod.n_live().unwrap(), 2);
    assert_eq!(prod.grad_accumulation().unwrap(), 4);
    assert_both_ok(
        prod.heal_spare(&rid("spare-0"), &rid("live-0")),
        refer.heal_spare(&rid("spare-0"), &rid("live-0")),
    );
    assert_eq!(
        prod.replica_state(&rid("spare-0")).unwrap(),
        ReplicaState::Healing
    );
    assert_eq!(prod.n_live().unwrap(), 2);
    assert_eq!(prod.grad_accumulation().unwrap(), 4);
    assert_eq!(
        prod.live_replicas().unwrap(),
        vec![rid("live-0"), rid("live-1")]
    );
    assert_ctrl_tokens(&prod, &cfg);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn rejoin_makes_live_n_live_up_accum_falls_tokens_held() {
    let cfg = default_config();
    let specs = vec![
        spec("a", "r0", false),
        spec("s0", "sr0", true),
        spec("b", "r1", false),
    ];
    let (mut prod, mut refer) = pair(cfg.clone(), specs);
    assert_both_ok(prod.fail_replica(&rid("b")), refer.fail_replica(&rid("b")));
    assert_eq!(prod.n_live().unwrap(), 1);
    assert_eq!(prod.grad_accumulation().unwrap(), 8);
    assert_both_ok(
        prod.heal_spare(&rid("s0"), &rid("a")),
        refer.heal_spare(&rid("s0"), &rid("a")),
    );
    assert_both_ok(prod.rejoin(&rid("s0"), 11), refer.rejoin(&rid("s0"), 11));
    assert_eq!(prod.replica_state(&rid("s0")).unwrap(), ReplicaState::Live);
    assert_eq!(prod.n_live().unwrap(), 2);
    // Spec order: a, s0, b. Live filter: [a, s0].
    assert_eq!(prod.live_replicas().unwrap(), vec![rid("a"), rid("s0")]);
    assert_eq!(prod.grad_accumulation().unwrap(), 4);
    assert_eq!(prod.effective_tokens_per_step().unwrap(), 1024);
    assert_ctrl_tokens(&prod, &cfg);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn heal_non_spare_is_not_spare() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 1));
    let err = assert_both_err(
        prod.heal_spare(&rid("live-0"), &rid("live-1")),
        refer.heal_spare(&rid("live-0"), &rid("live-1")),
    );
    assert_not_spare(&err, &rid("live-0"));
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn rejoin_before_heal_is_not_spare_or_not_step_boundary() {
    let (mut prod, _) = pair(default_config(), n_live_m_spare(2, 1));
    let err = prod.rejoin(&rid("spare-0"), 1).expect_err("before heal");
    assert_rejoin_before_heal(&err, &rid("spare-0"));
    assert_eq!(
        prod.replica_state(&rid("spare-0")).unwrap(),
        ReplicaState::Spare
    );
}

#[test]
fn heal_source_must_be_live() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(3, 1));
    assert_both_ok(
        prod.fail_replica(&rid("live-2")),
        refer.fail_replica(&rid("live-2")),
    );
    let err = assert_both_err(
        prod.heal_spare(&rid("spare-0"), &rid("live-2")),
        refer.heal_spare(&rid("spare-0"), &rid("live-2")),
    );
    assert_not_live(&err, &rid("live-2"));
    assert_eq!(
        prod.replica_state(&rid("spare-0")).unwrap(),
        ReplicaState::Spare
    );
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn heal_unknown_spare_is_not_found() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    let id = rid("no-spare");
    let err = assert_both_err(
        prod.heal_spare(&id, &rid("live-0")),
        refer.heal_spare(&id, &rid("live-0")),
    );
    assert_replica_not_found(&err, &id);
}

#[test]
fn heal_unknown_source_is_not_found() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 1));
    let src = rid("no-src");
    let err = assert_both_err(
        prod.heal_spare(&rid("spare-0"), &src),
        refer.heal_spare(&rid("spare-0"), &src),
    );
    assert_replica_not_found(&err, &src);
}

#[test]
fn rejoin_unknown_is_not_found() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 1));
    let id = rid("ghost");
    let err = assert_both_err(prod.rejoin(&id, 4), refer.rejoin(&id, 4));
    assert_replica_not_found(&err, &id);
}

#[test]
fn rejoin_of_live_is_not_spare() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    let err = assert_both_err(
        prod.rejoin(&rid("live-0"), 0),
        refer.rejoin(&rid("live-0"), 0),
    );
    assert_not_spare(&err, &rid("live-0"));
}

#[test]
fn heal_already_healing_is_not_spare() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 1));
    assert_both_ok(
        prod.heal_spare(&rid("spare-0"), &rid("live-0")),
        refer.heal_spare(&rid("spare-0"), &rid("live-0")),
    );
    let err = assert_both_err(
        prod.heal_spare(&rid("spare-0"), &rid("live-1")),
        refer.heal_spare(&rid("spare-0"), &rid("live-1")),
    );
    assert_not_spare(&err, &rid("spare-0"));
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn rejoin_twice_is_not_spare() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 1));
    assert_both_ok(
        prod.heal_spare(&rid("spare-0"), &rid("live-0")),
        refer.heal_spare(&rid("spare-0"), &rid("live-0")),
    );
    assert_both_ok(
        prod.rejoin(&rid("spare-0"), 5),
        refer.rejoin(&rid("spare-0"), 5),
    );
    let err = assert_both_err(
        prod.rejoin(&rid("spare-0"), 6),
        refer.rejoin(&rid("spare-0"), 6),
    );
    assert_not_spare(&err, &rid("spare-0"));
    assert_eq!(prod.n_live().unwrap(), 3);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn source_equal_spare_is_not_live() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 1));
    let err = assert_both_err(
        prod.heal_spare(&rid("spare-0"), &rid("spare-0")),
        refer.heal_spare(&rid("spare-0"), &rid("spare-0")),
    );
    assert_not_live(&err, &rid("spare-0"));
}
