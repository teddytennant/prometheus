//! Group: coordinator / token-broker leases: attempt 1, heartbeat extends,
//! expire then attempt 2, NotHolder, both roles holdable.

mod common;
mod reference;

use common::{
    assert_lease, assert_not_claimable, assert_not_holder, assert_state_eq, assert_untrusted,
    ephemeral, follower_index, fresh_base, short_config, short_ttl, start3, this_id_of,
    tick_until_leader,
};
use prometheus_raft::{Error, Role};
use reference::RefGroup;

#[test]
fn claim_coordinator_attempt_one_heartbeat_expire_then_attempt_two() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let mut refer = RefGroup::start(3, short_config()).expect("ref");
    let ttl = short_ttl();
    let t0 = 10_000;
    let leader = tick_until_leader(&mut group, t0);
    let rleader = refer.tick_until_leader(t0);
    let a = this_id_of(&mut group, 0);
    let b = this_id_of(&mut group, 1);

    let lease = group
        .get(leader)
        .expect("l")
        .claim_role(Role::Coordinator, &a, t0)
        .expect("claim 1");
    let rlease = refer
        .claim_role(rleader, Role::Coordinator, &a, t0)
        .expect("ref claim 1");
    assert_lease(&lease, Role::Coordinator, &a, 1, t0 + ttl);
    assert_eq!(lease, rlease);
    assert_eq!(
        group.get(leader).expect("l").state().coordinator.as_ref(),
        Some(&lease)
    );

    let t_hb = t0 + 500;
    let extended = group
        .get(leader)
        .expect("l")
        .heartbeat_role(Role::Coordinator, &a, t_hb)
        .expect("heartbeat");
    let rext = refer
        .heartbeat_role(rleader, Role::Coordinator, &a, t_hb)
        .expect("ref heartbeat");
    assert_lease(&extended, Role::Coordinator, &a, 1, t_hb + ttl);
    assert_eq!(extended.attempt, 1, "heartbeat does not bump attempt");
    assert_eq!(extended, rext);

    // Two missed periods from the heartbeat: due at t_hb + ttl.
    let t_due = t_hb + ttl;
    common::replicate(&mut group, t_hb);
    refer.tick(t_hb).expect("ref tick before due");

    let dropped = group
        .get(leader)
        .expect("l")
        .expire_roles(t_due)
        .expect("expire");
    let rdropped = refer.expire_roles(rleader, t_due).expect("ref expire");
    assert_eq!(dropped, vec![Role::Coordinator]);
    assert_eq!(dropped, rdropped);
    assert!(
        group.get(leader).expect("l").state().coordinator.is_none(),
        "expire_roles clears the lease without bumping attempt"
    );

    let t_claim2 = t_due + 1;
    let lease2 = group
        .get(leader)
        .expect("l")
        .claim_role(Role::Coordinator, &b, t_claim2)
        .expect("claim 2");
    let rlease2 = refer
        .claim_role(rleader, Role::Coordinator, &b, t_claim2)
        .expect("ref claim 2");
    assert_lease(&lease2, Role::Coordinator, &b, 2, t_claim2 + ttl);
    assert_eq!(lease2, rlease2);

    let err = group
        .get(leader)
        .expect("l")
        .heartbeat_role(Role::Coordinator, &a, t_claim2)
        .expect_err("old holder");
    assert_not_holder(&err, Role::Coordinator, "old holder heartbeat");
    let rerr = refer
        .heartbeat_role(rleader, Role::Coordinator, &a, t_claim2)
        .expect_err("ref old holder");
    assert_eq!(common::err_kind(&err), common::err_kind(&rerr));
}

#[test]
fn token_broker_is_a_separate_role_both_can_be_held() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let mut refer = RefGroup::start(3, short_config()).expect("ref");
    let ttl = short_ttl();
    let now = 20_000;
    let leader = tick_until_leader(&mut group, now);
    let rleader = refer.tick_until_leader(now);
    let a = this_id_of(&mut group, 0);
    let b = this_id_of(&mut group, 1);

    let coord = group
        .get(leader)
        .expect("l")
        .claim_role(Role::Coordinator, &a, now)
        .expect("coord");
    refer
        .claim_role(rleader, Role::Coordinator, &a, now)
        .expect("ref coord");
    let broker = group
        .get(leader)
        .expect("l")
        .claim_role(Role::TokenBroker, &b, now)
        .expect("broker on other node");
    refer
        .claim_role(rleader, Role::TokenBroker, &b, now)
        .expect("ref broker");
    assert_lease(&coord, Role::Coordinator, &a, 1, now + ttl);
    assert_lease(&broker, Role::TokenBroker, &b, 1, now + ttl);
    let st = group.get(leader).expect("l").state().clone();
    assert_eq!(st.coordinator.as_ref().map(|l| &l.holder), Some(&a));
    assert_eq!(st.token_broker.as_ref().map(|l| &l.holder), Some(&b));
    assert_state_eq(&st, &refer.committed_state(), "two roles different holders");

    // Same node may hold both.
    let now2 = now + 10;
    // Release by expiring both, then reclaim on a.
    group
        .get(leader)
        .expect("l")
        .expire_roles(now + ttl)
        .expect("expire both");
    refer.expire_roles(rleader, now + ttl).expect("ref expire");
    let c2 = group
        .get(leader)
        .expect("l")
        .claim_role(Role::Coordinator, &a, now2 + ttl)
        .expect("coord again");
    let b2 = group
        .get(leader)
        .expect("l")
        .claim_role(Role::TokenBroker, &a, now2 + ttl)
        .expect("broker same node");
    refer
        .claim_role(rleader, Role::Coordinator, &a, now2 + ttl)
        .expect("ref coord2");
    refer
        .claim_role(rleader, Role::TokenBroker, &a, now2 + ttl)
        .expect("ref broker2");
    assert_eq!(c2.holder, a);
    assert_eq!(b2.holder, a);
    assert_eq!(c2.attempt, 2);
    assert_eq!(b2.attempt, 2);
    assert_state_eq(
        group.get(leader).expect("l").state(),
        &refer.committed_state(),
        "same node both roles",
    );
}

#[test]
fn claim_held_unexpired_role_is_not_claimable() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 1_000;
    let leader = tick_until_leader(&mut group, now);
    let a = this_id_of(&mut group, 0);
    let b = this_id_of(&mut group, 1);
    group
        .get(leader)
        .expect("l")
        .claim_role(Role::Coordinator, &a, now)
        .expect("claim");
    let err = group
        .get(leader)
        .expect("l")
        .claim_role(Role::Coordinator, &b, now + 1)
        .expect_err("held");
    assert_not_claimable(&err, Role::Coordinator, "second claim");
}

#[test]
fn claim_unknown_node_is_not_found() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 1_000;
    let leader = tick_until_leader(&mut group, now);
    let err = group
        .get(leader)
        .expect("l")
        .claim_role(Role::Coordinator, &common::node_id("nope"), now)
        .expect_err("unknown");
    match err {
        Error::NotFound(s) => assert!(s.contains("nope"), "got {s}"),
        other => panic!("expected NotFound, got {other:?}"),
    }
}

#[test]
fn untrusted_member_cannot_claim_role() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 3_000;
    let leader = tick_until_leader(&mut group, now);
    let w = ephemeral("w-untrusted", "local:wu", false);
    group
        .get(leader)
        .expect("l")
        .add_worker(w.clone())
        .expect("add untrusted worker");
    let err = group
        .get(leader)
        .expect("l")
        .claim_role(Role::TokenBroker, &w.id, now)
        .expect_err("untrusted claim");
    assert_untrusted(&err, "untrusted claim");
}

#[test]
fn expire_roles_nothing_due_returns_empty() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 4_000;
    let leader = tick_until_leader(&mut group, now);
    let a = this_id_of(&mut group, 0);
    group
        .get(leader)
        .expect("l")
        .claim_role(Role::Coordinator, &a, now)
        .expect("claim");
    let dropped = group
        .get(leader)
        .expect("l")
        .expire_roles(now)
        .expect("not due");
    assert!(dropped.is_empty());
    assert!(group.get(leader).expect("l").state().coordinator.is_some());
}

#[test]
fn claim_can_take_over_expired_lease_without_expire_roles() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let ttl = short_ttl();
    let t0 = 8_000;
    let leader = tick_until_leader(&mut group, t0);
    let a = this_id_of(&mut group, 0);
    let b = this_id_of(&mut group, 1);
    group
        .get(leader)
        .expect("l")
        .claim_role(Role::Coordinator, &a, t0)
        .expect("claim 1");
    let t_due = t0 + ttl;
    let lease2 = group
        .get(leader)
        .expect("l")
        .claim_role(Role::Coordinator, &b, t_due)
        .expect("take over expired");
    assert_eq!(lease2.attempt, 2);
    assert_eq!(lease2.holder, b);
}

#[test]
fn heartbeat_on_vacant_role_is_not_holder() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 1_000;
    let leader = tick_until_leader(&mut group, now);
    let a = this_id_of(&mut group, 0);
    let err = group
        .get(leader)
        .expect("l")
        .heartbeat_role(Role::TokenBroker, &a, now)
        .expect_err("vacant");
    assert_not_holder(&err, Role::TokenBroker, "vacant heartbeat");
    let _ = follower_index(&mut group);
}

#[test]
fn missed_heartbeats_zero_ttl_is_one_period() {
    let (_parent, dir) = fresh_base();
    let cfg = prometheus_raft::ClusterConfig {
        heartbeat_period_ms: 1_000,
        missed_heartbeats: 0,
    };
    assert_eq!(cfg.lease_ttl_ms(), 1_000);
    let mut group = common::start_n(3, &dir, cfg);
    let now = 50_000;
    let leader = tick_until_leader(&mut group, now);
    let a = this_id_of(&mut group, 0);
    let lease = group
        .get(leader)
        .expect("l")
        .claim_role(Role::Coordinator, &a, now)
        .expect("claim");
    assert_eq!(lease.expires_at, now + 1_000);
}
