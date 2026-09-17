//! Group: crash leader, tick, new leader, committed state preserved.
//! crash/restart index stability.

mod common;
mod reference;

use common::{
    assert_not_found, assert_state_eq, follower_index, fresh_base, live_indices, replicate,
    short_config, start3, this_id_of, tick_until_leader, FAKE_TOKEN,
};
use prometheus_raft::Role;
use reference::RefGroup;

#[test]
fn crash_leader_new_leader_preserves_membership_token_kernel_roles() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let mut refer = RefGroup::start(3, short_config()).expect("ref");
    let now = 30_000;
    let old = tick_until_leader(&mut group, now);
    let rold = refer.tick_until_leader(now);
    let holder = this_id_of(&mut group, 0);

    group
        .get(old)
        .expect("l")
        .commit_token(FAKE_TOKEN.to_vec())
        .expect("token");
    group
        .get(old)
        .expect("l")
        .set_kernel_version("preserved".into())
        .expect("kernel");
    group
        .get(old)
        .expect("l")
        .claim_role(Role::Coordinator, &holder, now)
        .expect("coord");
    group
        .get(old)
        .expect("l")
        .claim_role(Role::TokenBroker, &holder, now)
        .expect("broker");
    refer
        .commit_token(rold, FAKE_TOKEN.to_vec())
        .expect("ref token");
    refer
        .set_kernel_version(rold, "preserved".into())
        .expect("ref kernel");
    refer
        .claim_role(rold, Role::Coordinator, &holder, now)
        .expect("ref coord");
    refer
        .claim_role(rold, Role::TokenBroker, &holder, now)
        .expect("ref broker");
    replicate(&mut group, now);
    refer.tick(now).expect("ref tick");

    let expected = group.get(old).expect("l").state().clone();
    assert_eq!(expected.membership.len(), 3);
    assert_eq!(expected.kernel_version, "preserved");
    assert_eq!(expected.token.as_ref().unwrap().generation, 1);
    assert!(expected.coordinator.is_some());
    assert!(expected.token_broker.is_some());

    group.crash(old).expect("crash leader");
    refer.crash(old).expect("ref crash");
    assert_eq!(group.len(), 3, "len stays the started n");
    assert_not_found(group.get(old).map(|_| ()), "get crashed leader");

    let survivors: Vec<usize> = (0..3).filter(|&i| i != old).collect();
    let new = common::tick_until_leader_among(&mut group, now, &survivors);
    refer.tick(now).expect("ref re-elect");
    assert_ne!(new, old);
    let got = group.get(new).expect("new leader").state().clone();
    assert_eq!(got.membership.len(), 3);
    assert_eq!(got.kernel_version, "preserved");
    assert_eq!(got.token.as_ref().unwrap().blob, FAKE_TOKEN);
    assert_eq!(
        got.coordinator.as_ref().map(|l| l.attempt),
        Some(1),
        "role leases survive if not expired"
    );
    assert_eq!(
        got.token_broker.as_ref().map(|l| l.holder.clone()),
        Some(holder)
    );
    assert_state_eq(&got, &expected, "new leader vs pre-crash");
    assert_state_eq(&got, &refer.committed_state(), "new leader vs ref");

    group.restart(old).expect("restart old leader");
    refer.restart(old).expect("ref restart");
    replicate(&mut group, now);
    refer.tick(now).expect("ref catch-up");
    assert_eq!(
        group.get(old).expect("restarted").state().kernel_version,
        "preserved"
    );
    assert_eq!(live_indices(&mut group), vec![0, 1, 2]);
}

#[test]
fn crash_then_crash_is_not_found_restart_live_fails() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 1_000;
    let _ = tick_until_leader(&mut group, now);
    let f = follower_index(&mut group);
    group.crash(f).expect("crash");
    assert_not_found(group.crash(f), "double crash");
    assert_not_found(group.crash(9), "crash oob");
    assert_not_found(group.restart(9), "restart oob");
    let live = (0..3).find(|&i| i != f).expect("live");
    common::assert_err(group.restart(live), "restart live node");
}

#[test]
fn crash_follower_len_and_other_indices_stable() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 2_000;
    let leader = tick_until_leader(&mut group, now);
    let f = follower_index(&mut group);
    let other = (0..3).find(|&i| i != f && i != leader).unwrap_or(leader);
    let other_id = this_id_of(&mut group, other);
    group.crash(f).expect("crash");
    assert_eq!(group.len(), 3);
    assert_eq!(this_id_of(&mut group, other), other_id);
    group.restart(f).expect("restart");
    assert_eq!(this_id_of(&mut group, f).0, f.to_string());
}
