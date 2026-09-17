//! Group: fault injection. No majority cannot commit. Clock via injected NowMs
//! (no wall clock). Crash two of three. Partition two of three.

mod common;
mod reference;

use common::{
    follower_index, fresh_base, replicate, short_config, short_ttl, start3, this_id_of,
    tick_until_leader, tick_until_leader_among, FAKE_TOKEN,
};
use prometheus_raft::{Error, Role};
use reference::RefGroup;

#[test]
fn injected_now_expires_role_without_sleep() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let mut refer = RefGroup::start(3, short_config()).expect("ref");
    let ttl = short_ttl();
    let t0 = 1_000;
    let leader = tick_until_leader(&mut group, t0);
    let rleader = refer.tick_until_leader(t0);
    let a = this_id_of(&mut group, 0);
    group
        .get(leader)
        .expect("l")
        .claim_role(Role::Coordinator, &a, t0)
        .expect("claim");
    refer
        .claim_role(rleader, Role::Coordinator, &a, t0)
        .expect("ref claim");

    let just_before = t0 + ttl - 1;
    replicate(&mut group, just_before);
    refer.tick(just_before).expect("ref before");
    let dropped = group
        .get(leader)
        .expect("l")
        .expire_roles(just_before)
        .expect("not due");
    assert!(dropped.is_empty());
    assert!(group.get(leader).expect("l").state().coordinator.is_some());

    let due = t0 + ttl;
    // tick(now) also expires due leases when a majority exists.
    replicate(&mut group, due);
    refer.tick(due).expect("ref due tick");
    assert!(
        group.get(leader).expect("l").state().coordinator.is_none() || {
            let extra = group
                .get(leader)
                .expect("l")
                .expire_roles(due)
                .expect("expire");
            extra.contains(&Role::Coordinator)
                || group.get(leader).expect("l").state().coordinator.is_none()
        },
        "due lease must drop via tick(now) and/or expire_roles(now); due means expires_at <= now"
    );
    let _ = refer.expire_roles(rleader, due);
}

#[test]
fn partition_two_nodes_no_majority_no_commit() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 40_000;
    let leader = tick_until_leader(&mut group, now);
    group
        .get(leader)
        .expect("l")
        .commit_token(FAKE_TOKEN.to_vec())
        .expect("pre-partition commit");
    replicate(&mut group, now);

    group.partition(&[0, 1]).expect("isolate two");
    group.tick(now).expect("tick after 1-node minority");

    let mut any_commit = false;
    for i in 0..3 {
        if let Ok(c) = group.get(i) {
            match c.set_kernel_version("no-quorum".into()) {
                Ok(()) => any_commit = true,
                Err(Error::NotLeader) => {}
                Err(other) => panic!("unexpected {other:?} on node {i}"),
            }
        }
    }
    if any_commit {
        replicate(&mut group, now);
        let mut saw = 0;
        for i in 0..3 {
            if let Ok(c) = group.get(i) {
                if c.state().kernel_version == "no-quorum" {
                    saw += 1;
                }
            }
        }
        assert!(
            saw < 2,
            "minority must not make a durable majority write (saw {saw})"
        );
    }

    group.heal().expect("heal");
    let _ = tick_until_leader(&mut group, now);
    for i in 0..3 {
        assert_ne!(
            group.get(i).expect("n").state().kernel_version,
            "no-quorum",
            "healed cluster must not have committed the minority write"
        );
    }
}

#[test]
fn crash_two_of_three_no_leader_restart_recovers() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let mut refer = RefGroup::start(3, short_config()).expect("ref");
    let now = 50_000;
    let leader = tick_until_leader(&mut group, now);
    let rleader = refer.tick_until_leader(now);
    group
        .get(leader)
        .expect("l")
        .set_kernel_version("before-double-crash".into())
        .expect("kernel");
    refer
        .set_kernel_version(rleader, "before-double-crash".into())
        .expect("ref kernel");
    replicate(&mut group, now);
    refer.tick(now).expect("ref tick");

    let f = follower_index(&mut group);
    group.crash(leader).expect("crash leader");
    group.crash(f).expect("crash follower");
    refer.crash(leader).expect("ref crash leader");
    refer.crash(f).expect("ref crash follower");
    group.tick(now).expect("tick with one voter");
    refer.tick(now).expect("ref tick one voter");
    assert!(
        common::leader_index(&mut group).is_none(),
        "one of three cannot elect"
    );

    let survivor = (0..3).find(|&i| i != leader && i != f).expect("survivor");
    match group
        .get(survivor)
        .expect("s")
        .commit_token(b"fake-token".to_vec())
    {
        Err(Error::NotLeader) => {}
        Ok(_) => panic!("single survivor must not commit"),
        Err(other) => panic!("expected NotLeader, got {other:?}"),
    }

    group.restart(f).expect("restart one");
    refer.restart(f).expect("ref restart one");
    let majority = {
        let mut v = vec![survivor, f];
        v.sort();
        v
    };
    let new = tick_until_leader_among(&mut group, now, &majority);
    refer.tick(now).expect("ref majority tick");
    assert_eq!(
        group.get(new).expect("nl").state().kernel_version,
        "before-double-crash"
    );
}

#[test]
fn clock_does_not_use_wall_time_for_expiry() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let ttl = short_ttl();
    let t0 = 1;
    let leader = tick_until_leader(&mut group, t0);
    let a = this_id_of(&mut group, 0);
    let lease = group
        .get(leader)
        .expect("l")
        .claim_role(Role::Coordinator, &a, t0)
        .expect("claim");
    assert_eq!(lease.expires_at, t0 + ttl);
    // A real wall-clock wait is forbidden. Advancing injected now by 1 ms must
    // not expire a ttl-ms lease.
    let dropped = group
        .get(leader)
        .expect("l")
        .expire_roles(t0 + 1)
        .expect("still valid");
    assert!(dropped.is_empty());
    assert_eq!(
        group
            .get(leader)
            .expect("l")
            .state()
            .coordinator
            .as_ref()
            .unwrap()
            .expires_at,
        t0 + ttl
    );
}
