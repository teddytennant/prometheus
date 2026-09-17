//! Group: LocalGroup::start elects a leader; followers reject mutations;
//! start(n < 3) is TooFewVoters.

mod common;
mod reference;

use common::{
    assert_initial_control_state, assert_min_voters_lock, assert_not_leader, assert_too_few_voters,
    follower_index, fresh_base, short_config, start3, this_id_of, tick_until_leader, voter,
    FAKE_TOKEN,
};
use prometheus_raft::{LocalGroup, NodeKind, Role, MIN_VOTERS};
use reference::RefGroup;

#[test]
fn start_three_elects_one_leader_after_tick() {
    assert_min_voters_lock();
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let mut refer = RefGroup::start(3, short_config()).expect("ref start");
    assert_eq!(group.len(), 3);
    assert!(!group.is_empty());

    for i in 0..3 {
        let c = group.get(i).expect("get");
        assert_eq!(c.this_id().0, i.to_string(), "locked slot id");
        assert_eq!(c.dir(), dir.join(i.to_string()).as_path());
        assert_eq!(c.config().lease_ttl_ms(), 2_000);
        assert_initial_control_state(c.state(), &format!("node {i}"));
        assert_eq!(c.state().membership.len(), 3);
        let me = c
            .state()
            .membership
            .iter()
            .find(|n| n.id == *c.this_id())
            .expect("self in membership");
        assert_eq!(me.kind, NodeKind::AlwaysOn);
        assert!(me.trusted);
    }

    let now = 1_000;
    let leader = tick_until_leader(&mut group, now);
    let rleader = refer.tick_until_leader(now);
    assert!(
        group.get(leader).expect("leader").is_leader(),
        "leader reports is_leader"
    );
    let _ = rleader;
    let follower = follower_index(&mut group);
    assert_ne!(leader, follower);
    assert!(!group.get(follower).expect("follower").is_leader());

    common::assert_state_eq(
        group.get(leader).expect("l").state(),
        &refer.get(rleader).expect("rl").state,
        "elected vs ref",
    );
}

#[test]
fn followers_add_voter_claim_role_commit_token_are_not_leader() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 5_000;
    let _leader = tick_until_leader(&mut group, now);
    let f = follower_index(&mut group);
    let fid = this_id_of(&mut group, f);

    assert_not_leader(
        group.get(f).expect("f").add_voter(voter("x", "local:x")),
        "follower add_voter",
    );
    assert_not_leader(
        group
            .get(f)
            .expect("f")
            .claim_role(Role::Coordinator, &fid, now),
        "follower claim_role",
    );
    assert_not_leader(
        group.get(f).expect("f").commit_token(FAKE_TOKEN.to_vec()),
        "follower commit_token",
    );
    assert_not_leader(
        group.get(f).expect("f").set_kernel_version("1".into()),
        "follower set_kernel_version",
    );
    assert_not_leader(
        group
            .get(f)
            .expect("f")
            .add_worker(common::ephemeral("w", "local:w", true)),
        "follower add_worker",
    );
    assert_not_leader(
        group.get(f).expect("f").remove_node(&fid),
        "follower remove_node",
    );
    assert_not_leader(
        group
            .get(f)
            .expect("f")
            .heartbeat_role(Role::Coordinator, &fid, now),
        "follower heartbeat_role",
    );
    assert_not_leader(
        group.get(f).expect("f").expire_roles(now),
        "follower expire_roles",
    );
}

#[test]
fn start_two_is_too_few_voters() {
    assert_eq!(MIN_VOTERS, 3);
    let (_parent, dir) = fresh_base();
    let err = common::unwrap_err(LocalGroup::start(2, &dir, short_config()), "start(2)");
    assert_too_few_voters(&err, 2, "start(2)");
    let rerr = RefGroup::start(2, short_config()).expect_err("ref start(2)");
    assert_eq!(common::err_kind(&err), common::err_kind(&rerr));
}

#[test]
fn start_zero_and_one_are_too_few_voters() {
    let (_parent, dir) = fresh_base();
    for n in [0usize, 1] {
        let err = common::unwrap_err(
            LocalGroup::start(n, dir.join(n.to_string()), short_config()),
            "start too few",
        );
        assert_too_few_voters(&err, n, &format!("start({n})"));
    }
}

#[test]
fn start_three_leader_mutations_are_not_not_leader() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 10_000;
    let leader = tick_until_leader(&mut group, now);
    let lid = this_id_of(&mut group, leader);
    group
        .get(leader)
        .expect("l")
        .set_kernel_version("ok".into())
        .expect("leader set_kernel_version");
    group
        .get(leader)
        .expect("l")
        .commit_token(FAKE_TOKEN.to_vec())
        .expect("leader commit_token");
    group
        .get(leader)
        .expect("l")
        .claim_role(Role::Coordinator, &lid, now)
        .expect("leader claim_role");
    assert_eq!(group.get(leader).expect("l").state().kernel_version, "ok");
}
