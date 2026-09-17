//! Group: add_worker ephemeral; add_voter ephemeral fails; duplicate id fails;
//! cannot drop below 3 voters once reached.

mod common;
mod reference;

use common::{
    assert_duplicate, assert_ephemeral_voter, assert_too_few_voters, assert_untrusted, ephemeral,
    fresh_base, member, short_config, start3, this_id_of, tick_until_leader, untrusted_voter,
    voter, voter_count,
};
use prometheus_raft::{Error, NodeKind};
use reference::RefGroup;

#[test]
fn add_worker_ephemeral_succeeds_add_voter_ephemeral_fails() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let mut refer = RefGroup::start(3, short_config()).expect("ref");
    let now = 1_000;
    let leader = tick_until_leader(&mut group, now);
    let rleader = refer.tick_until_leader(now);

    let w = ephemeral("ncshare-1", "local:w1", true);
    group
        .get(leader)
        .expect("l")
        .add_worker(w.clone())
        .expect("add_worker");
    refer
        .add_worker(rleader, w.clone())
        .expect("ref add_worker");
    let st = group.get(leader).expect("l").state().clone();
    let got = member(&st, &w.id);
    assert_eq!(got, &w);
    assert_eq!(got.kind, NodeKind::Ephemeral);
    assert_eq!(voter_count(&st), 3);

    let err = group
        .get(leader)
        .expect("l")
        .add_voter(ephemeral("ncshare-2", "local:w2", true))
        .expect_err("ephemeral voter");
    assert_ephemeral_voter(&err, "add_voter ephemeral");
    let rerr = refer
        .add_voter(rleader, ephemeral("ncshare-2", "local:w2", true))
        .expect_err("ref ephemeral voter");
    assert_eq!(common::err_kind(&err), common::err_kind(&rerr));
}

#[test]
fn duplicate_node_id_fails_across_voter_and_worker() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 2_000;
    let leader = tick_until_leader(&mut group, now);
    let existing = this_id_of(&mut group, 0);

    let err = group
        .get(leader)
        .expect("l")
        .add_voter(voter(&existing.0, "local:dup"))
        .expect_err("dup voter");
    assert_duplicate(&err, "duplicate voter id");

    let err = group
        .get(leader)
        .expect("l")
        .add_worker(ephemeral(&existing.0, "local:dupw", true))
        .expect_err("dup worker");
    assert_duplicate(&err, "duplicate worker id vs voter");

    let w = ephemeral("w", "local:w", true);
    group
        .get(leader)
        .expect("l")
        .add_worker(w.clone())
        .expect("worker");
    let err = group
        .get(leader)
        .expect("l")
        .add_worker(ephemeral("w", "local:w2", false))
        .expect_err("dup worker");
    assert_duplicate(&err, "duplicate worker id");
}

#[test]
fn add_voter_untrusted_always_on_fails() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 3_000;
    let leader = tick_until_leader(&mut group, now);
    let err = group
        .get(leader)
        .expect("l")
        .add_voter(untrusted_voter("u", "local:u"))
        .expect_err("untrusted");
    assert_untrusted(&err, "add_voter untrusted");
}

#[test]
fn add_worker_always_on_is_other() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 4_000;
    let leader = tick_until_leader(&mut group, now);
    let err = group
        .get(leader)
        .expect("l")
        .add_worker(voter("v4", "local:v4"))
        .expect_err("always-on worker");
    match err {
        Error::Other(_) => {}
        other => panic!("expected Other for AlwaysOn add_worker, got {other:?}"),
    }
}

#[test]
fn cannot_remove_voter_below_three_once_reached() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let mut refer = RefGroup::start(3, short_config()).expect("ref");
    let now = 5_000;
    let leader = tick_until_leader(&mut group, now);
    let rleader = refer.tick_until_leader(now);
    let slot = follower_not(leader, &mut group);
    let target = this_id_of(&mut group, slot);

    let err = group
        .get(leader)
        .expect("l")
        .remove_node(&target)
        .expect_err("below 3");
    assert_too_few_voters(&err, 2, "remove voter 3->2");
    let rerr = refer
        .remove_node(rleader, &target)
        .expect_err("ref below 3");
    assert_eq!(common::err_kind(&err), common::err_kind(&rerr));
    assert_eq!(voter_count(group.get(leader).expect("l").state()), 3);
}

#[test]
fn can_remove_worker_at_three_voters() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 6_000;
    let leader = tick_until_leader(&mut group, now);
    let w = ephemeral("w", "local:w", true);
    group
        .get(leader)
        .expect("l")
        .add_worker(w.clone())
        .expect("add");
    group
        .get(leader)
        .expect("l")
        .remove_node(&w.id)
        .expect("remove worker");
    let st = group.get(leader).expect("l").state();
    assert_eq!(voter_count(st), 3);
    assert!(st.membership.iter().all(|n| n.id != w.id));
}

#[test]
fn add_fourth_voter_then_remove_back_to_three() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 7_000;
    let leader = tick_until_leader(&mut group, now);
    let extra = voter("3", "local:3");
    group
        .get(leader)
        .expect("l")
        .add_voter(extra.clone())
        .expect("add 4th");
    assert_eq!(voter_count(group.get(leader).expect("l").state()), 4);
    group
        .get(leader)
        .expect("l")
        .remove_node(&extra.id)
        .expect("remove 4th");
    assert_eq!(voter_count(group.get(leader).expect("l").state()), 3);
}

#[test]
fn remove_unknown_is_not_found() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 8_000;
    let leader = tick_until_leader(&mut group, now);
    common::assert_not_found(
        group
            .get(leader)
            .expect("l")
            .remove_node(&common::node_id("ghost")),
        "remove unknown",
    );
}

fn follower_not(leader: usize, _group: &mut prometheus_raft::LocalGroup) -> usize {
    (0..3).find(|&i| i != leader).expect("follower slot")
}
