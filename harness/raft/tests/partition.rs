//! Group: partition one node; majority of 2 still commits; isolated node does
//! not accept client writes; heal catch-up.

mod common;
mod reference;

use common::{
    follower_index, fresh_base, replicate, short_config, start3, tick_until_leader,
    tick_until_leader_among, FAKE_TOKEN, FAKE_TOKEN_2,
};
use prometheus_raft::Error;
use reference::RefGroup;

#[test]
fn partition_follower_majority_commits_isolated_no_writes_heal_catch_up() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let mut refer = RefGroup::start(3, short_config()).expect("ref");
    let now = 11_000;
    let leader = tick_until_leader(&mut group, now);
    let _rleader = refer.tick_until_leader(now);
    let iso = follower_index(&mut group);
    assert_ne!(iso, leader);

    group.partition(&[iso]).expect("partition");
    refer.partition(&[iso]).expect("ref partition");
    replicate(&mut group, now);
    refer.tick(now).expect("ref tick");

    let leader = tick_until_leader_among(&mut group, now, &{
        let v: Vec<usize> = (0..3).filter(|&i| i != iso).collect();
        v
    });
    let rleader = refer.leader_index().expect("ref majority leader");
    let _ = rleader;

    let rec = group
        .get(leader)
        .expect("l")
        .commit_token(FAKE_TOKEN.to_vec())
        .expect("majority commit");
    refer
        .commit_token(refer.leader_index().expect("rl"), FAKE_TOKEN.to_vec())
        .expect("ref majority commit");
    assert_eq!(rec.generation, 1);
    replicate(&mut group, now);
    refer.tick(now).expect("ref replicate");

    for i in 0..3 {
        if i == iso {
            continue;
        }
        let tok = group
            .get(i)
            .expect("maj")
            .state()
            .token
            .clone()
            .expect("majority has token");
        assert_eq!(tok.blob, FAKE_TOKEN);
    }

    let isolated_write = group
        .get(iso)
        .expect("iso")
        .commit_token(FAKE_TOKEN_2.to_vec());
    match isolated_write {
        Err(Error::NotLeader) => {}
        Ok(_) => {
            replicate(&mut group, now);
            for i in 0..3 {
                if i == iso {
                    continue;
                }
                let tok = group.get(i).expect("m").state().token.clone();
                assert_eq!(
                    tok.as_ref().map(|t| t.blob.as_slice()),
                    Some(FAKE_TOKEN),
                    "isolated write must not become the majority token"
                );
            }
        }
        Err(other) => panic!("isolated write: expected NotLeader or no progress, got {other:?}"),
    }

    group.heal().expect("heal");
    refer.heal().expect("ref heal");
    replicate(&mut group, now);
    refer.tick(now).expect("ref heal tick");
    // Another tick so the former isolate applies committed entries.
    replicate(&mut group, now);
    refer.tick(now).expect("ref catch-up");

    for i in 0..3 {
        let tok = group
            .get(i)
            .expect("n")
            .state()
            .token
            .clone()
            .expect("catch-up token");
        assert_eq!(tok.generation, 1);
        assert_eq!(tok.blob, FAKE_TOKEN, "node {i} after heal");
    }
}

#[test]
fn partition_leader_majority_of_two_still_commits() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 12_000;
    let old_leader = tick_until_leader(&mut group, now);

    group.partition(&[old_leader]).expect("partition leader");
    let majority: Vec<usize> = (0..3).filter(|&i| i != old_leader).collect();
    let new_leader = tick_until_leader_among(&mut group, now, &majority);
    assert_ne!(new_leader, old_leader);

    group
        .get(new_leader)
        .expect("nl")
        .set_kernel_version("after-part".into())
        .expect("majority kernel");
    replicate(&mut group, now);
    for &i in &majority {
        assert_eq!(
            group.get(i).expect("m").state().kernel_version,
            "after-part"
        );
    }

    let isolated_write = group
        .get(old_leader)
        .expect("old")
        .set_kernel_version("stale".into());
    match isolated_write {
        Err(Error::NotLeader) => {}
        Ok(()) => {
            replicate(&mut group, now);
            for &i in &majority {
                assert_eq!(
                    group.get(i).expect("m").state().kernel_version,
                    "after-part",
                    "isolated kernel write must not win"
                );
            }
        }
        Err(other) => panic!("expected NotLeader or no progress, got {other:?}"),
    }

    group.heal().expect("heal");
    replicate(&mut group, now);
    replicate(&mut group, now);
    for i in 0..3 {
        assert_eq!(
            group.get(i).expect("n").state().kernel_version,
            "after-part",
            "node {i} catch-up"
        );
    }
}

#[test]
fn partition_out_of_range_is_not_found() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let _ = tick_until_leader(&mut group, 1);
    common::assert_not_found(group.partition(&[9]), "partition oob");
}
