//! Group: commit_token generations 1 then 2; successor chain; follower
//! crash+restart restores the committed blob.

mod common;
mod reference;

use common::{
    assert_state_eq, follower_index, fresh_base, replicate, short_config, start3, this_id_of,
    tick_until_leader, FAKE_TOKEN, FAKE_TOKEN_2, FAKE_TOKEN_3,
};
use prometheus_raft::TokenRecord;
use reference::RefGroup;

#[test]
fn commit_token_generation_one_then_two() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let mut refer = RefGroup::start(3, short_config()).expect("ref");
    let now = 1_000;
    let leader = tick_until_leader(&mut group, now);
    let rleader = refer.tick_until_leader(now);

    assert!(
        group.get(leader).expect("l").state().token.is_none(),
        "none committed at start"
    );

    let t1 = group
        .get(leader)
        .expect("l")
        .commit_token(FAKE_TOKEN.to_vec())
        .expect("gen 1");
    let r1 = refer
        .commit_token(rleader, FAKE_TOKEN.to_vec())
        .expect("ref gen 1");
    assert_eq!(
        t1,
        TokenRecord {
            generation: 1,
            blob: FAKE_TOKEN.to_vec()
        }
    );
    assert_eq!(t1, r1);
    assert_eq!(
        group.get(leader).expect("l").state().token.as_ref(),
        Some(&t1)
    );

    let t2 = group
        .get(leader)
        .expect("l")
        .commit_token(FAKE_TOKEN_2.to_vec())
        .expect("gen 2");
    let r2 = refer
        .commit_token(rleader, FAKE_TOKEN_2.to_vec())
        .expect("ref gen 2");
    assert_eq!(t2.generation, 2);
    assert_eq!(t2.generation, t1.generation + 1);
    assert_eq!(t2.blob, FAKE_TOKEN_2);
    assert_eq!(t2, r2);

    let t3 = group
        .get(leader)
        .expect("l")
        .commit_token(FAKE_TOKEN_3.to_vec())
        .expect("gen 3");
    assert_eq!(t3.generation, 3, "generation never skips or rewinds");
    refer
        .commit_token(rleader, FAKE_TOKEN_3.to_vec())
        .expect("ref gen 3");
    assert_state_eq(
        group.get(leader).expect("l").state(),
        &refer.committed_state(),
        "token chain",
    );
}

#[test]
fn follower_crash_restart_restores_committed_token() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let mut refer = RefGroup::start(3, short_config()).expect("ref");
    let now = 7_000;
    let leader = tick_until_leader(&mut group, now);
    let rleader = refer.tick_until_leader(now);

    let rec = group
        .get(leader)
        .expect("l")
        .commit_token(FAKE_TOKEN.to_vec())
        .expect("commit");
    refer
        .commit_token(rleader, FAKE_TOKEN.to_vec())
        .expect("ref commit");
    assert_eq!(rec.generation, 1);
    replicate(&mut group, now);
    refer.tick(now).expect("ref replicate");

    let f = follower_index(&mut group);
    assert_ne!(f, leader);
    let before = group
        .get(f)
        .expect("f")
        .state()
        .token
        .clone()
        .expect("follower had token after replicate");
    assert_eq!(before, rec);

    group.crash(f).expect("crash follower");
    refer.crash(f).expect("ref crash");
    common::assert_not_found(group.get(f).map(|_| ()), "get crashed follower");

    group.restart(f).expect("restart follower");
    refer.restart(f).expect("ref restart");
    let restored = group
        .get(f)
        .expect("restarted")
        .state()
        .token
        .clone()
        .expect("token on restarted follower");
    assert_eq!(restored, rec);
    assert_eq!(
        restored,
        refer.get(f).expect("ref f").state.token.clone().unwrap()
    );
    assert!(!group.get(f).expect("f").is_leader());
}

#[test]
fn commit_token_replicates_to_connected_followers_after_tick() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 2_000;
    let leader = tick_until_leader(&mut group, now);
    group
        .get(leader)
        .expect("l")
        .commit_token(FAKE_TOKEN.to_vec())
        .expect("commit");
    replicate(&mut group, now);
    for i in 0..3 {
        let tok = group
            .get(i)
            .expect("n")
            .state()
            .token
            .clone()
            .expect("token on all");
        assert_eq!(tok.generation, 1);
        assert_eq!(tok.blob, FAKE_TOKEN);
    }
    let _ = this_id_of(&mut group, 0);
}
