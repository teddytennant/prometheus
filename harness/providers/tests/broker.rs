//! Group: Broker::refresh generation 1 then 2 via H4, access-token derivation,
//! NotBroker on a LocalGroup follower.

mod common;
mod reference;

use common::{
    assert_access_token, assert_not_broker, assert_record, broker, follower_index, group3,
    provider, solo_cluster, tick_until_leader, FAKE_ACCESS_HEX, FAKE_BLOB,
};
use prometheus_providers::Broker;
use reference::{expected_refresh, sha256_hex};

#[test]
fn refresh_generation_one_then_two_via_h4() {
    let (_parent, mut cluster) = solo_cluster();
    let mut br = broker("xai", 60_000);
    let cfg = provider("xai", 60_000);
    let now = 1_000u64;

    let (rec1, tok1) = br
        .refresh(&mut cluster, FAKE_BLOB, now)
        .expect("refresh gen 1");
    let (want_rec1, want_tok1) = expected_refresh(&cfg, FAKE_BLOB, now, 1);
    assert_record(&rec1, 1, FAKE_BLOB);
    assert_eq!(rec1, want_rec1);
    assert_eq!(tok1, want_tok1);
    assert_eq!(tok1.token, FAKE_ACCESS_HEX);
    assert_eq!(tok1.token, sha256_hex(FAKE_BLOB));
    assert_access_token(&tok1, &cfg.id, FAKE_BLOB, now + 60_000);
    assert_eq!(
        cluster.state().token.as_ref(),
        Some(&rec1),
        "commit_token first: raft machine holds generation 1"
    );

    let now2 = now + 5_000;
    let (rec2, tok2) = br
        .refresh(&mut cluster, FAKE_BLOB, now2)
        .expect("refresh gen 2");
    let (want_rec2, want_tok2) = expected_refresh(&cfg, FAKE_BLOB, now2, 2);
    assert_record(&rec2, 2, FAKE_BLOB);
    assert_eq!(rec2, want_rec2);
    assert_eq!(tok2, want_tok2);
    assert_eq!(tok2.expires_at, now2 + 60_000);
    assert_eq!(tok2.token, tok1.token, "same blob → same access hex");
    assert_ne!(rec2.generation, rec1.generation);
    assert_eq!(cluster.state().token.as_ref(), Some(&rec2));
}

#[test]
fn access_token_is_lowercase_hex_sha256_not_the_blob() {
    let (_parent, mut cluster) = solo_cluster();
    let mut br = Broker::new(provider("anthropic", 1_000));
    let now = 42u64;
    let (_rec, tok) = br.refresh(&mut cluster, FAKE_BLOB, now).expect("refresh");
    assert_eq!(tok.token.len(), 64);
    assert_eq!(tok.token, FAKE_ACCESS_HEX);
    assert_ne!(tok.token.as_bytes(), FAKE_BLOB);
    assert!(!tok
        .token
        .as_bytes()
        .windows(FAKE_BLOB.len())
        .any(|w| w == FAKE_BLOB));
    assert_eq!(tok.provider, provider("anthropic", 1_000).id);
    assert_eq!(tok.expires_at, 1_042);
}

#[test]
fn expires_at_uses_injected_now_not_wall_clock() {
    let (_parent, mut cluster) = solo_cluster();
    let mut br = broker("p", 7);
    let (_rec, tok) = br.refresh(&mut cluster, FAKE_BLOB, 99).expect("refresh");
    assert_eq!(tok.expires_at, 106);
    let (_rec2, tok2) = br.refresh(&mut cluster, FAKE_BLOB, 0).expect("refresh 2");
    assert_eq!(tok2.expires_at, 7);
}

#[test]
fn refresh_on_follower_is_not_broker() {
    let (_parent, mut group) = group3();
    let now = 10_000u64;
    let leader = tick_until_leader(&mut group, now);
    let follower = follower_index(&mut group, leader);
    assert!(
        !group.get(follower).expect("f").is_leader(),
        "follower must not be leader"
    );
    let mut br = broker("xai", 60_000);
    let err = br
        .refresh(group.get(follower).expect("f"), FAKE_BLOB, now)
        .expect_err("follower refresh");
    assert_not_broker(&err);
}

#[test]
fn refresh_on_leader_of_three_node_group_commits() {
    let (_parent, mut group) = group3();
    let now = 10_000u64;
    let leader = tick_until_leader(&mut group, now);
    let mut br = broker("xai", 1_000);
    let (rec, tok) = br
        .refresh(group.get(leader).expect("l"), FAKE_BLOB, now)
        .expect("leader refresh");
    assert_record(&rec, 1, FAKE_BLOB);
    assert_eq!(tok.token, FAKE_ACCESS_HEX);
    assert_eq!(tok.expires_at, now + 1_000);
}

#[test]
fn bootstrap_leader_does_not_need_token_broker_role_claim() {
    let (_parent, mut cluster) = solo_cluster();
    assert!(cluster.is_leader());
    assert!(cluster.state().token_broker.is_none());
    let mut br = broker("xai", 10);
    let (rec, _) = br
        .refresh(&mut cluster, FAKE_BLOB, 1)
        .expect("leader without role claim");
    assert_eq!(rec.generation, 1);
}
