//! Group: propose — digest, quorum refuse, pin, kernel, events.

mod common;
mod reference;

use common::{
    assert_digest_mismatch, assert_has_event, assert_no_event, assert_no_pin, assert_no_quorum,
    assert_pin, assert_result_tag, assert_state_refused, assert_wrong_state, create_ops,
    create_ops_cfg, fresh_world, kernel_leader, membership, one_sig_release, ops_cfg,
    ops_cfg_quorum, release, sig, status_of, two_sig_release, v1_digest, v1_signed, v2_digest,
    ABC_BYTES, ABC_HEX, V1_BYTES, V1_HEX, V2_HEX,
};
use prometheus_cas::Digest;
use prometheus_ops::{
    ReleaseState, DEFAULT_QUORUM, EVENT_PROPOSE, EVENT_REFUSE, PIN_CURRENT, PIN_SHADOW,
};
use reference::RefOps;

#[test]
fn digest_mismatch_does_not_put_or_pin_or_change_kernel() {
    let mut world = fresh_world();
    let members = membership(&mut world);
    let kernel = kernel_leader(&mut world);
    let mut refer = RefOps::new(ops_cfg(), members, kernel.clone());
    let mut ops = create_ops(&world);
    let rel = two_sig_release("v1", v2_digest());
    let got = ops.propose(rel.clone(), V1_BYTES, &mut world.store, world.now);
    let exp = refer.propose(rel, V1_BYTES);
    assert_result_tag(&got, &exp, "mismatch");
    assert_digest_mismatch(got.as_ref().unwrap_err(), V2_HEX, V1_HEX);
    assert!(world.store.get(&v1_digest()).is_err(), "must not put");
    assert_no_pin(&world.store, PIN_SHADOW);
    assert_eq!(kernel_leader(&mut world), kernel);
    let st = status_of(&ops, &mut world);
    assert_eq!(st.state, ReleaseState::Idle);
    assert_no_event(&ops, EVENT_PROPOSE);
    assert_no_event(&ops, EVENT_REFUSE);
}

#[test]
fn no_quorum_does_not_put_or_pin_state_refused() {
    let mut world = fresh_world();
    let members = membership(&mut world);
    let kernel = kernel_leader(&mut world);
    let mut refer = RefOps::new(ops_cfg(), members, kernel.clone());
    let mut ops = create_ops(&world);
    let rel = one_sig_release("v1", v1_digest());
    let got = ops.propose(rel.clone(), V1_BYTES, &mut world.store, world.now);
    let exp = refer.propose(rel, V1_BYTES);
    assert_result_tag(&got, &exp, "no quorum");
    assert_no_quorum(got.as_ref().unwrap_err(), 1, DEFAULT_QUORUM);
    assert!(world.store.get(&v1_digest()).is_err(), "must not put");
    assert_no_pin(&world.store, PIN_SHADOW);
    assert_eq!(kernel_leader(&mut world), kernel);
    let st = status_of(&ops, &mut world);
    assert_state_refused(&st);
    assert_eq!(st.release.as_ref().map(|r| r.version.as_str()), Some("v1"));
    assert_has_event(&ops, EVENT_REFUSE);
    assert_no_event(&ops, EVENT_PROPOSE);
}

#[test]
fn duplicate_humans_are_not_quorum() {
    let mut world = fresh_world();
    let mut ops = create_ops(&world);
    let rel = release(
        "v1",
        v1_digest(),
        vec![sig("alice", b"one"), sig("alice", b"two")],
    );
    let err = ops
        .propose(rel, V1_BYTES, &mut world.store, world.now)
        .expect_err("dup humans");
    assert_no_quorum(&err, 1, 2);
    assert!(world.store.get(&v1_digest()).is_err());
    assert_no_pin(&world.store, PIN_SHADOW);
}

#[test]
fn empty_signature_bytes_do_not_count_for_propose() {
    let mut world = fresh_world();
    let mut ops = create_ops(&world);
    let rel = release(
        "v1",
        v1_digest(),
        vec![sig("alice", b""), sig("bob", b"ok")],
    );
    let err = ops
        .propose(rel, V1_BYTES, &mut world.store, world.now)
        .expect_err("empty bytes");
    assert_no_quorum(&err, 1, 2);
}

#[test]
fn refused_release_never_sets_kernel_version() {
    let mut world = fresh_world();
    let kernel = kernel_leader(&mut world);
    let mut ops = create_ops(&world);
    let _ = ops.propose(
        one_sig_release("v1", v1_digest()),
        V1_BYTES,
        &mut world.store,
        world.now,
    );
    assert_eq!(kernel_leader(&mut world), kernel);
    assert_eq!(kernel, "0");
}

#[test]
fn quorum_puts_pins_shadow_state_proposed() {
    let mut world = fresh_world();
    let members = membership(&mut world);
    let kernel = kernel_leader(&mut world);
    let mut refer = RefOps::new(ops_cfg(), members, kernel.clone());
    let mut ops = create_ops(&world);
    let rel = v1_signed();
    let got = ops.propose(rel.clone(), V1_BYTES, &mut world.store, world.now);
    let exp = refer.propose(rel, V1_BYTES);
    assert_result_tag(&got, &exp, "propose ok");
    got.expect("propose");
    assert_eq!(world.store.get(&v1_digest()).expect("get"), V1_BYTES);
    assert_pin(&world.store, PIN_SHADOW, V1_HEX);
    assert_no_pin(&world.store, PIN_CURRENT);
    assert_eq!(kernel_leader(&mut world), kernel);
    let st = status_of(&ops, &mut world);
    assert_eq!(st.state, ReleaseState::Proposed);
    assert_eq!(st.release.as_ref().map(|r| r.version.as_str()), Some("v1"));
    assert_has_event(&ops, EVENT_PROPOSE);
    assert_eq!(common::cas_pins(&world.store), refer.pins);
}

#[test]
fn empty_artifact_with_matching_digest_and_quorum() {
    let mut world = fresh_world();
    let mut ops = create_ops(&world);
    let empty = b"";
    let hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    let rel = two_sig_release("empty", Digest(hex.to_string()));
    ops.propose(rel, empty, &mut world.store, world.now)
        .expect("empty artifact");
    assert_eq!(
        world.store.get(&Digest(hex.to_string())).expect("get"),
        empty
    );
    assert_pin(&world.store, PIN_SHADOW, hex);
}

#[test]
fn abc_artifact_known_vector() {
    let mut world = fresh_world();
    let mut ops = create_ops(&world);
    let rel = two_sig_release("abc", Digest(ABC_HEX.to_string()));
    ops.propose(rel, ABC_BYTES, &mut world.store, world.now)
        .expect("abc");
    assert_pin(&world.store, PIN_SHADOW, ABC_HEX);
}

#[test]
fn custom_quorum_three_refuses_two_signers() {
    let mut world = fresh_world();
    let mut ops = create_ops_cfg(&world, ops_cfg_quorum(3));
    let err = ops
        .propose(v1_signed(), V1_BYTES, &mut world.store, world.now)
        .expect_err("need 3");
    assert_no_quorum(&err, 2, 3);
    assert_state_refused(&status_of(&ops, &mut world));
}

#[test]
fn custom_quorum_one_accepts_one_signer() {
    let mut world = fresh_world();
    let mut ops = create_ops_cfg(&world, ops_cfg_quorum(1));
    ops.propose(
        one_sig_release("v1", v1_digest()),
        V1_BYTES,
        &mut world.store,
        world.now,
    )
    .expect("quorum 1");
    assert_eq!(status_of(&ops, &mut world).state, ReleaseState::Proposed);
}

#[test]
fn propose_again_while_proposed_is_wrong_state() {
    let mut world = fresh_world();
    let mut ops = create_ops(&world);
    ops.propose(v1_signed(), V1_BYTES, &mut world.store, world.now)
        .expect("first");
    let err = ops
        .propose(v1_signed(), V1_BYTES, &mut world.store, world.now)
        .expect_err("second");
    assert_wrong_state(&err);
}

#[test]
fn propose_after_refuse_can_succeed() {
    let mut world = fresh_world();
    let mut ops = create_ops(&world);
    let _ = ops.propose(
        one_sig_release("v1", v1_digest()),
        V1_BYTES,
        &mut world.store,
        world.now,
    );
    ops.propose(v1_signed(), V1_BYTES, &mut world.store, world.now)
        .expect("retry with quorum");
    assert_eq!(status_of(&ops, &mut world).state, ReleaseState::Proposed);
    assert_pin(&world.store, PIN_SHADOW, V1_HEX);
}
