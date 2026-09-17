//! Group: rollback on a bad release; previous pin and kernel restored.

mod common;
mod reference;

use common::{
    assert_has_event, assert_live_applied, assert_no_pin, assert_pin, assert_result_tag,
    assert_state_rolled_back, assert_wrong_state, cas_pins, create_ops, fresh_world, kernel_leader,
    leader_idx, membership, ops_cfg, status_of, v1_signed, v2_signed, V1_BYTES, V1_HEX, V2_BYTES,
    V2_HEX,
};
use prometheus_ops::{ReleaseState, EVENT_ROLLBACK, PIN_CURRENT, PIN_PREVIOUS, PIN_SHADOW};
use reference::RefOps;

fn rollout_to_current(
    world: &mut common::World,
    ops: &mut prometheus_ops::Ops,
    refer: &mut RefOps,
    rel: prometheus_ops::Release,
    bytes: &[u8],
) {
    assert_result_tag(
        &ops.propose(rel.clone(), bytes, &mut world.store, world.now),
        &refer.propose(rel, bytes),
        "propose",
    );
    assert_result_tag(
        &ops.start_shadow(&mut world.store, world.now),
        &refer.start_shadow(),
        "shadow",
    );
    assert_result_tag(&ops.shadow_pass(world.now), &refer.shadow_pass(), "pass");
    let members = membership(world);
    for m in &members {
        assert_result_tag(
            &ops.apply_one(m, world.now),
            &refer.apply_one(m),
            &format!("apply {}", m.0),
        );
    }
    let i = leader_idx(world);
    assert_result_tag(
        &ops.promote(world.group.get(i).unwrap(), &mut world.store, world.now),
        &refer.promote(),
        "promote",
    );
}

#[test]
fn rollback_from_idle_is_wrong_state() {
    let mut world = fresh_world();
    let mut ops = create_ops(&world);
    let i = leader_idx(&mut world);
    let err = ops
        .rollback(
            "nothing",
            world.group.get(i).unwrap(),
            &mut world.store,
            world.now,
        )
        .expect_err("idle");
    assert_wrong_state(&err);
}

#[test]
fn rollback_in_progress_unpins_shadow_kernel_stays() {
    let mut world = fresh_world();
    let members = membership(&mut world);
    let kernel = kernel_leader(&mut world);
    let mut refer = RefOps::new(ops_cfg(), members.clone(), kernel.clone());
    let mut ops = create_ops(&world);
    ops.propose(v1_signed(), V1_BYTES, &mut world.store, world.now)
        .unwrap();
    refer.propose(v1_signed(), V1_BYTES).unwrap();
    ops.start_shadow(&mut world.store, world.now).unwrap();
    refer.start_shadow().unwrap();
    ops.shadow_pass(world.now).unwrap();
    refer.shadow_pass().unwrap();
    ops.apply_one(&members[0], world.now).unwrap();
    refer.apply_one(&members[0]).unwrap();
    let i = leader_idx(&mut world);
    let got = ops.rollback(
        "bad smoke",
        world.group.get(i).unwrap(),
        &mut world.store,
        world.now,
    );
    let exp = refer.rollback("bad smoke");
    assert_result_tag(&got, &exp, "rollback");
    got.expect("rollback");
    let st = status_of(&ops, &mut world);
    assert_state_rolled_back(&st, "bad smoke");
    assert_eq!(kernel_leader(&mut world), kernel);
    assert_no_pin(&world.store, PIN_SHADOW);
    assert_has_event(&ops, EVENT_ROLLBACK);
    assert_live_applied(&st, &kernel);
}

#[test]
fn rollback_after_promote_restores_previous_kernel_and_pin() {
    let mut world = fresh_world();
    let members = membership(&mut world);
    let kernel0 = kernel_leader(&mut world);
    assert_eq!(kernel0, "0");
    let mut refer = RefOps::new(ops_cfg(), members, kernel0);
    let mut ops = create_ops(&world);
    rollout_to_current(&mut world, &mut ops, &mut refer, v1_signed(), V1_BYTES);
    assert_eq!(kernel_leader(&mut world), "v1");
    assert_pin(&world.store, PIN_CURRENT, V1_HEX);

    rollout_to_current(&mut world, &mut ops, &mut refer, v2_signed(), V2_BYTES);
    assert_eq!(kernel_leader(&mut world), "v2");
    assert_pin(&world.store, PIN_CURRENT, V2_HEX);
    assert_pin(&world.store, PIN_PREVIOUS, V1_HEX);
    assert_no_pin(&world.store, PIN_SHADOW);

    let i = leader_idx(&mut world);
    let got = ops.rollback(
        "v2 is bad",
        world.group.get(i).unwrap(),
        &mut world.store,
        world.now,
    );
    let exp = refer.rollback("v2 is bad");
    assert_result_tag(&got, &exp, "rollback v2");
    got.expect("rollback");
    assert_eq!(kernel_leader(&mut world), "v1");
    assert_eq!(refer.kernel_version, "v1");
    assert_pin(&world.store, PIN_CURRENT, V1_HEX);
    assert_no_pin(&world.store, PIN_SHADOW);
    assert_eq!(
        cas_pins(&world.store).get(PIN_CURRENT).map(String::as_str),
        Some(V1_HEX)
    );
    let st = status_of(&ops, &mut world);
    assert_state_rolled_back(&st, "v2 is bad");
    assert_eq!(st.kernel_version, "v1");
    assert_live_applied(&st, "v1");
    assert_has_event(&ops, EVENT_ROLLBACK);
}

#[test]
fn rollback_from_refused_is_wrong_state() {
    let mut world = fresh_world();
    let mut ops = create_ops(&world);
    let _ = ops.propose(
        common::one_sig_release("v1", common::v1_digest()),
        V1_BYTES,
        &mut world.store,
        world.now,
    );
    assert!(matches!(
        status_of(&ops, &mut world).state,
        ReleaseState::Refused { .. }
    ));
    let i = leader_idx(&mut world);
    assert_wrong_state(
        &ops.rollback(
            "x",
            world.group.get(i).unwrap(),
            &mut world.store,
            world.now,
        )
        .unwrap_err(),
    );
}

#[test]
fn rollback_from_rolled_back_is_wrong_state() {
    let mut world = fresh_world();
    let mut ops = create_ops(&world);
    ops.propose(v1_signed(), V1_BYTES, &mut world.store, world.now)
        .unwrap();
    let i = leader_idx(&mut world);
    ops.rollback(
        "first",
        world.group.get(i).unwrap(),
        &mut world.store,
        world.now,
    )
    .unwrap();
    let i = leader_idx(&mut world);
    assert_wrong_state(
        &ops.rollback(
            "second",
            world.group.get(i).unwrap(),
            &mut world.store,
            world.now,
        )
        .unwrap_err(),
    );
}

#[test]
fn propose_after_rollback_starts_a_new_release() {
    let mut world = fresh_world();
    let mut ops = create_ops(&world);
    ops.propose(v1_signed(), V1_BYTES, &mut world.store, world.now)
        .unwrap();
    let i = leader_idx(&mut world);
    ops.rollback(
        "abort",
        world.group.get(i).unwrap(),
        &mut world.store,
        world.now,
    )
    .unwrap();
    ops.propose(v2_signed(), V2_BYTES, &mut world.store, world.now)
        .unwrap();
    assert_eq!(status_of(&ops, &mut world).state, ReleaseState::Proposed);
    assert_pin(&world.store, PIN_SHADOW, V2_HEX);
}
