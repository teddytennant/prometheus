//! Group: shadow node, start_shadow, shadow_pass, shadow_fail.

mod common;
mod reference;

use common::{
    assert_has_event, assert_live_applied, assert_membership_live, assert_no_event, assert_no_pin,
    assert_pin, assert_result_tag, assert_state_rolled_back, assert_wrong_state, create_ops,
    fresh_world, has_shadow_role, kernel_leader, live_nodes, membership, ops_cfg, status_of,
    v1_signed, V1_BYTES, V1_HEX,
};
use prometheus_ops::{
    NodeRole, ReleaseState, EVENT_APPLY, EVENT_SHADOW, EVENT_SHADOW_RESULT, PIN_CURRENT, PIN_SHADOW,
};
use reference::RefOps;

fn proposed(world: &mut common::World) -> (prometheus_ops::Ops, RefOps) {
    let members = membership(world);
    let kernel = kernel_leader(world);
    let mut refer = RefOps::new(ops_cfg(), members, kernel);
    let mut ops = create_ops(world);
    let rel = v1_signed();
    let got = ops.propose(rel.clone(), V1_BYTES, &mut world.store, world.now);
    let exp = refer.propose(rel, V1_BYTES);
    assert_result_tag(&got, &exp, "propose");
    (ops, refer)
}

#[test]
fn start_shadow_wrong_state_if_not_proposed() {
    let mut world = fresh_world();
    let mut ops = create_ops(&world);
    let err = ops.start_shadow(&mut world.store, world.now).expect_err("idle");
    assert_wrong_state(&err);
}

#[test]
fn start_shadow_pins_shadow_does_not_change_live_kernel_or_live_applied() {
    let mut world = fresh_world();
    let members = membership(&mut world);
    let kernel = kernel_leader(&mut world);
    let (mut ops, mut refer) = proposed(&mut world);
    let got = ops.start_shadow(&mut world.store, world.now);
    let exp = refer.start_shadow();
    assert_result_tag(&got, &exp, "start_shadow");
    got.expect("start_shadow");
    assert_eq!(kernel_leader(&mut world), kernel);
    assert_pin(&world.store, PIN_SHADOW, V1_HEX);
    assert_no_pin(&world.store, PIN_CURRENT);
    let st = status_of(&ops, &mut world);
    assert_eq!(st.state, ReleaseState::Shadowing);
    assert_eq!(st.kernel_version, kernel);
    assert_membership_live(&st, &members);
    assert_live_applied(&st, &kernel);
    assert!(has_shadow_role(&st), "shadow role must appear");
    for n in live_nodes(&st) {
        assert_ne!(n.applied, "v1", "live node must not yet run the proposal");
        assert_eq!(n.role, NodeRole::Live);
    }
    assert_has_event(&ops, EVENT_SHADOW);
}

#[test]
fn start_shadow_twice_is_wrong_state() {
    let mut world = fresh_world();
    let (mut ops, mut refer) = proposed(&mut world);
    ops.start_shadow(&mut world.store, world.now).expect("first");
    refer.start_shadow().expect("ref first");
    let err = ops.start_shadow(&mut world.store, world.now).expect_err("second");
    assert_wrong_state(&err);
    assert_wrong_state(&refer.start_shadow().unwrap_err());
}

#[test]
fn shadow_pass_wrong_state_if_not_shadowing() {
    let world = fresh_world();
    let mut ops = create_ops(&world);
    assert_wrong_state(&ops.shadow_pass(world.now).unwrap_err());
    let mut world = fresh_world();
    let (mut ops, _) = proposed(&mut world);
    assert_wrong_state(&ops.shadow_pass(world.now).unwrap_err());
}

#[test]
fn shadow_fail_wrong_state_if_not_shadowing() {
    let mut world = fresh_world();
    let mut ops = create_ops(&world);
    assert_wrong_state(&ops.shadow_fail("nope", &mut world.store, world.now).unwrap_err());
}

#[test]
fn shadow_pass_enables_apply_one_without_changing_kernel() {
    let mut world = fresh_world();
    let kernel = kernel_leader(&mut world);
    let members = membership(&mut world);
    let (mut ops, mut refer) = proposed(&mut world);
    ops.start_shadow(&mut world.store, world.now).unwrap();
    refer.start_shadow().unwrap();
    let got = ops.shadow_pass(world.now);
    let exp = refer.shadow_pass();
    assert_result_tag(&got, &exp, "pass");
    let st = status_of(&ops, &mut world);
    assert_eq!(st.state, ReleaseState::Shadowing);
    assert_eq!(kernel_leader(&mut world), kernel);
    assert_has_event(&ops, EVENT_SHADOW_RESULT);
    ops.apply_one(&members[0], world.now)
        .expect("apply after pass");
    assert_eq!(kernel_leader(&mut world), kernel);
}

#[test]
fn apply_one_before_shadow_pass_is_wrong_state() {
    let mut world = fresh_world();
    let members = membership(&mut world);
    let (mut ops, mut refer) = proposed(&mut world);
    ops.start_shadow(&mut world.store, world.now).unwrap();
    refer.start_shadow().unwrap();
    let err = ops.apply_one(&members[0], world.now).expect_err("no pass");
    assert_wrong_state(&err);
    assert_wrong_state(&refer.apply_one(&members[0]).unwrap_err());
    assert_no_event(&ops, EVENT_APPLY);
}

#[test]
fn shadow_fail_unpins_shadow_rolls_back_kernel_unchanged() {
    let mut world = fresh_world();
    let kernel = kernel_leader(&mut world);
    let (mut ops, mut refer) = proposed(&mut world);
    ops.start_shadow(&mut world.store, world.now).unwrap();
    refer.start_shadow().unwrap();
    let got = ops.shadow_fail("smoke failed", &mut world.store, world.now);
    let exp = refer.shadow_fail("smoke failed");
    assert_result_tag(&got, &exp, "fail");
    let st = status_of(&ops, &mut world);
    assert_state_rolled_back(&st, "smoke failed");
    assert_eq!(kernel_leader(&mut world), kernel);
    assert_no_pin(&world.store, PIN_SHADOW);
    assert_has_event(&ops, EVENT_SHADOW_RESULT);
    assert!(!has_shadow_role(&st));
    assert_live_applied(&st, &kernel);
}

#[test]
fn shadow_node_is_distinct_from_live() {
    let mut world = fresh_world();
    let (mut ops, _) = proposed(&mut world);
    ops.start_shadow(&mut world.store, world.now).unwrap();
    let st = status_of(&ops, &mut world);
    let live: Vec<_> = live_nodes(&st).into_iter().map(|n| n.id.clone()).collect();
    let shadows: Vec<_> = st
        .nodes
        .iter()
        .filter(|n| n.role == NodeRole::Shadow)
        .cloned()
        .collect();
    assert!(!shadows.is_empty(), "expected a shadow node");
    for s in &shadows {
        assert_eq!(s.role, NodeRole::Shadow);
        assert!(!live.contains(&s.id), "shadow id {} also listed as live", s.id.0);
    }
    for n in live_nodes(&st) {
        assert_eq!(n.role, NodeRole::Live);
        assert_ne!(n.role, NodeRole::Shadow);
    }
}
