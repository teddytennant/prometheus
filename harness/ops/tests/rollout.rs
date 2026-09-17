//! Group: apply_one one-node-at-a-time, promote pins and kernel, mark_unhealthy.

mod common;
mod reference;

use common::{
    assert_has_event, assert_live_applied, assert_no_pin, assert_not_found, assert_pin,
    assert_result_tag, assert_state_rolled_back, assert_wrong_state, cas_pins, create_ops,
    fresh_world, has_shadow_role, kernel_leader, leader_idx, membership, nid, ops_cfg, status_of,
    v1_signed, V1_BYTES, V1_HEX,
};
use prometheus_ops::{
    NodeRole, ReleaseState, EVENT_APPLY, EVENT_PROMOTE, EVENT_UNHEALTHY, PIN_CURRENT, PIN_PREVIOUS,
    PIN_SHADOW,
};
use reference::RefOps;

fn ready(world: &mut common::World) -> (prometheus_ops::Ops, RefOps, Vec<prometheus_raft::NodeId>) {
    let members = membership(world);
    let kernel = kernel_leader(world);
    let mut refer = RefOps::new(ops_cfg(), members.clone(), kernel);
    let mut ops = create_ops(world);
    let rel = v1_signed();
    assert_result_tag(
        &ops.propose(rel.clone(), V1_BYTES, &mut world.store, world.now),
        &refer.propose(rel, V1_BYTES),
        "propose",
    );
    assert_result_tag(&ops.start_shadow(&mut world.store, world.now), &refer.start_shadow(), "shadow");
    assert_result_tag(&ops.shadow_pass(world.now), &refer.shadow_pass(), "pass");
    (ops, refer, members)
}

#[test]
fn apply_one_wrong_state_before_shadow_pass() {
    let mut world = fresh_world();
    let members = membership(&mut world);
    let mut ops = create_ops(&world);
    assert_wrong_state(&ops.apply_one(&members[0], world.now).unwrap_err());
    ops.propose(v1_signed(), V1_BYTES, &mut world.store, world.now)
        .unwrap();
    assert_wrong_state(&ops.apply_one(&members[0], world.now).unwrap_err());
}

#[test]
fn apply_one_unknown_node_is_not_found() {
    let mut world = fresh_world();
    let (mut ops, mut refer, _) = ready(&mut world);
    let ghost = nid("no-such-node");
    let got = ops.apply_one(&ghost, world.now);
    let exp = refer.apply_one(&ghost);
    assert_result_tag(&got, &exp, "unknown");
    assert_not_found(got.as_ref().unwrap_err());
}

#[test]
fn apply_one_updates_exactly_one_live_node_and_does_not_set_kernel() {
    let mut world = fresh_world();
    let kernel = kernel_leader(&mut world);
    let (mut ops, mut refer, members) = ready(&mut world);
    let got = ops.apply_one(&members[0], world.now);
    let exp = refer.apply_one(&members[0]);
    assert_result_tag(&got, &exp, "apply0");
    got.expect("apply");
    assert_eq!(kernel_leader(&mut world), kernel);
    let st = status_of(&ops, &mut world);
    match &st.state {
        ReleaseState::Rolling { node } => assert_eq!(node, &members[0]),
        other => panic!("expected Rolling, got {other:?}"),
    }
    let n0 = st
        .nodes
        .iter()
        .find(|n| n.id == members[0])
        .expect("node 0");
    assert_eq!(n0.applied, "v1");
    assert!(n0.healthy);
    assert_eq!(n0.role, NodeRole::Live);
    for m in members.iter().skip(1) {
        let n = st.nodes.iter().find(|n| n.id == *m).expect("member");
        assert_eq!(n.applied, kernel, "other live nodes stay on old version");
        assert_eq!(n.role, NodeRole::Live);
    }
    assert_has_event(&ops, EVENT_APPLY);
}

#[test]
fn apply_one_on_shadow_node_is_wrong_state() {
    let mut world = fresh_world();
    let (mut ops, _, _) = ready(&mut world);
    let st = status_of(&ops, &mut world);
    let shadow = st
        .nodes
        .iter()
        .find(|n| n.role == NodeRole::Shadow)
        .expect("shadow")
        .id
        .clone();
    assert_wrong_state(&ops.apply_one(&shadow, world.now).unwrap_err());
}

#[test]
fn promote_requires_all_live_nodes_applied() {
    let mut world = fresh_world();
    let kernel = kernel_leader(&mut world);
    let (mut ops, mut refer, members) = ready(&mut world);
    let i = leader_idx(&mut world);
    let err = ops
        .promote(world.group.get(i).unwrap(), &mut world.store, world.now)
        .expect_err("none applied");
    assert_wrong_state(&err);
    assert_wrong_state(&refer.promote().unwrap_err());
    assert_eq!(kernel_leader(&mut world), kernel);

    ops.apply_one(&members[0], world.now).unwrap();
    refer.apply_one(&members[0]).unwrap();
    let i = leader_idx(&mut world);
    let err = ops
        .promote(world.group.get(i).unwrap(), &mut world.store, world.now)
        .expect_err("one of three");
    assert_wrong_state(&err);
    assert_eq!(kernel_leader(&mut world), kernel);
}

#[test]
fn promote_after_all_live_applied_sets_kernel_and_pins() {
    let mut world = fresh_world();
    let kernel = kernel_leader(&mut world);
    assert_eq!(kernel, "0");
    let (mut ops, mut refer, members) = ready(&mut world);
    for m in &members {
        assert_result_tag(
            &ops.apply_one(m, world.now),
            &refer.apply_one(m),
            &format!("apply {}", m.0),
        );
    }
    let i = leader_idx(&mut world);
    let got = ops.promote(world.group.get(i).unwrap(), &mut world.store, world.now);
    let exp = refer.promote();
    assert_result_tag(&got, &exp, "promote");
    got.expect("promote");
    assert_eq!(kernel_leader(&mut world), "v1");
    assert_eq!(refer.kernel_version, "v1");
    assert_pin(&world.store, PIN_CURRENT, V1_HEX);
    assert_no_pin(&world.store, PIN_SHADOW);
    assert_eq!(cas_pins(&world.store), refer.pins);
    let st = status_of(&ops, &mut world);
    assert_eq!(st.state, ReleaseState::Current);
    assert_eq!(st.kernel_version, "v1");
    assert_live_applied(&st, "v1");
    assert!(!has_shadow_role(&st), "shadow gone after promote");
    assert_has_event(&ops, EVENT_PROMOTE);
    assert_no_pin(&world.store, PIN_PREVIOUS); // first release: no previous artifact
}

#[test]
fn mark_unhealthy_during_rollout_triggers_rollback() {
    let mut world = fresh_world();
    let kernel = kernel_leader(&mut world);
    let (mut ops, mut refer, members) = ready(&mut world);
    ops.apply_one(&members[0], world.now).unwrap();
    refer.apply_one(&members[0]).unwrap();
    let i = leader_idx(&mut world);
    let got = ops.mark_unhealthy(
        &members[0],
        "node sick",
        world.group.get(i).unwrap(),
        &mut world.store,
        world.now,
    );
    let exp = refer.mark_unhealthy(&members[0], "node sick");
    assert_result_tag(&got, &exp, "unhealthy");
    got.expect("mark_unhealthy");
    let st = status_of(&ops, &mut world);
    assert_state_rolled_back(&st, "node sick");
    assert_eq!(kernel_leader(&mut world), kernel);
    assert_no_pin(&world.store, PIN_SHADOW);
    assert_has_event(&ops, EVENT_UNHEALTHY);
    assert_live_applied(&st, &kernel);
}

#[test]
fn mark_unhealthy_unknown_node() {
    let mut world = fresh_world();
    let (mut ops, mut refer, _) = ready(&mut world);
    let i = leader_idx(&mut world);
    let ghost = nid("ghost");
    let got = ops.mark_unhealthy(
        &ghost,
        "nope",
        world.group.get(i).unwrap(),
        &mut world.store,
        world.now,
    );
    let exp = refer.mark_unhealthy(&ghost, "nope");
    assert_result_tag(&got, &exp, "ghost");
    assert_not_found(got.as_ref().unwrap_err());
}

#[test]
fn apply_one_is_recorded_once_per_node() {
    let mut world = fresh_world();
    let (mut ops, _, members) = ready(&mut world);
    for m in &members {
        ops.apply_one(m, world.now).unwrap();
    }
    let applies = common::event_types(&ops)
        .into_iter()
        .filter(|t| t == EVENT_APPLY)
        .count();
    assert_eq!(applies, members.len());
}
