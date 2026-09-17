//! Group: status / status_json / status_text.

mod common;
mod reference;

use common::{
    assert_live_applied, assert_membership_live, create_ops, fresh_world, has_shadow_role,
    kernel_leader, leader_idx, membership, ops_cfg, status_of, v1_signed, V1_BYTES,
};
use prometheus_ops::{NodeRole, ReleaseState, Status};
use reference::RefOps;

#[test]
fn status_idle_shape() {
    let mut world = fresh_world();
    let members = membership(&mut world);
    let kernel = kernel_leader(&mut world);
    let ops = create_ops(&world);
    let st = status_of(&ops, &mut world);
    assert_eq!(st.kernel_version, kernel);
    assert_eq!(st.kernel_version, "0");
    assert!(st.release.is_none());
    assert_eq!(st.state, ReleaseState::Idle);
    assert_membership_live(&st, &members);
    assert_live_applied(&st, "0");
    assert!(!has_shadow_role(&st));
    for n in &st.nodes {
        assert_eq!(n.role, NodeRole::Live);
        assert!(n.healthy);
    }
}

#[test]
fn status_matches_cluster_kernel() {
    let mut world = fresh_world();
    let ops = create_ops(&world);
    let i = leader_idx(&mut world);
    let cluster = world.group.get(i).unwrap();
    let st = ops.status(cluster);
    assert_eq!(st.kernel_version, cluster.state().kernel_version);
}

#[test]
fn status_json_roundtrips_to_status() {
    let mut world = fresh_world();
    let mut ops = create_ops(&world);
    ops.propose(v1_signed(), V1_BYTES, &mut world.store, world.now)
        .unwrap();
    let i = leader_idx(&mut world);
    let cluster = world.group.get(i).unwrap();
    let st = ops.status(cluster);
    let json = ops.status_json(cluster).expect("status_json");
    let parsed: Status = serde_json::from_str(&json).expect("valid JSON of Status");
    assert_eq!(parsed, st);
    let v: serde_json::Value = serde_json::from_str(&json).expect("json value");
    assert_eq!(v["kernel_version"], st.kernel_version);
    assert_eq!(v["state"], "proposed");
}

#[test]
fn status_text_is_nonempty_and_contains_kernel() {
    let mut world = fresh_world();
    let ops = create_ops(&world);
    let i = leader_idx(&mut world);
    let cluster = world.group.get(i).unwrap();
    let text = ops.status_text(cluster);
    assert!(!text.is_empty());
    assert!(
        text.contains(&cluster.state().kernel_version),
        "status_text must contain kernel_version, got {text:?}"
    );
}

#[test]
fn status_after_propose_matches_reference() {
    let mut world = fresh_world();
    let members = membership(&mut world);
    let kernel = kernel_leader(&mut world);
    let mut refer = RefOps::new(ops_cfg(), members, kernel);
    let mut ops = create_ops(&world);
    ops.propose(v1_signed(), V1_BYTES, &mut world.store, world.now)
        .unwrap();
    refer.propose(v1_signed(), V1_BYTES).unwrap();
    let st = status_of(&ops, &mut world);
    assert_eq!(st.state, refer.state);
    assert_eq!(st.release, refer.release);
    assert_eq!(st.kernel_version, refer.kernel_version);
}

#[test]
fn status_json_after_shadowing_uses_snake_case_state() {
    let mut world = fresh_world();
    let mut ops = create_ops(&world);
    ops.propose(v1_signed(), V1_BYTES, &mut world.store, world.now)
        .unwrap();
    ops.start_shadow(&mut world.store, world.now).unwrap();
    let i = leader_idx(&mut world);
    let json = ops.status_json(world.group.get(i).unwrap()).expect("json");
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["state"], "shadowing");
    let nodes = v["nodes"].as_array().expect("nodes array");
    assert!(nodes.iter().any(|n| n["role"] == "shadow"));
    assert!(nodes.iter().any(|n| n["role"] == "live"));
}
