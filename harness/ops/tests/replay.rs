//! Group: drop + open replays the same state and status.

mod common;
mod reference;

use common::{
    assert_live_applied, cas_pins, create_ops, event_types, fresh_world, kernel_leader, leader_idx,
    membership, ops_cfg, status_of, v1_signed, v2_signed, V1_BYTES, V2_BYTES,
};
use prometheus_ops::{Ops, ReleaseState};
use reference::RefOps;

#[test]
fn replay_after_propose() {
    let mut world = fresh_world();
    let members = membership(&mut world);
    let kernel = kernel_leader(&mut world);
    let mut refer = RefOps::new(ops_cfg(), members, kernel);
    let snap;
    let events;
    {
        let mut ops = create_ops(&world);
        ops.propose(v1_signed(), V1_BYTES, &mut world.store, world.now)
            .unwrap();
        refer.propose(v1_signed(), V1_BYTES).unwrap();
        snap = status_of(&ops, &mut world);
        events = event_types(&ops);
        drop(ops);
    }
    let ops = Ops::open(&world.ops_dir, ops_cfg()).expect("open");
    let st = status_of(&ops, &mut world);
    assert_eq!(st.state, snap.state);
    assert_eq!(st.release, snap.release);
    assert_eq!(st.kernel_version, snap.kernel_version);
    assert_eq!(st.state, refer.state);
    assert_eq!(event_types(&ops), events);
    assert_eq!(event_types(&ops), refer.events);
}

#[test]
fn replay_after_full_rollout() {
    let mut world = fresh_world();
    let members = membership(&mut world);
    let kernel = kernel_leader(&mut world);
    let mut refer = RefOps::new(ops_cfg(), members.clone(), kernel);
    {
        let mut ops = create_ops(&world);
        ops.propose(v1_signed(), V1_BYTES, &mut world.store, world.now)
            .unwrap();
        refer.propose(v1_signed(), V1_BYTES).unwrap();
        ops.start_shadow(&mut world.store, world.now).unwrap();
        refer.start_shadow().unwrap();
        ops.shadow_pass(world.now).unwrap();
        refer.shadow_pass().unwrap();
        for m in &members {
            ops.apply_one(m, world.now).unwrap();
            refer.apply_one(m).unwrap();
        }
        let i = leader_idx(&mut world);
        ops.promote(world.group.get(i).unwrap(), &mut world.store, world.now)
            .unwrap();
        refer.promote().unwrap();
        drop(ops);
    }
    let ops = Ops::open(&world.ops_dir, ops_cfg()).expect("open");
    let st = status_of(&ops, &mut world);
    assert_eq!(st.state, ReleaseState::Current);
    assert_eq!(st.kernel_version, "v1");
    assert_eq!(kernel_leader(&mut world), "v1");
    assert_live_applied(&st, "v1");
    assert_eq!(st.state, refer.state);
    assert_eq!(cas_pins(&world.store), refer.pins);
}

#[test]
fn replay_after_rollback() {
    let mut world = fresh_world();
    let members = membership(&mut world);
    let kernel = kernel_leader(&mut world);
    let mut refer = RefOps::new(ops_cfg(), members, kernel);
    {
        let mut ops = create_ops(&world);
        ops.propose(v1_signed(), V1_BYTES, &mut world.store, world.now)
            .unwrap();
        refer.propose(v1_signed(), V1_BYTES).unwrap();
        let i = leader_idx(&mut world);
        ops.rollback(
            "abort",
            world.group.get(i).unwrap(),
            &mut world.store,
            world.now,
        )
        .unwrap();
        refer.rollback("abort").unwrap();
        drop(ops);
    }
    let ops = Ops::open(&world.ops_dir, ops_cfg()).expect("open");
    let st = status_of(&ops, &mut world);
    match &st.state {
        ReleaseState::RolledBack { reason } => assert_eq!(reason, "abort"),
        other => panic!("expected RolledBack, got {other:?}"),
    }
    assert_eq!(st.state, refer.state);
}

#[test]
fn replay_mid_rollout_preserves_which_nodes_applied() {
    let mut world = fresh_world();
    let members = membership(&mut world);
    let kernel = kernel_leader(&mut world);
    let mut refer = RefOps::new(ops_cfg(), members.clone(), kernel);
    {
        let mut ops = create_ops(&world);
        ops.propose(v2_signed(), V2_BYTES, &mut world.store, world.now)
            .unwrap();
        refer.propose(v2_signed(), V2_BYTES).unwrap();
        ops.start_shadow(&mut world.store, world.now).unwrap();
        refer.start_shadow().unwrap();
        ops.shadow_pass(world.now).unwrap();
        refer.shadow_pass().unwrap();
        ops.apply_one(&members[0], world.now).unwrap();
        refer.apply_one(&members[0]).unwrap();
        drop(ops);
    }
    let ops = Ops::open(&world.ops_dir, ops_cfg()).expect("open");
    let st = status_of(&ops, &mut world);
    let n0 = st.nodes.iter().find(|n| n.id == members[0]).expect("n0");
    assert_eq!(n0.applied, "v2");
    for m in members.iter().skip(1) {
        let n = st.nodes.iter().find(|n| n.id == *m).expect("member");
        assert_eq!(n.applied, "0");
    }
    match st.state {
        ReleaseState::Rolling { node } => assert_eq!(node, members[0]),
        other => panic!("expected Rolling, got {other:?}"),
    }
}
