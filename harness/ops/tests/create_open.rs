//! Group: create / open / default config / EventLog location.

mod common;
mod reference;

use common::{
    create_ops, event_types, fresh_world, log_dir, ops_cfg, ops_cfg_quorum, status_of, v1_signed,
    V1_BYTES,
};
use prometheus_ops::{Ops, OpsConfig, ReleaseState, DEFAULT_QUORUM};
use reference::RefOps;
use std::fs;

/// Allowed to pass against the stub: `Default` is implemented in the iface.
#[test]
fn ops_config_default_quorum_is_two() {
    let cfg = OpsConfig::default();
    assert_eq!(cfg.quorum, 2);
    assert_eq!(cfg.quorum, DEFAULT_QUORUM);
    assert_eq!(DEFAULT_QUORUM, 2);
}

#[test]
fn create_fails_if_dir_exists() {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("ops");
    fs::create_dir(&dir).expect("mkdir");
    assert!(
        Ops::create(&dir, ops_cfg()).is_err(),
        "create existing dir must fail"
    );
}

#[test]
fn create_fails_if_dir_exists_after_successful_create() {
    let world = fresh_world();
    let ops = create_ops(&world);
    drop(ops);
    assert!(
        Ops::create(&world.ops_dir, ops_cfg()).is_err(),
        "second create must fail"
    );
}

#[test]
fn create_makes_event_log_at_dir_log() {
    let world = fresh_world();
    let ops = create_ops(&world);
    assert_eq!(ops.dir(), world.ops_dir.as_path());
    assert_eq!(ops.config(), &ops_cfg());
    assert_eq!(ops.log().dir(), log_dir(&world.ops_dir));
    assert_eq!(ops.log().iter().count(), 0);
    assert!(log_dir(&world.ops_dir).is_dir(), "dir/log/ must exist");
}

#[test]
fn create_stores_custom_quorum() {
    let world = fresh_world();
    let cfg = ops_cfg_quorum(3);
    let ops = Ops::create(&world.ops_dir, cfg.clone()).expect("create");
    assert_eq!(ops.config(), &cfg);
    assert_eq!(ops.config().quorum, 3);
}

#[test]
fn open_missing_dir_fails() {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("nope");
    assert!(Ops::open(&dir, ops_cfg()).is_err(), "open missing must fail");
}

#[test]
fn open_replays_idle_after_create() {
    let mut world = fresh_world();
    let members = common::membership(&mut world);
    let kernel = common::kernel_leader(&mut world);
    {
        let ops = create_ops(&world);
        let st = status_of(&ops, &mut world);
        assert_eq!(st.state, ReleaseState::Idle);
        assert!(st.release.is_none());
        assert_eq!(st.kernel_version, kernel);
        common::assert_membership_live(&st, &members);
        common::assert_live_applied(&st, &kernel);
        assert!(!common::has_shadow_role(&st));
        drop(ops);
    }
    let ops = Ops::open(&world.ops_dir, ops_cfg()).expect("open");
    let st = status_of(&ops, &mut world);
    assert_eq!(st.state, ReleaseState::Idle);
    assert_eq!(st.kernel_version, kernel);
    assert_eq!(ops.config(), &ops_cfg());
    assert_eq!(event_types(&ops).len(), 0);
}

#[test]
fn open_replays_proposed_matches_reference() {
    let mut world = fresh_world();
    let members = common::membership(&mut world);
    let kernel = common::kernel_leader(&mut world);
    let mut refer = RefOps::new(ops_cfg(), members.clone(), kernel.clone());
    let rel = v1_signed();
    {
        let mut ops = create_ops(&world);
        let got = ops.propose(rel.clone(), V1_BYTES, &mut world.store, world.now);
        let exp = refer.propose(rel.clone(), V1_BYTES);
        common::assert_result_tag(&got, &exp, "propose");
        let st = status_of(&ops, &mut world);
        assert_eq!(st.state, ReleaseState::Proposed);
        assert_eq!(st.release.as_ref().map(|r| r.version.as_str()), Some("v1"));
        drop(ops);
    }
    let ops = Ops::open(&world.ops_dir, ops_cfg()).expect("open");
    let st = status_of(&ops, &mut world);
    assert_eq!(st.state, refer.state);
    assert_eq!(st.release, refer.release);
    assert_eq!(event_types(&ops), refer.events);
}
