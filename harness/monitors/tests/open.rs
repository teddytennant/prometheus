//! Group: `Monitors::open`. Absent freeze path is not frozen. A freeze file
//! written by kill (or a well-formed frozen snapshot) loads as frozen.

mod common;
mod reference;

use common::{assert_snap_eq, assert_unfrozen_pair, fresh_world, open_pair, write_snap, NOW0};
use prometheus_monitors::{KillSnapshot, Monitors};
use reference::{empty_snapshot, RefMonitors};

#[test]
fn open_without_freeze_file_is_not_frozen() {
    let world = fresh_world();
    assert!(!world.prod.freeze_path.exists());
    assert!(!world.refer.freeze_path.exists());
    let (prod, refer) = open_pair(&world);
    assert_unfrozen_pair(&prod, &refer);
    assert_eq!(prod.snapshot().violations.len(), 0);
    assert_eq!(prod.snapshot().frozen_at, None);
    assert_eq!(prod.snapshot().reason, None);
    assert_snap_eq(&prod.snapshot(), &empty_snapshot());
    assert!(
        !world.prod.freeze_path.exists(),
        "open must not create the freeze file"
    );
}

#[test]
fn open_does_not_create_parent_freeze_file() {
    let world = fresh_world();
    let mut cfg = world.prod.clone();
    cfg.freeze_path = world.tmp.path().join("nested").join("kill.freeze");
    let m = Monitors::open(cfg).expect("open missing nested");
    assert!(!m.frozen());
    assert!(!world.tmp.path().join("nested").exists());
}

#[test]
fn open_loads_hand_written_frozen_snapshot() {
    let world = fresh_world();
    let snap = KillSnapshot {
        frozen: true,
        frozen_at: Some(NOW0),
        reason: Some("preexisting".into()),
        violations: vec![common::planted_violation(
            prometheus_monitors::Boundary::Kernel,
            NOW0,
            "cap",
        )],
    };
    write_snap(&world.prod.freeze_path, &snap);
    write_snap(&world.refer.freeze_path, &snap);
    let prod = Monitors::open(world.prod.clone()).expect("prod open frozen");
    let refer = RefMonitors::open(world.refer.clone()).expect("ref open frozen");
    assert!(prod.frozen());
    assert!(refer.frozen());
    assert_snap_eq(&prod.snapshot(), &snap);
    assert_snap_eq(&refer.snapshot(), &snap);
}

#[test]
fn reopen_after_kill_is_frozen() {
    let world = fresh_world();
    {
        let (mut prod, mut refer) = open_pair(&world);
        let got = prod.kill("manual", NOW0);
        let exp = refer.kill("manual", NOW0);
        common::assert_result_eq(got, exp, "kill");
        assert!(prod.frozen());
    }
    let prod = Monitors::open(world.prod.clone()).expect("reopen prod");
    let refer = RefMonitors::open(world.refer.clone()).expect("reopen ref");
    assert!(prod.frozen(), "prod reopen must see the freeze file");
    assert!(refer.frozen());
    assert_eq!(prod.snapshot().frozen_at, Some(NOW0));
    assert_eq!(prod.snapshot().reason.as_deref(), Some("manual"));
    assert_snap_eq(&prod.snapshot(), &refer.snapshot());
}

#[test]
fn freeze_file_present_after_trip_loads_frozen() {
    let world = fresh_world();
    {
        let (mut prod, mut refer) = open_pair(&world);
        let _ = prod.note_held_out_access("agent", NOW0);
        let _ = refer.note_held_out_access("agent", NOW0);
        assert!(common::freeze_exists(&world.prod.freeze_path));
        assert!(common::freeze_exists(&world.refer.freeze_path));
    }
    let prod = Monitors::open(world.prod.clone()).expect("reopen");
    assert!(prod.frozen());
    assert_eq!(prod.snapshot().violations.len(), 1);
}

#[test]
fn two_opens_of_the_same_frozen_file_agree() {
    let world = fresh_world();
    {
        let mut m = Monitors::open(world.prod.clone()).expect("open");
        m.kill("once", NOW0).expect("kill");
    }
    let a = Monitors::open(world.prod.clone()).expect("a");
    let b = Monitors::open(world.prod.clone()).expect("b");
    assert!(a.frozen() && b.frozen());
    assert_snap_eq(&a.snapshot(), &b.snapshot());
}
