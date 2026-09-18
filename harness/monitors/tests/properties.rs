//! Group: properties. `frozen()` is true after any trip. Snapshot agrees
//! with `frozen()`. Durable reopen. Reference parity.

mod common;
mod reference;

use common::{
    assert_frozen_pair, assert_has_violation, assert_result_eq, assert_snap_eq,
    assert_unfrozen_pair, fresh_world, open_pair, ALL_BOUNDARIES, GRADER_ID, HASH_A, NOW0, NOW1,
};
use prometheus_monitors::{Boundary, Monitors};
use reference::RefMonitors;

#[test]
fn frozen_true_after_any_trip() {
    let trips: Vec<(&str, Box<dyn Fn(&mut Monitors, prometheus_monitors::NowMs)>)> = vec![
        (
            "grader mismatch",
            Box::new(|m, now| {
                let _ = m.note_grader_hash(GRADER_ID, "nope", now);
            }),
        ),
        (
            "held_out",
            Box::new(|m, now| {
                let _ = m.note_held_out_access("w", now);
            }),
        ),
        (
            "kernel",
            Box::new(|m, now| {
                let _ = m.note_kernel_escape("cap", now);
            }),
        ),
        (
            "self_check",
            Box::new(|m, now| {
                let _ = m.note_self_check("t", now);
            }),
        ),
        (
            "kill",
            Box::new(|m, now| {
                let _ = m.kill("k", now);
            }),
        ),
    ];
    for (name, step) in trips {
        let world = fresh_world();
        let mut prod = Monitors::open(world.prod.clone()).unwrap();
        assert!(!prod.frozen(), "{name} start");
        step(&mut prod, NOW0);
        assert!(prod.frozen(), "{name} must freeze");
        assert_eq!(
            prod.frozen(),
            prod.snapshot().frozen,
            "{name} frozen() vs snapshot"
        );
    }
}

#[test]
fn frozen_true_after_plant_of_each_boundary() {
    for boundary in ALL_BOUNDARIES {
        let world = fresh_world();
        let (mut prod, mut refer) = open_pair(&world);
        let _ = prod.plant(boundary, "p", NOW0);
        let _ = refer.plant(boundary, "p", NOW0);
        assert!(prod.frozen(), "plant({boundary:?})");
        assert_frozen_pair(&prod, &refer);
    }
}

#[test]
fn matching_hashes_never_freeze() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    for now in [0, NOW0, NOW1] {
        assert_result_eq(
            prod.note_grader_hash(GRADER_ID, HASH_A, now),
            refer.note_grader_hash(GRADER_ID, HASH_A, now),
            "match loop",
        );
        assert_unfrozen_pair(&prod, &refer);
    }
}

#[test]
fn frozen_equals_snapshot_frozen_across_ops() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    assert_eq!(prod.frozen(), prod.snapshot().frozen);
    let _ = prod.note_grader_hash(GRADER_ID, HASH_A, NOW0);
    assert_eq!(prod.frozen(), prod.snapshot().frozen);
    let _ = prod.kill("k", NOW0);
    assert_eq!(prod.frozen(), prod.snapshot().frozen);
    let _ = refer.kill("k", NOW0);
    assert_snap_eq(&prod.snapshot(), &refer.snapshot());
}

#[test]
fn reopen_snapshot_matches_live_snapshot() {
    let world = fresh_world();
    let live = {
        let (mut prod, mut refer) = open_pair(&world);
        let _ = prod.note_held_out_access("w", NOW0);
        let _ = refer.note_held_out_access("w", NOW0);
        let _ = prod.note_kernel_escape("c", NOW1);
        let _ = refer.note_kernel_escape("c", NOW1);
        assert_snap_eq(&prod.snapshot(), &refer.snapshot());
        prod.snapshot()
    };
    let prod = Monitors::open(world.prod.clone()).unwrap();
    let refer = RefMonitors::open(world.refer.clone()).unwrap();
    assert_snap_eq(&prod.snapshot(), &live);
    assert_snap_eq(&refer.snapshot(), &live);
}

#[test]
fn violations_append_in_call_order() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    let _ = prod.plant(Boundary::Kernel, "k", NOW0);
    let _ = prod.plant(Boundary::Graders, GRADER_ID, NOW1);
    let _ = refer.plant(Boundary::Kernel, "k", NOW0);
    let _ = refer.plant(Boundary::Graders, GRADER_ID, NOW1);
    assert_eq!(prod.snapshot().violations.len(), 2);
    assert_eq!(prod.snapshot().violations[0].boundary, Boundary::Kernel);
    assert_eq!(prod.snapshot().violations[1].boundary, Boundary::Graders);
    assert_snap_eq(&prod.snapshot(), &refer.snapshot());
}

#[test]
fn snapshot_clone_is_independent() {
    let world = fresh_world();
    let (mut prod, _) = open_pair(&world);
    let _ = prod.kill("r", NOW0);
    let mut snap = prod.snapshot();
    snap.frozen = false;
    snap.violations
        .push(common::planted_violation(Boundary::Monitors, NOW0, "x"));
    assert!(prod.frozen());
    assert!(prod.snapshot().violations.is_empty());
}

#[test]
fn four_boundaries_each_leave_their_own_violation() {
    for boundary in ALL_BOUNDARIES {
        let world = fresh_world();
        let mut prod = Monitors::open(world.prod.clone()).unwrap();
        let _ = prod.plant(boundary, "d", NOW0);
        assert_eq!(prod.snapshot().violations.len(), 1);
        assert_has_violation(&prod.snapshot(), boundary, NOW0, "d");
        for v in &prod.snapshot().violations {
            assert_eq!(v.boundary, boundary);
        }
    }
}
