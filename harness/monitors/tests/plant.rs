//! Group: L3 gate. `plant(boundary)` is the same path as a real trip for
//! each of the four boundaries.

mod common;
mod reference;

use common::{
    assert_frozen_pair, assert_has_violation, assert_result_eq, assert_snap_eq, fresh_world,
    open_pair, unwrap_err, ALL_BOUNDARIES, GRADER_ID, HASH_B, NOW0,
};
use prometheus_monitors::{Boundary, Error, Monitors};
use reference::{freeze_reason, trip_err, RefMonitors};

fn real_trip(
    m: &mut Monitors,
    boundary: Boundary,
    detail: &str,
    now: prometheus_monitors::NowMs,
) -> prometheus_monitors::Result<()> {
    match boundary {
        Boundary::Kernel => m.note_kernel_escape(detail, now),
        Boundary::Graders => m.note_grader_hash(detail, HASH_B, now),
        Boundary::HeldOut => m.note_held_out_access(detail, now),
        Boundary::Monitors => m.note_self_check(detail, now),
    }
}

fn real_trip_ref(
    m: &mut RefMonitors,
    boundary: Boundary,
    detail: &str,
    now: prometheus_monitors::NowMs,
) -> prometheus_monitors::Result<()> {
    match boundary {
        Boundary::Kernel => m.note_kernel_escape(detail, now),
        Boundary::Graders => m.note_grader_hash(detail, HASH_B, now),
        Boundary::HeldOut => m.note_held_out_access(detail, now),
        Boundary::Monitors => m.note_self_check(detail, now),
    }
}

fn detail_for(boundary: Boundary) -> &'static str {
    match boundary {
        Boundary::Kernel => "ptrace",
        Boundary::Graders => GRADER_ID,
        Boundary::HeldOut => "agent-x",
        Boundary::Monitors => "tamper",
    }
}

#[test]
fn plant_each_boundary_trips_kill() {
    for boundary in ALL_BOUNDARIES {
        let detail = detail_for(boundary);
        let world = fresh_world();
        let (mut prod, mut refer) = open_pair(&world);
        assert_result_eq(
            prod.plant(boundary, detail, NOW0),
            refer.plant(boundary, detail, NOW0),
            &format!("plant {boundary:?}"),
        );
        assert_frozen_pair(&prod, &refer);
        assert!(prod.frozen(), "L3 gate: plant({boundary:?}) must freeze");
        assert_has_violation(&prod.snapshot(), boundary, NOW0, detail);
        assert_eq!(prod.snapshot().frozen_at, Some(NOW0));
        assert_eq!(
            prod.snapshot().reason.as_deref(),
            Some(freeze_reason(boundary))
        );
        assert!(
            world.prod.freeze_path.is_file(),
            "plant({boundary:?}) must write the freeze file"
        );
        assert_eq!(
            unwrap_err(
                {
                    let world = fresh_world();
                    let mut m = Monitors::open(world.prod.clone()).unwrap();
                    m.plant(boundary, detail, NOW0)
                },
                &format!("prod plant {boundary:?}")
            ),
            trip_err(boundary)
        );
    }
}

#[test]
fn plant_is_the_same_path_as_a_real_trip() {
    for boundary in ALL_BOUNDARIES {
        let detail = detail_for(boundary);
        let planted = fresh_world();
        let real = fresh_world();
        let (mut prod_plant, mut ref_plant) = open_pair(&planted);
        let (mut prod_real, mut ref_real) = open_pair(&real);

        let p = prod_plant.plant(boundary, detail, NOW0);
        let r = real_trip(&mut prod_real, boundary, detail, NOW0);
        assert_result_eq(p, r, &format!("prod plant vs real {boundary:?}"));
        assert_snap_eq(&prod_plant.snapshot(), &prod_real.snapshot());

        let p = ref_plant.plant(boundary, detail, NOW0);
        let r = real_trip_ref(&mut ref_real, boundary, detail, NOW0);
        assert_result_eq(p, r, &format!("ref plant vs real {boundary:?}"));
        assert_snap_eq(&prod_plant.snapshot(), &ref_plant.snapshot());
        assert_snap_eq(&prod_real.snapshot(), &ref_real.snapshot());
    }
}

#[test]
fn plant_graders_returns_grader_hash_mismatch() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    assert_eq!(
        unwrap_err(prod.plant(Boundary::Graders, GRADER_ID, NOW0), "prod"),
        Error::GraderHashMismatch
    );
    assert_eq!(
        unwrap_err(refer.plant(Boundary::Graders, GRADER_ID, NOW0), "ref"),
        Error::GraderHashMismatch
    );
    assert!(prod.frozen());
}

#[test]
fn plant_held_out_returns_held_out_access() {
    let world = fresh_world();
    let mut prod = Monitors::open(world.prod.clone()).unwrap();
    assert_eq!(
        unwrap_err(prod.plant(Boundary::HeldOut, "who", NOW0), "prod"),
        Error::HeldOutAccess
    );
}

#[test]
fn plant_monitors_returns_monitor_self_check() {
    let world = fresh_world();
    let mut prod = Monitors::open(world.prod.clone()).unwrap();
    assert_eq!(
        unwrap_err(prod.plant(Boundary::Monitors, "x", NOW0), "prod"),
        Error::MonitorSelfCheck
    );
}

#[test]
fn plant_kernel_returns_message_kernel_escape() {
    let world = fresh_world();
    let mut prod = Monitors::open(world.prod.clone()).unwrap();
    assert_eq!(
        unwrap_err(prod.plant(Boundary::Kernel, "cap", NOW0), "prod"),
        Error::Message("kernel escape".into())
    );
}
