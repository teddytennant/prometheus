//! Group: escape. The genome cannot disable the four boundaries. Deleting
//! the freeze file does not thaw a live instance. Kill stays frozen.

mod common;
mod reference;

use common::{assert_frozen_pair, fresh_world, open_pair, ALL_BOUNDARIES, GRADER_ID, NOW0, NOW1};
use prometheus_monitors::{Boundary, Monitors};

#[test]
fn genome_cannot_skip_any_boundary() {
    for boundary in ALL_BOUNDARIES {
        let world = fresh_world();
        let (mut prod, mut refer) = open_pair(&world);
        let detail = match boundary {
            Boundary::Graders => GRADER_ID,
            _ => "escape",
        };
        let _ = prod.plant(boundary, detail, NOW0);
        let _ = refer.plant(boundary, detail, NOW0);
        assert!(
            prod.frozen(),
            "plant({boundary:?}) must still trip; no disable path"
        );
        assert_frozen_pair(&prod, &refer);
    }
}

#[test]
fn deleting_the_freeze_file_does_not_thaw_live_instance() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    prod.kill("k", NOW0).expect("kill");
    refer.kill("k", NOW0).expect("ref kill");
    std::fs::remove_file(&world.prod.freeze_path).unwrap();
    std::fs::remove_file(&world.refer.freeze_path).unwrap();
    assert!(prod.frozen(), "in-process kill switch must stay frozen");
    assert!(refer.frozen());
    match prod.kill("again", NOW1) {
        Err(prometheus_monitors::Error::AlreadyFrozen) => {}
        Ok(s) => assert!(s.frozen),
        other => panic!("{other:?}"),
    }
}

#[test]
fn held_out_cannot_be_read_without_trip() {
    let world = fresh_world();
    let mut prod = Monitors::open(world.prod.clone()).unwrap();
    match prod.note_held_out_access("genome", NOW0) {
        Err(prometheus_monitors::Error::HeldOutAccess) => {}
        other => panic!("held-out must block, got {other:?}"),
    }
    assert!(prod.frozen());
}

#[test]
fn self_check_cannot_be_swallowed() {
    let world = fresh_world();
    let mut prod = Monitors::open(world.prod.clone()).unwrap();
    match prod.note_self_check("patched monitors", NOW0) {
        Err(prometheus_monitors::Error::MonitorSelfCheck) => {}
        other => panic!("self-check must trip, got {other:?}"),
    }
    assert!(prod.frozen());
}

#[test]
fn grader_mismatch_cannot_be_ignored() {
    let world = fresh_world();
    let mut prod = Monitors::open(world.prod.clone()).unwrap();
    match prod.note_grader_hash(GRADER_ID, "deadbeef", NOW0) {
        Err(prometheus_monitors::Error::GraderHashMismatch) => {}
        other => panic!("mismatch must trip, got {other:?}"),
    }
    assert!(prod.frozen());
}

#[test]
fn kernel_escape_cannot_be_ignored() {
    let world = fresh_world();
    let mut prod = Monitors::open(world.prod.clone()).unwrap();
    let _ = prod.note_kernel_escape("unshare", NOW0);
    assert!(prod.frozen());
}
