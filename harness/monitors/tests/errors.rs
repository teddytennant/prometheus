//! Group: documented `Error` variants from the public methods.

mod common;
mod reference;

use common::{fresh_world, open_pair, unwrap_err, GRADER_ID, HASH_B, NOW0, NOW1};
use prometheus_monitors::{Boundary, Error, Monitors};

#[test]
fn methods_return_documented_errors() {
    let world = fresh_world();
    let mut m = Monitors::open(world.prod.clone()).unwrap();
    assert_eq!(
        unwrap_err(m.note_grader_hash(GRADER_ID, HASH_B, NOW0), "grader"),
        Error::GraderHashMismatch
    );
    assert_eq!(
        unwrap_err(m.note_held_out_access("w", NOW0), "held_out"),
        Error::HeldOutAccess
    );
    assert_eq!(
        unwrap_err(m.note_self_check("t", NOW0), "self"),
        Error::MonitorSelfCheck
    );
    assert_eq!(
        unwrap_err(m.note_kernel_escape("c", NOW0), "kernel"),
        Error::Message("kernel escape".into())
    );
    assert_eq!(unwrap_err(m.kill("x", NOW1), "kill"), Error::AlreadyFrozen);
}

#[test]
fn plant_errors_match_trip_err() {
    for (b, err) in [
        (Boundary::Kernel, Error::Message("kernel escape".into())),
        (Boundary::Graders, Error::GraderHashMismatch),
        (Boundary::HeldOut, Error::HeldOutAccess),
        (Boundary::Monitors, Error::MonitorSelfCheck),
    ] {
        let world = fresh_world();
        let (mut prod, mut refer) = open_pair(&world);
        assert_eq!(unwrap_err(prod.plant(b, "d", NOW0), "prod"), err);
        assert_eq!(
            unwrap_err(refer.plant(b, "d", NOW0), "ref"),
            reference::trip_err(b)
        );
    }
}

#[test]
fn message_from_corrupt_freeze_is_not_a_boundary_error() {
    let world = fresh_world();
    std::fs::write(&world.prod.freeze_path, b"{").unwrap();
    match Monitors::open(world.prod.clone()) {
        Err(Error::Message(_)) => {}
        Err(
            Error::UnknownBoundary(_)
            | Error::AlreadyFrozen
            | Error::GraderHashMismatch
            | Error::HeldOutAccess
            | Error::MonitorSelfCheck,
        ) => {
            panic!("corrupt freeze must be Message, not a boundary error")
        }
        Ok(_) => panic!("corrupt freeze must fail"),
    }
}
