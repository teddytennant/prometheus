//! Group: `note_kernel_escape`. Any recorded capability trips kill.

mod common;
mod reference;

use common::{
    assert_frozen_pair, assert_has_violation, assert_result_eq, fresh_world, open_pair, NOW0,
};
use prometheus_monitors::{Boundary, Error};
use reference::trip_err;

#[test]
fn kernel_escape_trips_kill() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    let got = prod.note_kernel_escape("ptrace", NOW0);
    let exp = refer.note_kernel_escape("ptrace", NOW0);
    assert_eq!(got, exp);
    assert_eq!(got, Err(Error::Message("kernel escape".into())));
    assert_eq!(
        trip_err(Boundary::Kernel),
        Error::Message("kernel escape".into())
    );
    assert_frozen_pair(&prod, &refer);
    assert!(prod.frozen());
    assert_has_violation(&prod.snapshot(), Boundary::Kernel, NOW0, "ptrace");
    assert_eq!(prod.snapshot().frozen_at, Some(NOW0));
}

#[test]
fn kernel_escape_empty_capability_still_trips() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    assert_result_eq(
        prod.note_kernel_escape("", NOW0),
        refer.note_kernel_escape("", NOW0),
        "empty cap",
    );
    assert_frozen_pair(&prod, &refer);
    assert_has_violation(&prod.snapshot(), Boundary::Kernel, NOW0, "");
}

#[test]
fn kernel_escape_does_not_report_another_boundary_error() {
    let world = fresh_world();
    let (mut prod, _) = open_pair(&world);
    match prod.note_kernel_escape("raw-socket", NOW0) {
        Err(Error::Message(s)) => assert_eq!(s, "kernel escape"),
        other => panic!("expected Message(\"kernel escape\"), got {other:?}"),
    }
    assert!(prod.frozen());
}
