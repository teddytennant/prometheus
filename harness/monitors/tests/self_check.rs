//! Group: `note_self_check`. Tamper with this crate is
//! `Error::MonitorSelfCheck` and trips kill.

mod common;
mod reference;

use common::{
    assert_frozen_pair, assert_has_violation, assert_result_eq, fresh_world, open_pair, NOW0,
};
use prometheus_monitors::{Boundary, Error};
use reference::trip_err;

#[test]
fn self_check_is_monitor_self_check_and_trips_kill() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    let got = prod.note_self_check("lib.rs rewritten", NOW0);
    let exp = refer.note_self_check("lib.rs rewritten", NOW0);
    assert_eq!(got, exp);
    assert_eq!(got, Err(Error::MonitorSelfCheck));
    assert_eq!(trip_err(Boundary::Monitors), Error::MonitorSelfCheck);
    assert_frozen_pair(&prod, &refer);
    assert!(prod.frozen());
    assert_has_violation(
        &prod.snapshot(),
        Boundary::Monitors,
        NOW0,
        "lib.rs rewritten",
    );
}

#[test]
fn empty_detail_still_trips() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    assert_result_eq(
        prod.note_self_check("", NOW0),
        refer.note_self_check("", NOW0),
        "empty",
    );
    assert_frozen_pair(&prod, &refer);
    assert_has_violation(&prod.snapshot(), Boundary::Monitors, NOW0, "");
}
