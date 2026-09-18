//! Group: `note_held_out_access`. Any attempt is `Error::HeldOutAccess` and
//! trips kill.

mod common;
mod reference;

use common::{
    assert_frozen_pair, assert_has_violation, assert_result_eq, fresh_world, open_pair, NOW0, NOW1,
};
use prometheus_monitors::{Boundary, Error};
use reference::trip_err;

#[test]
fn any_attempt_is_held_out_access_and_trips_kill() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    let got = prod.note_held_out_access("agent-7", NOW0);
    let exp = refer.note_held_out_access("agent-7", NOW0);
    assert_eq!(got, exp);
    assert_eq!(got, Err(Error::HeldOutAccess));
    assert_eq!(trip_err(Boundary::HeldOut), Error::HeldOutAccess);
    assert_frozen_pair(&prod, &refer);
    assert_has_violation(&prod.snapshot(), Boundary::HeldOut, NOW0, "agent-7");
    assert!(prod.frozen());
    assert_eq!(prod.snapshot().frozen_at, Some(NOW0));
}

#[test]
fn empty_who_still_trips() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    assert_result_eq(
        prod.note_held_out_access("", NOW0),
        refer.note_held_out_access("", NOW0),
        "empty who",
    );
    assert_frozen_pair(&prod, &refer);
    assert_has_violation(&prod.snapshot(), Boundary::HeldOut, NOW0, "");
}

#[test]
fn unicode_who_is_stored_verbatim() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    let who = "genome/试";
    assert_result_eq(
        prod.note_held_out_access(who, NOW0),
        refer.note_held_out_access(who, NOW0),
        "unicode",
    );
    assert_has_violation(&prod.snapshot(), Boundary::HeldOut, NOW0, who);
}

#[test]
fn second_attempt_stays_frozen_and_appends() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    let _ = prod.note_held_out_access("a", NOW0);
    let _ = refer.note_held_out_access("a", NOW0);
    assert_result_eq(
        prod.note_held_out_access("b", NOW1),
        refer.note_held_out_access("b", NOW1),
        "second",
    );
    assert_frozen_pair(&prod, &refer);
    assert_eq!(prod.snapshot().frozen_at, Some(NOW0));
    assert_eq!(prod.snapshot().violations.len(), 2);
    assert_has_violation(&prod.snapshot(), Boundary::HeldOut, NOW1, "b");
}
