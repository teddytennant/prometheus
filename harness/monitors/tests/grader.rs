//! Group: `note_grader_hash`. Matching hash is ok. Mismatch is
//! `Error::GraderHashMismatch` and trips kill.

mod common;
mod reference;

use std::collections::BTreeMap;

use common::{
    assert_frozen_pair, assert_has_violation, assert_result_eq, assert_unfrozen_pair, fresh_world,
    open_pair, unwrap_err, world_with_hashes, GRADER_ID, HASH_A, HASH_B, NOW0, NOW1,
};
use prometheus_monitors::{Boundary, Error};
use reference::trip_err;

#[test]
fn matching_hash_is_ok_and_does_not_freeze() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    assert_result_eq(
        prod.note_grader_hash(GRADER_ID, HASH_A, NOW0),
        refer.note_grader_hash(GRADER_ID, HASH_A, NOW0),
        "match",
    );
    assert_unfrozen_pair(&prod, &refer);
    assert!(prod.snapshot().violations.is_empty());
    assert!(!world.prod.freeze_path.exists());
}

#[test]
fn mismatch_is_grader_hash_mismatch_and_trips_kill() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    let got = prod.note_grader_hash(GRADER_ID, HASH_B, NOW0);
    let exp = refer.note_grader_hash(GRADER_ID, HASH_B, NOW0);
    assert_eq!(got, exp);
    assert_eq!(got, Err(Error::GraderHashMismatch));
    assert_eq!(trip_err(Boundary::Graders), Error::GraderHashMismatch);
    assert_frozen_pair(&prod, &refer);
    assert_has_violation(&prod.snapshot(), Boundary::Graders, NOW0, GRADER_ID);
    assert_eq!(
        prod.snapshot().reason.as_deref(),
        Some(reference::freeze_reason(Boundary::Graders))
    );
    assert!(prod.frozen());
    assert_eq!(
        unwrap_err(prod.note_grader_hash(GRADER_ID, HASH_B, NOW1), "again"),
        Error::GraderHashMismatch
    );
    assert!(prod.frozen());
}

#[test]
fn unknown_grader_id_is_mismatch_and_trips() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    assert_result_eq(
        prod.note_grader_hash("no-such", HASH_A, NOW0),
        refer.note_grader_hash("no-such", HASH_A, NOW0),
        "unknown id",
    );
    assert_frozen_pair(&prod, &refer);
    assert_has_violation(&prod.snapshot(), Boundary::Graders, NOW0, "no-such");
}

#[test]
fn empty_config_any_hash_trips() {
    let world = world_with_hashes(BTreeMap::new());
    let (mut prod, mut refer) = open_pair(&world);
    let got = prod.note_grader_hash(GRADER_ID, HASH_A, NOW0);
    let exp = refer.note_grader_hash(GRADER_ID, HASH_A, NOW0);
    assert_eq!(got, exp);
    assert_eq!(got, Err(Error::GraderHashMismatch));
    assert_frozen_pair(&prod, &refer);
}

#[test]
fn hash_compare_is_exact_including_case() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    let upper = HASH_A.to_uppercase();
    assert_ne!(upper, HASH_A);
    assert_result_eq(
        prod.note_grader_hash(GRADER_ID, &upper, NOW0),
        refer.note_grader_hash(GRADER_ID, &upper, NOW0),
        "case",
    );
    assert_frozen_pair(&prod, &refer);
}

#[test]
fn trailing_whitespace_is_mismatch() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    let padded = format!("{HASH_A} ");
    assert_result_eq(
        prod.note_grader_hash(GRADER_ID, &padded, NOW0),
        refer.note_grader_hash(GRADER_ID, &padded, NOW0),
        "pad",
    );
    assert_frozen_pair(&prod, &refer);
}

#[test]
fn matching_one_grader_does_not_excuse_another() {
    let hashes = BTreeMap::from([
        ("g1".into(), HASH_A.to_string()),
        ("g2".into(), HASH_B.to_string()),
    ]);
    let world = world_with_hashes(hashes);
    let (mut prod, mut refer) = open_pair(&world);
    assert_result_eq(
        prod.note_grader_hash("g1", HASH_A, NOW0),
        refer.note_grader_hash("g1", HASH_A, NOW0),
        "g1 ok",
    );
    assert_unfrozen_pair(&prod, &refer);
    assert_result_eq(
        prod.note_grader_hash("g2", HASH_A, NOW1),
        refer.note_grader_hash("g2", HASH_A, NOW1),
        "g2 bad",
    );
    assert_frozen_pair(&prod, &refer);
    assert_has_violation(&prod.snapshot(), Boundary::Graders, NOW1, "g2");
    assert_eq!(prod.snapshot().frozen_at, Some(NOW1));
}

#[test]
fn matching_hash_after_freeze_is_still_ok() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    let _ = prod.note_held_out_access("x", NOW0);
    let _ = refer.note_held_out_access("x", NOW0);
    assert_result_eq(
        prod.note_grader_hash(GRADER_ID, HASH_A, NOW1),
        refer.note_grader_hash(GRADER_ID, HASH_A, NOW1),
        "match while frozen",
    );
    assert_frozen_pair(&prod, &refer);
    assert_eq!(prod.snapshot().violations.len(), 1);
}
