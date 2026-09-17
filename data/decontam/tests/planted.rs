//! Group: planted_caught — every planted EvalItem must produce a Flagged hit.

mod common;
mod reference;

use prometheus_decontam::{planted_caught, scan_shard, Error, Status};

#[test]
fn planted_caught_ok_when_every_planted_item_is_in_the_shard() {
    let planted = vec![common::planted_item()];
    let index = common::index_with(&planted);
    let shard = vec![
        common::UNRELATED_TEXT.to_string(),
        common::PLANTED_TEXT.to_string(),
    ];
    planted_caught(&index, &planted, &shard).expect("planted item must be caught");
    let report = scan_shard(&index, &shard).unwrap();
    assert_eq!(report.status, Status::Flagged);
    assert!(report
        .hits
        .iter()
        .any(|h| h.suite == common::PLANTED_SUITE && h.item_id == common::PLANTED_ID));
}

#[test]
fn planted_caught_errors_planted_miss_when_shard_lacks_the_item() {
    let planted = vec![common::planted_item()];
    let index = common::index_with(&planted);
    let shard = vec![common::UNRELATED_TEXT.to_string()];
    match planted_caught(&index, &planted, &shard) {
        Err(Error::PlantedMiss(id)) => assert_eq!(id, common::PLANTED_ID),
        other => panic!(
            "expected PlantedMiss({}), got {other:?}",
            common::PLANTED_ID
        ),
    }
}

#[test]
fn planted_caught_misses_even_if_a_different_item_is_flagged() {
    let planted = common::planted_item();
    let other = common::unrelated_item();
    let index = common::index_with(&[planted.clone(), other.clone()]);
    let shard = vec![common::UNRELATED_TEXT.to_string()];
    match planted_caught(&index, &[planted], &shard) {
        Err(Error::PlantedMiss(id)) => assert_eq!(id, common::PLANTED_ID),
        other => panic!("expected PlantedMiss for planted-1, got {other:?}"),
    }
}

#[test]
fn planted_caught_requires_every_planted_item() {
    let a = common::planted_item();
    let b = common::unrelated_item();
    let index = common::index_with(&[a.clone(), b.clone()]);
    let shard = vec![common::PLANTED_TEXT.to_string()];
    match planted_caught(&index, &[a, b], &shard) {
        Err(Error::PlantedMiss(id)) => assert_eq!(id, common::UNRELATED_ID),
        other => panic!(
            "expected PlantedMiss({}), got {other:?}",
            common::UNRELATED_ID
        ),
    }
}

#[test]
fn planted_caught_empty_index_is_missing_index() {
    let planted = vec![common::planted_item()];
    let shard = vec![common::PLANTED_TEXT.to_string()];
    common::assert_missing_index(planted_caught(
        &prometheus_decontam::Index::new(),
        &planted,
        &shard,
    ));
}

#[test]
fn planted_caught_matches_reference_scan_hits() {
    let planted = vec![common::planted_item()];
    let index = common::index_with(&planted);
    let shard = vec![common::PLANTED_TEXT.to_string()];
    planted_caught(&index, &planted, &shard).unwrap();
    let report = reference::scan_shard(&planted, &shard).unwrap();
    assert_eq!(report.status, Status::Flagged);
    assert_eq!(report.hits.len(), 1);
}
