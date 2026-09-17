//! Group: Index::add — duplicate (suite, id) errors.

mod common;

use prometheus_decontam::{Index, NGRAM_N};

#[test]
fn add_increments_len_and_is_not_empty() {
    let mut index = Index::new();
    index.add(common::planted_item()).expect("add planted");
    assert_eq!(index.len(), 1);
    assert!(!index.is_empty());
    index.add(common::unrelated_item()).expect("add unrelated");
    assert_eq!(index.len(), 2);
}

#[test]
fn add_duplicate_suite_and_id_errors() {
    let mut index = Index::new();
    index.add(common::planted_item()).expect("first add");
    let dup = common::item(
        common::PLANTED_SUITE,
        common::PLANTED_ID,
        "different text but same suite and id must still be rejected",
    );
    assert!(index.add(dup).is_err(), "duplicate (suite, id) must error");
    assert_eq!(index.len(), 1, "failed add must not grow the index");
}

#[test]
fn add_same_id_different_suite_is_ok() {
    let mut index = Index::new();
    index
        .add(common::item("gpqa", "same-id", &common::words(8)))
        .expect("gpqa");
    index
        .add(common::item("hle", "same-id", &common::words(9)))
        .expect("hle with same id is a different key");
    assert_eq!(index.len(), 2);
}

#[test]
fn add_same_suite_different_id_is_ok() {
    let mut index = Index::new();
    index
        .add(common::item("gpqa", "a", &common::words(8)))
        .expect("a");
    index
        .add(common::item("gpqa", "b", &common::words(8)))
        .expect("b");
    assert_eq!(index.len(), 2);
}

#[test]
fn ngram_n_stays_eight_and_add_stores_items_for_matching() {
    assert_eq!(NGRAM_N, 8);
    let index = common::index_with(&[common::planted_item()]);
    let hits = prometheus_decontam::ngram_match(&index, common::PLANTED_TEXT).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].suite, common::PLANTED_SUITE);
    assert_eq!(hits[0].item_id, common::PLANTED_ID);
}
