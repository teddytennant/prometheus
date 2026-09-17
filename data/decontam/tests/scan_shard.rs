//! Group: scan_shard — any document flagged => Flagged.

mod common;
mod reference;

use prometheus_decontam::{scan_shard, Status};

#[test]
fn scan_shard_all_clean_documents_is_clean() {
    let items = vec![common::planted_item()];
    let index = common::index_with(&items);
    let shard = vec![
        common::UNRELATED_TEXT.to_string(),
        "zzzz yyyy xxxx wwww vvvv uuuu tttt ssss extra".to_string(),
    ];
    let got = scan_shard(&index, &shard).unwrap();
    assert_eq!(got.status, Status::Clean);
    assert!(got.hits.is_empty());
    common::assert_report_matches(&got, &reference::scan_shard(&items, &shard).unwrap());
}

#[test]
fn scan_shard_any_document_flagged_flags_the_shard() {
    let items = vec![common::planted_item()];
    let index = common::index_with(&items);
    let shard = vec![
        common::UNRELATED_TEXT.to_string(),
        common::PLANTED_TEXT.to_string(),
        "more unrelated filler words without eight gram overlap here".to_string(),
    ];
    let got = scan_shard(&index, &shard).unwrap();
    assert_eq!(got.status, Status::Flagged);
    assert_eq!(got.hits.len(), 1);
    assert_eq!(got.hits[0].item_id, common::PLANTED_ID);
    assert_eq!(got.hits[0].method, "ngram");
    common::assert_unique_hits(&got.hits);
    common::assert_report_matches(&got, &reference::scan_shard(&items, &shard).unwrap());
}

#[test]
fn scan_shard_unions_hits_across_documents_uniquely() {
    let items = vec![common::planted_item(), common::unrelated_item()];
    let index = common::index_with(&items);
    let shard = vec![
        common::PLANTED_TEXT.to_string(),
        common::UNRELATED_TEXT.to_string(),
        common::PLANTED_TEXT.to_string(),
    ];
    let got = scan_shard(&index, &shard).unwrap();
    assert_eq!(got.status, Status::Flagged);
    common::assert_unique_hits(&got.hits);
    assert_eq!(got.hits.len(), 2);
    common::assert_report_matches(&got, &reference::scan_shard(&items, &shard).unwrap());
}

#[test]
fn scan_shard_empty_documents_with_filled_index_is_clean() {
    let items = vec![common::planted_item()];
    let index = common::index_with(&items);
    let shard: Vec<String> = vec![];
    let got = scan_shard(&index, &shard).unwrap();
    assert_eq!(got.status, Status::Clean);
    assert!(got.hits.is_empty());
    common::assert_sha256_hex(&got.method_hash);
    common::assert_report_matches(&got, &reference::scan_shard(&items, &shard).unwrap());
}
