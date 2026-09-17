//! Group: fail-closed — empty index is MissingIndex, never clean.

mod common;

use prometheus_decontam::{embedding_match, ngram_match, scan_shard, scan_text, Index};

fn empty() -> Index {
    Index::new()
}

#[test]
fn scan_text_empty_index_is_missing_index_not_clean() {
    let result = scan_text(&empty(), "any document text at all");
    common::assert_missing_index(result);
}

#[test]
fn scan_shard_empty_index_is_missing_index_not_clean() {
    let shard = vec![common::UNRELATED_TEXT.to_string()];
    common::assert_missing_index(scan_shard(&empty(), &shard));
}

#[test]
fn ngram_match_empty_index_is_missing_index_not_clean() {
    common::assert_missing_index(ngram_match(&empty(), common::PLANTED_TEXT));
}

#[test]
fn embedding_match_empty_index_is_missing_index_not_clean() {
    common::assert_missing_index(embedding_match(&empty(), common::PLANTED_TEXT));
}

#[test]
fn empty_index_is_empty_but_scans_still_fail_closed() {
    let index = empty();
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    common::assert_missing_index(scan_text(&index, ""));
    common::assert_missing_index(scan_shard(&index, &[]));
    common::assert_missing_index(ngram_match(&index, ""));
    common::assert_missing_index(embedding_match(&index, ""));
}
