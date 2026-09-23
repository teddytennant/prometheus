//! Group: golden — planted eval sentence is caught; unrelated shard is clean.

mod common;
mod reference;

use prometheus_decontam::{planted_caught, scan_shard, scan_text, Status};

#[test]
fn planted_eval_sentence_in_a_shard_is_caught() {
    let planted = common::planted_item();
    let index = common::index_with(std::slice::from_ref(&planted));
    let shard = vec![format!(
        "training document wrapping the eval sentence: {}",
        common::PLANTED_TEXT
    )];
    let report = scan_shard(&index, &shard).expect("scan_shard");
    assert_eq!(report.status, Status::Flagged);
    assert!(
        report.hits.iter().any(|h| {
            h.suite == common::PLANTED_SUITE
                && h.item_id == common::PLANTED_ID
                && h.method == "ngram"
        }),
        "planted sentence must produce a flagged ngram hit: {:?}",
        report.hits
    );
    planted_caught(&index, std::slice::from_ref(&planted), &shard).expect("planted_caught");
    common::assert_report_matches(&report, &reference::scan_shard(&[planted], &shard).unwrap());
}

#[test]
fn unrelated_shard_is_clean() {
    let planted = common::planted_item();
    let index = common::index_with(std::slice::from_ref(&planted));
    let shard = vec![common::UNRELATED_TEXT.to_string()];
    let report = scan_shard(&index, &shard).expect("scan_shard");
    assert_eq!(report.status, Status::Clean);
    assert!(report.hits.is_empty());
    common::assert_sha256_hex(&report.method_hash);
    assert_eq!(
        common::sorted_strings(report.against.clone()),
        vec![common::PLANTED_SUITE.to_string()]
    );
    common::assert_report_matches(&report, &reference::scan_shard(&[planted], &shard).unwrap());
}

#[test]
fn scan_text_golden_matches_reference_ngrams() {
    let text = common::PLANTED_TEXT;
    assert_eq!(prometheus_decontam::ngrams(text), reference::ngrams(text));
    let items = vec![common::planted_item()];
    let index = common::index_with(&items);
    let got = scan_text(&index, text).unwrap();
    common::assert_report_matches(&got, &reference::scan_text(&items, text).unwrap());
    assert_eq!(got.status, Status::Flagged);
}
