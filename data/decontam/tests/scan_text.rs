//! Group: scan_text — no hits => Clean; hits => Flagged with hits filled.

mod common;
mod reference;

use prometheus_decontam::{scan_text, Status};

#[test]
fn scan_text_unrelated_is_clean_with_against_and_method_hash() {
    let items = vec![common::planted_item(), common::unrelated_item()];
    let index = common::index_with(&items);
    let got = scan_text(&index, "zzzz yyyy xxxx wwww vvvv uuuu tttt ssss").unwrap();
    let expected = reference::scan_text(&items, "zzzz yyyy xxxx wwww vvvv uuuu tttt ssss").unwrap();
    assert_eq!(got.status, Status::Clean);
    assert!(got.hits.is_empty());
    assert_eq!(
        common::sorted_strings(got.against.clone()),
        vec!["gpqa".to_string(), "hle".to_string()]
    );
    common::assert_sha256_hex(&got.method_hash);
    common::assert_report_matches(&got, &expected);
}

#[test]
fn scan_text_planted_is_flagged_with_hits_filled() {
    let items = vec![common::planted_item()];
    let index = common::index_with(&items);
    let got = scan_text(&index, common::PLANTED_TEXT).unwrap();
    let expected = reference::scan_text(&items, common::PLANTED_TEXT).unwrap();
    assert_eq!(got.status, Status::Flagged);
    assert_eq!(got.hits.len(), 1);
    assert_eq!(got.hits[0].suite, common::PLANTED_SUITE);
    assert_eq!(got.hits[0].item_id, common::PLANTED_ID);
    assert_eq!(got.hits[0].method, "ngram");
    assert_eq!(got.against, vec![common::PLANTED_SUITE.to_string()]);
    common::assert_sha256_hex(&got.method_hash);
    common::assert_report_matches(&got, &expected);
}

#[test]
fn scan_text_against_is_full_suite_list_even_when_one_suite_hits() {
    let items = vec![common::planted_item(), common::unrelated_item()];
    let index = common::index_with(&items);
    let got = scan_text(&index, common::PLANTED_TEXT).unwrap();
    assert_eq!(got.status, Status::Flagged);
    assert_eq!(
        common::sorted_strings(got.against.clone()),
        vec!["gpqa".to_string(), "hle".to_string()]
    );
    assert_eq!(got.hits.len(), 1);
    assert_eq!(got.hits[0].suite, "gpqa");
    common::assert_report_matches(
        &got,
        &reference::scan_text(&items, common::PLANTED_TEXT).unwrap(),
    );
}

#[test]
fn scan_text_empty_query_with_filled_index_is_clean() {
    let items = vec![common::planted_item()];
    let index = common::index_with(&items);
    let got = scan_text(&index, "").unwrap();
    assert_eq!(got.status, Status::Clean);
    assert!(got.hits.is_empty());
    common::assert_sha256_hex(&got.method_hash);
    common::assert_report_matches(&got, &reference::scan_text(&items, "").unwrap());
}
