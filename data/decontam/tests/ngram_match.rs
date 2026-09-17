//! Group: ngram_match — shared 8-gram is a Hit with suite, item_id, method "ngram".

mod common;
mod reference;

use prometheus_decontam::{ngram_match, Index};

#[test]
fn ngram_match_empty_index_is_missing_index() {
    common::assert_missing_index(ngram_match(&Index::new(), common::PLANTED_TEXT));
}

#[test]
fn shared_eight_gram_is_ngram_hit() {
    let items = vec![common::planted_item()];
    let index = common::index_with(&items);
    let hits = ngram_match(&index, common::PLANTED_TEXT).expect("match");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].suite, common::PLANTED_SUITE);
    assert_eq!(hits[0].item_id, common::PLANTED_ID);
    assert_eq!(hits[0].method, "ngram");
    assert_eq!(
        common::sorted_hits(hits.clone()),
        common::sorted_hits(reference::ngram_match(&items, common::PLANTED_TEXT).unwrap())
    );
}

#[test]
fn shared_eight_gram_in_a_longer_document_is_a_hit() {
    let items = vec![common::planted_item()];
    let index = common::index_with(&items);
    let wrapped = format!("prefix noise {} trailing words here", common::PLANTED_TEXT);
    let hits = ngram_match(&index, &wrapped).expect("match");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].method, "ngram");
    assert_eq!(
        common::sorted_hits(hits),
        common::sorted_hits(reference::ngram_match(&items, &wrapped).unwrap())
    );
}

#[test]
fn seven_word_overlap_is_not_a_hit() {
    // 9-word item; query shares 7 interior words — not an 8-gram.
    let item_text = "alpha bravo charlie delta echo foxtrot golf hotel india";
    let query = "zulu bravo charlie delta echo foxtrot golf hotel yankee";
    let items = vec![common::item("gpqa", "overlap7", item_text)];
    let index = common::index_with(&items);
    let hits = ngram_match(&index, query).expect("no panic");
    assert!(
        hits.is_empty(),
        "7-word overlap must not count as an 8-gram hit: {hits:?}"
    );
    assert_eq!(hits, reference::ngram_match(&items, query).unwrap());
}

#[test]
fn identical_seven_word_text_is_not_a_hit() {
    let items = vec![common::item("gpqa", "short", common::SEVEN_WORDS)];
    let index = common::index_with(&items);
    let hits = ngram_match(&index, common::SEVEN_WORDS).expect("no panic");
    assert!(
        hits.is_empty(),
        "texts shorter than 8 words have no 8-grams: {hits:?}"
    );
}

#[test]
fn no_shared_gram_is_empty_hits_not_an_error() {
    let items = vec![common::planted_item()];
    let index = common::index_with(&items);
    let hits = ngram_match(&index, common::UNRELATED_TEXT).expect("clean query");
    assert!(hits.is_empty());
    assert_eq!(
        hits,
        reference::ngram_match(&items, common::UNRELATED_TEXT).unwrap()
    );
}

#[test]
fn two_items_can_both_hit() {
    let items = vec![common::planted_item(), common::unrelated_item()];
    let index = common::index_with(&items);
    let mixed = format!("{} {}", common::PLANTED_TEXT, common::UNRELATED_TEXT);
    let hits = ngram_match(&index, &mixed).expect("match");
    assert_eq!(hits.len(), 2);
    common::assert_unique_hits(&hits);
    assert!(hits.iter().all(|h| h.method == "ngram"));
    assert_eq!(
        common::sorted_hits(hits),
        common::sorted_hits(reference::ngram_match(&items, &mixed).unwrap())
    );
}

#[test]
fn match_is_case_and_whitespace_insensitive() {
    let items = vec![common::planted_item()];
    let index = common::index_with(&items);
    let noisy = "  ALPHA\tBRAVO\nCHARLIE   DELTA echo foxtrot golf hotel planted eval item  ";
    let hits = ngram_match(&index, noisy).expect("match");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].item_id, common::PLANTED_ID);
}
