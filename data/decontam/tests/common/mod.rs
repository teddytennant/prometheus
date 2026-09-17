//! Shared builders and assertions for B4 decontam integration tests.
#![allow(dead_code)]

use prometheus_decontam::{Error, EvalItem, Hit, Index, Report};

pub const PLANTED_SUITE: &str = "gpqa";
pub const PLANTED_ID: &str = "planted-1";
/// Distinctive 11-word eval sentence used as a planted item.
pub const PLANTED_TEXT: &str =
    "alpha bravo charlie delta echo foxtrot golf hotel planted eval item";

pub const UNRELATED_SUITE: &str = "hle";
pub const UNRELATED_ID: &str = "unrelated-1";
/// No 8-gram overlap with [`PLANTED_TEXT`].
pub const UNRELATED_TEXT: &str = "one two three four five six seven eight nine ten eleven";

/// Seven whitespace-separated words: not enough for an 8-gram.
pub const SEVEN_WORDS: &str = "only seven words in this short line";

pub fn item(suite: &str, id: &str, text: &str) -> EvalItem {
    EvalItem {
        suite: suite.to_string(),
        id: id.to_string(),
        text: text.to_string(),
    }
}

pub fn planted_item() -> EvalItem {
    item(PLANTED_SUITE, PLANTED_ID, PLANTED_TEXT)
}

pub fn unrelated_item() -> EvalItem {
    item(UNRELATED_SUITE, UNRELATED_ID, UNRELATED_TEXT)
}

/// Build an index by adding each item in order. Panics on add error.
pub fn index_with(items: &[EvalItem]) -> Index {
    let mut index = Index::new();
    for it in items {
        index
            .add(it.clone())
            .unwrap_or_else(|e| panic!("add {}/{}: {e}", it.suite, it.id));
    }
    index
}

pub fn assert_missing_index<T: std::fmt::Debug>(result: Result<T, Error>) {
    match result {
        Err(Error::MissingIndex) => {}
        other => panic!("expected Error::MissingIndex, got {other:?}"),
    }
}

pub fn assert_sha256_hex(s: &str) {
    assert_eq!(s.len(), 64, "method_hash must be 64 hex chars, got {s:?}");
    assert!(
        s.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')),
        "method_hash must be lowercase hex SHA-256, got {s:?}"
    );
}

pub fn sorted_hits(mut hits: Vec<Hit>) -> Vec<Hit> {
    hits.sort_by(|a, b| (&a.suite, &a.item_id, &a.method).cmp(&(&b.suite, &b.item_id, &b.method)));
    hits
}

pub fn sorted_strings(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v
}

pub fn assert_report_matches(got: &Report, expected: &Report) {
    assert_eq!(got.status, expected.status, "status");
    assert_eq!(
        sorted_strings(got.against.clone()),
        sorted_strings(expected.against.clone()),
        "against"
    );
    assert_eq!(got.method_hash, expected.method_hash, "method_hash");
    assert_eq!(
        sorted_hits(got.hits.clone()),
        sorted_hits(expected.hits.clone()),
        "hits"
    );
}

pub fn assert_unique_hits(hits: &[Hit]) {
    let mut keys: Vec<_> = hits
        .iter()
        .map(|h| (h.suite.clone(), h.item_id.clone()))
        .collect();
    let n = keys.len();
    keys.sort();
    keys.dedup();
    assert_eq!(
        n,
        keys.len(),
        "hits must be unique per (suite, item_id): {hits:?}"
    );
}

pub fn ngram_hit(suite: &str, item_id: &str) -> Hit {
    Hit {
        suite: suite.to_string(),
        item_id: item_id.to_string(),
        method: "ngram".to_string(),
    }
}

/// `n` synthetic words `w1 .. wn` joined by a single space.
pub fn words(n: usize) -> String {
    (1..=n)
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>()
        .join(" ")
}
