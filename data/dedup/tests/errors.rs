//! Group: empty corpus / bad config errors.

mod common;

use common::doc;
use prometheus_dedup::{dedup_corpus, dedup_exact, dedup_near, lsh_band_keys, DedupConfig, Grain};

#[test]
fn empty_corpus_exact_document() {
    common::assert_err_empty_corpus(dedup_exact(&[], Grain::Document));
}

#[test]
fn empty_corpus_exact_paragraph() {
    common::assert_err_empty_corpus(dedup_exact(&[], Grain::Paragraph));
}

#[test]
fn empty_corpus_near() {
    common::assert_err_empty_corpus(dedup_near(&[], DedupConfig::standard()));
}

#[test]
fn empty_corpus_corpus() {
    common::assert_err_empty_corpus(dedup_corpus(&[], DedupConfig::standard()));
}

#[test]
fn bad_config_near_num_hashes() {
    let docs = vec![doc("a", "alpha bravo charlie delta echo foxtrot", "01")];
    common::assert_err_config(dedup_near(
        &docs,
        DedupConfig {
            shingle_size: 2,
            num_hashes: 7,
            bands: 4,
            rows: 2,
        },
    ));
}

#[test]
fn bad_config_corpus_num_hashes() {
    let docs = vec![doc("a", "alpha bravo charlie delta echo foxtrot", "01")];
    common::assert_err_config(dedup_corpus(
        &docs,
        DedupConfig {
            shingle_size: 5,
            num_hashes: 128,
            bands: 16,
            rows: 4,
        },
    ));
}

#[test]
fn bad_config_shingle_size_zero() {
    let docs = vec![doc("a", "alpha bravo charlie delta echo foxtrot", "01")];
    common::assert_err_config(dedup_near(
        &docs,
        DedupConfig {
            shingle_size: 0,
            num_hashes: 8,
            bands: 4,
            rows: 2,
        },
    ));
}

#[test]
fn bad_config_does_not_affect_exact() {
    // Exact pass has no config; a non-empty corpus must still run.
    let docs = vec![doc("a", "hello", "01")];
    let got = dedup_exact(&docs, Grain::Document).expect("exact ignores config");
    assert_eq!(got.kept, vec!["a"]);
}

#[test]
fn lsh_rejects_mismatched_signature() {
    common::assert_err_config(lsh_band_keys(&[1, 2, 3], DedupConfig::standard()));
}
