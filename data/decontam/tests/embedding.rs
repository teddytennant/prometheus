//! Group: embedding_match — without an embedder must error (fail closed).

mod common;

use prometheus_decontam::{embedding_match, scan_text, Error, Status};

#[test]
fn embedding_match_empty_index_is_missing_index() {
    common::assert_missing_index(embedding_match(
        &prometheus_decontam::Index::new(),
        common::PLANTED_TEXT,
    ));
}

#[test]
fn embedding_match_without_embedder_errors_not_clean() {
    let index = common::index_with(&[common::planted_item()]);
    match embedding_match(&index, common::PLANTED_TEXT) {
        Ok(hits) => {
            panic!("embedding_match without an embedder must fail closed, got Ok({hits:?})")
        }
        Err(Error::MissingIndex) => panic!(
            "non-empty index must not report MissingIndex; missing embedder is a config/other error"
        ),
        Err(Error::PlantedMiss(_)) => {
            panic!("missing embedder is not PlantedMiss")
        }
        Err(Error::Config(_)) | Err(Error::Other(_)) => {}
    }
}

#[test]
fn scan_text_does_not_require_an_embedder() {
    // n-gram scan is the default method; a missing embedder must not poison scan_text.
    let items = vec![common::planted_item()];
    let index = common::index_with(&items);
    let clean = scan_text(&index, common::UNRELATED_TEXT).unwrap();
    assert_eq!(clean.status, Status::Clean);
    let flagged = scan_text(&index, common::PLANTED_TEXT).unwrap();
    assert_eq!(flagged.status, Status::Flagged);
}
