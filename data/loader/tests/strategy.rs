//! Group: Strategy::as_str contract values.
//!
//! This helper is implemented on the stub; these tests may pass before pack
//! / shuffle / loader exist.

use prometheus_loader::Strategy;

#[test]
fn as_str_concat_eos() {
    assert_eq!(Strategy::ConcatEos.as_str(), "concat_eos");
}

#[test]
fn as_str_document_mask() {
    assert_eq!(Strategy::DocumentMask.as_str(), "document_mask");
}

#[test]
fn as_str_single_document() {
    assert_eq!(Strategy::SingleDocument.as_str(), "single_document");
}

#[test]
fn as_str_matches_serde_rename() {
    for strategy in [
        Strategy::ConcatEos,
        Strategy::DocumentMask,
        Strategy::SingleDocument,
    ] {
        let rendered = serde_json::to_string(&strategy).expect("serialize Strategy");
        assert_eq!(rendered, format!("\"{}\"", strategy.as_str()));
    }
}
