//! Group: dedup_corpus — exact document, exact paragraph, then near.

mod common;
mod reference;

use common::doc;
use prometheus_dedup::{dedup_corpus, dedup_exact, DedupConfig, Grain};

#[test]
fn later_passes_see_earlier_keeps_only() {
    let cfg = DedupConfig::standard();
    let docs = common::fixture_corpus();
    let got = dedup_corpus(&docs, cfg).expect("corpus");
    let exp = reference::dedup_corpus(&docs, cfg).expect("ref");
    assert_eq!(got, exp);

    let doc_pass = dedup_exact(&docs, Grain::Document).unwrap();
    assert!(
        doc_pass
            .dropped_exact
            .contains(&"drop-exact-doc".to_string()),
        "document grain must drop the normalized duplicate"
    );
    assert!(
        !got.kept.contains(&"drop-exact-doc".to_string()),
        "exact-dropped docs must not be kept later"
    );
    assert!(
        got.dropped_exact.contains(&"drop-exact-doc".to_string()),
        "document-grain drop stays in dropped_exact"
    );
    assert!(
        got.dropped_exact.contains(&"drop-exact-para".to_string()),
        "paragraph-grain drop stays in dropped_exact"
    );
    assert!(
        !got.dropped_near.contains(&"drop-exact-doc".to_string()),
        "exact drops must not be re-listed as near"
    );
    assert_eq!(got.dropped_near, vec!["drop-near"]);
    common::assert_partition(&docs, &got);
    common::assert_report_canonical(&got);
}

#[test]
fn exact_then_near_on_remaining() {
    // A is exact-dup of B (B kept). C is a near-dup of B. After exact, only B
    // and C remain; near then drops C against B, never against A.
    let cfg = DedupConfig::standard();
    let docs = vec![
        doc("A", common::alphabet(), "02"),
        doc("B", common::alphabet(), "01"),
        doc("C", common::alphabet_near(), "03"),
        doc("D", common::alphabet_unrelated(), "04"),
    ];
    let got = dedup_corpus(&docs, cfg).expect("corpus");
    assert_eq!(got.kept, vec!["B", "D"]);
    assert_eq!(got.dropped_exact, vec!["A"]);
    assert_eq!(got.dropped_near, vec!["C"]);
    assert_eq!(got.clusters.len(), 1);
    assert_eq!(got.clusters[0].kept, "B");
    assert_eq!(got.clusters[0].dropped, vec!["C"]);
}

#[test]
fn paragraph_pass_runs_after_document_pass() {
    let cfg = DedupConfig::standard();
    let docs = vec![
        doc(
            "src",
            "shared paragraph about cats\n\nunique src tail lives here",
            "01",
        ),
        doc(
            "para-dup",
            "shared paragraph about cats\n\nunique other tail about zinc",
            "02",
        ),
        doc("near", common::alphabet_near(), "05"),
        doc("src-near", common::alphabet(), "04"),
    ];
    let got = dedup_corpus(&docs, cfg).expect("corpus");
    assert!(got.dropped_exact.contains(&"para-dup".to_string()));
    assert!(got.kept.contains(&"src".to_string()));
    assert!(got.dropped_near.contains(&"near".to_string()));
    assert!(got.kept.contains(&"src-near".to_string()));
}

#[test]
fn single_document_kept() {
    let docs = vec![doc(
        "only",
        "a reasonably long unique document about widgets",
        "01",
    )];
    let got = dedup_corpus(&docs, DedupConfig::standard()).expect("corpus");
    assert_eq!(got.kept, vec!["only"]);
    assert!(got.dropped_exact.is_empty());
    assert!(got.dropped_near.is_empty());
    assert!(got.clusters.is_empty());
}

#[test]
fn partition_property_on_fixture() {
    let docs = common::fixture_corpus();
    let got = dedup_corpus(&docs, DedupConfig::standard()).expect("corpus");
    common::assert_partition(&docs, &got);
    assert_eq!(
        got,
        reference::dedup_corpus(&docs, DedupConfig::standard()).unwrap()
    );
}
