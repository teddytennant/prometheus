//! Group: frozen golden report on the small fixture corpus.

mod common;
mod reference;

use prometheus_dedup::{dedup_corpus, DedupConfig};

#[test]
fn frozen_report_ids_kept_and_dropped() {
    let docs = common::fixture_corpus();
    let want = common::fixture_report();
    let got = dedup_corpus(&docs, DedupConfig::standard()).expect("corpus");
    assert_eq!(
        got.kept, want.kept,
        "kept ids must match goldens/report.json"
    );
    assert_eq!(
        got.dropped_exact, want.dropped_exact,
        "dropped_exact ids must match goldens/report.json"
    );
    assert_eq!(
        got.dropped_near, want.dropped_near,
        "dropped_near ids must match goldens/report.json"
    );
    assert_eq!(got.clusters, want.clusters);
    assert_eq!(got, want);
    common::assert_partition(&docs, &got);
    common::assert_report_canonical(&got);
}

#[test]
fn frozen_report_matches_reference() {
    let docs = common::fixture_corpus();
    let want = common::fixture_report();
    let public = dedup_corpus(&docs, DedupConfig::standard()).expect("public");
    let got = reference::dedup_corpus(&docs, DedupConfig::standard()).expect("ref");
    assert_eq!(got, want, "reference must reproduce goldens/report.json");
    assert_eq!(public, got);
}

#[test]
fn fixture_story() {
    // keep-cats wins the exact-document duel with drop-exact-doc (same
    // normalize, lower content_hash). drop-exact-para shares a paragraph with
    // keep-cats so the whole document is dropped. keep-near-src / drop-near
    // survive exact passes and then cluster; lowest content_hash is kept.
    // keep-unique and keep-short (too short to shingle) stay.
    let docs = common::fixture_corpus();
    let got = dedup_corpus(&docs, DedupConfig::standard()).expect("corpus");
    assert_eq!(
        got.kept,
        vec!["keep-cats", "keep-near-src", "keep-short", "keep-unique"]
    );
    assert_eq!(got.dropped_exact, vec!["drop-exact-doc", "drop-exact-para"]);
    assert_eq!(got.dropped_near, vec!["drop-near"]);
    assert_eq!(got.clusters[0].kept, "keep-near-src");
    assert_eq!(got.clusters[0].dropped, vec!["drop-near"]);
}
