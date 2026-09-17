//! Shared fixtures and assertions for B2 integration tests.
#![allow(dead_code)]

use std::collections::HashSet;
use std::fmt::Debug;

use prometheus_dedup::{DedupConfig, DedupReport, Document, Error, Result};

pub fn doc(id: &str, text: &str, content_hash: &str) -> Document {
    Document {
        id: id.to_string(),
        text: text.to_string(),
        content_hash: content_hash.to_string(),
    }
}

/// Tiny LSH config for fast tests: 2-shingles, 8 hashes, 4 bands × 2 rows.
pub fn cfg_small() -> DedupConfig {
    DedupConfig {
        shingle_size: 2,
        num_hashes: 8,
        bands: 4,
        rows: 2,
    }
}

pub fn fixture_corpus() -> Vec<Document> {
    serde_json::from_str(include_str!("../goldens/corpus.json"))
        .expect("goldens/corpus.json must deserialize as Vec<Document>")
}

pub fn fixture_report() -> DedupReport {
    serde_json::from_str(include_str!("../goldens/report.json"))
        .expect("goldens/report.json must deserialize as DedupReport")
}

pub fn assert_err_config<T: Debug>(result: Result<T>) {
    match result {
        Err(Error::Config(msg)) => {
            assert!(!msg.is_empty(), "Error::Config must explain the problem");
        }
        Ok(v) => panic!("expected Error::Config, got Ok({v:?})"),
        Err(other) => panic!("expected Error::Config, got {other:?}"),
    }
}

pub fn assert_err_other<T: Debug>(result: Result<T>) {
    match result {
        Err(Error::Other(msg)) => {
            assert!(!msg.is_empty(), "Error::Other must explain the problem");
        }
        Ok(v) => panic!("expected Error::Other, got Ok({v:?})"),
        Err(other) => panic!("expected Error::Other, got {other:?}"),
    }
}

pub fn assert_err_empty_corpus<T: Debug>(result: Result<T>) {
    match result {
        Err(Error::Other(msg)) => {
            assert!(
                msg.to_ascii_lowercase().contains("empty"),
                "empty corpus error should mention empty, got {msg:?}"
            );
        }
        Ok(v) => panic!("expected empty-corpus Error::Other, got Ok({v:?})"),
        Err(other) => panic!("expected empty-corpus Error::Other, got {other:?}"),
    }
}

pub fn ids_of(docs: &[Document]) -> HashSet<String> {
    docs.iter().map(|d| d.id.clone()).collect()
}

/// kept ∪ dropped_exact ∪ dropped_near is a partition of the input ids.
pub fn assert_partition(docs: &[Document], report: &DedupReport) {
    let input = ids_of(docs);
    let mut seen = HashSet::new();
    for (label, ids) in [
        ("kept", &report.kept),
        ("dropped_exact", &report.dropped_exact),
        ("dropped_near", &report.dropped_near),
    ] {
        for id in ids {
            assert!(
                seen.insert(id.clone()),
                "{label} id {id:?} appears more than once in the report"
            );
            assert!(
                input.contains(id),
                "{label} id {id:?} was not in the input corpus"
            );
        }
    }
    assert_eq!(
        seen, input,
        "report ids must be a partition of the input corpus"
    );
}

pub fn assert_sorted(ids: &[String], label: &str) {
    let mut sorted = ids.to_vec();
    sorted.sort();
    assert_eq!(ids, &sorted, "{label} must be sorted lexicographically");
}

pub fn assert_report_canonical(report: &DedupReport) {
    assert_sorted(&report.kept, "kept");
    assert_sorted(&report.dropped_exact, "dropped_exact");
    assert_sorted(&report.dropped_near, "dropped_near");
    let mut cluster_kept: Vec<&str> = report.clusters.iter().map(|c| c.kept.as_str()).collect();
    let mut cluster_sorted = cluster_kept.clone();
    cluster_sorted.sort();
    assert_eq!(
        cluster_kept, cluster_sorted,
        "clusters must be sorted by kept id"
    );
    cluster_kept.sort();
    cluster_kept.dedup();
    assert_eq!(
        cluster_kept.len(),
        report.clusters.len(),
        "cluster kept ids must be unique"
    );
    for c in &report.clusters {
        assert_sorted(&c.dropped, &format!("cluster {} dropped", c.kept));
        assert!(
            !c.dropped.is_empty(),
            "clusters list only groups with at least one drop"
        );
        assert!(
            report.kept.contains(&c.kept),
            "cluster kept {} must be in report.kept",
            c.kept
        );
        for d in &c.dropped {
            assert!(
                report.dropped_near.contains(d) || report.dropped_exact.contains(d),
                "cluster drop {d} must appear in a dropped_* list"
            );
        }
    }
}

pub fn alphabet() -> &'static str {
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima \
     mike november oscar papa quebec romeo sierra tango"
}

pub fn alphabet_near() -> &'static str {
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima \
     mike november oscar papa quebec romeo sierra ultra"
}

pub fn alphabet_unrelated() -> &'static str {
    "red orange yellow green blue indigo violet purple brown black white gray \
     silver gold copper zinc nickel cobalt"
}
