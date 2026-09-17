//! Group: dedup_near — shared band key ⇒ same cluster; keep lowest content_hash.

mod common;
mod reference;

use common::doc;
use prometheus_dedup::{dedup_near, lsh_band_keys, minhash_signature, DedupConfig};

#[test]
fn shared_band_key_same_cluster_keep_lowest_hash() {
    let cfg = DedupConfig::standard();
    let docs = vec![
        doc("high", common::alphabet_near(), "05"),
        doc("low", common::alphabet(), "04"),
        doc("other", common::alphabet_unrelated(), "01"),
    ];
    let got = dedup_near(&docs, cfg).expect("near");
    assert_eq!(got, reference::dedup_near(&docs, cfg).unwrap());
    assert!(got.dropped_exact.is_empty());
    common::assert_partition(&docs, &got);
    common::assert_report_canonical(&got);

    assert!(got.kept.contains(&"low".to_string()));
    assert!(got.kept.contains(&"other".to_string()));
    assert_eq!(got.dropped_near, vec!["high"]);
    assert_eq!(got.clusters.len(), 1);
    assert_eq!(got.clusters[0].kept, "low");
    assert_eq!(got.clusters[0].dropped, vec!["high"]);

    let sig_low = minhash_signature(common::alphabet(), cfg);
    let sig_high = minhash_signature(common::alphabet_near(), cfg);
    let k_low = lsh_band_keys(&sig_low, cfg).unwrap();
    let k_high = lsh_band_keys(&sig_high, cfg).unwrap();
    assert!(
        k_low.iter().zip(&k_high).any(|(a, b)| a == b),
        "near-dup fixtures must share at least one LSH band key"
    );
}

#[test]
fn disjoint_texts_do_not_cluster() {
    let cfg = DedupConfig::standard();
    let docs = vec![
        doc("a", common::alphabet(), "01"),
        doc("b", common::alphabet_unrelated(), "02"),
    ];
    let got = dedup_near(&docs, cfg).expect("near");
    assert_eq!(got.kept, vec!["a", "b"]);
    assert!(got.dropped_near.is_empty());
    assert!(got.clusters.is_empty());
}

#[test]
fn empty_shingle_docs_do_not_join_each_other() {
    let cfg = DedupConfig::standard();
    let docs = vec![doc("hi", "hi", "01"), doc("yo", "yo", "02")];
    let got = dedup_near(&docs, cfg).expect("near");
    assert_eq!(got.kept, vec!["hi", "yo"]);
    assert!(got.dropped_near.is_empty());
    assert!(got.clusters.is_empty());
}

#[test]
fn keep_is_lowest_content_hash_not_input_order() {
    let cfg = DedupConfig::standard();
    let docs = vec![
        doc("first-in-list", common::alphabet(), "zz"),
        doc("should-keep", common::alphabet_near(), "aa"),
    ];
    let got = dedup_near(&docs, cfg).expect("near");
    assert_eq!(got.kept, vec!["should-keep"]);
    assert_eq!(got.dropped_near, vec!["first-in-list"]);
    assert_eq!(got.clusters[0].kept, "should-keep");
}

#[test]
fn transitivity_three_near_dups() {
    let cfg = DedupConfig::standard();
    let a = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima \
             mike november oscar papa quebec romeo sierra tango";
    let b = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima \
             mike november oscar papa quebec romeo sierra ultra";
    let c = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima \
             mike november oscar papa quebec romeo sierra victor";
    let docs = vec![doc("a", a, "01"), doc("b", b, "02"), doc("c", c, "03")];
    let got = dedup_near(&docs, cfg).expect("near");
    assert_eq!(got.kept, vec!["a"]);
    assert_eq!(got.dropped_near, vec!["b", "c"]);
    assert_eq!(got.clusters.len(), 1);
    assert_eq!(got.clusters[0].kept, "a");
}

#[test]
fn bad_config_is_error() {
    let docs = vec![doc("a", common::alphabet(), "01")];
    common::assert_err_config(dedup_near(
        &docs,
        DedupConfig {
            shingle_size: 5,
            num_hashes: 10,
            bands: 32,
            rows: 4,
        },
    ));
}

#[test]
fn matches_reference_on_fixture() {
    let docs = common::fixture_corpus();
    let cfg = DedupConfig::standard();
    let got = dedup_near(&docs, cfg).expect("near");
    assert_eq!(got, reference::dedup_near(&docs, cfg).unwrap());
}
