//! Group: minhash_signature length, equality, Jaccard vs matching hashes.

mod common;
mod reference;

use prometheus_dedup::{minhash_signature, DedupConfig};

#[test]
fn length_equals_num_hashes_standard() {
    let cfg = DedupConfig::standard();
    assert_eq!(cfg.shingle_size, 5, "B2 standard is 5-shingles");
    assert_eq!(cfg.num_hashes, 128, "B2 standard is 128 MinHash hashes");
    assert_eq!(cfg.bands, 32, "B2 standard is 32 LSH bands");
    assert_eq!(cfg.rows, 4, "B2 standard is 4 rows per band");
    assert_eq!(cfg.num_hashes, cfg.bands * cfg.rows);
    let sig = minhash_signature(common::alphabet(), cfg);
    assert_eq!(sig.len(), 128);
    assert_eq!(sig, reference::minhash_signature(common::alphabet(), cfg));
}

#[test]
fn length_equals_num_hashes_small() {
    let cfg = common::cfg_small();
    let sig = minhash_signature("alpha bravo charlie delta", cfg);
    assert_eq!(sig.len(), cfg.num_hashes);
}

#[test]
fn identical_texts_equal() {
    let cfg = DedupConfig::standard();
    let a = minhash_signature("The quick brown fox jumps over the lazy dog today", cfg);
    let b = minhash_signature("the  QUICK brown fox jumps over the lazy dog today", cfg);
    assert_eq!(a, b);
}

#[test]
fn identical_normalized_empty_equal() {
    let cfg = common::cfg_small();
    assert_eq!(minhash_signature("", cfg), minhash_signature("  \n", cfg));
    assert_eq!(minhash_signature("", cfg), vec![u64::MAX; cfg.num_hashes]);
}

#[test]
fn different_texts_differ() {
    let cfg = DedupConfig::standard();
    let a = minhash_signature(common::alphabet(), cfg);
    let b = minhash_signature(common::alphabet_unrelated(), cfg);
    assert_ne!(a, b);
}

#[test]
fn higher_jaccard_means_more_matching_hashes() {
    let cfg = DedupConfig::standard();
    let n = cfg.shingle_size;
    let pairs = [
        (common::alphabet(), common::alphabet()),
        (common::alphabet(), common::alphabet_near()),
        (
            "alpha bravo charlie delta echo foxtrot golf hotel",
            "alpha bravo charlie zebra yankee xray whiskey victor",
        ),
        (common::alphabet(), common::alphabet_unrelated()),
    ];
    let mut scored: Vec<(f64, usize)> = pairs
        .iter()
        .map(|(a, b)| {
            let j = reference::jaccard(a, b, n);
            let sa = minhash_signature(a, cfg);
            let sb = minhash_signature(b, cfg);
            assert_eq!(sa.len(), cfg.num_hashes);
            (j, reference::signature_matches(&sa, &sb))
        })
        .collect();
    // Property: on these fixtures, more Jaccard ⇒ at least as many matching hashes.
    scored.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    for w in scored.windows(2) {
        assert!(
            w[0].1 <= w[1].1,
            "Jaccard {} had {} matches but Jaccard {} had {}; expected monotonic",
            w[0].0,
            w[0].1,
            w[1].0,
            w[1].1
        );
    }
    assert_eq!(
        scored.last().unwrap().1,
        cfg.num_hashes,
        "identical → all match"
    );
}

#[test]
fn matches_reference_on_fixture_corpus() {
    let cfg = DedupConfig::standard();
    for d in common::fixture_corpus() {
        assert_eq!(
            minhash_signature(&d.text, cfg),
            reference::minhash_signature(&d.text, cfg),
            "minhash mismatch on {}",
            d.id
        );
    }
}
