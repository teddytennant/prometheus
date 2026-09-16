//! Group: embedding neighbourhood search (CPU, no network).
//!
//! Any deterministic bag-of-words / hashing-trick embedder should rank a
//! hypothesis that shares "Reverie loss" / "token-only" above an unrelated
//! Firecracker hypothesis. No GPU.

mod common;

use prometheus_ledger::Ledger;
use serde_json::json;

#[test]
fn search_similar_ranks_related_hypothesis_above_unrelated() {
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");

    let mut related = common::sample_record("exp-reverie");
    related.hypothesis =
        "Switching the decoder to Reverie loss versus a token-only training objective".to_string();
    related.prediction = "Reverie loss beats the token-only baseline on ΔRCI".to_string();

    let mut unrelated = common::sample_record("exp-firecracker");
    unrelated.hypothesis =
        "Firecracker microVM snapshot restore latency on the ARM sandbox fleet".to_string();
    unrelated.prediction = "fork latency stays under 15ms at p99".to_string();

    let mut other = common::sample_record("exp-muon");
    other.hypothesis = "MuonClip versus AdamW at equal H200 hours".to_string();
    other.prediction = "optimizer swap is neutral on rung 0".to_string();

    ledger.append(unrelated).unwrap();
    ledger.append(related).unwrap();
    ledger.append(other).unwrap();

    let hits = ledger
        .search_similar("Reverie loss vs token-only", 3)
        .expect("search_similar");
    assert!(
        hits.len() >= 2,
        "expected at least related+unrelated hits, got {}",
        hits.len()
    );

    let related_pos = hits
        .iter()
        .position(|h| h.record.experiment_id == "exp-reverie")
        .expect("related row must be in the neighbourhood");
    let unrelated_pos = hits
        .iter()
        .position(|h| h.record.experiment_id == "exp-firecracker")
        .expect("unrelated row should still be returned when k covers the store");
    assert!(
        related_pos < unrelated_pos,
        "related Reverie hypothesis must rank above unrelated Firecracker; order: {:?}",
        hits.iter()
            .map(|h| (h.record.experiment_id.as_str(), h.score))
            .collect::<Vec<_>>()
    );
    assert!(
        hits[related_pos].score > hits[unrelated_pos].score,
        "related score {} must exceed unrelated score {}",
        hits[related_pos].score,
        hits[unrelated_pos].score
    );
}

#[test]
fn search_similar_surfaces_prior_negatives() {
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");

    let mut negative = common::sample_record("exp-reverie-neg");
    negative.hypothesis =
        "Reverie loss vs token-only did not improve RCI on the held-out mix".to_string();
    negative.prediction = "null result: no gain over token-only".to_string();
    negative.results = Some(json!({"passed": false, "negative": true}));

    let mut unrelated = common::sample_record("exp-sandbox");
    unrelated.hypothesis = "cgroup memory.max tuning for the eval sandbox".to_string();
    unrelated.prediction = "OOM rate drops".to_string();
    unrelated.results = Some(json!({"passed": true}));

    ledger.append(unrelated).unwrap();
    ledger.append(negative).unwrap();

    let hits = ledger
        .search_similar("Reverie loss vs token-only", 2)
        .expect("search_similar");
    assert!(
        !hits.is_empty(),
        "neighbourhood must include the prior negative"
    );
    assert_eq!(
        hits[0].record.experiment_id,
        "exp-reverie-neg",
        "the semantically related *negative* must surface first, got {:?}",
        hits.iter()
            .map(|h| h.record.experiment_id.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn search_similar_empty_store_and_k_zero_are_empty() {
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");
    let empty = ledger
        .search_similar("Reverie loss vs token-only", 5)
        .expect("search empty");
    assert!(empty.is_empty());

    ledger.append(common::sample_record("exp-k0")).unwrap();
    let none = ledger
        .search_similar("Reverie loss vs token-only", 0)
        .expect("k=0");
    assert!(none.is_empty(), "k=0 must return no hits");
}
