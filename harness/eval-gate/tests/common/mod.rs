//! Shared fixtures and assertions for the L2 eval-gate oracle.
//!
//! Canary strings are planted in held-out item bodies. Production return
//! values, serde, Debug, Display, and error strings must never contain them.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use prometheus_eval_gate::{GateConfig, Scores, Subject};
use tempfile::TempDir;

use crate::reference::{self, Item};

pub const HV: &str = "test-harness-1";
pub const CANARY_PROMPT: &str = "CANARY_PROMPT_NEVER_LEAVE_GATE_7f3a9c2e";
pub const CANARY_ANSWER: &str = "CANARY_ANSWER_NEVER_LEAVE_GATE_d41d8cd9";
pub const CANARY_BODY: &str = "CANARY_ITEM_BODY_NEVER_LEAVE_GATE_deadbeef";

pub const PUBLIC_NOT_HELD_OUT: &[&str] =
    &["gpqa", "aime", "swe_bench_verified", "arc_agi_1", "hle"];

pub fn fresh_root() -> (TempDir, PathBuf) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().to_path_buf();
    (tmp, root)
}

pub fn cfg(root: &Path, harness_version: &str) -> GateConfig {
    GateConfig {
        suite_root: root.to_path_buf(),
        harness_version: harness_version.to_string(),
    }
}

pub fn canary_items() -> Vec<Item> {
    vec![
        Item {
            id: "i1".into(),
            prompt: CANARY_BODY.into(),
            answer: "alpha".into(),
            weight_milli: 400,
        },
        Item {
            id: "i2".into(),
            prompt: CANARY_PROMPT.into(),
            answer: CANARY_ANSWER.into(),
            weight_milli: 600,
        },
    ]
}

pub fn bland_items(suite: &str, n: usize) -> Vec<Item> {
    (0..n)
        .map(|i| Item {
            id: format!("{suite}-{i}"),
            prompt: format!("q-{suite}-{i}"),
            answer: format!("a-{suite}-{i}"),
            weight_milli: 1000,
        })
        .collect()
}

pub fn pending_items(suite: &str, n: usize) -> Vec<Item> {
    (0..n)
        .map(|i| Item {
            id: format!("{suite}-p{i}"),
            prompt: format!("pending-{suite}-{i}-{CANARY_PROMPT}"),
            answer: format!("pend-a-{suite}-{i}"),
            weight_milli: 250,
        })
        .collect()
}

pub fn seed_re_bench(root: &Path) -> String {
    reference::install_current(root, "re_bench", canary_items(), 1, 0, 5_000, 0)
        .expect("install re_bench")
}

pub fn seed_held_out_suite(root: &Path, suite: &str, n_items: usize, compute_ms: u64) -> String {
    let items = if suite == "re_bench" {
        canary_items()
    } else {
        bland_items(suite, n_items)
    };
    reference::install_current(root, suite, items, 1, 0, compute_ms, 0).expect("install suite")
}

pub fn seed_all_held_out(root: &Path) {
    seed_re_bench(root);
    seed_held_out_suite(root, "mle_bench", 3, 3_000);
    seed_held_out_suite(root, "paperbench", 1, 1_000);
    seed_held_out_suite(root, "speedrun", 2, 2_000);
    seed_held_out_suite(root, "metr_time_horizon", 4, 4_000);
}

pub fn seed_perfect_genome(root: &Path) {
    let mut answers = BTreeMap::new();
    answers.insert("i1".into(), "alpha".into());
    answers.insert("i2".into(), CANARY_ANSWER.into());
    let sub = Subject::GenomeRev("perfect".into());
    reference::install_subject(root, &sub, &answers).expect("perfect genome");
}

pub fn seed_partial_genome(root: &Path) {
    let mut answers = BTreeMap::new();
    answers.insert("i1".into(), "alpha".into());
    let sub = Subject::GenomeRev("partial".into());
    reference::install_subject(root, &sub, &answers).expect("partial genome");
}

pub fn seed_zero_genome(root: &Path) {
    let sub = Subject::GenomeRev("zero".into());
    reference::install_subject(root, &sub, &BTreeMap::new()).expect("zero genome");
}

pub fn seed_perfect_checkpoint(root: &Path) {
    let mut answers = BTreeMap::new();
    answers.insert("i1".into(), "alpha".into());
    answers.insert("i2".into(), CANARY_ANSWER.into());
    let sub = Subject::Checkpoint("ckpt-perfect".into());
    reference::install_subject(root, &sub, &answers).expect("perfect ckpt");
}

pub fn seed_standard(root: &Path) {
    seed_all_held_out(root);
    seed_perfect_genome(root);
    seed_partial_genome(root);
    seed_zero_genome(root);
    seed_perfect_checkpoint(root);
    seed_bland_subjects(root);
}

fn seed_bland_subjects(root: &Path) {
    for suite in ["mle_bench", "paperbench", "speedrun", "metr_time_horizon"] {
        let items = bland_items(
            suite,
            match suite {
                "mle_bench" => 3,
                "paperbench" => 1,
                "speedrun" => 2,
                "metr_time_horizon" => 4,
                _ => 1,
            },
        );
        let mut answers = BTreeMap::new();
        for it in &items {
            answers.insert(it.id.clone(), it.answer.clone());
        }
        // Re-use genome/perfect for re_bench only; other suites get their own.
        let sub = Subject::GenomeRev(format!("perfect-{suite}"));
        reference::install_subject(root, &sub, &answers).expect("bland subject");
    }
}

pub fn assert_no_leak(hay: &str, where_: &str) {
    for c in [CANARY_PROMPT, CANARY_ANSWER, CANARY_BODY] {
        assert!(
            !hay.contains(c),
            "task text leaked in {where_}: found {c} in {hay:?}"
        );
    }
}

pub fn assert_scores_eq(got: &Scores, exp: &Scores) {
    assert_eq!(got.suite, exp.suite, "suite");
    assert_eq!(got.subject, exp.subject, "subject");
    assert_eq!(got.rci_milli, exp.rci_milli, "rci_milli");
    assert_eq!(got.n_items, exp.n_items, "n_items");
    assert_eq!(got.compute_ms, exp.compute_ms, "compute_ms");
    assert_eq!(got.harness_version, exp.harness_version, "harness_version");
    assert_eq!(got.suite_hash, exp.suite_hash, "suite_hash");
}

pub fn assert_hash_shape(hash: &str) {
    assert_eq!(
        hash.len(),
        64,
        "suite_hash must be 64 hex chars, got {hash}"
    );
    assert!(
        hash.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')),
        "suite_hash must be lowercase hex, got {hash}"
    );
}

pub fn scores_json(scores: &Scores) -> String {
    serde_json::to_string(scores).expect("serde Scores")
}

pub fn leak_scan_scores(scores: &Scores, where_: &str) {
    assert_no_leak(&scores_json(scores), &format!("{where_} json"));
    assert_no_leak(&format!("{scores:?}"), &format!("{where_} debug"));
    assert_no_leak(&scores.suite, &format!("{where_} suite"));
    assert_no_leak(scores.subject.as_str(), &format!("{where_} subject"));
    assert_no_leak(
        &scores.harness_version,
        &format!("{where_} harness_version"),
    );
    assert_no_leak(&scores.suite_hash, &format!("{where_} suite_hash"));
    let v: serde_json::Value = serde_json::from_str(&scores_json(scores)).unwrap();
    let obj = v.as_object().expect("Scores json object");
    let keys: std::collections::BTreeSet<&str> = obj.keys().map(|s| s.as_str()).collect();
    let expected: std::collections::BTreeSet<&str> = [
        "suite",
        "subject",
        "rci_milli",
        "n_items",
        "compute_ms",
        "harness_version",
        "suite_hash",
    ]
    .into_iter()
    .collect();
    assert_eq!(keys, expected, "Scores serde keys");
    for banned in [
        "prompt",
        "answer",
        "items",
        "item",
        "task",
        "body",
        "text",
        "payload",
        "item_body",
        "task_text",
    ] {
        assert!(
            !obj.contains_key(banned),
            "Scores serde must not contain {banned}"
        );
    }
}

pub fn generation_items_path(root: &Path, suite: &str, hash: &str) -> PathBuf {
    root.join(suite)
        .join("generations")
        .join(hash)
        .join("items.jsonl")
}
