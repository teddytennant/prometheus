//! Group: public constants, `is_held_out_suite`, `Subject::as_str`, Scores
//! serde field set, and the public-method inventory.
//!
//! These tests do **not** call `EvalGate::{open,request,rotate,suite_hash}`
//! and are allowed to pass against the unimplemented stub.

use prometheus_eval_gate::{is_held_out_suite, Scores, Subject, HELD_OUT_SUITES, ROTATION_MS};

mod reference;

#[test]
fn held_out_suites_exact_slugs_in_order() {
    assert_eq!(
        HELD_OUT_SUITES,
        &[
            "re_bench",
            "mle_bench",
            "paperbench",
            "speedrun",
            "metr_time_horizon",
        ]
    );
    assert_eq!(HELD_OUT_SUITES, reference::HELD_OUT_SUITES);
}

#[test]
fn is_held_out_suite_true_for_every_held_out_slug() {
    for slug in HELD_OUT_SUITES {
        assert!(is_held_out_suite(slug), "{slug}");
    }
}

#[test]
fn is_held_out_suite_false_for_public_not_held_out() {
    for slug in [
        "gpqa",
        "aime",
        "swe_bench_verified",
        "swe_bench_pro",
        "terminal_bench",
        "competitive_programming",
        "hmmt",
        "frontiermath",
        "minif2f",
        "putnambench",
        "arc_agi_1",
        "arc_agi_2",
        "arc_agi_3",
        "hle",
        "long_context_128k",
        "long_context_1m",
        "forecasting",
    ] {
        assert!(
            !is_held_out_suite(slug),
            "{slug} is public F3, not held-out"
        );
        assert!(
            reference::PUBLIC_SUITES.contains(&slug),
            "{slug} missing from reference public list"
        );
    }
}

#[test]
fn is_held_out_suite_false_for_unknown_empty_and_wrong_case() {
    assert!(!is_held_out_suite(""));
    assert!(!is_held_out_suite("nope"));
    assert!(!is_held_out_suite("RE_BENCH"));
    assert!(!is_held_out_suite("re-bench"));
    assert!(!is_held_out_suite("re_bench "));
}

#[test]
fn rotation_ms_is_ninety_days() {
    assert_eq!(ROTATION_MS, 90 * 24 * 60 * 60 * 1000);
    assert_eq!(ROTATION_MS, 7_776_000_000);
}

#[test]
fn subject_as_str_genome_and_checkpoint() {
    assert_eq!(Subject::GenomeRev("abc".into()).as_str(), "abc");
    assert_eq!(Subject::Checkpoint("ckpt".into()).as_str(), "ckpt");
    assert_eq!(Subject::GenomeRev(String::new()).as_str(), "");
}

#[test]
fn scores_serde_keys_are_exactly_the_seven_fields() {
    let scores = Scores {
        suite: "re_bench".into(),
        subject: Subject::GenomeRev("rev".into()),
        rci_milli: 0,
        n_items: 0,
        compute_ms: 0,
        harness_version: "hv".into(),
        suite_hash: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into(),
    };
    let v = serde_json::to_value(&scores).expect("serde");
    let obj = v.as_object().expect("object");
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
    assert_eq!(keys, expected);
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
        "n_correct",
    ] {
        assert!(!obj.contains_key(banned), "unexpected field {banned}");
    }
}

#[test]
fn evalgate_public_methods_are_exactly_open_request_rotate_suite_hash() {
    let src = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs"));
    let impl_idx = src.find("impl EvalGate").expect("impl EvalGate block");
    let rest = &src[impl_idx..];
    let mut names = Vec::new();
    for line in rest.lines() {
        let t = line.trim();
        if t.starts_with("pub fn ") {
            let after = t.trim_start_matches("pub fn ");
            let name = after.split(['<', '(']).next().unwrap();
            names.push(name.to_string());
        }
        // Stop at the next top-level impl/fn after this block's methods —
        // the iface keeps methods inside a single impl EvalGate.
        if t.starts_with("impl ") && !t.starts_with("impl EvalGate") {
            break;
        }
    }
    names.sort();
    assert_eq!(
        names,
        ["open", "request", "rotate", "suite_hash"],
        "EvalGate public methods changed; a method that could return task text is forbidden"
    );
}

#[test]
fn lib_src_has_no_public_item_or_prompt_api() {
    let src = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs"));
    for needle in [
        "pub struct Item",
        "pub struct EvalItem",
        "pub struct Task",
        "pub fn prompt",
        "pub fn answer",
        "pub fn items",
        "pub fn item",
        "pub fn task_text",
        "pub fn item_body",
        "pub fn get_prompt",
        "pub fn get_item",
        "pub fn dump_suite",
        "pub fn read_task",
        "pub fn leak",
    ] {
        assert!(
            !src.contains(needle),
            "public API must not expose task text ({needle})"
        );
    }
}

#[test]
fn reference_golden_hash_matches_python_bytes() {
    let h = reference::hash_items(&[reference::golden_item()]);
    assert_eq!(h, reference::GOLDEN_A_HASH);
    assert_eq!(reference::hash_items(&[]), reference::EMPTY_HASH);
    let mut reversed = vec![
        reference::Item {
            id: "b".into(),
            prompt: "pb".into(),
            answer: "sb".into(),
            weight_milli: 2,
        },
        reference::golden_item(),
    ];
    let h1 = reference::hash_items(&reversed);
    reversed.reverse();
    assert_eq!(h1, reference::hash_items(&reversed));
}
