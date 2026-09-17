//! Group: F1 TaskSpec golden round-trip and snake_case domain/split.

mod common;
mod reference;

use std::fs;

use common::CREATED;
use prometheus_tasks::{Provenance, Split, TaskDomain, TaskSpec, SCHEMA_TASK_SPEC, SCHEMA_VERSION};
use serde_json::{json, Value};

fn load_golden() -> Value {
    let p = common::golden_path("task_spec.default.json");
    let text = fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    serde_json::from_str(&text).expect("golden json")
}

fn required_fields() -> [&'static str; 7] {
    [
        "schema_id",
        "schema_version",
        "task_id",
        "domain",
        "verifier_id",
        "hidden_tests_hash",
        "provenance",
    ]
}

#[test]
fn golden_required_fields_present() {
    let g = load_golden();
    for key in required_fields() {
        assert!(g.get(key).is_some(), "golden missing {key}");
    }
    assert_eq!(g["schema_id"], json!(SCHEMA_TASK_SPEC));
    assert_eq!(g["schema_version"], json!(SCHEMA_VERSION));
    assert_eq!(g["domain"], json!("math"));
    assert!(g["task_id"].as_str().is_some_and(|s| !s.is_empty()));
    assert!(g["verifier_id"].as_str().is_some_and(|s| !s.is_empty()));
    assert!(g["hidden_tests_hash"]
        .as_str()
        .is_some_and(|s| s.len() == 64));
    assert!(g["provenance"]["source"]
        .as_str()
        .is_some_and(|s| !s.is_empty()));
}

#[test]
fn task_spec_round_trip_shares_required_fields_with_golden() {
    let g = load_golden();
    let spec = TaskSpec {
        schema_id: g["schema_id"].as_str().unwrap().to_string(),
        schema_version: g["schema_version"].as_u64().unwrap() as u32,
        task_id: g["task_id"].as_str().unwrap().to_string(),
        domain: serde_json::from_value(g["domain"].clone()).unwrap(),
        split: Split::Train,
        statement_hash: g["public_statement_hash"].as_str().unwrap().to_string(),
        hidden_tests_hash: g["hidden_tests_hash"].as_str().unwrap().to_string(),
        verifier_id: g["verifier_id"].as_str().unwrap().to_string(),
        env_image: None,
        horizon_s: g["horizon"]["walltime_s"].as_u64().unwrap() as u32,
        max_tool_calls: g["horizon"]["max_tool_calls"].as_u64().unwrap() as u32,
        provenance: serde_json::from_value(g["provenance"].clone()).unwrap(),
        created_at: CREATED.to_string(),
    };
    spec.schema_ok().unwrap();
    let ser = serde_json::to_value(&spec).unwrap();
    for key in required_fields() {
        assert_eq!(ser[key], g[key], "required field {key}");
    }
    // Extra golden fields (factory_id, difficulty, tags, env_snapshot_id, horizon
    // object, public_statement_hash) are ignored.
    let back: TaskSpec = serde_json::from_value(ser).unwrap();
    assert_eq!(back, spec);
}

#[test]
fn domain_serde_snake_case() {
    let cases = [
        (TaskDomain::Math, "math"),
        (TaskDomain::Code, "code"),
        (TaskDomain::Science, "science"),
        (TaskDomain::Security, "security"),
        (TaskDomain::Arc, "arc"),
        (TaskDomain::Agent, "agent"),
        (TaskDomain::Other, "other"),
    ];
    for (dom, name) in cases {
        assert_eq!(serde_json::to_value(dom).unwrap(), json!(name));
        let got: TaskDomain = serde_json::from_value(json!(name)).unwrap();
        assert_eq!(got, dom);
    }
}

#[test]
fn split_serde_snake_case() {
    let cases = [
        (Split::Train, "train"),
        (Split::Val, "val"),
        (Split::Test, "test"),
        (Split::Holdout, "holdout"),
    ];
    for (split, name) in cases {
        assert_eq!(serde_json::to_value(split).unwrap(), json!(name));
        let got: Split = serde_json::from_value(json!(name)).unwrap();
        assert_eq!(got, split);
    }
}

#[test]
fn provenance_skips_none_license() {
    let p = Provenance::new("gsm8k");
    let v = serde_json::to_value(&p).unwrap();
    assert_eq!(v["source"], json!("gsm8k"));
    assert!(v.get("license").is_none());
}

#[test]
fn sha256_empty_is_the_published_digest() {
    assert_eq!(
        reference::sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    let g = load_golden();
    assert!(reference::is_sha256_hex(
        g["hidden_tests_hash"].as_str().unwrap()
    ));
    assert!(reference::is_sha256_hex(
        g["public_statement_hash"].as_str().unwrap()
    ));
}
