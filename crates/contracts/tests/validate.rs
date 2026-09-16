//! Implementer-facing Rust tests for `prometheus-contracts`.
//!
//! Against the current `unimplemented!` stubs these panic (fail). Once
//! `validate` is implemented they must succeed for goldens and return
//! `Error::Validation` or `Error::UnknownSchema` for a missing schema_id —
//! never panic.

use std::fs;
use std::path::PathBuf;

use prometheus_contracts::{validate, Error};
use serde_json::{json, Value};

fn goldens_v1() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../contracts/goldens/v1")
}

#[test]
fn validate_all_v1_goldens() {
    let dir = goldens_v1();
    let mut n = 0usize;
    for entry in fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display())) {
        let path = entry.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let text = fs::read_to_string(&path).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        validate(&value).unwrap_or_else(|e| panic!("{} should validate: {e}", path.display()));
        n += 1;
    }
    assert!(n >= 16, "expected a golden per schema, found {n}");
}

#[test]
fn validate_missing_schema_id_is_error_not_panic() {
    let payload = json!({ "schema_version": 1, "packing_id": "x" });
    let err = validate(&payload).expect_err("missing schema_id must be Err, not Ok");
    match err {
        Error::Validation(_) | Error::UnknownSchema { .. } => {}
        other => panic!("expected Validation or UnknownSchema, got {other:?}"),
    }
}
