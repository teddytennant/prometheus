//! Group: iface helpers that do **not** call unimplemented Checkpointer / Store
//! methods.
//!
//! These may pass against the stub. Every other test file exercises save,
//! restore, Store, or HostRamOffload and must fail until A5 is implemented.

use prometheus_ckpt::{sha256_hex, Dtype, Manifest, MEMORY_REPLICAS, SCHEMA_ID, SCHEMA_VERSION};
use serde_json::Value;

const GOLDEN_JSON: &str = include_str!("../../contracts/goldens/v1/checkpoint.default.json");

const NIST_SHA256_EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
const NIST_SHA256_ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

#[test]
fn memory_replicas_is_two() {
    assert_eq!(MEMORY_REPLICAS, 2);
}

#[test]
fn sha256_hex_matches_nist_empty_and_abc() {
    assert_eq!(sha256_hex(b""), NIST_SHA256_EMPTY);
    assert_eq!(sha256_hex(b"abc"), NIST_SHA256_ABC);
    assert_eq!(sha256_hex(&[]), NIST_SHA256_EMPTY);
    assert_eq!(NIST_SHA256_EMPTY.len(), 64);
    assert_eq!(NIST_SHA256_ABC.len(), 64);
}

#[test]
fn golden_manifest_from_json_to_json_roundtrip() {
    let value: Value = serde_json::from_str(GOLDEN_JSON).expect("golden parses as JSON");
    assert_eq!(value["schema_id"], "prometheus.checkpoint");
    assert_eq!(value["schema_version"], 1);
    assert!(value["parent_checkpoint_id"].is_null());
    assert_eq!(value["precision"]["param_dtype"], "fp32");
    assert_eq!(value["precision"]["remat"], "full");
    assert_eq!(value["weights"][0]["dtype"], "fp32");
    assert!(value["rng"][0]["state_hash"].is_string());

    let m = Manifest::from_json(&value).expect("Manifest::from_json of F1 golden");
    assert_eq!(m.parent_checkpoint_id, None);
    assert_eq!(m.checkpoint_id, "r0-step-1024");
    assert_eq!(m.step, 1024);
    assert_eq!(m.rung, 0);
    let precision = m.precision.as_ref().expect("precision present");
    assert_eq!(precision.param_dtype, Dtype::Fp32);
    assert_eq!(precision.remat, "full");
    assert!(!precision.remat.is_empty());
    assert_eq!(m.weights[0].dtype, Dtype::Fp32);
    assert_eq!(m.weights[0].name, "embed");
    assert_eq!(m.rng.len(), 1);
    assert_eq!(
        m.rng[0].state_hash,
        "b4ce30dbac92e34ead9e3986d13dec9b59e8664c3961f41db7226adef8e48390"
    );
    assert_eq!(m.rng[0].state_hash.len(), 64);
    assert_eq!(m.rng[0].scope, "dropout");

    let encoded = m.to_json().expect("to_json");
    assert_eq!(encoded["schema_id"], SCHEMA_ID);
    assert_eq!(encoded["schema_id"], "prometheus.checkpoint");
    assert_eq!(encoded["schema_version"], SCHEMA_VERSION);
    assert!(encoded["parent_checkpoint_id"].is_null());
    assert_eq!(encoded["precision"]["remat"], "full");
    assert_eq!(encoded["weights"][0]["dtype"], "fp32");
    assert!(encoded["rng"][0]["state_hash"].is_string());

    let m2 = Manifest::from_json(&encoded).expect("from_json of to_json");
    assert_eq!(m, m2);
}

#[test]
fn from_json_rejects_wrong_schema_id() {
    let mut value: Value = serde_json::from_str(GOLDEN_JSON).unwrap();
    value["schema_id"] = Value::String("prometheus.not_checkpoint".into());
    let err = Manifest::from_json(&value).expect_err("wrong schema_id");
    let msg = err.to_string();
    assert!(
        msg.contains("prometheus.not_checkpoint") || msg.contains("schema"),
        "schema error should mention the id, got {msg}"
    );
}
