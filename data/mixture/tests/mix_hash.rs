//! Group: mix_hash / canonical_json — lowercase hex SHA-256 of serde_json Mix.

mod common;
mod reference;

use common::{rung2_like, two_source, uniform8};
use prometheus_mixture::{canonical_json, mix_hash, Mix, Phase, Source};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// serde_json compact Mix of two_source() (web 0.75, code 0.25).
const TWO_SOURCE_JSON: &str = r#"{"mix_id":"two","mix_bucket":"pretrain-r0","phase":"pretrain","weights":{"web":0.75,"code":0.25}}"#;

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

#[test]
fn two_source_canonical_json_matches_serde_json_and_reference() {
    let mix = two_source();
    let got = canonical_json(&mix).expect("public");
    let want = reference::canonical_json(&mix).expect("reference");
    assert_eq!(got, want);
    assert_eq!(got, TWO_SOURCE_JSON.as_bytes());
    assert_eq!(got, serde_json::to_vec(&mix).expect("serde_json"));
}

#[test]
fn two_source_mix_hash_is_sha256_of_canonical_json() {
    let mix = two_source();
    let got = mix_hash(&mix).expect("public");
    let want = reference::mix_hash(&mix).expect("reference");
    assert_eq!(got, want);
    common::assert_sha256_hex(&got);
    let bytes = canonical_json(&mix).expect("canonical");
    assert_eq!(got, sha256_hex(&bytes));
    assert_eq!(got, sha256_hex(TWO_SOURCE_JSON.as_bytes()));
}

#[test]
fn uniform8_hash_matches_reference_and_is_lowercase_hex() {
    let mix = uniform8();
    let got = mix_hash(&mix).expect("public");
    let want = reference::mix_hash(&mix).expect("reference");
    assert_eq!(got, want);
    common::assert_sha256_hex(&got);
    assert_eq!(got, sha256_hex(&canonical_json(&mix).expect("canonical")));
}

#[test]
fn rung2_like_hash_matches_reference() {
    let mix = rung2_like();
    assert_eq!(
        mix_hash(&mix).expect("public"),
        reference::mix_hash(&mix).expect("reference")
    );
}

#[test]
fn btree_insertion_order_does_not_change_hash() {
    let mut a = BTreeMap::new();
    a.insert(Source::Code, 0.25);
    a.insert(Source::Web, 0.75);
    let mut b = BTreeMap::new();
    b.insert(Source::Web, 0.75);
    b.insert(Source::Code, 0.25);
    let ma = Mix {
        mix_id: "two".into(),
        mix_bucket: "pretrain-r0".into(),
        phase: Phase::Pretrain,
        weights: a,
    };
    let mb = Mix {
        mix_id: "two".into(),
        mix_bucket: "pretrain-r0".into(),
        phase: Phase::Pretrain,
        weights: b,
    };
    let ha = mix_hash(&ma).expect("a");
    let hb = mix_hash(&mb).expect("b");
    assert_eq!(ha, hb);
    assert_eq!(ha, reference::mix_hash(&ma).expect("ref"));
}

#[test]
fn mix_id_change_changes_hash() {
    let mut other = two_source();
    other.mix_id = "other".into();
    let a = mix_hash(&two_source()).expect("a");
    let b = mix_hash(&other).expect("b");
    assert_ne!(a, b);
    assert_eq!(b, reference::mix_hash(&other).expect("ref"));
}

#[test]
fn phase_change_changes_hash() {
    let mut decay = two_source();
    decay.phase = Phase::Decay;
    assert_ne!(
        mix_hash(&two_source()).expect("pretrain"),
        mix_hash(&decay).expect("decay")
    );
}

#[test]
fn non_ascii_free_compact_json_has_no_whitespace() {
    let bytes = canonical_json(&two_source()).expect("canonical");
    let s = String::from_utf8(bytes).expect("utf8");
    assert!(!s.chars().any(|c| c.is_whitespace()), "{s}");
    assert!(s.contains("\"web\":0.75"));
    assert!(s.find("\"mix_id\"").unwrap() < s.find("\"mix_bucket\"").unwrap());
    assert!(s.find("\"web\"").unwrap() < s.find("\"code\"").unwrap());
}

#[test]
fn agentic_key_follows_enum_order_not_alpha() {
    let mix = common::mix(
        "m",
        "b",
        Phase::Pretrain,
        &[(Source::AgenticTrajectories, 0.5), (Source::Web, 0.5)],
    );
    let s = String::from_utf8(canonical_json(&mix).expect("canonical")).expect("utf8");
    let want = reference::canonical_json(&mix).expect("ref");
    assert_eq!(s.as_bytes(), want.as_slice());
    // BTreeMap / Source Ord: web before agentic_trajectories, not alphabetical.
    assert!(
        s.find("\"web\"").unwrap() < s.find("\"agentic_trajectories\"").unwrap(),
        "{s}"
    );
}

#[test]
fn nan_weight_canonical_json_errors() {
    let mix = common::mix("m", "b", Phase::Pretrain, &[(Source::Web, f64::NAN)]);
    assert!(canonical_json(&mix).is_err());
    assert!(mix_hash(&mix).is_err());
    assert!(reference::canonical_json(&mix).is_err());
    assert!(reference::mix_hash(&mix).is_err());
}
