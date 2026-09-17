//! Group: F1 RewardRequest / RewardResponse goldens and Flag::Tampering serde.

mod common;
mod reference;

use common::golden_path;
use prometheus_rewards::{
    Evidence, EvidenceKind, Flag, HackKind, RewardRequest, RewardResponse, SCHEMA_REWARD_RESPONSE,
    SCHEMA_VERSION,
};

fn load_golden(name: &str) -> serde_json::Value {
    let p = golden_path(name);
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    serde_json::from_str(&text).expect("golden json")
}

#[test]
fn reward_request_golden_round_trip() {
    let golden = load_golden("reward_request.default.json");
    let req: RewardRequest = serde_json::from_value(golden.clone()).expect("decode request");
    let back = serde_json::to_value(&req).expect("encode request");
    assert_eq!(back, golden);
}

#[test]
fn reward_response_golden_shares_scalar_fields() {
    let v = load_golden("reward_response.default.json");
    assert_eq!(v["schema_id"], "prometheus.reward_response");
    assert_eq!(v["schema_version"], 1);
    assert_eq!(v["request_id"], "rew-0001");
    assert_eq!(v["score"], 1.0);
    assert_eq!(v["passed"], true);
    assert_eq!(v["scored_at"], "2026-09-16T12:00:00Z");
    assert!(v["flags"].as_array().unwrap().is_empty());
}

#[test]
fn reward_response_constructed_roundtrip_includes_tampering() {
    let resp = RewardResponse {
        schema_id: SCHEMA_REWARD_RESPONSE.to_string(),
        schema_version: SCHEMA_VERSION,
        request_id: "rew-0001".into(),
        score: 0.0,
        passed: false,
        flags: vec![Flag::Tampering],
        evidence: vec![Evidence {
            kind: EvidenceKind::Rubric,
            hash: reference::sha256_hex(b"the answer is four"),
            summary: None,
        }],
        scored_at: "not-a-wall-clock".into(),
    };
    let v = serde_json::to_value(&resp).unwrap();
    assert_eq!(v["schema_id"], "prometheus.reward_response");
    assert_eq!(v["flags"][0], "tampering");
    assert_eq!(v["evidence"][0]["kind"], "rubric");
    assert!(v["evidence"][0]["hash"].as_str().unwrap().len() == 64);
    assert!(v.get("task_id").is_none());
    assert!(v.get("scorer_id").is_none());
    let back: RewardResponse = serde_json::from_value(v).unwrap();
    assert_eq!(resp, back);
}

#[test]
fn flag_tampering_serde_is_snake_case() {
    let v = serde_json::to_value(Flag::Tampering).unwrap();
    assert_eq!(v, serde_json::json!("tampering"));
    let back: Flag = serde_json::from_value(serde_json::json!("tampering")).unwrap();
    assert_eq!(back, Flag::Tampering);
}

#[test]
fn hack_kinds_are_snake_case() {
    assert_eq!(
        serde_json::to_value(HackKind::TestWrite).unwrap(),
        serde_json::json!("test_write")
    );
    assert_eq!(
        serde_json::to_value(HackKind::SpecialCase).unwrap(),
        serde_json::json!("special_case")
    );
    assert_eq!(
        serde_json::to_value(HackKind::VisiblePassHiddenFail).unwrap(),
        serde_json::json!("visible_pass_hidden_fail")
    );
}

#[test]
fn evidence_kind_rubric_is_snake_case() {
    assert_eq!(
        serde_json::to_value(EvidenceKind::Rubric).unwrap(),
        serde_json::json!("rubric")
    );
}
