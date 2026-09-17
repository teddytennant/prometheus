//! Group: F1 RewardRequest / RewardResponse serde, flag and evidence names.

mod common;
mod reference;

use common::golden_path;
use prometheus_verifiers::{
    Evidence, EvidenceKind, Flag, RewardRequest, RewardResponse, SCHEMA_REWARD_RESPONSE,
    SCHEMA_VERSION,
};

#[test]
fn reward_request_golden_roundtrip() {
    let raw = std::fs::read_to_string(golden_path("reward_request.default.json")).unwrap();
    let req: RewardRequest = serde_json::from_str(&raw).unwrap();
    assert_eq!(req.schema_id, "prometheus.reward_request");
    assert_eq!(req.schema_version, 1);
    assert_eq!(req.request_id, "rew-0001");
    assert_eq!(req.task_id, "task-math-0001");
    assert_eq!(req.rollout_id, "roll-0001");
    assert_eq!(
        req.trajectory_hash,
        "df4d832092e8878123a97d334652a1b44bfc48e5c4035adcceb00b25487e7001"
    );
    assert_eq!(req.verifier_id, "sympy-exact");
    assert_eq!(req.env_snapshot_id, "snap-0001");
    let back: RewardRequest = serde_json::from_value(serde_json::to_value(&req).unwrap()).unwrap();
    assert_eq!(req, back);
    let v = serde_json::to_value(&req).unwrap();
    for key in [
        "schema_id",
        "schema_version",
        "request_id",
        "task_id",
        "rollout_id",
        "trajectory_hash",
        "verifier_id",
        "env_snapshot_id",
    ] {
        assert!(v.get(key).is_some(), "missing {key}");
    }
}

#[test]
fn flag_names_match_f1_schema() {
    let names: Vec<String> = [
        Flag::Tampering,
        Flag::TestWrite,
        Flag::Timeout,
        Flag::EnvCrash,
        Flag::Unverifiable,
    ]
    .into_iter()
    .map(|f| {
        serde_json::to_value(f)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    })
    .collect();
    assert_eq!(
        names,
        [
            "tampering",
            "test_write",
            "timeout",
            "env_crash",
            "unverifiable"
        ]
    );
}

#[test]
fn evidence_kind_names_match_schema() {
    let names: Vec<String> = [
        EvidenceKind::HiddenTests,
        EvidenceKind::Mutation,
        EvidenceKind::Symbolic,
        EvidenceKind::Lean,
        EvidenceKind::Grid,
        EvidenceKind::Rubric,
        EvidenceKind::Human,
    ]
    .into_iter()
    .map(|k| {
        serde_json::to_value(k)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    })
    .collect();
    assert_eq!(
        names,
        [
            "hidden_tests",
            "mutation",
            "symbolic",
            "lean",
            "grid",
            "rubric",
            "human"
        ]
    );
}

#[test]
fn reward_response_constructed_roundtrip() {
    let resp = RewardResponse {
        schema_id: SCHEMA_REWARD_RESPONSE.to_string(),
        schema_version: SCHEMA_VERSION,
        request_id: "rew-0001".into(),
        score: 1.0,
        passed: true,
        flags: vec![Flag::TestWrite],
        evidence: vec![Evidence {
            kind: EvidenceKind::HiddenTests,
            hash: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into(),
            summary: Some("hidden tests".into()),
        }],
        scored_at: "2020-01-01T00:00:00Z".into(),
    };
    let v = serde_json::to_value(&resp).unwrap();
    assert_eq!(v["schema_id"], "prometheus.reward_response");
    assert_eq!(v["request_id"], "rew-0001");
    assert_eq!(v["score"], 1.0);
    assert_eq!(v["passed"], true);
    assert_eq!(v["flags"][0], "test_write");
    assert!(v["evidence"].is_array(), "interface evidence is an array");
    assert_eq!(v["evidence"][0]["kind"], "hidden_tests");
    assert_eq!(v["scored_at"], "2020-01-01T00:00:00Z");
    assert!(v.get("task_id").is_none());
    assert!(v.get("scorer_id").is_none());
    let back: RewardResponse = serde_json::from_value(v).unwrap();
    assert_eq!(resp, back);
}

#[test]
fn reward_response_golden_shares_scalar_fields() {
    // F1 golden evidence is a single object (kind exact_match) plus task_id/scorer_id.
    // The Rust interface is Vec<Evidence> without ExactMatch; scalars still match.
    let raw = std::fs::read_to_string(golden_path("reward_response.default.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(v["schema_id"], "prometheus.reward_response");
    assert_eq!(v["schema_version"], 1);
    assert_eq!(v["request_id"], "rew-0001");
    assert_eq!(v["score"], 1.0);
    assert_eq!(v["passed"], true);
    assert_eq!(v["scored_at"], "2026-09-16T12:00:00Z");
    assert!(v["flags"].as_array().unwrap().is_empty());
}

#[test]
fn production_equivalent_matches_pinned_reference_on_golden_math() {
    assert!(reference::equivalent("1+2", "3"));
    assert!(prometheus_verifiers::SymbolicMath::equivalent("1+2", "3"));
}
