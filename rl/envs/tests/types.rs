//! Group: F1 serde goldens, `ToolRequest::new`, `Image::new`, `ForkResult`, getters.

use prometheus_envs::{
    Artifact, CallId, ForkResult, Image, ImageId, SandboxId, SnapshotId, Tool, ToolRequest,
    ToolResponse, FORK_BUDGET_MS, SCHEMA_TOOL_REQUEST, SCHEMA_TOOL_RESPONSE, SCHEMA_VERSION,
};

fn golden(name: &str) -> String {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts/goldens/v1")
        .join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

#[test]
fn tool_request_new_sets_envelope() {
    let r = ToolRequest::new("call-1", "snap-1", Tool::Python);
    assert_eq!(r.schema_id, SCHEMA_TOOL_REQUEST);
    assert_eq!(r.schema_version, SCHEMA_VERSION);
    assert_eq!(r.call_id, "call-1");
    assert_eq!(r.env_snapshot_id, "snap-1");
    assert_eq!(r.tool, Tool::Python);
    assert_eq!(r.payload, serde_json::json!({}));
    assert_eq!(r.timeout_s, None);
    assert_eq!(r.parent_call_id, None);
}

#[test]
fn tool_request_golden_roundtrip() {
    let raw = golden("env_tool_request.default.json");
    let req: ToolRequest = serde_json::from_str(&raw).expect("request golden");
    assert_eq!(req.schema_id, SCHEMA_TOOL_REQUEST);
    assert_eq!(req.schema_version, 1);
    assert_eq!(req.call_id, "call-0001");
    assert_eq!(req.env_snapshot_id, "snap-0001");
    assert_eq!(req.tool, Tool::Python);
    assert_eq!(req.payload["code"], "print(1+1)");
    assert_eq!(req.timeout_s, Some(5));
    assert_eq!(req.parent_call_id, None);
    let again: ToolRequest =
        serde_json::from_value(serde_json::to_value(&req).unwrap()).expect("roundtrip");
    assert_eq!(req, again);
}

#[test]
fn tool_response_golden_roundtrip() {
    let raw = golden("env_tool_response.default.json");
    let resp: ToolResponse = serde_json::from_str(&raw).expect("response golden");
    assert_eq!(resp.schema_id, SCHEMA_TOOL_RESPONSE);
    assert_eq!(resp.schema_version, 1);
    assert_eq!(resp.call_id, "call-0001");
    assert!(resp.ok);
    assert_eq!(resp.exit_code, Some(0));
    assert_eq!(resp.stdout_artifact.bytes, 2);
    assert_eq!(resp.stderr_artifact.bytes, 0);
    assert!(!resp.truncated);
    assert_eq!(resp.duration_ms, 12);
    assert_eq!(resp.snapshot_id_after, "snap-0001");
    assert_eq!(resp.error, None);
    let again: ToolResponse =
        serde_json::from_value(serde_json::to_value(&resp).unwrap()).expect("roundtrip");
    assert_eq!(resp, again);
}

#[test]
fn image_new_starts_empty() {
    let img = Image::new("img-1");
    assert_eq!(img.id, ImageId("img-1".into()));
    assert!(img.env.is_empty());
    assert!(img.agent_files.is_empty());
    assert!(img.hidden_tests.is_empty());
    assert!(img.hidden_tests_hash.is_empty());
}

#[test]
fn fork_result_all_under_budget() {
    let ok = ForkResult {
        sandboxes: vec![SandboxId("a".into()), SandboxId("b".into())],
        durations_ms: vec![0, FORK_BUDGET_MS],
    };
    assert!(ok.all_under_budget());
    let slow = ForkResult {
        sandboxes: vec![SandboxId("a".into())],
        durations_ms: vec![FORK_BUDGET_MS + 1],
    };
    assert!(!slow.all_under_budget());
}

#[test]
fn tool_enum_serde_snake_case() {
    for (t, name) in [
        (Tool::Shell, "shell"),
        (Tool::Editor, "editor"),
        (Tool::Python, "python"),
        (Tool::Browser, "browser"),
        (Tool::Lean, "lean"),
        (Tool::Notes, "notes"),
        (Tool::Subagent, "subagent"),
    ] {
        let v = serde_json::to_value(t).unwrap();
        assert_eq!(v, serde_json::Value::String(name.into()));
        let back: Tool = serde_json::from_value(v).unwrap();
        assert_eq!(back, t);
    }
}

#[test]
fn artifact_and_id_serde() {
    let a = Artifact {
        content_hash: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into(),
        bytes: 0,
        path: None,
        media_type: Some("text/plain".into()),
    };
    let v = serde_json::to_value(&a).unwrap();
    let b: Artifact = serde_json::from_value(v).unwrap();
    assert_eq!(a, b);
    let id = CallId("c1".into());
    assert_eq!(
        serde_json::from_value::<CallId>(serde_json::to_value(&id).unwrap()).unwrap(),
        id
    );
    let sid = SnapshotId("s1".into());
    assert_eq!(
        serde_json::from_value::<SnapshotId>(serde_json::to_value(&sid).unwrap()).unwrap(),
        sid
    );
}

#[test]
fn snapshot_id_getter_roundtrip() {
    let sid = SnapshotId("s1".into());
    assert_eq!(sid.0, "s1");
    let sb = SandboxId("sb1".into());
    assert_eq!(sb.0, "sb1");
}
