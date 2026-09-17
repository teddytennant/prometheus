//! Shared fixtures and assertions for D1 `prometheus-envs` oracle tests.
//!
//! Production must never import `tests/`. GPU-only coverage lives under
//! `#[cfg(feature = "gpu")]`; Firecracker under `#[cfg(feature = "kvm")]`.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use prometheus_envs::{
    Error, Image, OfflinePage, OfflineWeb, PoolConfig, Result as EnvResult, Tool, ToolRequest,
    GRADER_ROOT,
};
use serde_json::json;

pub const WORKSPACE_README: &str = "/workspace/readme.txt";
pub const WORKSPACE_PY: &str = "/workspace/main.py";
pub const HIDDEN_TEST: &str = "/grader/test.py";
pub const README_BODY: &[u8] = b"hello";
pub const HIDDEN_BODY: &[u8] = b"assert True\n";

pub fn cpu_cfg() -> PoolConfig {
    PoolConfig::default()
}

pub fn cfg_cap(capacity: usize) -> PoolConfig {
    PoolConfig {
        capacity,
        ..PoolConfig::default()
    }
}

pub fn cfg_timeout(default_timeout_s: u32) -> PoolConfig {
    PoolConfig {
        default_timeout_s,
        ..PoolConfig::default()
    }
}

pub fn cfg_with_web(pages: BTreeMap<String, OfflinePage>, cutoff_unix_s: u64) -> PoolConfig {
    PoolConfig {
        offline_web: OfflineWeb {
            pages,
            cutoff_unix_s,
        },
        ..PoolConfig::default()
    }
}

pub fn sample_image(id: &str) -> Image {
    let mut img = Image::new(id);
    img.agent_files
        .insert(WORKSPACE_README.to_string(), README_BODY.to_vec());
    img.agent_files
        .insert(WORKSPACE_PY.to_string(), b"print(1)\n".to_vec());
    img.hidden_tests
        .insert(HIDDEN_TEST.to_string(), HIDDEN_BODY.to_vec());
    img
}

pub fn req(call: &str, snap: &str, tool: Tool, payload: serde_json::Value) -> ToolRequest {
    let mut r = ToolRequest::new(call, snap, tool);
    r.payload = payload;
    r
}

pub fn shell(call: &str, snap: &str, command: &str) -> ToolRequest {
    req(call, snap, Tool::Shell, json!({ "command": command }))
}

pub fn python(call: &str, snap: &str, code: &str) -> ToolRequest {
    req(call, snap, Tool::Python, json!({ "code": code }))
}

pub fn editor_write(call: &str, snap: &str, path: &str, content: &str) -> ToolRequest {
    req(
        call,
        snap,
        Tool::Editor,
        json!({ "op": "write", "path": path, "content": content }),
    )
}

pub fn editor_read(call: &str, snap: &str, path: &str) -> ToolRequest {
    req(
        call,
        snap,
        Tool::Editor,
        json!({ "op": "read", "path": path }),
    )
}

pub fn notes_write(call: &str, snap: &str, path: &str, content: &str) -> ToolRequest {
    req(
        call,
        snap,
        Tool::Notes,
        json!({ "op": "write", "path": path, "content": content }),
    )
}

pub fn browser(call: &str, snap: &str, url: &str) -> ToolRequest {
    req(call, snap, Tool::Browser, json!({ "url": url }))
}

pub fn lean(call: &str, snap: &str, code: &str) -> ToolRequest {
    req(call, snap, Tool::Lean, json!({ "code": code }))
}

pub fn subagent(call: &str, snap: &str, task: &str, parent: Option<&str>) -> ToolRequest {
    let mut r = req(call, snap, Tool::Subagent, json!({ "task": task }));
    r.parent_call_id = parent.map(str::to_string);
    r
}

pub fn assert_err_eq<T: std::fmt::Debug, U: std::fmt::Debug>(
    prod: EnvResult<T>,
    refer: EnvResult<U>,
) {
    match (prod, refer) {
        (Err(a), Err(b)) => assert_eq!(a, b, "prod/ref errors must match"),
        (Ok(a), Err(b)) => panic!("prod Ok({a:?}) ref Err({b:?})"),
        (Err(a), Ok(b)) => panic!("prod Err({a:?}) ref Ok({b:?})"),
        (Ok(a), Ok(b)) => panic!("both Ok, expected error: {a:?} {b:?}"),
    }
}

pub fn assert_no_grader_paths(files: &BTreeMap<String, Vec<u8>>) {
    for p in files.keys() {
        assert!(
            p != GRADER_ROOT && !p.starts_with("/grader/"),
            "agent view leaked grader path {p}"
        );
    }
}

pub fn assert_has_grader_paths(files: &BTreeMap<String, Vec<u8>>) {
    assert!(
        files.keys().any(|p| p == GRADER_ROOT || p.starts_with("/grader/")),
        "grader view missing GRADER_ROOT paths: {:?}",
        files.keys().collect::<Vec<_>>()
    );
}

pub fn is_egress_or_live(err: &Error) -> bool {
    matches!(err, Error::EgressDenied | Error::LiveInternet)
}

pub fn src_does_not_import_tests() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk_rs(&src, &mut files);
    for f in files {
        let text = fs::read_to_string(&f).unwrap_or_else(|e| panic!("read {f:?}: {e}"));
        for line in text.lines() {
            let t = line.trim();
            if t.starts_with("//") {
                continue;
            }
            assert!(
                !t.contains("tests::") && !t.contains("tests/reference") && !t.contains("crate::tests"),
                "{} must not import tests/: {t}",
                f.display()
            );
        }
    }
}

fn walk_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    for ent in fs::read_dir(dir).expect("read src") {
        let ent = ent.expect("entry");
        let p = ent.path();
        if p.is_dir() {
            walk_rs(&p, out);
        } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}
