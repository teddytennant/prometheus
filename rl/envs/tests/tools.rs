//! Group: F1 tools (shell, editor, python, lean, notes) and timeout vs injected NowMs.

mod common;
mod reference;

use std::time::{Duration, Instant};

use common::{
    editor_read, editor_write, lean, notes_write, python, sample_image, shell,
    src_does_not_import_tests, README_BODY,
};
use prometheus_envs::{Error, Pool, Tool, ToolRequest};
use reference::{artifact_for, RefPool};

fn pair() -> (
    Pool<prometheus_envs::InProcess>,
    RefPool,
    prometheus_envs::SandboxId,
    prometheus_envs::SandboxId,
    prometheus_envs::SnapshotId,
) {
    src_does_not_import_tests();
    let cfg = common::cpu_cfg();
    let mut prod = Pool::in_process(cfg.clone());
    let mut refer = RefPool::new(cfg);
    let img = sample_image("img-tools");
    let pid = prod.register_image(img.clone()).unwrap();
    let rid = refer.register_image(img).unwrap();
    let ps = prod.snapshot_from_image(&pid, 0).unwrap();
    let rs = refer.snapshot_from_image(&rid, 0).unwrap();
    let pg = prod.fork_group(&ps, 1, 0).unwrap();
    let rg = refer.fork_group(&rs, 1, 0).unwrap();
    (
        prod,
        refer,
        pg.sandboxes[0].clone(),
        rg.sandboxes[0].clone(),
        ps,
    )
}

fn assert_stdout_eq(
    prod: &prometheus_envs::ToolResponse,
    refer: &prometheus_envs::ToolResponse,
    bytes: &[u8],
) {
    let art = artifact_for(bytes);
    assert!(prod.ok, "prod not ok: {:?}", prod.error);
    assert!(refer.ok);
    assert_eq!(prod.call_id, refer.call_id);
    assert_eq!(prod.stdout_artifact.content_hash, art.content_hash);
    assert_eq!(refer.stdout_artifact.content_hash, art.content_hash);
    assert_eq!(prod.stdout_artifact.bytes, art.bytes);
    assert!(!prod.truncated);
    assert_eq!(prod.schema_id, prometheus_envs::SCHEMA_TOOL_RESPONSE);
}

#[test]
fn python_print_literals_and_arithmetic() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let req = python("py-add", &ps.0, "print(1+1)");
    let pr = prod.call(&a, req.clone(), 0).unwrap();
    let rr = refer.call(&ra, req, 0).unwrap();
    assert_stdout_eq(&pr, &rr, b"2\n");
}

#[test]
fn python_print_string_literal() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let req = python("py-str", &ps.0, "print('hi')");
    let pr = prod.call(&a, req.clone(), 0).unwrap();
    let rr = refer.call(&ra, req, 0).unwrap();
    assert_stdout_eq(&pr, &rr, b"hi\n");
}

#[test]
fn python_reads_agent_files() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let req = python(
        "py-open",
        &ps.0,
        "print(open('/workspace/readme.txt').read())",
    );
    let pr = prod.call(&a, req.clone(), 0).unwrap();
    let rr = refer.call(&ra, req, 0).unwrap();
    let mut expected = README_BODY.to_vec();
    if !expected.ends_with(b"\n") {
        expected.push(b'\n');
    }
    assert_stdout_eq(&pr, &rr, &expected);
}

#[test]
fn python_does_not_read_grader_files() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let req = python("py-hidden", &ps.0, "print(open('/grader/test.py').read())");
    let pr = prod.call(&a, req.clone(), 0).unwrap();
    let rr = refer.call(&ra, req, 0).unwrap();
    assert!(!pr.ok && !rr.ok);
}

#[test]
fn shell_echo_and_cat() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let echo = shell("sh-echo", &ps.0, "echo hello");
    let pr = prod.call(&a, echo.clone(), 0).unwrap();
    let rr = refer.call(&ra, echo, 0).unwrap();
    assert_stdout_eq(&pr, &rr, b"hello\n");
    let cat = shell("sh-cat", &ps.0, "cat /workspace/readme.txt");
    let pr = prod.call(&a, cat.clone(), 0).unwrap();
    let rr = refer.call(&ra, cat, 0).unwrap();
    assert_stdout_eq(&pr, &rr, README_BODY);
}

#[test]
fn editor_read_write_roundtrip() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let w = editor_write("ed-w", &ps.0, "/workspace/f.txt", "abc");
    assert!(prod.call(&a, w.clone(), 0).unwrap().ok);
    assert!(refer.call(&ra, w, 0).unwrap().ok);
    let r = editor_read("ed-r", &ps.0, "/workspace/f.txt");
    let pr = prod.call(&a, r.clone(), 1).unwrap();
    let rr = refer.call(&ra, r, 1).unwrap();
    assert_stdout_eq(&pr, &rr, b"abc");
}

#[test]
fn notes_write_to_agent_notes_path() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let w = notes_write("n-w", &ps.0, "/notes/memo.txt", "remember");
    assert!(prod.call(&a, w.clone(), 0).unwrap().ok);
    assert!(refer.call(&ra, w, 0).unwrap().ok);
    assert_eq!(
        prod.agent_view(&a)
            .unwrap()
            .get("/notes/memo.txt")
            .map(Vec::as_slice),
        Some(&b"remember"[..])
    );
    assert_eq!(
        prod.agent_view(&a).unwrap().get("/notes/memo.txt"),
        refer.agent_view(&ra).unwrap().get("/notes/memo.txt")
    );
}

#[test]
fn lean_accepts_complete_rejects_sorry() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let ok = lean("ln-ok", &ps.0, "theorem t : True := trivial");
    let pr = prod.call(&a, ok.clone(), 0).unwrap();
    let rr = refer.call(&ra, ok, 0).unwrap();
    assert_stdout_eq(&pr, &rr, b"ok\n");
    let bad = lean("ln-bad", &ps.0, "theorem t : True := sorry");
    let pr = prod.call(&a, bad.clone(), 0).unwrap();
    let rr = refer.call(&ra, bad, 0).unwrap();
    assert!(!pr.ok && !rr.ok);
}

#[test]
fn timeout_uses_request_timeout_s_without_wall_sleep() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let mut req = shell("sleep-req", &ps.0, "sleep 10");
    req.timeout_s = Some(1);
    let t0 = Instant::now();
    assert_eq!(prod.call(&a, req.clone(), 0), Err(Error::Timeout));
    assert_eq!(refer.call(&ra, req, 0), Err(Error::Timeout));
    assert!(
        t0.elapsed() < Duration::from_secs(2),
        "timeout must use injected NowMs, not wall sleep"
    );
}

#[test]
fn timeout_uses_default_timeout_s_when_request_omits_it() {
    src_does_not_import_tests();
    let cfg = common::cfg_timeout(2);
    let mut prod = Pool::in_process(cfg.clone());
    let mut refer = RefPool::new(cfg);
    let img = sample_image("img-to");
    let pid = prod.register_image(img.clone()).unwrap();
    let rid = refer.register_image(img).unwrap();
    let ps = prod.snapshot_from_image(&pid, 0).unwrap();
    let rs = refer.snapshot_from_image(&rid, 0).unwrap();
    let a = prod.fork_group(&ps, 1, 0).unwrap().sandboxes[0].clone();
    let ra = refer.fork_group(&rs, 1, 0).unwrap().sandboxes[0].clone();
    let req = shell("sleep-def", &ps.0, "sleep 10");
    assert_eq!(req.timeout_s, None);
    assert_eq!(prod.call(&a, req.clone(), 0), Err(Error::Timeout));
    assert_eq!(refer.call(&ra, req, 0), Err(Error::Timeout));
}

#[test]
fn unknown_shell_command_fails_closed() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let req = shell("unk", &ps.0, "rm -rf /");
    let pr = prod.call(&a, req.clone(), 0).unwrap();
    let rr = refer.call(&ra, req, 0).unwrap();
    assert!(!pr.ok && !rr.ok);
}

#[test]
fn all_seven_tools_are_dispatchable() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    for tool in [
        Tool::Shell,
        Tool::Editor,
        Tool::Python,
        Tool::Browser,
        Tool::Lean,
        Tool::Notes,
        Tool::Subagent,
    ] {
        let mut r = ToolRequest::new(format!("t-{tool:?}"), ps.0.clone(), tool);
        r.payload = match tool {
            Tool::Shell => serde_json::json!({"command": "echo x"}),
            Tool::Python => serde_json::json!({"code": "print(1)"}),
            Tool::Editor | Tool::Notes => {
                serde_json::json!({"op": "read", "path": "/workspace/readme.txt"})
            }
            Tool::Browser => serde_json::json!({"url": "offline://missing"}),
            Tool::Lean => serde_json::json!({"code": "theorem t : True := trivial"}),
            Tool::Subagent => serde_json::json!({"task": "child"}),
        };
        let pr = prod.call(&a, r.clone(), 0);
        let rr = refer.call(&ra, r, 0);
        assert!(pr.is_ok(), "{tool:?} prod {pr:?}");
        assert!(rr.is_ok(), "{tool:?} ref {rr:?}");
    }
}
