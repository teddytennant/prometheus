//! Group: crash/replay on a handful of `_exit` / SIGKILL, not gate D0's 1,000.
//!
//! Linux-only. Each test hits `Queue::create`/`open` on the parent path so a
//! stub cannot pass. Child work uses `_exit(0)` or SIGKILL so Drop/fsync-on-drop
//! cannot hide a missing durability barrier.

mod common;
mod reference;

use common::{assert_completed, assert_leased, assert_queued, fresh_queue_dir, short_config};
use prometheus_leases::{Queue, QueueConfig, TaskId, WorkItem, WorkerId};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::time::{Duration, Instant};

const CHILD_KIND: &str = "PROMETHEUS_LEASES_H3_CRASH_CHILD";
const CHILD_DIR: &str = "PROMETHEUS_LEASES_H3_CRASH_DIR";

fn child_mode() -> Option<String> {
    std::env::var(CHILD_KIND).ok()
}

fn cfg() -> QueueConfig {
    short_config()
}

fn work_item() -> WorkItem {
    WorkItem {
        task_id: TaskId("crash1".into()),
        payload: serde_json::json!({"crash": true}),
    }
}

fn worker() -> WorkerId {
    WorkerId("crash-w".into())
}

#[cfg(unix)]
extern "C" {
    fn _exit(code: i32) -> !;
}

#[cfg(unix)]
fn exit_hard() -> ! {
    unsafe { _exit(0) }
}

fn child_command(kind: &str, dir: &Path) -> Command {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = Command::new(exe);
    cmd.env(CHILD_KIND, kind)
        .env(CHILD_DIR, dir)
        .env("RUST_TEST_THREADS", "1")
        .arg("--test-threads=1");
    if let Some(name) = std::thread::current().name() {
        cmd.arg(name).arg("--exact");
    }
    cmd
}

fn spawn_child(kind: &str, dir: &Path) -> ExitStatus {
    child_command(kind, dir)
        .status()
        .unwrap_or_else(|e| panic!("spawn child {kind}: {e}"))
}

fn spawn_child_killable(kind: &str, dir: &Path) -> std::process::Child {
    child_command(kind, dir)
        .spawn()
        .unwrap_or_else(|e| panic!("spawn killable {kind}: {e}"))
}

#[cfg(unix)]
fn run_child(kind: &str) {
    let dir = PathBuf::from(std::env::var(CHILD_DIR).expect("CHILD_DIR"));
    match kind {
        "after_enqueue" => {
            let mut q = Queue::create(&dir, cfg()).expect("child create");
            q.enqueue(work_item(), 0).expect("child enqueue");
            exit_hard();
        }
        "after_claim" => {
            let mut q = Queue::create(&dir, cfg()).expect("child create");
            q.enqueue(work_item(), 0).expect("child enqueue");
            q.claim(&worker(), 0).expect("child claim");
            exit_hard();
        }
        "after_complete" => {
            let mut q = Queue::create(&dir, cfg()).expect("child create");
            q.enqueue(work_item(), 0).expect("child enqueue");
            q.claim(&worker(), 0).expect("child claim");
            q.complete(&TaskId("crash1".into()), &worker(), 1, b"crash-out", 1)
                .expect("child complete");
            exit_hard();
        }
        "kill9_after_complete" => {
            let mut q = Queue::create(&dir, cfg()).expect("child create");
            q.enqueue(work_item(), 0).expect("child enqueue");
            q.claim(&worker(), 0).expect("child claim");
            q.complete(&TaskId("crash1".into()), &worker(), 1, b"kill9-out", 1)
                .expect("child complete");
            let sentinel = dir.parent().expect("parent").join("complete.ok");
            std::fs::write(&sentinel, b"ok").expect("sentinel");
            loop {
                std::thread::sleep(Duration::from_secs(30));
            }
        }
        other => panic!("unknown child kind {other}"),
    }
}

#[cfg(unix)]
#[test]
fn crash_after_enqueue_is_durable() {
    if let Some(kind) = child_mode() {
        run_child(&kind);
        return;
    }
    let (_parent, dir) = fresh_queue_dir();
    let status = spawn_child("after_enqueue", &dir);
    assert!(status.success(), "child after_enqueue: {status}");
    let q = Queue::open(&dir, cfg()).expect("parent open");
    assert_queued(q.get(&TaskId("crash1".into())).expect("get"), "crash1", 0);
}

#[cfg(unix)]
#[test]
fn crash_after_claim_is_durable() {
    if let Some(kind) = child_mode() {
        run_child(&kind);
        return;
    }
    let (_parent, dir) = fresh_queue_dir();
    let status = spawn_child("after_claim", &dir);
    assert!(status.success(), "child after_claim: {status}");
    let q = Queue::open(&dir, cfg()).expect("parent open");
    assert_leased(
        q.get(&TaskId("crash1".into())).expect("get"),
        "crash1",
        "crash-w",
        1,
        cfg().lease_ttl_ms(),
    );
}

#[cfg(unix)]
#[test]
fn crash_after_complete_is_durable_with_output() {
    if let Some(kind) = child_mode() {
        run_child(&kind);
        return;
    }
    let (_parent, dir) = fresh_queue_dir();
    let status = spawn_child("after_complete", &dir);
    assert!(status.success(), "child after_complete: {status}");
    let q = Queue::open(&dir, cfg()).expect("parent open");
    let hash = reference::output_hash(b"crash-out");
    assert_completed(
        q.get(&TaskId("crash1".into())).expect("get"),
        "crash1",
        1,
        &hash,
    );
    assert_eq!(
        q.get_output(&TaskId("crash1".into()), 1)
            .expect("get_output")
            .as_deref(),
        Some(&b"crash-out"[..])
    );
}

#[cfg(unix)]
#[test]
fn kill9_after_complete_is_durable() {
    if let Some(kind) = child_mode() {
        run_child(&kind);
        return;
    }
    let (parent, dir) = fresh_queue_dir();
    let sentinel = parent.path().join("complete.ok");
    let mut child = spawn_child_killable("kill9_after_complete", &dir);
    let start = Instant::now();
    while !sentinel.exists() {
        if start.elapsed() > Duration::from_secs(5) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("child did not write sentinel (stub or hung complete)");
        }
        if let Some(status) = child.try_wait().expect("try_wait") {
            panic!("child exited before sentinel: {status}");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    child.kill().expect("SIGKILL");
    let _ = child.wait();
    let q = Queue::open(&dir, cfg()).expect("parent open after kill -9");
    let hash = reference::output_hash(b"kill9-out");
    assert_completed(
        q.get(&TaskId("crash1".into())).expect("get"),
        "crash1",
        1,
        &hash,
    );
    assert_eq!(
        q.get_output(&TaskId("crash1".into()), 1)
            .expect("get_output")
            .as_deref(),
        Some(&b"kill9-out"[..])
    );
}

#[cfg(not(unix))]
#[test]
fn crash_tests_require_unix_and_still_hit_the_stub() {
    let (_parent, dir) = fresh_queue_dir();
    let _ = Queue::create(&dir, cfg());
    panic!("H3 crash tests are unix-only; this path still calls create so a stub fails");
}
