//! Group: crash / replay. Small kill counts (not the H11 1,000). Linux only.

#![cfg(unix)]

mod common;
mod reference;

use common::{
    append_input, assert_torn_does_not_invent, events_path, f1_append, fresh_log_dir,
};
use prometheus_log::EventLog;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

const CHILD_KIND: &str = "PROMETHEUS_LOG_H2_CRASH_CHILD";
const CHILD_DIR: &str = "PROMETHEUS_LOG_H2_CRASH_DIR";

fn child_kind() -> Option<String> {
    std::env::var(CHILD_KIND).ok()
}

fn child_dir() -> PathBuf {
    PathBuf::from(std::env::var(CHILD_DIR).expect("child dir"))
}

fn exit_hard() -> ! {
    extern "C" {
        fn _exit(code: i32) -> !;
    }
    unsafe { _exit(0) }
}

fn spawn_child(test_name: &str, kind: &str, dir: &Path) -> ExitStatus {
    let exe = std::env::current_exe().expect("current_exe");
    Command::new(exe)
        .args(["--exact", "--nocapture", test_name])
        .env(CHILD_KIND, kind)
        .env(CHILD_DIR, dir)
        .status()
        .expect("spawn crash child")
}

fn child_create_append_n(n: usize) {
    let dir = child_dir();
    let mut log = EventLog::create(&dir).expect("child create");
    for i in 0..n {
        log.append(append_input("crash", json!({"i": i})))
            .expect("child append");
    }
    std::fs::write(dir.join("appended.ok"), format!("{n}")).expect("sentinel");
}

#[test]
fn crash_after_append_exit_preserves_record() {
    if child_kind().as_deref() == Some("after_append_exit") {
        child_create_append_n(1);
        exit_hard();
    }
    let (_parent, dir) = fresh_log_dir();
    let status = spawn_child(
        "crash_after_append_exit_preserves_record",
        "after_append_exit",
        &dir,
    );
    assert!(status.success(), "child _exit(0) after append, got {status}");
    assert!(dir.join("appended.ok").is_file(), "append returned");
    let log = EventLog::open(&dir).expect("open after _exit");
    assert_eq!(log.len(), 1);
    assert_eq!(log.get(0).expect("get").seq, 0);
    log.verify().expect("verify");
}

#[test]
fn crash_after_append_abort_preserves_record() {
    if child_kind().as_deref() == Some("after_append_abort") {
        child_create_append_n(1);
        std::process::abort();
    }
    let (_parent, dir) = fresh_log_dir();
    let _status = spawn_child(
        "crash_after_append_abort_preserves_record",
        "after_append_abort",
        &dir,
    );
    assert!(dir.join("appended.ok").is_file(), "append returned before abort");
    let log = EventLog::open(&dir).expect("open after abort");
    assert_eq!(log.len(), 1);
    assert_eq!(log.get(0).expect("get").payload, json!({"i": 0}));
    log.verify().expect("verify");
}

#[test]
fn crash_after_two_appends_preserves_both() {
    if child_kind().as_deref() == Some("after_two") {
        child_create_append_n(2);
        exit_hard();
    }
    let (_parent, dir) = fresh_log_dir();
    let status = spawn_child(
        "crash_after_two_appends_preserves_both",
        "after_two",
        &dir,
    );
    assert!(status.success(), "child _exit(0), got {status}");
    let log = EventLog::open(&dir).expect("open after two");
    assert_eq!(log.len(), 2);
    assert_eq!(log.get(0).unwrap().seq, 0);
    assert_eq!(log.get(1).unwrap().seq, 1);
    assert_eq!(log.get(1).unwrap().prev_hash, log.get(0).unwrap().hash);
    log.verify().expect("verify");
}

#[test]
fn partial_write_does_not_lose_committed_or_claim_later_seq() {
    let (_parent, dir) = fresh_log_dir();
    {
        let mut log = EventLog::create(&dir).expect("create");
        log.append(f1_append()).expect("commit 0");
        log.append(append_input("second", json!({"k": 2})))
            .expect("commit 1");
    }
    let mut body = std::fs::read(events_path(&dir)).expect("read committed");
    body.extend_from_slice(b"{\"schema_id\":\"prometheus.event_log\",\"seq\":2");
    std::fs::write(events_path(&dir), body).expect("partial line");
    assert_torn_does_not_invent(&dir, 2);
}

#[test]
fn child_partial_line_does_not_claim_later_seq() {
    if child_kind().as_deref() == Some("partial_line") {
        let dir = child_dir();
        let mut log = EventLog::create(&dir).expect("child create");
        log.append(f1_append()).expect("commit");
        drop(log);
        let mut body = std::fs::read(events_path(&dir)).expect("read");
        body.extend_from_slice(b"{\"seq\":1");
        std::fs::write(events_path(&dir), &body).expect("torn");
        std::fs::write(dir.join("appended.ok"), b"1").expect("sentinel");
        exit_hard();
    }
    let (_parent, dir) = fresh_log_dir();
    let status = spawn_child(
        "child_partial_line_does_not_claim_later_seq",
        "partial_line",
        &dir,
    );
    assert!(status.success(), "child _exit, got {status}");
    assert_torn_does_not_invent(&dir, 1);
}

#[test]
fn reference_chain_after_crash_matches_production() {
    if child_kind().as_deref() == Some("ref_check") {
        child_create_append_n(3);
        exit_hard();
    }
    let (_parent, dir) = fresh_log_dir();
    let status = spawn_child(
        "reference_chain_after_crash_matches_production",
        "ref_check",
        &dir,
    );
    assert!(status.success());
    let log = EventLog::open(&dir).expect("open");
    let events: Vec<_> = (0..3u64)
        .map(|i| log.get(i).expect("get").clone())
        .collect();
    reference::verify_chain(&events).expect("reference chain");
    let disk = reference::replay_jsonl(&events_path(&dir)).expect("replay jsonl");
    assert_eq!(disk.len(), 3);
}
