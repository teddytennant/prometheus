//! Group: EventLog::create / open directory layout and empty-log invariants.

mod common;
mod reference;

use common::{assert_err, events_path, fresh_log_dir, EVENTS_JSONL};
use prometheus_log::{EventLog, GENESIS_PREV_HASH};
use std::fs;
use std::path::Path;

#[test]
fn create_empty_log_invariants() {
    let (_parent, dir) = fresh_log_dir();
    let log = EventLog::create(&dir).expect("create");
    assert_eq!(log.dir(), Path::new(&dir));
    assert!(log.is_empty());
    assert_eq!(log.len(), 0);
    assert_eq!(log.last_index(), None);
    assert!(log.last().is_none());
    assert!(log.get(0).is_none());
    log.verify().expect("empty log verifies");
    assert!(dir.is_dir(), "create must create the directory");
    assert!(
        events_path(&dir).is_file(),
        "create must materialize {EVENTS_JSONL} so open is not a missing-file error"
    );
    assert_eq!(GENESIS_PREV_HASH.len(), 64);
}

#[test]
fn open_after_create_replays_empty() {
    let (_parent, dir) = fresh_log_dir();
    {
        let log = EventLog::create(&dir).expect("create");
        assert!(log.is_empty());
    }
    let log = EventLog::open(&dir).expect("open empty");
    assert!(log.is_empty());
    assert_eq!(log.len(), 0);
    assert_eq!(log.last_index(), None);
    assert!(log.get(0).is_none());
    log.verify().expect("verify empty");
}

#[test]
fn create_fails_if_directory_exists() {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("log");
    fs::create_dir(&dir).expect("mkdir");
    assert_err(EventLog::create(&dir), "create existing dir");
}

#[test]
fn create_fails_if_path_is_file() {
    let parent = tempfile::tempdir().expect("tempdir");
    let path = parent.path().join("notadir");
    fs::write(&path, b"x").expect("write file");
    assert_err(EventLog::create(&path), "create on file");
}

#[test]
fn create_fails_if_parent_missing() {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("nope").join("log");
    assert_err(EventLog::create(&dir), "create missing parent");
}

#[test]
fn open_fails_if_dir_missing() {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("missing");
    assert_err(EventLog::open(&dir), "open missing dir");
}

#[test]
fn open_fails_if_events_jsonl_missing() {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("log");
    fs::create_dir(&dir).expect("mkdir");
    assert_err(EventLog::open(&dir), "open missing events.jsonl");
}

#[test]
fn open_fails_after_deleting_jsonl() {
    let (_parent, dir) = fresh_log_dir();
    let mut log = EventLog::create(&dir).expect("create");
    log.append(common::f1_append()).expect("append");
    drop(log);
    fs::remove_file(events_path(&dir)).expect("unlink jsonl");
    assert_err(EventLog::open(&dir), "open after missing file");
}

#[test]
fn open_fails_if_path_is_file() {
    let parent = tempfile::tempdir().expect("tempdir");
    let path = parent.path().join("file");
    fs::write(&path, b"{}").expect("write");
    assert_err(EventLog::open(&path), "open file");
}

#[test]
fn create_then_reference_replay_empty_file() {
    let (_parent, dir) = fresh_log_dir();
    let _log = EventLog::create(&dir).expect("create");
    let bytes = fs::read(events_path(&dir)).unwrap_or_default();
    let parsed = reference::parse_jsonl_bytes(&bytes).expect("empty jsonl");
    assert!(parsed.is_empty());
}
