//! Group: F1 record shape, schema_version, optional fields.

mod common;
mod reference;

use common::{
    append_input, assert_err, assert_event_fields, assert_f1_field_names, f1_append, fresh_log_dir,
    write_jsonl_lines,
};
use prometheus_log::{EventLog, SCHEMA_ID, SCHEMA_VERSION};
use serde_json::json;

#[test]
fn appended_event_matches_f1_shape() {
    let (_parent, dir) = fresh_log_dir();
    let mut log = EventLog::create(&dir).expect("create");
    let ev = log.append(f1_append()).expect("append");
    assert_event_fields(&ev);
    assert_f1_field_names(&ev);
    assert_eq!(ev.schema_id, SCHEMA_ID);
    assert_eq!(ev.schema_version, SCHEMA_VERSION);
    assert_eq!(ev.schema_version, 1);
}

#[test]
fn v1_records_load() {
    let (_parent, dir) = fresh_log_dir();
    let ev = reference::make_event(
        0,
        reference::GENESIS_PREV_HASH,
        reference::F1_TIMESTAMP,
        reference::F1_EVENT_TYPE,
        None,
        None,
        Some("local".into()),
        json!({"cycle": 1}),
    );
    assert_eq!(ev.schema_version, 1);
    write_jsonl_lines(&dir, &[reference::event_to_jsonl_line(&ev)]);
    let log = EventLog::open(&dir).expect("open v1");
    assert_eq!(log.len(), 1);
    let got = log.get(0).expect("get");
    assert_eq!(got.schema_version, 1);
    assert_eq!(got.hash, reference::F1_GENESIS_EVENT_HASH);
    log.verify().expect("verify v1");
}

#[test]
fn unknown_schema_version_is_rejected() {
    let (_parent, dir) = fresh_log_dir();
    let mut ev = reference::make_event(
        0,
        reference::GENESIS_PREV_HASH,
        reference::F1_TIMESTAMP,
        "x",
        None,
        None,
        None,
        json!({}),
    );
    ev.schema_version = 99;
    write_jsonl_lines(&dir, &[reference::event_to_jsonl_line(&ev)]);
    assert_err(EventLog::open(&dir), "unknown schema_version");
}

#[test]
fn wrong_schema_id_is_rejected() {
    let (_parent, dir) = fresh_log_dir();
    let mut ev = reference::make_event(
        0,
        reference::GENESIS_PREV_HASH,
        reference::F1_TIMESTAMP,
        "x",
        None,
        None,
        None,
        json!({}),
    );
    ev.schema_id = "prometheus.not_the_log".into();
    write_jsonl_lines(&dir, &[reference::event_to_jsonl_line(&ev)]);
    assert_err(EventLog::open(&dir), "wrong schema_id");
}

#[test]
fn optional_fields_null() {
    let (_parent, dir) = fresh_log_dir();
    let mut log = EventLog::create(&dir).expect("create");
    let mut inp = append_input("opt", json!({"a": 1}));
    inp.task_id = None;
    inp.attempt = None;
    inp.node_id = None;
    let ev = log.append(inp).expect("append");
    assert_eq!(ev.task_id, None);
    assert_eq!(ev.attempt, None);
    assert_eq!(ev.node_id, None);
    drop(log);
    let log = EventLog::open(&dir).expect("open");
    let got = log.get(0).expect("get");
    assert_eq!(got.task_id, None);
    assert_eq!(got.attempt, None);
    assert_eq!(got.node_id, None);
}

#[test]
fn optional_fields_populated() {
    let (_parent, dir) = fresh_log_dir();
    let mut log = EventLog::create(&dir).expect("create");
    let mut inp = append_input("opt", json!({"a": 1}));
    inp.task_id = Some("task-9".into());
    inp.attempt = Some(3);
    inp.node_id = Some("node-a".into());
    let ev = log.append(inp).expect("append");
    assert_eq!(ev.task_id.as_deref(), Some("task-9"));
    assert_eq!(ev.attempt, Some(3));
    assert_eq!(ev.node_id.as_deref(), Some("node-a"));
    drop(log);
    let log = EventLog::open(&dir).expect("open");
    let got = log.get(0).expect("get");
    assert_eq!(got.task_id.as_deref(), Some("task-9"));
    assert_eq!(got.attempt, Some(3));
    assert_eq!(got.node_id.as_deref(), Some("node-a"));
}
