//! Group: append, get, len, last, last_index, iter, verify, JSONL write.

mod common;
mod reference;

use common::{
    append_input, assert_err, assert_event_fields, events_path, f1_append, f1_payload,
    fresh_log_dir,
};
use prometheus_log::{canonical_json, event_hash, payload_hash, EventLog, GENESIS_PREV_HASH};
use serde_json::json;

fn assert_matches_reference(ev: &prometheus_log::Event, expected_prev: &str) {
    assert_event_fields(ev);
    assert_eq!(ev.prev_hash, expected_prev);
    let ph = payload_hash(&ev.payload);
    assert_eq!(ev.payload_hash, ph);
    assert_eq!(ph, reference::payload_hash(&ev.payload));
    let eh = event_hash(
        ev.seq,
        &ev.prev_hash,
        &ev.payload_hash,
        &ev.timestamp,
        &ev.event_type,
    );
    assert_eq!(ev.hash, eh);
    assert_eq!(
        eh,
        reference::event_hash(
            ev.seq,
            &ev.prev_hash,
            &ev.payload_hash,
            &ev.timestamp,
            &ev.event_type,
        )
    );
    reference::verify_event(ev, expected_prev).expect("reference verify_event");
}

#[test]
fn first_append_is_seq_zero_genesis() {
    let (_parent, dir) = fresh_log_dir();
    let mut log = EventLog::create(&dir).expect("create");
    let ev = log.append(f1_append()).expect("append");
    assert_eq!(ev.seq, 0, "H2 first seq is 0, not F2's in-memory 1");
    assert_eq!(ev.prev_hash, GENESIS_PREV_HASH);
    assert_eq!(ev.event_type, "session.start");
    assert_eq!(ev.payload, f1_payload());
    assert_eq!(ev.node_id.as_deref(), Some("local"));
    assert_eq!(ev.task_id, None);
    assert_eq!(ev.attempt, None);
    assert_eq!(ev.payload_hash, reference::F1_GENESIS_PAYLOAD_HASH);
    assert_eq!(ev.hash, reference::F1_GENESIS_EVENT_HASH);
    assert_matches_reference(&ev, GENESIS_PREV_HASH);
    assert_eq!(log.len(), 1);
    assert!(!log.is_empty());
    assert_eq!(log.last_index(), Some(0));
    assert_eq!(log.last().map(|e| e.seq), Some(0));
    let got = log.get(0).expect("get(0)");
    assert_eq!(got, &ev);
    assert!(log.get(1).is_none());
    log.verify().expect("verify");
}

#[test]
fn append_assigns_monotonic_seq_and_chain() {
    let (_parent, dir) = fresh_log_dir();
    let mut log = EventLog::create(&dir).expect("create");
    let mut prev = GENESIS_PREV_HASH.to_string();
    for i in 0..5u64 {
        let ev = log
            .append(append_input("tick", json!({"i": i})))
            .expect("append");
        assert_eq!(ev.seq, i);
        assert_matches_reference(&ev, &prev);
        prev = ev.hash.clone();
    }
    assert_eq!(log.len(), 5);
    assert_eq!(log.last_index(), Some(4));
    reference::verify_chain(&log.iter().cloned().collect::<Vec<_>>()).expect("in-memory chain");
    log.verify().expect("verify");
}

#[test]
fn get_out_of_range_is_none() {
    let (_parent, dir) = fresh_log_dir();
    let mut log = EventLog::create(&dir).expect("create");
    assert!(log.get(0).is_none());
    assert!(log.get(99).is_none());
    log.append(f1_append()).expect("append");
    assert!(log.get(0).is_some());
    assert!(log.get(1).is_none());
    assert!(log.get(u64::MAX).is_none());
}

#[test]
fn append_after_open() {
    let (_parent, dir) = fresh_log_dir();
    {
        let mut log = EventLog::create(&dir).expect("create");
        log.append(append_input("a", json!({"k": 1}))).expect("a");
    }
    let mut log = EventLog::open(&dir).expect("open");
    assert_eq!(log.len(), 1);
    let ev = log.append(append_input("b", json!({"k": 2}))).expect("b");
    assert_eq!(ev.seq, 1);
    assert_eq!(ev.prev_hash, log.get(0).unwrap().hash);
    assert_matches_reference(&ev, &log.get(0).unwrap().hash);
    assert_eq!(log.len(), 2);
}

#[test]
fn last_len_iter_match_get() {
    let (_parent, dir) = fresh_log_dir();
    let mut log = EventLog::create(&dir).expect("create");
    for i in 0..3u64 {
        log.append(append_input("n", json!({"i": i})))
            .expect("append");
    }
    assert_eq!(log.len(), 3);
    assert_eq!(log.last().unwrap().seq, 2);
    assert_eq!(log.last_index(), Some(2));
    let via_iter: Vec<_> = log.iter().map(|e| e.seq).collect();
    assert_eq!(via_iter, vec![0, 1, 2]);
    for i in 0..3u64 {
        assert_eq!(log.get(i).unwrap().seq, i);
        assert_eq!(log.get(i).unwrap(), log.iter().nth(i as usize).unwrap());
    }
}

#[test]
fn sequential_appends_are_totally_ordered() {
    let (_parent, dir) = fresh_log_dir();
    let mut log = EventLog::create(&dir).expect("create");
    let n = 8u64;
    for i in 0..n {
        log.append(append_input("seq", json!({"i": i})))
            .expect("append");
    }
    for i in 0..n {
        let ev = log.get(i).expect("get");
        assert_eq!(ev.seq, i);
        if i == 0 {
            assert_eq!(ev.prev_hash, GENESIS_PREV_HASH);
        } else {
            assert_eq!(ev.prev_hash, log.get(i - 1).unwrap().hash);
        }
        assert_matches_reference(
            ev,
            if i == 0 {
                GENESIS_PREV_HASH
            } else {
                &log.get(i - 1).unwrap().hash
            },
        );
    }
}

#[test]
fn empty_event_type_is_rejected() {
    let (_parent, dir) = fresh_log_dir();
    let mut log = EventLog::create(&dir).expect("create");
    let mut inp = f1_append();
    inp.event_type.clear();
    assert_err(log.append(inp), "empty event_type");
    assert!(log.is_empty(), "rejected append must not commit a record");
}

#[test]
fn jsonl_one_object_per_line_after_append() {
    let (_parent, dir) = fresh_log_dir();
    let mut log = EventLog::create(&dir).expect("create");
    log.append(f1_append()).expect("append");
    log.append(append_input("next", json!({"x": true})))
        .expect("append 2");
    drop(log);
    let bytes = common::read_jsonl_bytes(&dir);
    let events = reference::parse_jsonl_bytes(&bytes).expect("parse jsonl");
    assert_eq!(events.len(), 2);
    reference::verify_chain(&events).expect("disk chain");
    assert!(events_path(&dir).is_file());
    let text = String::from_utf8(bytes).expect("utf-8 jsonl");
    let lines: Vec<_> = text.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 2);
    for line in &lines {
        let v: serde_json::Value = serde_json::from_str(line).expect("json line");
        assert!(v.is_object());
    }
}

#[test]
fn returned_event_hashes_use_canonical_payload() {
    let (_parent, dir) = fresh_log_dir();
    let mut log = EventLog::create(&dir).expect("create");
    let payload = json!({"z": 1, "a": 2});
    let ev = log
        .append(append_input("k", payload.clone()))
        .expect("append");
    assert_eq!(ev.payload_hash, payload_hash(&payload));
    assert_eq!(
        canonical_json(&payload),
        reference::canonical_json_bytes(&payload)
    );
}
