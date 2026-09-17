//! Group: fault injection. Torn writes, hash flips, seq gaps, payload mutation.

mod common;
mod reference;

use common::{
    append_input, assert_err, assert_torn_does_not_invent, events_path, f1_append, fresh_log_dir,
    write_jsonl_lines,
};
use prometheus_log::{EventLog, GENESIS_PREV_HASH};
use serde_json::json;
use std::fs;

fn two_ref_events() -> (prometheus_log::Event, prometheus_log::Event) {
    let e0 = reference::make_event(
        0,
        reference::GENESIS_PREV_HASH,
        reference::F1_TIMESTAMP,
        "a",
        None,
        None,
        Some("local".into()),
        json!({"n": 0}),
    );
    let e1 = reference::make_event(
        1,
        &e0.hash,
        reference::F1_TIMESTAMP,
        "b",
        None,
        None,
        Some("local".into()),
        json!({"n": 1}),
    );
    (e0, e1)
}

#[test]
fn open_valid_reference_jsonl() {
    let (_parent, dir) = fresh_log_dir();
    let (e0, e1) = two_ref_events();
    write_jsonl_lines(
        &dir,
        &[
            reference::event_to_jsonl_line(&e0),
            reference::event_to_jsonl_line(&e1),
        ],
    );
    reference::replay_jsonl(&events_path(&dir)).expect("reference replay");
    let log = EventLog::open(&dir).expect("open valid jsonl");
    assert_eq!(log.len(), 2);
    assert_eq!(log.get(0).unwrap().hash, e0.hash);
    assert_eq!(log.get(1).unwrap().hash, e1.hash);
    assert_eq!(log.get(0).unwrap().prev_hash, GENESIS_PREV_HASH);
    log.verify().expect("verify");
}

#[test]
fn torn_last_line_does_not_invent_a_record() {
    let (_parent, dir) = fresh_log_dir();
    let (e0, e1) = two_ref_events();
    let mut body = String::new();
    body.push_str(&reference::event_to_jsonl_line(&e0));
    body.push('\n');
    body.push_str(&reference::event_to_jsonl_line(&e1));
    body.push('\n');
    body.push_str(r#"{"schema_id":"prometheus.event_log","seq":2"#);
    fs::create_dir_all(&dir).expect("mkdir");
    fs::write(events_path(&dir), body).expect("write torn jsonl");
    match reference::parse_jsonl_bytes(&fs::read(events_path(&dir)).unwrap()) {
        Err(reference::ReplayError::TornLastLine) => {}
        other => panic!("reference must detect torn last line, got {other:?}"),
    }
    assert_torn_does_not_invent(&dir, 2);
}

#[test]
fn torn_first_line_does_not_invent_seq_zero() {
    let (_parent, dir) = fresh_log_dir();
    fs::create_dir_all(&dir).expect("mkdir");
    fs::write(events_path(&dir), b"{\"seq\":0,\"hash\":").expect("torn");
    assert_torn_does_not_invent(&dir, 0);
}

#[test]
fn flipped_byte_in_hash_field_fails_open() {
    let (_parent, dir) = fresh_log_dir();
    let (e0, e1) = two_ref_events();
    let mut e1_bad = e1.clone();
    let mut chars: Vec<char> = e1_bad.hash.chars().collect();
    chars[0] = if chars[0] == '0' { '1' } else { '0' };
    e1_bad.hash = chars.into_iter().collect();
    write_jsonl_lines(
        &dir,
        &[
            reference::event_to_jsonl_line(&e0),
            reference::event_to_jsonl_line(&e1_bad),
        ],
    );
    assert_err(EventLog::open(&dir), "flipped hash");
}

#[test]
fn flipped_byte_in_payload_hash_field_fails_open() {
    let (_parent, dir) = fresh_log_dir();
    let e0 = reference::make_event(
        0,
        reference::GENESIS_PREV_HASH,
        reference::F1_TIMESTAMP,
        "a",
        None,
        None,
        None,
        json!({"n": 0}),
    );
    let mut bad = e0.clone();
    let mut chars: Vec<char> = bad.payload_hash.chars().collect();
    chars[3] = if chars[3] == 'a' { 'b' } else { 'a' };
    bad.payload_hash = chars.into_iter().collect();
    write_jsonl_lines(&dir, &[reference::event_to_jsonl_line(&bad)]);
    assert_err(EventLog::open(&dir), "flipped payload_hash");
}

#[test]
fn seq_gap_fails_open() {
    let (_parent, dir) = fresh_log_dir();
    let e0 = reference::make_event(
        0,
        reference::GENESIS_PREV_HASH,
        reference::F1_TIMESTAMP,
        "a",
        None,
        None,
        None,
        json!({"n": 0}),
    );
    let e2 = reference::make_event(
        2,
        &e0.hash,
        reference::F1_TIMESTAMP,
        "c",
        None,
        None,
        None,
        json!({"n": 2}),
    );
    write_jsonl_lines(
        &dir,
        &[
            reference::event_to_jsonl_line(&e0),
            reference::event_to_jsonl_line(&e2),
        ],
    );
    assert_err(EventLog::open(&dir), "seq gap");
}

#[test]
fn mutated_payload_with_old_payload_hash_fails_open() {
    let (_parent, dir) = fresh_log_dir();
    let e0 = reference::make_event(
        0,
        reference::GENESIS_PREV_HASH,
        reference::F1_TIMESTAMP,
        "a",
        None,
        None,
        None,
        json!({"cycle": 1}),
    );
    let mut bad = e0.clone();
    bad.payload = json!({"cycle": 2});
    write_jsonl_lines(&dir, &[reference::event_to_jsonl_line(&bad)]);
    assert_ne!(
        reference::payload_hash(&bad.payload),
        bad.payload_hash,
        "stored payload_hash must be the old digest"
    );
    assert_err(EventLog::open(&dir), "mutated payload");
}

#[test]
fn broken_prev_hash_fails_open() {
    let (_parent, dir) = fresh_log_dir();
    let (e0, e1) = two_ref_events();
    let mut bad = e1.clone();
    bad.prev_hash = "c".repeat(64);
    bad.hash = reference::event_hash(
        bad.seq,
        &bad.prev_hash,
        &bad.payload_hash,
        &bad.timestamp,
        &bad.event_type,
    );
    write_jsonl_lines(
        &dir,
        &[
            reference::event_to_jsonl_line(&e0),
            reference::event_to_jsonl_line(&bad),
        ],
    );
    assert_err(EventLog::open(&dir), "broken prev_hash link");
}

#[test]
fn append_then_corrupt_disk_open_fails() {
    let (_parent, dir) = fresh_log_dir();
    {
        let mut log = EventLog::create(&dir).expect("create");
        log.append(f1_append()).expect("append");
        log.append(append_input("next", json!({"x": 1})))
            .expect("append");
    }
    let mut bytes = fs::read(events_path(&dir)).expect("read");
    if let Some(pos) = bytes.iter().position(|&b| b == b'0') {
        bytes[pos] = b'1';
    } else {
        bytes[0] ^= 0x01;
    }
    fs::write(events_path(&dir), bytes).expect("overwrite");
    assert_err(EventLog::open(&dir), "corrupt disk");
}
