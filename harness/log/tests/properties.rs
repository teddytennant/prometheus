//! Group: reopen identity, key-order invariance, unicode, empty payload, nesting.

mod common;
mod reference;

use common::{append_input, f1_append, fresh_log_dir};
use prometheus_log::{event_hash, payload_hash, EventLog, GENESIS_PREV_HASH};
use serde_json::json;

fn reopen_and_compare(dir: &std::path::Path, n: usize) {
    let opened = EventLog::open(dir).expect("reopen");
    assert_eq!(opened.len(), n);
    opened.verify().expect("verify after reopen");
    if n == 0 {
        assert!(opened.is_empty());
        assert_eq!(opened.last_index(), None);
        return;
    }
    assert_eq!(opened.last_index(), Some((n as u64) - 1));
    let events: Vec<_> = (0..n as u64)
        .map(|i| opened.get(i).expect("get").clone())
        .collect();
    reference::verify_chain(&events).expect("reopened chain");
    for (i, ev) in events.iter().enumerate() {
        assert_eq!(ev.seq, i as u64);
        assert_eq!(ev.payload_hash, payload_hash(&ev.payload));
        assert_eq!(
            ev.hash,
            event_hash(
                ev.seq,
                &ev.prev_hash,
                &ev.payload_hash,
                &ev.timestamp,
                &ev.event_type,
            )
        );
    }
}

#[test]
fn reopen_identical_chain_for_n_events() {
    for n in [0usize, 1, 5, 17] {
        let (_parent, dir) = fresh_log_dir();
        {
            let mut log = EventLog::create(&dir).expect("create");
            for i in 0..n {
                log.append(append_input("p", json!({"i": i, "n": n})))
                    .expect("append");
            }
            assert_eq!(log.len(), n);
            log.verify().expect("verify before drop");
        }
        reopen_and_compare(&dir, n);
    }
}

#[test]
fn payload_key_order_does_not_change_payload_hash() {
    let (_parent, dir) = fresh_log_dir();
    let mut log = EventLog::create(&dir).expect("create");
    let a = json!({"b": 1, "a": 2});
    let b = json!({"a": 2, "b": 1});
    assert_eq!(payload_hash(&a), payload_hash(&b));
    let ea = log.append(append_input("k", a)).expect("a");
    let eb = log.append(append_input("k", b)).expect("b");
    assert_eq!(ea.payload_hash, eb.payload_hash);
    assert_eq!(
        ea.payload_hash,
        reference::payload_hash(&json!({"a": 2, "b": 1}))
    );
    assert_ne!(ea.hash, eb.hash, "seq differs so event_hash differs");
}

#[test]
fn unicode_payloads_roundtrip() {
    let (_parent, dir) = fresh_log_dir();
    let mut log = EventLog::create(&dir).expect("create");
    let payload = json!({
        "note": "ΔRCI café",
        "jp": "日本語",
        "emoji": "🦀"
    });
    let ev = log
        .append(append_input("uni", payload.clone()))
        .expect("append");
    assert_eq!(ev.payload, payload);
    assert_eq!(ev.payload_hash, payload_hash(&payload));
    assert_eq!(ev.payload_hash, reference::payload_hash(&payload));
    drop(log);
    let log = EventLog::open(&dir).expect("reopen");
    let got = log.get(0).expect("get");
    assert_eq!(got.payload, payload);
}

#[test]
fn empty_payload_object() {
    let (_parent, dir) = fresh_log_dir();
    let mut log = EventLog::create(&dir).expect("create");
    let ev = log
        .append(append_input("empty", json!({})))
        .expect("append");
    assert_eq!(ev.payload, json!({}));
    assert_eq!(ev.payload_hash, reference::GOLDEN_EMPTY_OBJECT_HASH);
    assert_eq!(ev.seq, 0);
    assert_eq!(ev.prev_hash, GENESIS_PREV_HASH);
}

#[test]
fn nested_objects_sorted_for_hash() {
    let p1 = json!({"outer": {"z": 1, "a": [3, 1]}, "k": 0});
    let p2 = json!({"k": 0, "outer": {"a": [3, 1], "z": 1}});
    assert_eq!(payload_hash(&p1), payload_hash(&p2));
    assert_eq!(payload_hash(&p1), reference::payload_hash(&p1));
    let p3 = json!({"outer": {"z": 1, "a": [1, 3]}, "k": 0});
    assert_ne!(
        payload_hash(&p1),
        payload_hash(&p3),
        "array order is significant"
    );
}

#[test]
fn integer_not_hashed_as_float() {
    let i = json!({"n": 1});
    let f = json!({"n": 1.0});
    assert_ne!(payload_hash(&i), payload_hash(&f));
    assert_eq!(payload_hash(&i), reference::payload_hash(&i));
}

#[test]
fn f1_append_then_reopen_matches_reference_genesis() {
    let (_parent, dir) = fresh_log_dir();
    {
        let mut log = EventLog::create(&dir).expect("create");
        let ev = log.append(f1_append()).expect("append");
        assert_eq!(ev.hash, reference::F1_GENESIS_EVENT_HASH);
    }
    let log = EventLog::open(&dir).expect("open");
    let ev = log.get(0).expect("get");
    assert_eq!(ev.hash, reference::F1_GENESIS_EVENT_HASH);
    assert_eq!(ev.payload_hash, reference::F1_GENESIS_PAYLOAD_HASH);
}
