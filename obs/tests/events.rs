//! Group: append_event hash chain, seq, schema, timestamps, events().

mod common;
mod reference;

use prometheus_obs::{
    Event, RunRegistry, EVENT_SCHEMA_ID, EVENT_SCHEMA_VERSION, GENESIS_HASH,
};
use serde_json::json;

use common::{assert_rfc3339ish, assert_unknown_run, is_sha256_hex, sample_run};

fn register(reg: &mut RunRegistry, run_id: &str) {
    reg.register(sample_run(run_id)).expect("register run");
}

fn assert_event_hashes(ev: &Event) {
    assert!(is_sha256_hex(&ev.payload_hash), "payload_hash hex: {}", ev.payload_hash);
    assert!(is_sha256_hex(&ev.hash), "hash hex: {}", ev.hash);
    let expected_ph = reference::payload_hash(&ev.payload);
    assert_eq!(ev.payload_hash, expected_ph, "payload_hash vs tests/reference");
    let expected_eh = reference::event_hash(
        ev.seq,
        &ev.prev_hash,
        &ev.payload_hash,
        &ev.timestamp,
        &ev.event_type,
    );
    assert_eq!(ev.hash, expected_eh, "hash vs tests/reference event_hash");
}

#[test]
fn first_event_uses_genesis_and_seq_one() {
    let mut reg = RunRegistry::new();
    register(&mut reg, "run-1");
    let payload = json!({"cycle": 1});
    let ev = reg
        .append_event("run-1", "session.start", payload.clone())
        .expect("append");

    assert_eq!(ev.schema_id, EVENT_SCHEMA_ID);
    assert_eq!(ev.schema_id, "prometheus.event_log");
    assert_eq!(ev.schema_version, EVENT_SCHEMA_VERSION);
    assert_eq!(ev.schema_version, 1);
    assert_eq!(ev.seq, 1, "seq starts at 1");
    assert_eq!(ev.prev_hash, GENESIS_HASH);
    assert_eq!(ev.prev_hash, "0".repeat(64));
    assert_eq!(GENESIS_HASH.len(), 64);
    assert_eq!(ev.event_type, "session.start");
    assert_eq!(ev.payload, payload);
    assert_rfc3339ish(&ev.timestamp);
    assert_event_hashes(&ev);
}

#[test]
fn seq_increases_by_one_and_prev_hash_chains() {
    let mut reg = RunRegistry::new();
    register(&mut reg, "run-chain");
    let a = reg
        .append_event("run-chain", "session.start", json!({"i": 1}))
        .expect("e1");
    let b = reg
        .append_event("run-chain", "train.step", json!({"i": 2}))
        .expect("e2");
    let c = reg
        .append_event("run-chain", "session.end", json!({"i": 3}))
        .expect("e3");

    assert_eq!(a.seq, 1);
    assert_eq!(b.seq, 2);
    assert_eq!(c.seq, 3);
    assert_eq!(a.prev_hash, GENESIS_HASH);
    assert_eq!(b.prev_hash, a.hash);
    assert_eq!(c.prev_hash, b.hash);
    assert_ne!(a.hash, b.hash);
    assert_ne!(b.hash, c.hash);
    assert_rfc3339ish(&a.timestamp);
    assert_rfc3339ish(&b.timestamp);
    assert_rfc3339ish(&c.timestamp);
    assert_event_hashes(&a);
    assert_event_hashes(&b);
    assert_event_hashes(&c);
}

#[test]
fn events_returns_the_chain_in_order() {
    let mut reg = RunRegistry::new();
    register(&mut reg, "run-list");
    let e1 = reg
        .append_event("run-list", "a", json!({"k": 1}))
        .expect("a");
    let e2 = reg
        .append_event("run-list", "b", json!({"k": 2}))
        .expect("b");
    let chain = reg.events("run-list").expect("events");
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[0], e1);
    assert_eq!(chain[1], e2);
}

#[test]
fn events_unknown_run_errors() {
    let reg = RunRegistry::new();
    assert_unknown_run(reg.events("no-such"), "no-such");
}

#[test]
fn append_event_unknown_run_errors() {
    let mut reg = RunRegistry::new();
    assert_unknown_run(
        reg.append_event("no-such", "x", json!({})),
        "no-such",
    );
}

#[test]
fn registered_run_with_no_events_returns_empty_chain() {
    let mut reg = RunRegistry::new();
    register(&mut reg, "empty");
    let chain = reg.events("empty").expect("events");
    assert!(chain.is_empty(), "no append_event yet => empty chain");
}

#[test]
fn event_chains_are_per_run() {
    let mut reg = RunRegistry::new();
    register(&mut reg, "rA");
    register(&mut reg, "rB");
    let a = reg
        .append_event("rA", "a", json!({"who": "A"}))
        .expect("A");
    let b = reg
        .append_event("rB", "b", json!({"who": "B"}))
        .expect("B");
    assert_eq!(a.seq, 1);
    assert_eq!(b.seq, 1);
    assert_eq!(a.prev_hash, GENESIS_HASH);
    assert_eq!(b.prev_hash, GENESIS_HASH);
    assert_eq!(reg.events("rA").expect("A").len(), 1);
    assert_eq!(reg.events("rB").expect("B").len(), 1);
    assert_ne!(a.hash, b.hash);
}

#[test]
fn payload_hash_tracks_payload_not_event_type() {
    let mut reg = RunRegistry::new();
    register(&mut reg, "run-ph");
    let payload = json!({"msg": "hello", "n": 1, "ok": true});
    let ev = reg
        .append_event("run-ph", "custom.type", payload.clone())
        .expect("append");
    assert_eq!(ev.payload_hash, reference::payload_hash(&payload));
    assert_eq!(
        ev.payload_hash,
        reference::GOLDEN_FIXED_PAYLOAD_HASH,
        "fixed payload golden"
    );
    assert_event_hashes(&ev);
}

#[test]
fn timestamps_are_rfc3339ish_non_empty_with_t() {
    let mut reg = RunRegistry::new();
    register(&mut reg, "run-ts");
    let ev = reg
        .append_event("run-ts", "tick", json!({"n": 0}))
        .expect("append");
    assert_rfc3339ish(&ev.timestamp);
    assert!(
        !ev.timestamp.contains(' '),
        "RFC3339-ish timestamps do not contain spaces, got {:?}",
        ev.timestamp
    );
}
