//! Group: fault injection. Corrupt / torn / wrong-typed freeze files fail
//! closed as `Error::Message`.

mod common;
mod reference;

use common::{fresh_world, open_pair, unwrap_err, write_snap, NOW0};
use prometheus_monitors::{Error, KillSnapshot, Monitors};
use reference::RefMonitors;

fn assert_open_message(bytes: &[u8], ctx: &str) {
    let world = fresh_world();
    std::fs::write(&world.prod.freeze_path, bytes).unwrap();
    std::fs::write(&world.refer.freeze_path, bytes).unwrap();
    match Monitors::open(world.prod.clone()) {
        Err(Error::Message(s)) => assert!(!s.is_empty(), "{ctx}: empty Message"),
        Err(e) => panic!("{ctx}: expected Message, got error {e:?}"),
        Ok(_) => panic!("{ctx}: expected Message, open succeeded"),
    }
    match RefMonitors::open(world.refer.clone()) {
        Err(Error::Message(_)) => {}
        Err(e) => panic!("{ctx}: reference expected Message, got error {e:?}"),
        Ok(_) => panic!("{ctx}: reference expected Message, open succeeded"),
    }
}

#[test]
fn torn_json_fails_closed() {
    assert_open_message(b"{", "torn");
}

#[test]
fn empty_freeze_file_fails_closed() {
    assert_open_message(b"", "empty");
}

#[test]
fn non_json_fails_closed() {
    assert_open_message(b"not-json", "garbage");
}

#[test]
fn json_array_fails_closed() {
    assert_open_message(b"[]", "array");
}

#[test]
fn json_null_fails_closed() {
    assert_open_message(b"null", "null");
}

#[test]
fn missing_frozen_field_fails_closed() {
    assert_open_message(
        br#"{"frozen_at":1,"reason":"r","violations":[]}"#,
        "missing frozen",
    );
}

#[test]
fn frozen_wrong_type_fails_closed() {
    assert_open_message(
        br#"{"frozen":"yes","frozen_at":1,"reason":"r","violations":[]}"#,
        "frozen string",
    );
}

#[test]
fn directory_at_freeze_path_fails_closed() {
    let world = fresh_world();
    std::fs::create_dir_all(&world.prod.freeze_path).unwrap();
    std::fs::create_dir_all(&world.refer.freeze_path).unwrap();
    match Monitors::open(world.prod.clone()) {
        Err(Error::Message(_)) => {}
        Err(e) => panic!("dir freeze_path: error {e:?}"),
        Ok(_) => panic!("dir freeze_path: open succeeded"),
    }
    match RefMonitors::open(world.refer.clone()) {
        Err(Error::Message(_)) => {}
        Err(e) => panic!("ref dir: error {e:?}"),
        Ok(_) => panic!("ref dir: open succeeded"),
    }
}

#[test]
fn extra_json_keys_are_ignored() {
    let world = fresh_world();
    let body = serde_json::json!({
        "frozen": true,
        "frozen_at": NOW0,
        "reason": "x",
        "violations": [],
        "extra": "ignored",
    });
    std::fs::write(&world.prod.freeze_path, serde_json::to_vec(&body).unwrap()).unwrap();
    std::fs::write(&world.refer.freeze_path, serde_json::to_vec(&body).unwrap()).unwrap();
    let prod = Monitors::open(world.prod.clone()).expect("extra keys");
    let refer = RefMonitors::open(world.refer.clone()).expect("ref extra keys");
    assert!(prod.frozen() && refer.frozen());
}

#[test]
fn unfrozen_snapshot_on_disk_loads_as_not_frozen() {
    let world = fresh_world();
    let snap = KillSnapshot {
        frozen: false,
        frozen_at: None,
        reason: None,
        violations: vec![],
    };
    write_snap(&world.prod.freeze_path, &snap);
    write_snap(&world.refer.freeze_path, &snap);
    let (prod, refer) = open_pair(&world);
    assert!(!prod.frozen());
    assert!(!refer.frozen());
}

#[test]
fn corrupt_file_does_not_get_replaced_by_open() {
    let world = fresh_world();
    std::fs::write(&world.prod.freeze_path, b"{").unwrap();
    let _ = unwrap_err(Monitors::open(world.prod.clone()).map(|_| ()), "open");
    let left = std::fs::read(&world.prod.freeze_path).unwrap();
    assert_eq!(left, b"{");
}
