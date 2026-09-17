//! Group: drop + open replay, same ids / log_len / invariants; fail-closed.

mod common;
mod reference;

use common::{default_config, events_jsonl, fresh_world_dir, replica, task};
use prometheus_chaos::World;
use reference::RefWorld;

#[test]
fn drop_and_open_same_replica_ids_log_len_invariants() {
    let (_parent, dir) = fresh_world_dir();
    let config = default_config();
    let (ids, lens, inv) = {
        let mut w = World::create(&dir, config.clone()).expect("create");
        w.enqueue_task(&task("keep"), w.now()).unwrap();
        w.complete_task(&task("keep"), 1, b"ok", w.now()).unwrap();
        let ids = w.replicas();
        let lens: Vec<usize> = ids.iter().map(|id| w.log_len(id).expect("len")).collect();
        let inv = w.invariants().expect("inv");
        (ids, lens, inv)
    };
    let w = World::open(&dir, config).expect("open");
    assert_eq!(w.replicas(), ids);
    for (id, len) in ids.iter().zip(lens.iter()) {
        assert_eq!(w.log_len(id).expect("open len"), *len, "log_len {id:?}");
    }
    assert_eq!(w.invariants().expect("open inv"), inv);
    assert!(inv.hold());
}

#[test]
fn open_replays_to_match_reference() {
    let (_pw, dir_w) = fresh_world_dir();
    let (_pr, dir_r) = fresh_world_dir();
    let config = default_config();
    {
        let mut w = World::create(&dir_w, config.clone()).expect("w");
        let mut r = RefWorld::create(&dir_r, config.clone()).expect("r");
        w.enqueue_task(&task("a"), 0).unwrap();
        r.enqueue_task(&task("a"), 0).unwrap();
        w.complete_task(&task("a"), 1, b"x", 0).unwrap();
        r.complete_task(&task("a"), 1, b"x", 0).unwrap();
    }
    let w = World::open(&dir_w, config.clone()).expect("open w");
    let r = RefWorld::open(&dir_r, config).expect("open r");
    assert_eq!(w.replicas(), r.replicas());
    assert_eq!(
        w.log_len(&replica("0")).unwrap(),
        r.log_len(&replica("0")).unwrap()
    );
    assert_eq!(w.invariants().unwrap(), r.invariants().unwrap());
}

#[test]
fn recover_is_startup_replay() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.enqueue_task(&task("t"), w.now()).unwrap();
    w.inject(&prometheus_chaos::Fault::Kill {
        process: prometheus_chaos::ProcessId("coordinator".into()),
    })
    .unwrap();
    w.recover().expect("recover");
    w.complete_task(&task("t"), 1, b"ok", w.now()).unwrap();
    assert!(w.invariants().unwrap().hold());
}

#[test]
fn open_broken_chain_fail_closed() {
    let (_parent, dir) = fresh_world_dir();
    let config = default_config();
    {
        let mut w = World::create(&dir, config.clone()).expect("create");
        w.enqueue_task(&task("t"), w.now()).unwrap();
    }
    let path = events_jsonl(&dir, &replica("1"));
    let bytes = std::fs::read(&path).unwrap();
    let mut broken = bytes.clone();
    if let Some(b) = broken.last_mut() {
        *b ^= 0xff;
    }
    std::fs::write(&path, broken).unwrap();
    assert!(World::open(&dir, config).is_err());
}
