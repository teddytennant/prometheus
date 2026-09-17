//! Group: partitions, asymmetric reverse-delivery, HealPartition.

mod common;
mod reference;

use common::{
    assert_no_replica, default_config, event_types, fresh_world_dir, replica, task,
};
use prometheus_chaos::{Fault, World, EVENT_COMPLETED, EVENT_ENQUEUED};
use reference::RefWorld;

fn partition(from: &str, to: &str, asymmetric: bool) -> Fault {
    Fault::Partition {
        from: replica(from),
        to: replica(to),
        asymmetric,
    }
}

#[test]
fn partition_unknown_replica_is_no_replica() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    let err = w
        .inject(&partition("0", "nope", false))
        .expect_err("unknown to");
    assert_no_replica(&err, "nope");
}

#[test]
fn symmetric_partition_drops_both_directions_for_enqueue() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.inject(&partition("0", "1", false)).unwrap();
    w.enqueue_task(&task("t"), w.now()).unwrap();
    let t0 = event_types(w.replica_log(&replica("0")).unwrap());
    let t1 = event_types(w.replica_log(&replica("1")).unwrap());
    assert!(
        t0.iter().any(|e| e == EVENT_ENQUEUED),
        "origin replica 0 still records enqueue"
    );
    assert!(
        !t1.iter().any(|e| e == EVENT_ENQUEUED),
        "replica 1 must not see enqueue under symmetric 0↔1 partition"
    );
}

#[test]
fn asymmetric_partition_reverse_still_delivers() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    // D2: drop 0→1 only; reverse 1→0 still delivers.
    w.inject(&partition("0", "1", true)).unwrap();
    w.enqueue_task(&task("t"), w.now()).unwrap();
    let t0 = event_types(w.replica_log(&replica("0")).unwrap());
    let t1 = event_types(w.replica_log(&replica("1")).unwrap());
    assert!(t0.iter().any(|e| e == EVENT_ENQUEUED));
    assert!(
        !t1.iter().any(|e| e == EVENT_ENQUEUED),
        "forward 0→1 dropped"
    );
    w.complete_task(&task("t"), 1, b"ok", w.now()).unwrap();
    let t0 = event_types(w.replica_log(&replica("0")).unwrap());
    let t1 = event_types(w.replica_log(&replica("1")).unwrap());
    assert!(
        t1.iter().any(|e| e == EVENT_COMPLETED),
        "complete originates at replica 1"
    );
    assert!(
        t0.iter().any(|e| e == EVENT_COMPLETED),
        "asymmetric: reverse 1→0 still delivers the complete"
    );
}

#[test]
fn heal_partition_restores_replication() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.inject(&partition("0", "1", false)).unwrap();
    w.inject(&Fault::HealPartition {
        from: replica("0"),
        to: replica("1"),
    })
    .unwrap();
    w.enqueue_task(&task("t"), w.now()).unwrap();
    let t0 = event_types(w.replica_log(&replica("0")).unwrap());
    let t1 = event_types(w.replica_log(&replica("1")).unwrap());
    assert!(t0.iter().any(|e| e == EVENT_ENQUEUED));
    assert!(
        t1.iter().any(|e| e == EVENT_ENQUEUED),
        "healed partition must replicate enqueue to replica 1"
    );
}

#[test]
fn partition_does_not_lose_tasks() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.enqueue_task(&task("t"), w.now()).unwrap();
    w.inject(&partition("0", "1", true)).unwrap();
    w.complete_task(&task("t"), 1, b"ok", w.now()).unwrap();
    assert!(w.invariants().unwrap().hold());
}

#[test]
fn partition_matches_reference() {
    let (_pw, dir_w) = fresh_world_dir();
    let (_pr, dir_r) = fresh_world_dir();
    let config = default_config();
    let mut w = World::create(&dir_w, config.clone()).expect("w");
    let mut r = RefWorld::create(&dir_r, config).expect("r");
    let f = partition("0", "1", true);
    w.inject(&f).unwrap();
    r.inject(&f).unwrap();
    w.enqueue_task(&task("t"), 0).unwrap();
    r.enqueue_task(&task("t"), 0).unwrap();
    assert_eq!(
        w.log_len(&replica("0")).unwrap(),
        r.log_len(&replica("0")).unwrap()
    );
    assert_eq!(
        w.log_len(&replica("1")).unwrap(),
        r.log_len(&replica("1")).unwrap()
    );
}
