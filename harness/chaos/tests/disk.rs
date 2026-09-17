//! Group: DiskCorrupt, DiskFull, fail-closed open on a broken hash chain.

mod common;
mod reference;

use common::{
    assert_no_replica, default_config, event_types, events_jsonl, fresh_world_dir, replica, task,
};
use prometheus_chaos::{Fault, World, EVENT_ENQUEUED, MIN_REPLICAS};
use reference::RefWorld;

#[test]
fn disk_corrupt_unknown_replica() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    let err = w
        .inject(&Fault::DiskCorrupt {
            replica: replica("nope"),
        })
        .expect_err("unknown");
    assert_no_replica(&err, "nope");
}

#[test]
fn disk_full_unknown_replica() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    let err = w
        .inject(&Fault::DiskFull {
            replica: replica("nope"),
        })
        .expect_err("unknown");
    assert_no_replica(&err, "nope");
}

#[test]
fn disk_corrupt_other_replica_still_works() {
    assert_eq!(MIN_REPLICAS, 2);
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.enqueue_task(&task("t"), w.now()).unwrap();
    w.inject(&Fault::DiskCorrupt {
        replica: replica("0"),
    })
    .unwrap();
    assert!(
        w.replica_log(&replica("0")).is_err(),
        "corrupt replica has no handle"
    );
    let log1 = w.replica_log(&replica("1")).expect("survivor");
    assert!(event_types(log1).iter().any(|e| e == EVENT_ENQUEUED));
    w.complete_task(&task("t"), 1, b"ok", w.now()).unwrap();
    assert!(
        w.invariants().unwrap().hold(),
        "the other replica recovers; no lost task"
    );
}

#[test]
fn disk_full_skips_writes_on_that_replica() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.inject(&Fault::DiskFull {
        replica: replica("1"),
    })
    .unwrap();
    w.enqueue_task(&task("t"), w.now()).unwrap();
    let t0 = event_types(w.replica_log(&replica("0")).unwrap());
    let t1 = event_types(w.replica_log(&replica("1")).unwrap());
    assert!(t0.iter().any(|e| e == EVENT_ENQUEUED));
    assert!(
        !t1.iter().any(|e| e == EVENT_ENQUEUED),
        "disk-full replica must not append"
    );
}

#[test]
fn open_fails_closed_on_broken_hash_chain() {
    let (_parent, dir) = fresh_world_dir();
    let config = default_config();
    {
        let mut w = World::create(&dir, config.clone()).expect("create");
        w.enqueue_task(&task("t"), w.now()).unwrap();
    }
    std::fs::write(events_jsonl(&dir, &replica("0")), b"{not a valid event\n").unwrap();
    assert!(
        World::open(&dir, config).is_err(),
        "open must fail closed on a broken hash chain"
    );
}

#[test]
fn disk_corrupt_then_open_fails_closed() {
    let (_parent, dir) = fresh_world_dir();
    let config = default_config();
    {
        let mut w = World::create(&dir, config.clone()).expect("create");
        w.inject(&Fault::DiskCorrupt {
            replica: replica("1"),
        })
        .unwrap();
    }
    assert!(
        World::open(&dir, config).is_err(),
        "DiskCorrupt is durable; open fails closed"
    );
}

#[test]
fn disk_full_matches_reference() {
    let (_pw, dir_w) = fresh_world_dir();
    let (_pr, dir_r) = fresh_world_dir();
    let config = default_config();
    let mut w = World::create(&dir_w, config.clone()).expect("w");
    let mut r = RefWorld::create(&dir_r, config).expect("r");
    let f = Fault::DiskFull {
        replica: replica("1"),
    };
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
