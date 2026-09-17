//! Group: World::create / open, layout, TooFewReplicas, process/replica names.

mod common;
mod reference;

use common::{
    assert_default_shape, assert_err, assert_too_few, cfg, coordinator, default_config,
    events_jsonl, fresh_world_dir, process_ids, replica, replica_ids, replica_path, worker,
};
use prometheus_chaos::{
    World, WorldConfig, DEFAULT_PROCESSES, MIN_REPLICAS,
};
use reference::RefWorld;

#[test]
fn create_fails_if_dir_exists() {
    let (_parent, dir) = fresh_world_dir();
    std::fs::create_dir_all(&dir).unwrap();
    assert_err(World::create(&dir, default_config()), "create existing dir");
}

#[test]
fn create_fails_too_few_replicas_zero() {
    let (_parent, dir) = fresh_world_dir();
    let err = match World::create(&dir, cfg(0, DEFAULT_PROCESSES, 0)) {
        Ok(_) => panic!("n=0 must fail"),
        Err(e) => e,
    };
    assert_too_few(&err, 0);
}

#[test]
fn create_fails_too_few_replicas_one() {
    let (_parent, dir) = fresh_world_dir();
    let err = match World::create(&dir, cfg(1, DEFAULT_PROCESSES, 0)) {
        Ok(_) => panic!("n=1 must fail"),
        Err(e) => e,
    };
    assert_too_few(&err, 1);
    assert_eq!(MIN_REPLICAS, 2);
}

#[test]
fn create_two_replicas_layout_and_names() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    assert_default_shape(w.config());
    assert_eq!(w.dir(), dir.as_path());
    assert_eq!(w.now(), 0);
    assert_eq!(w.replicas(), replica_ids(MIN_REPLICAS));
    assert_eq!(w.processes(), process_ids(DEFAULT_PROCESSES));
    assert!(!w.processes().is_empty(), "processes() nonempty after create");
    assert_eq!(w.processes()[0], coordinator());
    assert_eq!(w.processes()[1], worker(0));
    for id in w.replicas() {
        let p = replica_path(&dir, &id);
        assert!(p.is_dir(), "missing replica dir {}", p.display());
        assert!(events_jsonl(&dir, &id).exists(), "missing events.jsonl");
        assert_eq!(w.replica_dir(&id), p);
        assert_eq!(w.log_len(&id).expect("log_len"), 0);
    }
    // Touch an unimplemented path so this cannot pass a stub by accident
    // if create were ever a no-op: enqueue requires a live implementation.
    w.enqueue_task(&common::task("warm"), w.now())
        .expect("enqueue after create");
}

#[test]
fn create_three_replicas() {
    let (_parent, dir) = fresh_world_dir();
    let w = World::create(&dir, cfg(3, 2, 7)).expect("create 3");
    assert_eq!(w.replicas(), replica_ids(3));
    assert_eq!(w.processes(), process_ids(2));
    assert_eq!(w.config().seed, 7);
}

#[test]
fn open_replays_same_replica_ids() {
    let (_parent, dir) = fresh_world_dir();
    let config = default_config();
    {
        let mut w = World::create(&dir, config.clone()).expect("create");
        w.enqueue_task(&common::task("a"), w.now()).expect("enq");
    }
    let w = World::open(&dir, config).expect("open");
    assert_eq!(w.replicas(), replica_ids(MIN_REPLICAS));
    assert!(w.log_len(&replica("0")).expect("len") >= 1);
}

#[test]
fn open_missing_dir_fails() {
    let (_parent, dir) = fresh_world_dir();
    assert_err(World::open(&dir, default_config()), "open missing");
}

#[test]
fn create_matches_reference_layout() {
    let (_pw, dir_w) = fresh_world_dir();
    let (_pr, dir_r) = fresh_world_dir();
    let config = cfg(2, 4, 0);
    let w = World::create(&dir_w, config.clone()).expect("world");
    let r = RefWorld::create(&dir_r, config).expect("ref");
    assert_eq!(w.replicas(), r.replicas());
    assert_eq!(w.processes(), r.processes());
    assert_eq!(w.now(), r.now());
    assert_eq!(w.config().n_replicas, r.config().n_replicas);
}

#[test]
fn default_world_config_via_create() {
    let (_parent, dir) = fresh_world_dir();
    let w = World::create(&dir, WorldConfig::default()).expect("create");
    assert_eq!(w.config().n_replicas, MIN_REPLICAS);
    assert_eq!(w.config().n_processes, DEFAULT_PROCESSES);
    assert_eq!(w.config().seed, 0);
    assert!(!w.processes().is_empty());
}
