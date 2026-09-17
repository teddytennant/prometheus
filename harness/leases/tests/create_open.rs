//! Group: create / open, dir layout, create-fails-if-exists, empty replay.

mod common;
mod reference;

use common::{default_config, default_ttl, event_types, fresh_queue_dir, log_dir, short_config};
use prometheus_leases::{Queue, DEFAULT_HEARTBEAT_PERIOD_MS, MISSED_HEARTBEATS};
use reference::RefQueue;

#[test]
fn create_uses_given_dir_and_empty_log() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    assert_eq!(cfg.lease_ttl_ms(), 2_000);
    let q = Queue::create(&dir, cfg.clone()).expect("create");
    assert_eq!(q.dir(), dir.as_path());
    assert_eq!(q.log().dir(), log_dir(&dir));
    assert_eq!(q.config().heartbeat_period_ms, cfg.heartbeat_period_ms);
    assert_eq!(q.config().missed_heartbeats, cfg.missed_heartbeats);
    assert_eq!(q.config().lease_ttl_ms(), cfg.lease_ttl_ms());
    assert_eq!(q.log().len(), 0, "create writes no events");
    assert!(event_types(&q).is_empty());
    let _refer = RefQueue::new(cfg);
}

#[test]
fn create_default_config_ttl_is_period_times_missed() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = default_config();
    assert_eq!(cfg.heartbeat_period_ms, DEFAULT_HEARTBEAT_PERIOD_MS);
    assert_eq!(cfg.missed_heartbeats, MISSED_HEARTBEATS);
    assert_eq!(cfg.lease_ttl_ms(), default_ttl());
    let q = Queue::create(&dir, cfg).expect("create");
    assert_eq!(q.config().lease_ttl_ms(), default_ttl());
}

#[test]
fn create_fails_if_dir_exists() {
    let (_parent, dir) = fresh_queue_dir();
    std::fs::create_dir_all(&dir).expect("mkdir");
    let err = Queue::create(&dir, short_config());
    common::assert_err(err, "create on existing dir");
}

#[test]
fn create_twice_fails() {
    let (_parent, dir) = fresh_queue_dir();
    let _q = Queue::create(&dir, short_config()).expect("create");
    let err = Queue::create(&dir, short_config());
    common::assert_err(err, "second create");
}

#[test]
fn open_missing_dir_fails() {
    let (_parent, dir) = fresh_queue_dir();
    let err = Queue::open(&dir, short_config());
    common::assert_err(err, "open missing dir");
}

#[test]
fn open_empty_queue_replays_zero_tasks() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    {
        let q = Queue::create(&dir, cfg.clone()).expect("create");
        assert_eq!(q.log().len(), 0);
    }
    let q = Queue::open(&dir, cfg).expect("open");
    assert_eq!(q.dir(), dir.as_path());
    assert_eq!(q.log().len(), 0);
    assert!(q.get(&prometheus_leases::TaskId("nope".into())).is_none());
}
