//! Group: enqueue, claim attempt 1, FIFO, empty claim, duplicate ids.

mod common;
mod reference;

use common::{
    assert_duplicate, assert_lease_eq, assert_queued, assert_task_eq, fresh_queue_dir, item,
    item_n, short_config, short_ttl, worker,
};
use prometheus_leases::{Queue, TaskId, TaskState};
use reference::RefQueue;

#[test]
fn enqueue_then_get_is_queued_attempt_zero() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefQueue::new(cfg);
    let now = 1_000u64;
    let work = item("alpha", serde_json::json!({"k": "v"}));
    let id = q.enqueue(work.clone(), now).expect("enqueue");
    let rid = refer.enqueue(work.clone(), now).expect("ref enqueue");
    assert_eq!(id.0, "alpha");
    assert_eq!(rid.0, "alpha");
    let got = q.get(&id).expect("get").clone();
    let expect = refer.get(&rid).expect("ref get").clone();
    assert_task_eq(&got, &expect);
    assert_queued(&got, "alpha", 0);
    assert_eq!(got.payload, serde_json::json!({"k": "v"}));
    assert!(got.output_hash.is_none());
}

#[test]
fn get_missing_returns_none() {
    let (_parent, dir) = fresh_queue_dir();
    let q = Queue::create(&dir, short_config()).expect("create");
    assert!(q.get(&TaskId("missing".into())).is_none());
}

#[test]
fn claim_empty_returns_ok_none() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    let mut refer = RefQueue::new(short_config());
    let got = q.claim(&worker("w1"), 0).expect("claim empty");
    let expect = refer.claim(&worker("w1"), 0).expect("ref claim empty");
    assert!(got.is_none(), "empty queue claim");
    assert!(expect.is_none());
}

#[test]
fn first_claim_is_attempt_one_and_sets_expiry() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefQueue::new(cfg);
    let now = 5_000u64;
    q.enqueue(item_n(1), now).expect("enqueue");
    refer.enqueue(item_n(1), now).expect("ref enqueue");
    let lease = q
        .claim(&worker("alice"), now)
        .expect("claim")
        .expect("some lease");
    let rlease = refer
        .claim(&worker("alice"), now)
        .expect("ref claim")
        .expect("ref some");
    assert_lease_eq(&lease, "t1", "alice", 1, now + short_ttl());
    assert_eq!(lease.attempt, rlease.attempt);
    assert_eq!(lease.expires_at, rlease.expires_at);
    let got = q.get(&lease.task_id).expect("get");
    common::assert_leased(got, "t1", "alice", 1, now + short_ttl());
    assert_eq!(
        Queue::checkpoint_branch(&lease.task_id, lease.attempt),
        "task/t1/attempt/1"
    );
}

#[test]
fn claim_is_fifo_enqueue_order() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    let now = 10u64;
    q.enqueue(item_n(1), now).expect("e1");
    q.enqueue(item_n(2), now).expect("e2");
    q.enqueue(item_n(3), now).expect("e3");
    let a = q.claim(&worker("w"), now).expect("c1").expect("t1");
    let b = q.claim(&worker("w"), now).expect("c2").expect("t2");
    let c = q.claim(&worker("w"), now).expect("c3").expect("t3");
    assert_eq!(a.task_id.0, "t1");
    assert_eq!(b.task_id.0, "t2");
    assert_eq!(c.task_id.0, "t3");
    assert_eq!(a.attempt, 1);
    assert_eq!(b.attempt, 1);
    assert_eq!(c.attempt, 1);
    assert!(q.claim(&worker("w"), now).expect("empty").is_none());
}

#[test]
fn duplicate_task_id_enqueue_fails() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item("dup", serde_json::json!(1)), 0)
        .expect("first");
    let err = q
        .enqueue(item("dup", serde_json::json!(2)), 1)
        .expect_err("duplicate");
    assert_duplicate(&err, "dup");
    let got = q.get(&TaskId("dup".into())).expect("still there");
    assert_eq!(got.payload, serde_json::json!(1), "payload unchanged");
    assert_eq!(got.state, TaskState::Queued);
}

#[test]
fn duplicate_after_claim_still_fails() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    q.claim(&worker("w"), 0).expect("claim");
    let err = q.enqueue(item_n(1), 1).expect_err("duplicate");
    assert_duplicate(&err, "t1");
}

#[test]
fn claim_skips_nothing_when_all_leased() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    let first = q.claim(&worker("a"), 0).expect("claim").expect("lease");
    assert_eq!(first.attempt, 1);
    assert!(q.claim(&worker("b"), 0).expect("second").is_none());
}
