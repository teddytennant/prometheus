//! Group: open replays the EventLog; durable state matches pre-crash memory.

mod common;
mod reference;

use common::{
    assert_completed, assert_failed, assert_leased, assert_queued, assert_task_eq, fresh_queue_dir,
    item, item_n, short_config, short_ttl, worker,
};
use prometheus_leases::{Queue, TaskId};
use reference::{output_hash, RefQueue};

#[test]
fn reopen_sees_enqueued_task() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let work = item("alpha", serde_json::json!({"x": 1}));
    {
        let mut q = Queue::create(&dir, cfg.clone()).expect("create");
        q.enqueue(work.clone(), 42).expect("enqueue");
    }
    let q = Queue::open(&dir, cfg).expect("open");
    let got = q.get(&TaskId("alpha".into())).expect("get after open");
    assert_queued(got, "alpha", 0);
    assert_eq!(got.payload, serde_json::json!({"x": 1}));
}

#[test]
fn reopen_sees_lease_attempt_and_expiry() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let t0 = 9_000u64;
    {
        let mut q = Queue::create(&dir, cfg.clone()).expect("create");
        q.enqueue(item_n(1), t0).expect("enqueue");
        q.claim(&worker("alice"), t0).expect("claim");
    }
    let q = Queue::open(&dir, cfg).expect("open");
    let got = q.get(&TaskId("t1".into())).expect("get");
    assert_leased(got, "t1", "alice", 1, t0 + short_ttl());
}

#[test]
fn reopen_sees_completed_with_output() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let out = b"durable-bytes";
    {
        let mut q = Queue::create(&dir, cfg.clone()).expect("create");
        q.enqueue(item_n(1), 0).expect("enqueue");
        q.claim(&worker("w"), 0).expect("claim");
        q.complete(&TaskId("t1".into()), &worker("w"), 1, out, 1)
            .expect("complete");
    }
    let q = Queue::open(&dir, cfg).expect("open");
    let hash = output_hash(out);
    assert_completed(q.get(&TaskId("t1".into())).expect("get"), "t1", 1, &hash);
    assert_eq!(
        q.get_output(&TaskId("t1".into()), 1)
            .expect("get_output")
            .as_deref(),
        Some(&out[..])
    );
}

#[test]
fn reopen_sees_failed_terminal() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    {
        let mut q = Queue::create(&dir, cfg.clone()).expect("create");
        q.enqueue(item_n(1), 0).expect("enqueue");
        q.claim(&worker("w"), 0).expect("claim");
        q.fail(&TaskId("t1".into()), &worker("w"), 1, "nope", 1)
            .expect("fail");
    }
    let mut q = Queue::open(&dir, cfg).expect("open");
    assert_failed(q.get(&TaskId("t1".into())).expect("get"), "t1", 1);
    assert!(q.claim(&worker("other"), 99).expect("claim").is_none());
}

#[test]
fn reopen_matches_reference_after_heartbeat() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut refer = RefQueue::new(cfg.clone());
    let t0 = 100u64;
    {
        let mut q = Queue::create(&dir, cfg.clone()).expect("create");
        q.enqueue(item_n(1), t0).expect("enqueue");
        refer.enqueue(item_n(1), t0).expect("ref enqueue");
        q.claim(&worker("w"), t0).expect("claim");
        refer.claim(&worker("w"), t0).expect("ref claim");
        q.heartbeat(&TaskId("t1".into()), &worker("w"), 1, t0 + 50)
            .expect("heartbeat");
        refer
            .heartbeat(&TaskId("t1".into()), &worker("w"), 1, t0 + 50)
            .expect("ref heartbeat");
    }
    let q = Queue::open(&dir, cfg).expect("open");
    assert_task_eq(
        q.get(&TaskId("t1".into())).expect("get"),
        refer.get(&TaskId("t1".into())).expect("ref get"),
    );
}

#[test]
fn reopen_after_reclaim_keeps_attempt_two() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let t0 = 0u64;
    let now = short_ttl();
    {
        let mut q = Queue::create(&dir, cfg.clone()).expect("create");
        q.enqueue(item_n(1), t0).expect("enqueue");
        q.claim(&worker("A"), t0).expect("claim A");
        let _ = q.expire_due(now).expect("expire_due");
        q.claim(&worker("B"), now).expect("claim B");
    }
    let q = Queue::open(&dir, cfg).expect("open");
    let got = q.get(&TaskId("t1".into())).expect("get");
    assert_leased(got, "t1", "B", 2, now + short_ttl());
}
