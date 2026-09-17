//! Group: heartbeat extends expiry; two missed periods; reclaim attempt 2.

mod common;
mod reference;

use common::{
    assert_lease_eq, assert_stale_or_not_holder, fresh_queue_dir, item_n, short_config, short_ttl,
    worker,
};
use prometheus_leases::{Queue, TaskId};
use reference::RefQueue;

#[test]
fn heartbeat_extends_expiry_to_now_plus_ttl() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefQueue::new(cfg);
    let t0 = 10_000u64;
    q.enqueue(item_n(1), t0).expect("enqueue");
    refer.enqueue(item_n(1), t0).expect("ref enqueue");
    let lease = q.claim(&worker("w"), t0).expect("claim").expect("lease");
    refer.claim(&worker("w"), t0).expect("ref claim");
    assert_eq!(lease.expires_at, t0 + short_ttl());

    let t1 = t0 + 500;
    let new_lease = q
        .heartbeat(&lease.task_id, &worker("w"), lease.attempt, t1)
        .expect("heartbeat");
    let ref_lease = refer
        .heartbeat(&lease.task_id, &worker("w"), lease.attempt, t1)
        .expect("ref heartbeat");
    assert_eq!(new_lease.expires_at, t1 + short_ttl());
    assert_eq!(new_lease.expires_at, ref_lease.expires_at);
    assert_eq!(new_lease.attempt, 1);
    assert_eq!(new_lease.worker_id.0, "w");
    let got = q.get(&lease.task_id).expect("get");
    assert_eq!(got.expires_at, Some(new_lease.expires_at));
    assert_eq!(got.attempt, 1);
}

#[test]
fn heartbeat_wrong_worker_is_not_holder() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    let lease = q.claim(&worker("owner"), 0).expect("claim").expect("lease");
    let err = q
        .heartbeat(&lease.task_id, &worker("intruder"), 1, 1)
        .expect_err("not holder");
    match err {
        prometheus_leases::Error::NotHolder(task_id, worker_id, attempt) => {
            assert_eq!(task_id, "t1");
            assert_eq!(worker_id, "intruder");
            assert_eq!(attempt, 1);
        }
        other => panic!("expected NotHolder, got {other:?}"),
    }
}

#[test]
fn heartbeat_wrong_attempt_is_stale() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    q.claim(&worker("w"), 0).expect("claim");
    let err = q
        .heartbeat(&TaskId("t1".into()), &worker("w"), 99, 1)
        .expect_err("stale");
    match err {
        prometheus_leases::Error::StaleAttempt {
            task_id,
            attempt,
            current,
        } => {
            assert_eq!(task_id, "t1");
            assert_eq!(attempt, 99);
            assert_eq!(current, 1);
        }
        other => panic!("expected StaleAttempt, got {other:?}"),
    }
}

#[test]
fn before_ttl_expire_due_does_not_requeue() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    let t0 = 100u64;
    q.enqueue(item_n(1), t0).expect("enqueue");
    let lease = q.claim(&worker("a"), t0).expect("claim").expect("lease");
    let now = lease.expires_at.saturating_sub(1);
    let _ = q.expire_due(now).expect("expire_due");
    assert!(
        q.claim(&worker("b"), now).expect("claim").is_none(),
        "lease still held just before expiry"
    );
    let got = q.get(&lease.task_id).expect("get");
    assert_eq!(got.attempt, 1);
    assert_eq!(got.worker_id.as_ref().map(|w| w.0.as_str()), Some("a"));
}

#[test]
fn two_missed_periods_then_claim_is_attempt_two() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefQueue::new(cfg);
    let t0 = 1_000u64;
    q.enqueue(item_n(1), t0).expect("enqueue");
    refer.enqueue(item_n(1), t0).expect("ref enqueue");
    let lease = q.claim(&worker("a"), t0).expect("claim").expect("lease");
    refer.claim(&worker("a"), t0).expect("ref claim");
    assert_eq!(lease.attempt, 1);
    assert_eq!(lease.expires_at, t0 + short_ttl());

    // Two missed heartbeat periods, no heartbeat: now == claim + ttl.
    let now = t0 + short_ttl();
    let _ = q.expire_due(now).expect("expire_due");
    let _ = refer.expire_due(now).expect("ref expire_due");
    let reclaim = q.claim(&worker("b"), now).expect("reclaim").expect("some");
    let rreclaim = refer
        .claim(&worker("b"), now)
        .expect("ref reclaim")
        .expect("ref some");
    assert_lease_eq(&reclaim, "t1", "b", 2, now + short_ttl());
    assert_eq!(reclaim.attempt, rreclaim.attempt);
    assert_eq!(
        Queue::checkpoint_branch(&reclaim.task_id, reclaim.attempt),
        "task/t1/attempt/2"
    );
    common::assert_leased(
        q.get(&reclaim.task_id).expect("get"),
        "t1",
        "b",
        2,
        now + short_ttl(),
    );

    let hb = q.heartbeat(&lease.task_id, &worker("a"), 1, now + 1);
    assert_stale_or_not_holder(&hb.expect_err("old heartbeat"), "old heartbeat");
    let complete = q.complete(&lease.task_id, &worker("a"), 1, b"old", now + 1);
    assert_stale_or_not_holder(&complete.expect_err("old complete"), "old complete");
}

#[test]
fn expired_tasks_append_behind_already_queued() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    let t0 = 0u64;
    q.enqueue(item_n(1), t0).expect("e1");
    q.enqueue(item_n(2), t0).expect("e2");
    let a = q.claim(&worker("w"), t0).expect("claim a").expect("t1");
    assert_eq!(a.task_id.0, "t1");
    // t2 still queued. Expire t1; it goes to the back, so next claim is t2.
    let now = t0 + short_ttl();
    let _ = q.expire_due(now).expect("expire_due");
    let next = q.claim(&worker("w"), now).expect("claim").expect("some");
    assert_eq!(next.task_id.0, "t2");
    assert_eq!(next.attempt, 1);
    let reclaimed = q
        .claim(&worker("w"), now)
        .expect("claim")
        .expect("t1 again");
    assert_eq!(reclaimed.task_id.0, "t1");
    assert_eq!(reclaimed.attempt, 2);
}

#[test]
fn same_worker_can_reclaim_after_expiry() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    q.claim(&worker("solo"), 0).expect("claim");
    let now = short_ttl();
    let _ = q.expire_due(now).expect("expire_due");
    let lease = q
        .claim(&worker("solo"), now)
        .expect("reclaim")
        .expect("some");
    assert_eq!(lease.attempt, 2);
    assert_eq!(lease.worker_id.0, "solo");
}
