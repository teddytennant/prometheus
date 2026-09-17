//! Group: zombie worker cannot overwrite a newer attempt.

mod common;
mod reference;

use common::{
    assert_completed, assert_stale_or_not_holder, fresh_queue_dir, item_n, output_path, short_config,
    short_ttl, worker,
};
use prometheus_leases::{Queue, TaskId};
use reference::output_hash;

#[test]
fn zombie_complete_after_reclaim_is_refused_and_new_output_wins() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    let t0 = 0u64;
    q.enqueue(item_n(1), t0).expect("enqueue");
    let a = q
        .claim(&worker("A"), t0)
        .expect("claim A")
        .expect("lease A");
    assert_eq!(a.attempt, 1);

    let now = t0 + short_ttl();
    let _ = q.expire_due(now).expect("expire_due");
    let b = q
        .claim(&worker("B"), now)
        .expect("claim B")
        .expect("lease B");
    assert_eq!(b.attempt, 2);
    assert_eq!(b.task_id.0, a.task_id.0);

    let err = q
        .complete(&a.task_id, &worker("A"), 1, b"from-A", now + 1)
        .expect_err("zombie complete");
    assert_stale_or_not_holder(&err, "zombie complete");
    assert!(
        q.get_output(&a.task_id, 1)
            .expect("out1")
            .is_none(),
        "refused zombie complete must not write attempt 1"
    );
    assert!(
        q.get_output(&a.task_id, 2)
            .expect("out2")
            .is_none(),
        "refused zombie complete must not write attempt 2"
    );

    q.complete(&b.task_id, &worker("B"), 2, b"from-B", now + 2)
        .expect("B complete");
    let hash = output_hash(b"from-B");
    assert_completed(q.get(&b.task_id).expect("get"), "t1", 2, &hash);
    assert_eq!(
        q.get_output(&b.task_id, 2).expect("B bytes").as_deref(),
        Some(&b"from-B"[..])
    );
    assert!(
        q.get_output(&b.task_id, 1)
            .expect("A bytes")
            .is_none(),
        "attempt 1 was never successfully completed"
    );

    let err = q
        .complete(&a.task_id, &worker("A"), 1, b"from-A-late", now + 3)
        .expect_err("zombie after B");
    assert_stale_or_not_holder(&err, "zombie after B");
    assert_eq!(
        q.get_output(&b.task_id, 2).expect("B still").as_deref(),
        Some(&b"from-B"[..]),
        "zombie must not replace attempt 2"
    );

    let on_disk = std::fs::read(output_path(&dir, &b.task_id, 2)).expect("disk attempt 2");
    assert_eq!(on_disk, b"from-B");
}

#[test]
fn zombie_heartbeat_after_reclaim_is_refused() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    q.claim(&worker("A"), 0).expect("claim A");
    let now = short_ttl();
    let _ = q.expire_due(now).expect("expire_due");
    let b = q.claim(&worker("B"), now).expect("claim B").expect("lease B");
    assert_eq!(b.attempt, 2);
    let err = q
        .heartbeat(&TaskId("t1".into()), &worker("A"), 1, now + 1)
        .expect_err("zombie heartbeat");
    assert_stale_or_not_holder(&err, "zombie heartbeat");
    let got = q.get(&TaskId("t1".into())).expect("get");
    assert_eq!(got.attempt, 2);
    assert_eq!(got.worker_id.as_ref().map(|w| w.0.as_str()), Some("B"));
}

#[test]
fn zombie_complete_wrong_attempt_cannot_clobber_disk() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    q.claim(&worker("A"), 0).expect("claim A");
    let now = short_ttl();
    let _ = q.expire_due(now).expect("expire_due");
    q.claim(&worker("B"), now).expect("claim B");
    q.complete(&TaskId("t1".into()), &worker("B"), 2, b"keep", now + 1)
        .expect("B complete");

    let _ = q.complete(&TaskId("t1".into()), &worker("A"), 2, b"steal", now + 2);
    let _ = q.complete(&TaskId("t1".into()), &worker("A"), 1, b"old", now + 2);
    assert_eq!(
        q.get_output(&TaskId("t1".into()), 2)
            .expect("out")
            .as_deref(),
        Some(&b"keep"[..])
    );
}
