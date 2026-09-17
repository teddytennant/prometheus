//! Group: complete / fail holder checks; fail is terminal; duplicate after terminal.

mod common;
mod reference;

use common::{
    assert_completed, assert_duplicate, assert_failed, assert_stale_or_not_holder, fresh_queue_dir,
    item_n, short_config, worker,
};
use prometheus_leases::{Error, Queue, TaskId};
use reference::{output_hash, RefQueue};

#[test]
fn complete_writes_hash_and_bytes() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefQueue::new(cfg);
    let now = 7u64;
    q.enqueue(item_n(1), now).expect("enqueue");
    refer.enqueue(item_n(1), now).expect("ref enqueue");
    let lease = q.claim(&worker("w"), now).expect("claim").expect("lease");
    refer.claim(&worker("w"), now).expect("ref claim");
    let out = b"hello-output";
    q.complete(&lease.task_id, &worker("w"), 1, out, now + 1)
        .expect("complete");
    refer
        .complete(&lease.task_id, &worker("w"), 1, out, now + 1)
        .expect("ref complete");
    let hash = output_hash(out);
    assert_completed(q.get(&lease.task_id).expect("get"), "t1", 1, &hash);
    assert_eq!(
        q.get(&lease.task_id).expect("get").output_hash,
        refer.get(&lease.task_id).expect("ref get").output_hash
    );
    let got = q.get_output(&lease.task_id, 1).expect("get_output");
    assert_eq!(got.as_deref(), Some(&out[..]));
    assert_eq!(refer.get_output(&lease.task_id, 1).expect("ref out"), got);
}

#[test]
fn complete_wrong_worker_refused() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    q.claim(&worker("owner"), 0).expect("claim");
    let err = q
        .complete(&TaskId("t1".into()), &worker("other"), 1, b"x", 1)
        .expect_err("not holder");
    match err {
        Error::NotHolder(task_id, worker_id, attempt) => {
            assert_eq!(task_id, "t1");
            assert_eq!(worker_id, "other");
            assert_eq!(attempt, 1);
        }
        other => panic!("expected NotHolder, got {other:?}"),
    }
    assert!(
        q.get_output(&TaskId("t1".into()), 1)
            .expect("get_output")
            .is_none(),
        "refused complete must not write bytes"
    );
}

#[test]
fn complete_unknown_task_is_not_found() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    let err = q
        .complete(&TaskId("ghost".into()), &worker("w"), 1, b"x", 0)
        .expect_err("not found");
    common::assert_not_found(&err, "ghost");
}

#[test]
fn complete_twice_refused() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    q.claim(&worker("w"), 0).expect("claim");
    q.complete(&TaskId("t1".into()), &worker("w"), 1, b"once", 1)
        .expect("complete");
    let err = q
        .complete(&TaskId("t1".into()), &worker("w"), 1, b"twice", 2)
        .expect_err("second complete");
    assert_stale_or_not_holder(&err, "second complete");
    let got = q.get_output(&TaskId("t1".into()), 1).expect("out");
    assert_eq!(got.as_deref(), Some(&b"once"[..]), "first output kept");
}

#[test]
fn fail_is_terminal_not_requeued() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefQueue::new(cfg);
    q.enqueue(item_n(1), 0).expect("enqueue");
    refer.enqueue(item_n(1), 0).expect("ref enqueue");
    q.claim(&worker("w"), 0).expect("claim");
    refer.claim(&worker("w"), 0).expect("ref claim");
    q.fail(&TaskId("t1".into()), &worker("w"), 1, "boom", 1)
        .expect("fail");
    refer
        .fail(&TaskId("t1".into()), &worker("w"), 1, "boom", 1)
        .expect("ref fail");
    assert_failed(q.get(&TaskId("t1".into())).expect("get"), "t1", 1);
    assert_eq!(
        q.get(&TaskId("t1".into())).expect("get").state,
        refer.get(&TaskId("t1".into())).expect("ref get").state
    );
    let _ = q.expire_due(1_000_000).expect("expire_due");
    assert!(
        q.claim(&worker("other"), 1_000_000)
            .expect("claim")
            .is_none(),
        "failed tasks are not auto-requeued"
    );
}

#[test]
fn fail_then_reenqueue_same_id_is_duplicate() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    q.claim(&worker("w"), 0).expect("claim");
    q.fail(&TaskId("t1".into()), &worker("w"), 1, "no", 1)
        .expect("fail");
    let err = q.enqueue(item_n(1), 2).expect_err("duplicate");
    assert_duplicate(&err, "t1");
}

#[test]
fn complete_then_reenqueue_same_id_is_duplicate() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    q.claim(&worker("w"), 0).expect("claim");
    q.complete(&TaskId("t1".into()), &worker("w"), 1, b"ok", 1)
        .expect("complete");
    let err = q.enqueue(item_n(1), 2).expect_err("duplicate");
    assert_duplicate(&err, "t1");
}

#[test]
fn fail_wrong_attempt_refused() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    q.claim(&worker("w"), 0).expect("claim");
    let err = q
        .fail(&TaskId("t1".into()), &worker("w"), 7, "no", 1)
        .expect_err("stale fail");
    match err {
        Error::StaleAttempt {
            attempt, current, ..
        } => {
            assert_eq!(attempt, 7);
            assert_eq!(current, 1);
        }
        other => panic!("expected StaleAttempt, got {other:?}"),
    }
    let got = q.get(&TaskId("t1".into())).expect("get");
    assert_eq!(got.state, prometheus_leases::TaskState::Leased);
}
