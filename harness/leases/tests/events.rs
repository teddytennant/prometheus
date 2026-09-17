//! Group: locked EventLog event_type strings and task_id / attempt fields.

mod common;
mod reference;

use common::{event_types, fresh_queue_dir, item_n, short_config, short_ttl, worker};
use prometheus_leases::{
    Queue, TaskId, EVENT_CLAIMED, EVENT_COMPLETED, EVENT_ENQUEUED, EVENT_EXPIRED, EVENT_FAILED,
    EVENT_HEARTBEAT, EVENT_OUTPUT,
};
use reference::RefQueue;

fn find<'a>(
    q: &'a Queue,
    event_type: &str,
) -> &'a prometheus_log::Event {
    q.log()
        .iter()
        .find(|e| e.event_type == event_type)
        .unwrap_or_else(|| panic!("missing event_type {event_type}"))
}

#[test]
fn enqueue_then_claim_events_use_locked_strings() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefQueue::new(cfg);
    q.enqueue(item_n(1), 0).expect("enqueue");
    refer.enqueue(item_n(1), 0).expect("ref enqueue");
    q.claim(&worker("w"), 0).expect("claim");
    refer.claim(&worker("w"), 0).expect("ref claim");

    assert_eq!(event_types(&q), refer.event_types());
    assert_eq!(
        event_types(&q),
        vec![EVENT_ENQUEUED.to_string(), EVENT_CLAIMED.to_string()]
    );

    let enq = find(&q, EVENT_ENQUEUED);
    assert_eq!(enq.task_id.as_deref(), Some("t1"));
    assert!(
        enq.attempt.is_none() || enq.attempt == Some(0),
        "enqueued attempt is None or 0, got {:?}",
        enq.attempt
    );

    let claimed = find(&q, EVENT_CLAIMED);
    assert_eq!(claimed.task_id.as_deref(), Some("t1"));
    assert_eq!(claimed.attempt, Some(1));
}

#[test]
fn heartbeat_event_has_task_id_and_attempt() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    q.claim(&worker("w"), 0).expect("claim");
    q.heartbeat(&TaskId("t1".into()), &worker("w"), 1, 10)
        .expect("heartbeat");
    let hb = find(&q, EVENT_HEARTBEAT);
    assert_eq!(hb.task_id.as_deref(), Some("t1"));
    assert_eq!(hb.attempt, Some(1));
}

#[test]
fn complete_emits_output_then_completed() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefQueue::new(cfg);
    q.enqueue(item_n(1), 0).expect("enqueue");
    refer.enqueue(item_n(1), 0).expect("ref enqueue");
    q.claim(&worker("w"), 0).expect("claim");
    refer.claim(&worker("w"), 0).expect("ref claim");
    q.complete(&TaskId("t1".into()), &worker("w"), 1, b"x", 1)
        .expect("complete");
    refer
        .complete(&TaskId("t1".into()), &worker("w"), 1, b"x", 1)
        .expect("ref complete");
    assert_eq!(event_types(&q), refer.event_types());
    assert_eq!(
        event_types(&q),
        vec![
            EVENT_ENQUEUED.to_string(),
            EVENT_CLAIMED.to_string(),
            EVENT_OUTPUT.to_string(),
            EVENT_COMPLETED.to_string(),
        ]
    );
    let out = find(&q, EVENT_OUTPUT);
    assert_eq!(out.task_id.as_deref(), Some("t1"));
    assert_eq!(out.attempt, Some(1));
    let done = find(&q, EVENT_COMPLETED);
    assert_eq!(done.task_id.as_deref(), Some("t1"));
    assert_eq!(done.attempt, Some(1));
}

#[test]
fn fail_emits_task_failed() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    q.claim(&worker("w"), 0).expect("claim");
    q.fail(&TaskId("t1".into()), &worker("w"), 1, "reason", 1)
        .expect("fail");
    assert_eq!(
        event_types(&q),
        vec![
            EVENT_ENQUEUED.to_string(),
            EVENT_CLAIMED.to_string(),
            EVENT_FAILED.to_string(),
        ]
    );
    let failed = find(&q, EVENT_FAILED);
    assert_eq!(failed.task_id.as_deref(), Some("t1"));
    assert_eq!(failed.attempt, Some(1));
}

#[test]
fn expire_due_or_reclaim_records_expired_and_claimed_attempt_two() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    q.claim(&worker("A"), 0).expect("claim");
    let now = short_ttl();
    let _ = q.expire_due(now).expect("expire_due");
    let lease = q
        .claim(&worker("B"), now)
        .expect("reclaim")
        .expect("some");
    assert_eq!(lease.attempt, 2);

    let claimed_two = q
        .log()
        .iter()
        .filter(|e| e.event_type == EVENT_CLAIMED && e.attempt == Some(2))
        .count();
    assert_eq!(claimed_two, 1, "exactly one claim event at attempt 2");
    let claimed = q
        .log()
        .iter()
        .find(|e| e.event_type == EVENT_CLAIMED && e.attempt == Some(2))
        .expect("claimed 2");
    assert_eq!(claimed.task_id.as_deref(), Some("t1"));

    // expire_due may emit task.expired, or claim may fold that append.
    let expired: Vec<_> = q
        .log()
        .iter()
        .filter(|e| e.event_type == EVENT_EXPIRED)
        .cloned()
        .collect();
    if let Some(exp) = expired.first() {
        assert_eq!(exp.task_id.as_deref(), Some("t1"));
        assert_eq!(exp.attempt, Some(1), "expired event carries the old attempt");
    }
}
