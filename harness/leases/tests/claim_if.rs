//! Group: `Queue::claim_if` — skip cap-mismatches without claiming, FIFO preserved.
//!
//! Locked: expire_due first, walk currently Queued FIFO, first pred-match is
//! claimed with the same attempt/ttl/EVENT_CLAIMED rules as `claim`. Non-matches
//! stay Queued at the same attempt with no CLAIMED/EXPIRED events. Ok(None) if
//! empty or nothing matches. Pred is not stored.
//!
//! CPU-only. No `gpu` marker.

mod common;
mod reference;

use common::{
    assert_leased, assert_queued, event_types, fresh_queue_dir, item, item_n, short_config,
    short_ttl, worker,
};
use prometheus_leases::{
    Queue, Task, TaskId, TaskState, EVENT_CLAIMED, EVENT_ENQUEUED, EVENT_EXPIRED,
};
use reference::RefQueue;

fn always(_: &Task) -> bool {
    true
}

fn never(_: &Task) -> bool {
    false
}

fn id_is(want: &'static str) -> impl Fn(&Task) -> bool {
    move |t| t.id.0 == want
}

fn cap_is(want: &'static str) -> impl Fn(&Task) -> bool {
    move |t| t.payload.get("cap").and_then(|v| v.as_str()) == Some(want)
}

fn item_cap(id: &str, cap: &str) -> prometheus_leases::WorkItem {
    item(id, serde_json::json!({"cap": cap}))
}

fn claimed_task_ids(q: &Queue) -> Vec<String> {
    q.log()
        .iter()
        .filter(|e| e.event_type == EVENT_CLAIMED)
        .filter_map(|e| e.task_id.clone())
        .collect()
}

fn claimed_attempts_for<'a>(q: &'a Queue, task_id: &str) -> Vec<u64> {
    q.log()
        .iter()
        .filter(|e| e.event_type == EVENT_CLAIMED && e.task_id.as_deref() == Some(task_id))
        .filter_map(|e| e.attempt)
        .collect()
}

/// 1. Empty queue → Ok(None), not an error.
#[test]
fn claim_if_empty_queue_returns_ok_none() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefQueue::new(cfg);
    let got = q
        .claim_if(&worker("w"), 0, always)
        .expect("claim_if empty must be Ok");
    let want = refer
        .claim_if(&worker("w"), 0, always)
        .expect("ref claim_if empty");
    assert!(got.is_none(), "empty queue yields None, got {got:?}");
    assert!(want.is_none());
    assert!(event_types(&q).is_empty());
}

/// 2. Head matches → same lease as `claim` (attempt 1, worker, expires_at).
#[test]
fn claim_if_head_matches_same_as_claim() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut via_claim = Queue::create(&dir.join("via-claim"), cfg.clone()).expect("create claim");
    let mut refer = RefQueue::new(cfg);
    let t0 = 1_000u64;
    q.enqueue(item_n(1), t0).expect("enqueue");
    via_claim.enqueue(item_n(1), t0).expect("enqueue claim twin");
    refer.enqueue(item_n(1), t0).expect("ref enqueue");

    let lease = q
        .claim_if(&worker("alice"), t0, always)
        .expect("claim_if")
        .expect("some");
    let claimed = via_claim
        .claim(&worker("alice"), t0)
        .expect("claim")
        .expect("claim some");
    let rlease = refer
        .claim_if(&worker("alice"), t0, always)
        .expect("ref claim_if")
        .expect("ref some");

    assert_eq!(lease.task_id.0, claimed.task_id.0);
    assert_eq!(lease.worker_id.0, claimed.worker_id.0);
    assert_eq!(lease.attempt, claimed.attempt);
    assert_eq!(lease.expires_at, claimed.expires_at);
    assert_eq!(lease.attempt, 1);
    assert_eq!(lease.worker_id.0, "alice");
    assert_eq!(lease.expires_at, t0 + short_ttl());
    assert_eq!(lease.attempt, rlease.attempt);
    assert_eq!(lease.expires_at, rlease.expires_at);
    assert_leased(
        q.get(&lease.task_id).expect("get"),
        "t1",
        "alice",
        1,
        t0 + short_ttl(),
    );
    assert_eq!(claimed_attempts_for(&q, "t1"), vec![1]);
}

/// 3. Head fails pred, second matches → second leased attempt 1; head still
///    Queued attempt 0; log has no CLAIMED for the head.
#[test]
fn claim_if_skips_head_claims_second() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefQueue::new(cfg);
    q.enqueue(item_n(1), 0).expect("enqueue t1");
    q.enqueue(item_n(2), 0).expect("enqueue t2");
    refer.enqueue(item_n(1), 0).expect("ref t1");
    refer.enqueue(item_n(2), 0).expect("ref t2");

    let lease = q
        .claim_if(&worker("w"), 10, id_is("t2"))
        .expect("claim_if")
        .expect("t2");
    let rlease = refer
        .claim_if(&worker("w"), 10, id_is("t2"))
        .expect("ref claim_if")
        .expect("ref t2");

    assert_eq!(lease.task_id.0, "t2");
    assert_eq!(lease.attempt, 1);
    assert_eq!(lease.worker_id.0, "w");
    assert_eq!(lease.expires_at, 10 + short_ttl());
    assert_eq!(lease.task_id.0, rlease.task_id.0);
    assert_eq!(lease.attempt, rlease.attempt);

    assert_queued(q.get(&TaskId("t1".into())).expect("t1"), "t1", 0);
    assert_leased(
        q.get(&TaskId("t2".into())).expect("t2"),
        "t2",
        "w",
        1,
        10 + short_ttl(),
    );
    assert_eq!(claimed_task_ids(&q), vec!["t2".to_string()]);
    assert!(
        claimed_attempts_for(&q, "t1").is_empty(),
        "no CLAIMED event for skipped head"
    );
    let t1 = q.get(&TaskId("t1".into())).expect("t1 again");
    assert_eq!(t1.state, TaskState::Queued);
    assert_eq!(t1.attempt, 0);
    assert!(t1.worker_id.is_none());
    assert!(t1.expires_at.is_none());
}

/// 4. None match → Ok(None), all stay Queued attempt 0.
#[test]
fn claim_if_none_match_returns_none_all_stay_queued() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefQueue::new(cfg);
    q.enqueue(item_n(1), 0).expect("enqueue t1");
    q.enqueue(item_n(2), 0).expect("enqueue t2");
    refer.enqueue(item_n(1), 0).expect("ref t1");
    refer.enqueue(item_n(2), 0).expect("ref t2");

    let got = q
        .claim_if(&worker("w"), 0, never)
        .expect("claim_if none must be Ok");
    let want = refer
        .claim_if(&worker("w"), 0, never)
        .expect("ref claim_if none");
    assert!(got.is_none(), "no match yields None, got {got:?}");
    assert!(want.is_none());
    assert_queued(q.get(&TaskId("t1".into())).expect("t1"), "t1", 0);
    assert_queued(q.get(&TaskId("t2".into())).expect("t2"), "t2", 0);
    assert!(claimed_task_ids(&q).is_empty());
    assert_eq!(
        event_types(&q),
        vec![EVENT_ENQUEUED.to_string(), EVENT_ENQUEUED.to_string()]
    );
}

/// 5. After claim_if, a later `claim()` still takes the skipped head (FIFO).
#[test]
fn claim_if_then_claim_takes_skipped_head_fifo_preserved() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("t1");
    q.enqueue(item_n(2), 0).expect("t2");
    q.enqueue(item_n(3), 0).expect("t3");

    let skipped = q
        .claim_if(&worker("mesh"), 0, id_is("t2"))
        .expect("claim_if")
        .expect("t2");
    assert_eq!(skipped.task_id.0, "t2");

    let head = q
        .claim(&worker("fifo"), 1)
        .expect("claim")
        .expect("skipped head");
    assert_eq!(head.task_id.0, "t1", "FIFO head after skip is still t1");
    assert_eq!(head.attempt, 1);
    assert_eq!(head.worker_id.0, "fifo");

    let tail = q.claim(&worker("fifo"), 2).expect("claim t3").expect("t3");
    assert_eq!(tail.task_id.0, "t3");
    assert!(q.claim(&worker("fifo"), 3).expect("empty").is_none());
}

/// 6. expire_due still runs first: an expired lease at `now` is re-queued
///    before matching, so pred can claim it at attempt 2.
#[test]
fn claim_if_expire_due_runs_first_requeues_expired_before_matching() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefQueue::new(cfg);
    let t0 = 5_000u64;
    q.enqueue(item_n(1), t0).expect("t1");
    q.enqueue(item_n(2), t0).expect("t2");
    refer.enqueue(item_n(1), t0).expect("ref t1");
    refer.enqueue(item_n(2), t0).expect("ref t2");
    q.claim(&worker("A"), t0).expect("claim t1");
    refer.claim(&worker("A"), t0).expect("ref claim t1");

    let now = t0 + short_ttl();
    // Without expire_due, t1 is still Leased and would not match. After
    // expire_due it is Queued at the back (t2, t1); pred selects t1.
    let lease = q
        .claim_if(&worker("B"), now, id_is("t1"))
        .expect("claim_if")
        .expect("reclaimed t1");
    let rlease = refer
        .claim_if(&worker("B"), now, id_is("t1"))
        .expect("ref claim_if")
        .expect("ref reclaimed t1");

    assert_eq!(lease.task_id.0, "t1");
    assert_eq!(lease.attempt, 2);
    assert_eq!(lease.worker_id.0, "B");
    assert_eq!(lease.expires_at, now + short_ttl());
    assert_eq!(lease.attempt, rlease.attempt);
    assert_queued(q.get(&TaskId("t2".into())).expect("t2"), "t2", 0);
    assert_leased(
        q.get(&TaskId("t1".into())).expect("t1"),
        "t1",
        "B",
        2,
        now + short_ttl(),
    );

    let expired = q
        .log()
        .iter()
        .filter(|e| e.event_type == EVENT_EXPIRED)
        .count();
    assert!(
        expired >= 1,
        "expire_due must run inside claim_if and emit task.expired"
    );
    assert_eq!(claimed_attempts_for(&q, "t1"), vec![1, 2]);
}

/// expire_due first even when nothing matches: expired lease is re-queued,
/// attempt unchanged, no CLAIMED.
#[test]
fn claim_if_none_match_still_expires_due_leases() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    let t0 = 100u64;
    q.enqueue(item_n(1), t0).expect("t1");
    q.enqueue(item_n(2), t0).expect("t2");
    q.claim(&worker("A"), t0).expect("claim t1");
    let now = t0 + short_ttl();
    let got = q
        .claim_if(&worker("B"), now, never)
        .expect("claim_if none");
    assert!(got.is_none());
    assert_queued(q.get(&TaskId("t1".into())).expect("t1"), "t1", 1);
    assert_queued(q.get(&TaskId("t2".into())).expect("t2"), "t2", 0);
    assert!(
        q.log().iter().any(|e| e.event_type == EVENT_EXPIRED
            && e.task_id.as_deref() == Some("t1")
            && e.attempt == Some(1)),
        "expired event for t1 attempt 1"
    );
    assert!(claimed_attempts_for(&q, "t1") == vec![1]);
    // Later FIFO claim takes the already-queued t2, then the requeued t1.
    let first = q.claim(&worker("C"), now).expect("claim").expect("some");
    assert_eq!(first.task_id.0, "t2");
    let second = q.claim(&worker("C"), now).expect("claim").expect("t1");
    assert_eq!(second.task_id.0, "t1");
    assert_eq!(second.attempt, 2);
}

/// After expire_due, FIFO of queued is [already-queued, requeued-expired].
/// Pred-true claims the already-queued head, not the expired tail.
#[test]
fn claim_if_after_expire_matching_all_takes_queued_head_not_expired_tail() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    let t0 = 7_000u64;
    q.enqueue(item_n(1), t0).expect("t1");
    q.enqueue(item_n(2), t0).expect("t2");
    q.claim(&worker("A"), t0).expect("lease t1");
    let now = t0 + short_ttl();
    let lease = q
        .claim_if(&worker("B"), now, always)
        .expect("claim_if")
        .expect("some");
    assert_eq!(
        lease.task_id.0, "t2",
        "t2 was already queued; expired t1 is appended behind it"
    );
    assert_eq!(lease.attempt, 1);
    assert_queued(q.get(&TaskId("t1".into())).expect("t1"), "t1", 1);
}

/// Held unexpired lease is not in the Queued FIFO, so pred cannot steal it.
#[test]
fn claim_if_does_not_claim_held_unexpired_lease() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    let t0 = 3_000u64;
    q.enqueue(item_n(1), t0).expect("t1");
    q.enqueue(item_n(2), t0).expect("t2");
    q.claim(&worker("owner"), t0).expect("lease t1");
    let now = t0 + short_ttl() - 1;
    let lease = q
        .claim_if(&worker("other"), now, always)
        .expect("claim_if")
        .expect("t2");
    assert_eq!(lease.task_id.0, "t2");
    assert_leased(
        q.get(&TaskId("t1".into())).expect("t1"),
        "t1",
        "owner",
        1,
        t0 + short_ttl(),
    );
}

/// Pred is not stored: a later call with a different closure can match a
/// previously skipped head.
#[test]
fn claim_if_pred_not_stored_second_call_can_match_skipped() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("t1");
    q.enqueue(item_n(2), 0).expect("t2");
    let first = q
        .claim_if(&worker("w"), 0, never)
        .expect("claim_if never");
    assert!(first.is_none());
    let second = q
        .claim_if(&worker("w"), 1, always)
        .expect("claim_if always")
        .expect("head");
    assert_eq!(second.task_id.0, "t1");
    assert_eq!(second.attempt, 1);
}

/// Two skips, third matches; subsequent claim() walks the remaining FIFO.
#[test]
fn claim_if_skips_two_claims_third_then_claim_fifo() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("t1");
    q.enqueue(item_n(2), 0).expect("t2");
    q.enqueue(item_n(3), 0).expect("t3");
    let got = q
        .claim_if(&worker("w"), 0, id_is("t3"))
        .expect("claim_if")
        .expect("t3");
    assert_eq!(got.task_id.0, "t3");
    assert_queued(q.get(&TaskId("t1".into())).expect("t1"), "t1", 0);
    assert_queued(q.get(&TaskId("t2".into())).expect("t2"), "t2", 0);
    let a = q.claim(&worker("w"), 1).expect("claim").expect("t1");
    let b = q.claim(&worker("w"), 2).expect("claim").expect("t2");
    assert_eq!(a.task_id.0, "t1");
    assert_eq!(b.task_id.0, "t2");
}

/// H5 mesh-pull shape: skip cap-mismatch, claim a later fit.
#[test]
fn claim_if_skips_cap_mismatch_claims_later_fit() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefQueue::new(cfg);
    for (id, cap) in [("gpu", "gpu"), ("cpu", "cpu"), ("gpu2", "gpu")] {
        q.enqueue(item_cap(id, cap), 0).expect("enqueue");
        refer.enqueue(item_cap(id, cap), 0).expect("ref enqueue");
    }
    let lease = q
        .claim_if(&worker("cpu-node"), 0, cap_is("cpu"))
        .expect("claim_if")
        .expect("cpu");
    let rlease = refer
        .claim_if(&worker("cpu-node"), 0, cap_is("cpu"))
        .expect("ref")
        .expect("ref cpu");
    assert_eq!(lease.task_id.0, "cpu");
    assert_eq!(lease.attempt, 1);
    assert_eq!(rlease.task_id.0, "cpu");
    assert_queued(q.get(&TaskId("gpu".into())).expect("gpu"), "gpu", 0);
    assert_queued(q.get(&TaskId("gpu2".into())).expect("gpu2"), "gpu2", 0);
    let next = q
        .claim(&worker("gpu-node"), 1)
        .expect("claim")
        .expect("gpu head");
    assert_eq!(next.task_id.0, "gpu");
}

/// Durable: reopen after claim_if sees skipped head still queued and the
/// match leased (same event rules as claim).
#[test]
fn claim_if_reopen_preserves_skip_and_lease() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let t0 = 9_000u64;
    {
        let mut q = Queue::create(&dir, cfg.clone()).expect("create");
        q.enqueue(item_n(1), t0).expect("t1");
        q.enqueue(item_n(2), t0).expect("t2");
        let lease = q
            .claim_if(&worker("alice"), t0, id_is("t2"))
            .expect("claim_if")
            .expect("t2");
        assert_eq!(lease.attempt, 1);
        assert_eq!(lease.expires_at, t0 + short_ttl());
    }
    let q = Queue::open(&dir, cfg).expect("open");
    assert_queued(q.get(&TaskId("t1".into())).expect("t1"), "t1", 0);
    assert_leased(
        q.get(&TaskId("t2".into())).expect("t2"),
        "t2",
        "alice",
        1,
        t0 + short_ttl(),
    );
    assert_eq!(claimed_task_ids(&q), vec!["t2".to_string()]);
}

/// Production claim_if must match the slow in-memory reference on mixed ops.
#[test]
fn claim_if_random_ops_match_reference() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefQueue::new(cfg);
    let mut now = 1_000u64;
    for n in 1..=4u32 {
        let work = item_n(n);
        q.enqueue(work.clone(), now).expect("enqueue");
        refer.enqueue(work, now).expect("ref enqueue");
    }
    now += 5;
    let a = q
        .claim_if(&worker("w0"), now, |t| t.id.0 == "t2" || t.id.0 == "t4")
        .expect("claim_if");
    let b = refer
        .claim_if(&worker("w0"), now, |t| t.id.0 == "t2" || t.id.0 == "t4")
        .expect("ref claim_if");
    assert_eq!(a.as_ref().map(|l| l.task_id.0.as_str()), Some("t2"));
    assert_eq!(
        a.as_ref().map(|l| (l.attempt, l.expires_at, l.worker_id.0.as_str())),
        b.as_ref().map(|l| (l.attempt, l.expires_at, l.worker_id.0.as_str()))
    );
    now += 5;
    let c = q
        .claim_if(&worker("w1"), now, |t| t.id.0 == "t4")
        .expect("claim_if t4");
    let d = refer
        .claim_if(&worker("w1"), now, |t| t.id.0 == "t4")
        .expect("ref t4");
    assert_eq!(c.as_ref().map(|l| l.task_id.0.as_str()), Some("t4"));
    assert_eq!(
        c.as_ref().map(|l| l.attempt),
        d.as_ref().map(|l| l.attempt)
    );
    let e = q.claim(&worker("w2"), now).expect("claim head");
    let f = refer.claim(&worker("w2"), now).expect("ref claim head");
    assert_eq!(
        e.as_ref().map(|l| l.task_id.0.as_str()),
        f.as_ref().map(|l| l.task_id.0.as_str())
    );
    assert_eq!(e.as_ref().map(|l| l.task_id.0.as_str()), Some("t1"));
    for id in ["t1", "t2", "t3", "t4"] {
        let tid = TaskId(id.into());
        let pt = q.get(&tid).expect("prod get");
        let rt = refer.get(&tid).expect("ref get");
        assert_eq!(pt.state, rt.state, "{id} state");
        assert_eq!(pt.attempt, rt.attempt, "{id} attempt");
        assert_eq!(
            pt.worker_id.as_ref().map(|w| w.0.as_str()),
            rt.worker_id.as_ref().map(|w| w.0.as_str()),
            "{id} worker"
        );
        assert_eq!(pt.expires_at, rt.expires_at, "{id} expires");
    }
    assert_eq!(event_types(&q), refer.event_types());
}
