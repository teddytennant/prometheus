//! Group: property checks of production vs the in-memory reference.

mod common;
mod reference;

use common::{assert_task_eq, fresh_queue_dir, item, short_config, short_ttl, worker};
use prometheus_leases::{Queue, QueueConfig, TaskId, WorkItem, WorkerId};
use reference::RefQueue;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn pick(&mut self, n: usize) -> usize {
        (self.next() as usize) % n.max(1)
    }
}

fn payload(n: u32) -> WorkItem {
    item(
        &format!("p{n}"),
        serde_json::json!({"n": n, "blob": "opaque"}),
    )
}

fn workers() -> [WorkerId; 3] {
    [worker("w0"), worker("w1"), worker("w2")]
}

fn compare_known(q: &Queue, refer: &RefQueue, ids: &[TaskId]) {
    for id in ids {
        match (q.get(id), refer.get(id)) {
            (None, None) => {}
            (Some(a), Some(b)) => assert_task_eq(a, b),
            (a, b) => panic!("get({:?}) prod={a:?} ref={b:?}", id.0),
        }
        if q.get(id).is_some() {
            for attempt in 1..=4u64 {
                let po = q.get_output(id, attempt);
                let ro = refer.get_output(id, attempt);
                match (po, ro) {
                    (Ok(a), Ok(b)) => assert_eq!(a, b, "output {} attempt {attempt}", id.0),
                    (Err(_), Err(_)) => {}
                    (a, b) => panic!("get_output mismatch prod={a:?} ref={b:?}"),
                }
            }
        }
    }
}

#[test]
fn random_ops_match_reference() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefQueue::new(cfg);
    let ws = workers();
    let mut rng = Lcg(0xC0FFEE);
    let mut now = 1_000u64;
    let mut next_item = 0u32;
    let mut known: Vec<TaskId> = Vec::new();
    let mut last_lease: Option<(TaskId, WorkerId, u64)> = None;

    for step in 0..40 {
        now += 3;
        match rng.pick(6) {
            0 if next_item < 5 => {
                let work = payload(next_item);
                next_item += 1;
                let a = q.enqueue(work.clone(), now);
                let b = refer.enqueue(work, now);
                match (a, b) {
                    (Ok(id), Ok(rid)) => {
                        assert_eq!(id.0, rid.0);
                        known.push(id);
                    }
                    (Err(_), Err(_)) => {}
                    (a, b) => panic!("step {step} enqueue prod={a:?} ref={b:?}"),
                }
            }
            1 => {
                let w = &ws[rng.pick(3)];
                let a = q.claim(w, now);
                let b = refer.claim(w, now);
                match (a, b) {
                    (Ok(Some(la)), Ok(Some(lb))) => {
                        assert_eq!(la.task_id.0, lb.task_id.0);
                        assert_eq!(la.attempt, lb.attempt);
                        assert_eq!(la.expires_at, lb.expires_at);
                        last_lease = Some((la.task_id.clone(), la.worker_id.clone(), la.attempt));
                    }
                    (Ok(None), Ok(None)) => {}
                    (a, b) => panic!("step {step} claim prod={a:?} ref={b:?}"),
                }
            }
            2 => {
                if let Some((id, w, att)) = last_lease.clone() {
                    let a = q.heartbeat(&id, &w, att, now);
                    let b = refer.heartbeat(&id, &w, att, now);
                    match (a, b) {
                        (Ok(la), Ok(lb)) => {
                            assert_eq!(la.task_id.0, lb.task_id.0, "heartbeat task");
                            assert_eq!(la.attempt, lb.attempt, "heartbeat attempt");
                            assert_eq!(la.expires_at, lb.expires_at, "heartbeat expiry");
                        }
                        (Err(_), Err(_)) => {}
                        (a, b) => panic!("step {step} heartbeat prod={a:?} ref={b:?}"),
                    }
                }
            }
            3 => {
                now = now.saturating_add(short_ttl());
                let _ = q.expire_due(now);
                let _ = refer.expire_due(now);
                let w = &ws[rng.pick(3)];
                let a = q.claim(w, now);
                let b = refer.claim(w, now);
                match (a, b) {
                    (Ok(Some(la)), Ok(Some(lb))) => {
                        assert_eq!(la.task_id.0, lb.task_id.0);
                        assert_eq!(la.attempt, lb.attempt);
                        last_lease = Some((la.task_id.clone(), la.worker_id.clone(), la.attempt));
                    }
                    (Ok(None), Ok(None)) => {}
                    (a, b) => panic!("step {step} reclaim prod={a:?} ref={b:?}"),
                }
            }
            4 => {
                if let Some((id, w, att)) = last_lease.clone() {
                    let body = format!("out-{step}").into_bytes();
                    let a = q.complete(&id, &w, att, &body, now);
                    let b = refer.complete(&id, &w, att, &body, now);
                    match (a, b) {
                        (Ok(()), Ok(())) => {}
                        (Err(_), Err(_)) => {}
                        (a, b) => panic!("step {step} complete prod={a:?} ref={b:?}"),
                    }
                }
            }
            _ => {
                if let Some((id, w, att)) = last_lease.clone() {
                    let a = q.fail(&id, &w, att, "prop", now);
                    let b = refer.fail(&id, &w, att, "prop", now);
                    match (a, b) {
                        (Ok(()), Ok(())) => {}
                        (Err(_), Err(_)) => {}
                        (a, b) => panic!("step {step} fail prod={a:?} ref={b:?}"),
                    }
                }
            }
        }
        compare_known(&q, &refer, &known);
    }
}

#[test]
fn default_ttl_matches_config_helper() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = QueueConfig::default();
    let ttl = cfg.lease_ttl_ms();
    let mut q = Queue::create(&dir, cfg).expect("create");
    q.enqueue(payload(0), 0).expect("enqueue");
    let lease = q.claim(&worker("w"), 100).expect("claim").expect("lease");
    assert_eq!(lease.expires_at, 100 + ttl);
}
