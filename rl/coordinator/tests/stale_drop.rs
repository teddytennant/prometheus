//! Group 4: stale-prefix drop on consume. k==max_staleness is kept;
//! k==max_staleness+1 is dropped. Only a stale prefix is dropped.

mod common;
mod reference;

use common::*;
use prometheus_coordinator::CoordError;

fn consume_both(
    prod: &mut prometheus_coordinator::Coordinator,
    refer: &mut reference::RefCoordinator,
) -> prometheus_coordinator::Result<Option<prometheus_coordinator::RolloutBatch>> {
    let p = prod.consume_batch();
    let r = refer.consume_batch();
    match (p, r) {
        (Ok(a), Ok(b)) => {
            assert_eq!(a, b, "consumed batch");
            Ok(a)
        }
        (Err(e), Err(re)) => {
            assert_err_eq(&e, &re);
            Err(e)
        }
        other => panic!("consume mismatch {other:?}"),
    }
}

#[test]
fn staleness_equal_to_max_is_kept() {
    // trainer=0, batch v=0, k=0 <= max. After publishing up to 4, k=4 == max.
    let mut cfg = default_config();
    cfg.max_staleness = 4;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    assert_both_ok(
        prod.enqueue_batch(batch("keep", "p", 0, 2, false)),
        refer.enqueue_batch(batch("keep", "p", 0, 2, false)),
    );
    for v in 1..=4 {
        assert_both_ok(prod.publish_weights(v), refer.publish_weights(v));
    }
    assert_eq!(prod.trainer_version().unwrap(), 4);
    let got = consume_both(&mut prod, &mut refer).unwrap().expect("kept");
    assert_eq!(got.id, bid("keep"));
    assert_eq!(got.policy_version, 0);
    assert_eq!(prod.queue_depth().unwrap(), 0);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn staleness_one_past_max_is_dropped() {
    let mut cfg = default_config();
    cfg.max_staleness = 4;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    assert_both_ok(
        prod.enqueue_batch(batch("drop", "p", 0, 2, false)),
        refer.enqueue_batch(batch("drop", "p", 0, 2, false)),
    );
    for v in 1..=5 {
        assert_both_ok(prod.publish_weights(v), refer.publish_weights(v));
    }
    // k = 5-0 = 5 = max+1 → dropped, queue empty → None.
    match consume_both(&mut prod, &mut refer) {
        Ok(None) => {}
        other => panic!("expected Ok(None) after dropping stale, got {other:?}"),
    }
    assert_eq!(prod.queue_depth().unwrap(), 0);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn mixed_queue_drops_only_stale_prefix_then_returns_fresh() {
    let mut cfg = default_config();
    cfg.max_staleness = 1;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    // After publish 2: v0 has k=2 > 1 stale; v1 has k=1 kept; v0 again stale.
    assert_both_ok(
        prod.enqueue_batch(batch("stale-head", "p", 0, 1, false)),
        refer.enqueue_batch(batch("stale-head", "p", 0, 1, false)),
    );
    assert_both_ok(
        prod.enqueue_batch(batch("fresh", "p", 1, 1, true)),
        refer.enqueue_batch(batch("fresh", "p", 1, 1, true)),
    );
    assert_both_ok(
        prod.enqueue_batch(batch("stale-tail", "p", 0, 1, false)),
        refer.enqueue_batch(batch("stale-tail", "p", 0, 1, false)),
    );
    assert_both_ok(prod.publish_weights(1), refer.publish_weights(1));
    assert_both_ok(prod.publish_weights(2), refer.publish_weights(2));
    let got = consume_both(&mut prod, &mut refer).unwrap().expect("fresh");
    assert_eq!(got.id, bid("fresh"));
    assert_eq!(got.policy_version, 1);
    assert!(got.routing_present);
    // Prefix-only drop: the last stale remains after the first consume.
    assert_eq!(prod.queue_depth().unwrap(), 1);
    // Second consume sees an all-stale remainder (k=2 > 1): drop it, return None.
    match consume_both(&mut prod, &mut refer) {
        Ok(None) => {}
        other => panic!("stale tail must be dropped, got {other:?}"),
    }
    assert_eq!(prod.queue_depth().unwrap(), 0);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn all_remaining_stale_drops_them_and_returns_none() {
    let mut cfg = default_config();
    cfg.max_staleness = 0;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    assert_both_ok(
        prod.enqueue_batch(batch("a", "p", 0, 1, false)),
        refer.enqueue_batch(batch("a", "p", 0, 1, false)),
    );
    assert_both_ok(
        prod.enqueue_batch(batch("b", "p", 0, 1, false)),
        refer.enqueue_batch(batch("b", "p", 0, 1, false)),
    );
    assert_both_ok(prod.publish_weights(1), refer.publish_weights(1));
    match consume_both(&mut prod, &mut refer) {
        Ok(None) => {}
        other => panic!("expected None after dropping all stale, got {other:?}"),
    }
    assert_eq!(prod.queue_depth().unwrap(), 0);
    // Dropped ids may be reused.
    assert_both_ok(
        prod.enqueue_batch(batch("a", "p", 1, 1, false)),
        refer.enqueue_batch(batch("a", "p", 1, 1, false)),
    );
    let got = consume_both(&mut prod, &mut refer)
        .unwrap()
        .expect("reused");
    assert_eq!(got.id, bid("a"));
    assert_eq!(got.policy_version, 1);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn future_batch_at_front_is_from_the_future_queue_unchanged() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    assert_both_ok(
        prod.enqueue_batch(batch("future", "p", 1, 2, false)),
        refer.enqueue_batch(batch("future", "p", 1, 2, false)),
    );
    assert_both_ok(
        prod.enqueue_batch(batch("fresh", "p", 0, 2, false)),
        refer.enqueue_batch(batch("fresh", "p", 0, 2, false)),
    );
    let err = match consume_both(&mut prod, &mut refer) {
        Err(e) => e,
        other => panic!("expected FromTheFuture, got {other:?}"),
    };
    match err {
        CoordError::FromTheFuture => {}
        other => panic!("expected FromTheFuture, got {other:?}"),
    }
    assert_eq!(prod.queue_depth().unwrap(), 2);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn stale_prefix_then_future_does_not_drop_the_prefix() {
    // Scan hits FromTheFuture after a would-be-stale prefix: queue unchanged.
    let mut cfg = default_config();
    cfg.max_staleness = 0;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    assert_both_ok(
        prod.enqueue_batch(batch("stale", "p", 0, 1, false)),
        refer.enqueue_batch(batch("stale", "p", 0, 1, false)),
    );
    assert_both_ok(
        prod.enqueue_batch(batch("future", "p", 2, 1, false)),
        refer.enqueue_batch(batch("future", "p", 2, 1, false)),
    );
    assert_both_ok(prod.publish_weights(1), refer.publish_weights(1));
    match consume_both(&mut prod, &mut refer) {
        Err(CoordError::FromTheFuture) => {}
        other => panic!("expected FromTheFuture, got {other:?}"),
    }
    assert_eq!(prod.queue_depth().unwrap(), 2);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn max_staleness_zero_keeps_only_matching_version() {
    let mut cfg = default_config();
    cfg.max_staleness = 0;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    assert_both_ok(
        prod.enqueue_batch(batch("old", "p", 0, 1, false)),
        refer.enqueue_batch(batch("old", "p", 0, 1, false)),
    );
    assert_both_ok(
        prod.enqueue_batch(batch("cur", "p", 1, 1, false)),
        refer.enqueue_batch(batch("cur", "p", 1, 1, false)),
    );
    assert_both_ok(prod.publish_weights(1), refer.publish_weights(1));
    let got = consume_both(&mut prod, &mut refer).unwrap().expect("cur");
    assert_eq!(got.id, bid("cur"));
    assert_eq!(prod.queue_depth().unwrap(), 0);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn two_stale_then_fresh_drops_both_stale() {
    let mut cfg = default_config();
    cfg.max_staleness = 0;
    let racks = default_racks();
    let (mut prod, mut refer) = pair(cfg, racks.clone());
    assert_both_ok(
        prod.enqueue_batch(batch("s1", "p", 0, 1, false)),
        refer.enqueue_batch(batch("s1", "p", 0, 1, false)),
    );
    assert_both_ok(
        prod.enqueue_batch(batch("s2", "p", 0, 1, false)),
        refer.enqueue_batch(batch("s2", "p", 0, 1, false)),
    );
    assert_both_ok(
        prod.enqueue_batch(batch("ok", "p", 1, 1, true)),
        refer.enqueue_batch(batch("ok", "p", 1, 1, true)),
    );
    assert_both_ok(prod.publish_weights(1), refer.publish_weights(1));
    let got = consume_both(&mut prod, &mut refer).unwrap().expect("ok");
    assert_eq!(got.id, bid("ok"));
    assert_eq!(prod.queue_depth().unwrap(), 0);
    assert_prod_matches_ref(&prod, &refer, &racks);
}
