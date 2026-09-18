//! Group 3: enqueue/consume FIFO, prompt_id and routing_present pass-through,
//! EmptyGroup, DuplicateBatch, reuse of consumed ids.

mod common;
mod reference;

use common::*;
use prometheus_coordinator::{CoordError, GROUP_SIZE};

#[test]
fn fifo_consume_returns_oldest_first() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    assert_both_ok(
        prod.enqueue_batch(batch("a", "p", 0, 3, false)),
        refer.enqueue_batch(batch("a", "p", 0, 3, false)),
    );
    assert_both_ok(
        prod.enqueue_batch(batch("b", "p", 0, 5, true)),
        refer.enqueue_batch(batch("b", "p", 0, 5, true)),
    );
    assert_both_ok(
        prod.enqueue_batch(batch("c", "q", 0, 1, false)),
        refer.enqueue_batch(batch("c", "q", 0, 1, false)),
    );
    assert_eq!(prod.queue_depth().unwrap(), 3);
    assert_prod_matches_ref(&prod, &refer, &racks);

    let got = {
        let p = prod.consume_batch();
        let r = refer.consume_batch();
        match (p, r) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a, b);
                a
            }
            other => panic!("consume mismatch {other:?}"),
        }
    };
    let got = got.expect("first");
    assert_eq!(got.id, bid("a"));
    assert_eq!(got.prompt_id, pid("p"));
    assert_eq!(got.n_samples, 3);
    assert!(!got.routing_present);
    assert_eq!(prod.queue_depth().unwrap(), 2);

    let got = {
        let p = prod.consume_batch();
        let r = refer.consume_batch();
        match (p, r) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a, b);
                a
            }
            other => panic!("consume mismatch {other:?}"),
        }
    }
    .expect("second");
    assert_eq!(got.id, bid("b"));
    assert!(got.routing_present);

    let got = {
        let p = prod.consume_batch();
        let r = refer.consume_batch();
        match (p, r) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a, b);
                a
            }
            other => panic!("consume mismatch {other:?}"),
        }
    }
    .expect("third");
    assert_eq!(got.id, bid("c"));
    assert_eq!(got.prompt_id, pid("q"));
    assert_eq!(prod.queue_depth().unwrap(), 0);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn consume_empty_is_none() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    match (prod.consume_batch(), refer.consume_batch()) {
        (Ok(None), Ok(None)) => {}
        other => panic!("expected Ok(None), got {other:?}"),
    }
    assert_eq!(prod.queue_depth().unwrap(), 0);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn prompt_id_shared_across_group_is_passed_through() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    let prompt = "shared-prompt";
    assert_both_ok(
        prod.enqueue_batch(batch("g0", prompt, 0, 8, false)),
        refer.enqueue_batch(batch("g0", prompt, 0, 8, false)),
    );
    assert_both_ok(
        prod.enqueue_batch(batch("g1", prompt, 0, 8, false)),
        refer.enqueue_batch(batch("g1", prompt, 0, 8, false)),
    );
    let a = {
        let p = prod.consume_batch().unwrap();
        let r = refer.consume_batch().unwrap();
        assert_eq!(p, r);
        p.unwrap()
    };
    let b = {
        let p = prod.consume_batch().unwrap();
        let r = refer.consume_batch().unwrap();
        assert_eq!(p, r);
        p.unwrap()
    };
    assert_eq!(a.prompt_id, pid(prompt));
    assert_eq!(b.prompt_id, pid(prompt));
    assert_ne!(a.id, b.id);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn routing_present_true_and_false_round_trip() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    assert_both_ok(
        prod.enqueue_batch(batch("with", "p", 0, 2, true)),
        refer.enqueue_batch(batch("with", "p", 0, 2, true)),
    );
    assert_both_ok(
        prod.enqueue_batch(batch("without", "p", 0, 2, false)),
        refer.enqueue_batch(batch("without", "p", 0, 2, false)),
    );
    let a = prod.consume_batch().unwrap().unwrap();
    let ar = refer.consume_batch().unwrap().unwrap();
    assert_eq!(a, ar);
    assert!(a.routing_present);
    let b = prod.consume_batch().unwrap().unwrap();
    let br = refer.consume_batch().unwrap().unwrap();
    assert_eq!(b, br);
    assert!(!b.routing_present);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn n_samples_need_not_equal_group_size() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    for (id, n) in [
        ("one", 1u32),
        ("fifteen", 15),
        ("seventeen", 17),
        ("many", 100),
    ] {
        assert_ne!(n, GROUP_SIZE);
        assert_both_ok(
            prod.enqueue_batch(batch(id, "p", 0, n, false)),
            refer.enqueue_batch(batch(id, "p", 0, n, false)),
        );
    }
    // GROUP_SIZE itself is also allowed.
    assert_both_ok(
        prod.enqueue_batch(batch("exact", "p", 0, GROUP_SIZE, true)),
        refer.enqueue_batch(batch("exact", "p", 0, GROUP_SIZE, true)),
    );
    assert_eq!(prod.queue_depth().unwrap(), 5);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn empty_group_rejected() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    let err = assert_both_err(
        prod.enqueue_batch(batch("z", "p", 0, 0, false)),
        refer.enqueue_batch(batch("z", "p", 0, 0, false)),
    );
    match err {
        CoordError::EmptyGroup => {}
        other => panic!("expected EmptyGroup, got {other:?}"),
    }
    assert_eq!(prod.queue_depth().unwrap(), 0);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn empty_group_wins_over_empty_ids() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    let b = batch("", "", 0, 0, false);
    let err = assert_both_err(prod.enqueue_batch(b.clone()), refer.enqueue_batch(b));
    match err {
        CoordError::EmptyGroup => {}
        other => panic!("expected EmptyGroup before Message, got {other:?}"),
    }
}

#[test]
fn empty_batch_id_is_message_containing_id() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    let b = batch("", "prompt", 0, 4, false);
    let err = assert_both_err(prod.enqueue_batch(b.clone()), refer.enqueue_batch(b));
    assert_message_contains(&err, "id");
    assert_eq!(prod.queue_depth().unwrap(), 0);
}

#[test]
fn empty_prompt_id_is_message_containing_prompt() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    let b = batch("b1", "", 0, 4, false);
    let err = assert_both_err(prod.enqueue_batch(b.clone()), refer.enqueue_batch(b));
    assert_message_contains(&err, "prompt");
    assert_eq!(prod.queue_depth().unwrap(), 0);
}

#[test]
fn both_ids_empty_reports_batch_id_first() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    let b = batch("", "", 0, 2, false);
    let err = assert_both_err(prod.enqueue_batch(b.clone()), refer.enqueue_batch(b));
    assert_message_contains(&err, "id");
}

#[test]
fn duplicate_batch_id_still_in_queue() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    assert_both_ok(
        prod.enqueue_batch(batch("dup", "p", 0, 2, false)),
        refer.enqueue_batch(batch("dup", "p", 0, 2, false)),
    );
    let err = assert_both_err(
        prod.enqueue_batch(batch("dup", "other", 0, 3, true)),
        refer.enqueue_batch(batch("dup", "other", 0, 3, true)),
    );
    match err {
        CoordError::DuplicateBatch(id) => assert_eq!(id, bid("dup")),
        other => panic!("expected DuplicateBatch, got {other:?}"),
    }
    assert_eq!(prod.queue_depth().unwrap(), 1);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn consumed_batch_id_may_be_reused() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    assert_both_ok(
        prod.enqueue_batch(batch("reuse", "p", 0, 2, false)),
        refer.enqueue_batch(batch("reuse", "p", 0, 2, false)),
    );
    assert_both_ok(
        prod.consume_batch().map(|_| ()),
        refer.consume_batch().map(|_| ()),
    );
    assert_eq!(prod.queue_depth().unwrap(), 0);
    assert_both_ok(
        prod.enqueue_batch(batch("reuse", "p2", 0, 7, true)),
        refer.enqueue_batch(batch("reuse", "p2", 0, 7, true)),
    );
    let got = prod.consume_batch().unwrap().unwrap();
    let got_r = refer.consume_batch().unwrap().unwrap();
    assert_eq!(got, got_r);
    assert_eq!(got.prompt_id, pid("p2"));
    assert_eq!(got.n_samples, 7);
    assert!(got.routing_present);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn enqueue_does_not_validate_policy_version() {
    // Future versions sit in the queue; consume (not enqueue) reports FromTheFuture.
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    assert_both_ok(
        prod.enqueue_batch(batch("future", "p", 9, 2, false)),
        refer.enqueue_batch(batch("future", "p", 9, 2, false)),
    );
    assert_eq!(prod.queue_depth().unwrap(), 1);
    assert_prod_matches_ref(&prod, &refer, &racks);
}
