//! Group 2: `staleness` free function, including `FromTheFuture`.

mod common;
mod reference;

use common::*;
use prometheus_coordinator::{staleness as prod_staleness, CoordError};
use reference::staleness as ref_staleness;

fn both(trainer: u64, batch: u64) -> prometheus_coordinator::Result<u64> {
    let p = prod_staleness(trainer, batch);
    let r = ref_staleness(trainer, batch);
    match (p, r) {
        (Ok(a), Ok(b)) => {
            assert_eq!(a, b, "staleness({trainer}, {batch})");
            Ok(a)
        }
        (Err(e), Err(re)) => {
            assert_err_eq(&e, &re);
            Err(e)
        }
        (Ok(a), Err(re)) => panic!("prod Ok({a}), ref Err {re:?}"),
        (Err(e), Ok(b)) => panic!("prod Err {e:?}, ref Ok({b})"),
    }
}

#[test]
fn equal_versions_staleness_zero() {
    assert_eq!(both(0, 0).unwrap(), 0);
    assert_eq!(both(4, 4).unwrap(), 0);
    assert_eq!(both(99, 99).unwrap(), 0);
}

#[test]
fn batch_behind_trainer_is_difference() {
    assert_eq!(both(4, 0).unwrap(), 4);
    assert_eq!(both(4, 3).unwrap(), 1);
    assert_eq!(both(1, 0).unwrap(), 1);
    assert_eq!(both(10, 6).unwrap(), 4);
}

#[test]
fn batch_ahead_of_trainer_is_from_the_future() {
    match both(0, 1) {
        Err(CoordError::FromTheFuture) => {}
        other => panic!("expected FromTheFuture, got {other:?}"),
    }
    match both(3, 4) {
        Err(CoordError::FromTheFuture) => {}
        other => panic!("expected FromTheFuture, got {other:?}"),
    }
    match both(0, 99) {
        Err(CoordError::FromTheFuture) => {}
        other => panic!("expected FromTheFuture, got {other:?}"),
    }
}

#[test]
fn max_staleness_boundary_values() {
    // k==4 is kept by consume; k==5 is dropped. The free function itself
    // only reports the difference.
    assert_eq!(both(5, 1).unwrap(), 4);
    assert_eq!(both(5, 0).unwrap(), 5);
}
