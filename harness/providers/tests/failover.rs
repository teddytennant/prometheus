//! Group: failover order, NotFound, idempotence, wait_backoff never errors.

mod common;
mod reference;

use common::{assert_backoff, assert_no_provider, assert_not_found, failover, ids, pid};
use prometheus_providers::Failover;
use reference::RefFailover;
use std::time::Duration;

fn pair(names: &[&str]) -> (Failover, RefFailover) {
    (failover(names), RefFailover::new(ids(names)))
}

#[test]
fn current_is_first_provider_not_down() {
    let (mut prod, mut refer) = pair(&["a", "b", "c"]);
    assert_eq!(prod.current().expect("current"), &pid("a"));
    assert_eq!(refer.current().expect("ref"), &pid("a"));
    prod.mark_down(&pid("a")).expect("down a");
    refer.mark_down(&pid("a")).expect("ref down a");
    assert_eq!(prod.current().expect("current"), &pid("b"));
    assert_eq!(refer.current().expect("ref"), &pid("b"));
    prod.mark_down(&pid("b")).expect("down b");
    refer.mark_down(&pid("b")).expect("ref down b");
    assert_eq!(prod.current().expect("current"), &pid("c"));
    assert_eq!(refer.current().expect("ref"), &pid("c"));
}

#[test]
fn current_skips_down_preserving_original_order() {
    let (mut prod, mut refer) = pair(&["a", "b", "c"]);
    prod.mark_down(&pid("b")).unwrap();
    refer.mark_down(&pid("b")).unwrap();
    assert_eq!(prod.current().unwrap(), &pid("a"));
    prod.mark_down(&pid("a")).unwrap();
    refer.mark_down(&pid("a")).unwrap();
    assert_eq!(prod.current().unwrap(), &pid("c"));
    assert_eq!(prod.current().unwrap(), refer.current().unwrap());
}

#[test]
fn all_down_is_no_provider() {
    let (mut prod, mut refer) = pair(&["a", "b"]);
    prod.mark_down(&pid("a")).unwrap();
    prod.mark_down(&pid("b")).unwrap();
    refer.mark_down(&pid("a")).unwrap();
    refer.mark_down(&pid("b")).unwrap();
    assert_no_provider(&prod.current().unwrap_err());
    assert_no_provider(&refer.current().unwrap_err());
}

#[test]
fn empty_providers_current_is_no_provider() {
    let (prod, refer) = pair(&[]);
    assert_no_provider(&prod.current().unwrap_err());
    assert_no_provider(&refer.current().unwrap_err());
}

#[test]
fn mark_down_unknown_is_not_found() {
    let (mut prod, mut refer) = pair(&["a"]);
    let err = prod.mark_down(&pid("ghost")).expect_err("unknown");
    let rerr = refer.mark_down(&pid("ghost")).expect_err("ref unknown");
    assert_not_found(&err, &pid("ghost"));
    assert_not_found(&rerr, &pid("ghost"));
}

#[test]
fn mark_down_already_down_is_idempotent() {
    let (mut prod, mut refer) = pair(&["a", "b"]);
    prod.mark_down(&pid("a")).unwrap();
    prod.mark_down(&pid("a")).unwrap();
    refer.mark_down(&pid("a")).unwrap();
    refer.mark_down(&pid("a")).unwrap();
    assert_eq!(prod.current().unwrap(), &pid("b"));
    assert_eq!(refer.current().unwrap(), &pid("b"));
    // unique down count: backoff for 1, not 2
    assert_backoff(prod.wait_backoff(), 1);
    assert_eq!(prod.wait_backoff(), refer.wait_backoff());
}

#[test]
fn mark_up_restores_first_slot() {
    let (mut prod, mut refer) = pair(&["a", "b", "c"]);
    prod.mark_down(&pid("a")).unwrap();
    refer.mark_down(&pid("a")).unwrap();
    assert_eq!(prod.current().unwrap(), &pid("b"));
    prod.mark_up(&pid("a")).unwrap();
    refer.mark_up(&pid("a")).unwrap();
    assert_eq!(prod.current().unwrap(), &pid("a"));
    assert_eq!(prod.current().unwrap(), refer.current().unwrap());
}

#[test]
fn mark_up_unknown_is_not_found() {
    let (mut prod, mut refer) = pair(&["a"]);
    let err = prod.mark_up(&pid("ghost")).expect_err("unknown");
    let rerr = refer.mark_up(&pid("ghost")).expect_err("ref unknown");
    assert_not_found(&err, &pid("ghost"));
    assert_not_found(&rerr, &pid("ghost"));
}

#[test]
fn mark_up_of_already_up_is_ok() {
    let (mut prod, mut refer) = pair(&["a", "b"]);
    prod.mark_up(&pid("a")).unwrap();
    refer.mark_up(&pid("a")).unwrap();
    assert_eq!(prod.current().unwrap(), &pid("a"));
}

#[test]
fn wait_backoff_zero_down_is_100ms() {
    let (prod, refer) = pair(&["a"]);
    let d = prod.wait_backoff();
    assert_eq!(d, Duration::from_millis(100));
    assert_eq!(d, refer.wait_backoff());
    assert_backoff(d, 0);
}

#[test]
fn wait_backoff_doubles_per_down_capped_at_eight() {
    let names: Vec<&str> = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"].into();
    let (mut prod, mut refer) = pair(&names);
    for (i, name) in names.iter().enumerate() {
        prod.mark_down(&pid(name)).unwrap();
        refer.mark_down(&pid(name)).unwrap();
        let d = prod.wait_backoff();
        assert_eq!(d, refer.wait_backoff());
        assert_backoff(d, i + 1);
    }
    // 9 down → still 2^8
    assert_eq!(prod.wait_backoff(), Duration::from_millis(100 * 256));
}

#[test]
fn wait_backoff_never_errors_when_all_down() {
    let (mut prod, mut refer) = pair(&["a", "b", "c"]);
    for name in ["a", "b", "c"] {
        prod.mark_down(&pid(name)).unwrap();
        refer.mark_down(&pid(name)).unwrap();
    }
    assert_no_provider(&prod.current().unwrap_err());
    // The contract: all-down is not process exit. Backoff is always a Duration.
    let d = prod.wait_backoff();
    assert_eq!(d, Duration::from_millis(100 * 8));
    assert_eq!(d, refer.wait_backoff());
    let again = prod.wait_backoff();
    assert_eq!(again, d);
}
