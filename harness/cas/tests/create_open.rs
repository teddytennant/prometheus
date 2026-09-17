//! Group: Store::create / open layout, TooFewBackends, exists / missing.

mod common;
mod reference;

use common::{
    assert_err, assert_too_few, cfg, cfg_min, events_jsonl, fresh_store, fresh_store_n, log_dir,
    unwrap_err,
};
use prometheus_cas::{Store, StoreConfig, MIN_REPLICAS};
use std::fs;

#[test]
fn create_empty_store_has_log_and_no_events() {
    let h = fresh_store();
    let store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
    reference::assert_self_consistent_goldens();
    assert!(h.dir.is_dir(), "create must mkdir the store dir");
    assert!(
        log_dir(&h.dir).is_dir(),
        "create must create H2 EventLog at dir/log/"
    );
    assert!(
        events_jsonl(&h.dir).is_file(),
        "create must materialize dir/log/events.jsonl"
    );
    assert_eq!(store.log().len(), 0, "no events on create");
    assert!(store.log().is_empty());
    assert_eq!(store.config().min_replicas, MIN_REPLICAS);
    assert_eq!(store.backends(), h.backends.as_slice());
    assert_eq!(store.dir(), h.dir.as_path());
    assert_eq!(store.log().dir(), log_dir(&h.dir).as_path());
}

#[test]
fn open_after_create_replays_empty() {
    let h = fresh_store();
    {
        let store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
        assert_eq!(store.log().len(), 0);
    }
    let store = Store::open(&h.dir, h.backends.clone(), cfg()).expect("open empty");
    assert_eq!(store.log().len(), 0);
    assert_eq!(store.backends(), h.backends.as_slice());
}

#[test]
fn create_fails_if_directory_exists() {
    let h = fresh_store();
    fs::create_dir(&h.dir).expect("mkdir");
    assert_err(
        Store::create(&h.dir, h.backends.clone(), cfg()),
        "create existing dir",
    );
}

#[test]
fn create_fails_if_path_is_a_file() {
    let h = fresh_store();
    fs::write(&h.dir, b"not a dir").expect("write file");
    assert_err(
        Store::create(&h.dir, h.backends.clone(), cfg()),
        "create on file",
    );
}

#[test]
fn create_second_time_fails() {
    let h = fresh_store();
    let _store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
    assert_err(
        Store::create(&h.dir, h.backends.clone(), cfg()),
        "create if exists",
    );
}

#[test]
fn open_fails_if_dir_missing() {
    let h = fresh_store();
    assert_err(
        Store::open(&h.dir, h.backends.clone(), cfg()),
        "open missing dir",
    );
}

#[test]
fn create_zero_backends_is_too_few_and_does_not_mkdir() {
    let h = fresh_store_n(0);
    let err = unwrap_err(Store::create(&h.dir, h.backends.clone(), cfg()), "too few");
    assert_too_few(&err, 0);
    assert!(
        !h.dir.exists(),
        "TooFewBackends create must not create the store dir"
    );
}

#[test]
fn create_one_backend_is_too_few() {
    let h = fresh_store_n(1);
    let err = unwrap_err(Store::create(&h.dir, h.backends.clone(), cfg()), "too few");
    assert_too_few(&err, 1);
    assert!(!h.dir.exists(), "TooFewBackends create must not mkdir");
}

#[test]
fn create_two_backends_min_replicas_three_is_too_few() {
    let h = fresh_store_n(2);
    let err = unwrap_err(
        Store::create(&h.dir, h.backends.clone(), cfg_min(3)),
        "too few",
    );
    assert_too_few(&err, 2);
}

#[test]
fn open_one_backend_is_too_few() {
    let h = fresh_store();
    {
        let _store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
    }
    let one = vec![h.backends[0].clone()];
    let err = unwrap_err(Store::open(&h.dir, one, cfg()), "too few on open");
    assert_too_few(&err, 1);
}

#[test]
fn open_zero_backends_is_too_few() {
    let h = fresh_store();
    {
        let _store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
    }
    let err = unwrap_err(
        Store::open(&h.dir, Vec::new(), StoreConfig::default()),
        "too few",
    );
    assert_too_few(&err, 0);
}

#[test]
fn create_three_backends_is_ok() {
    let h = fresh_store_n(3);
    let store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create 3");
    assert_eq!(store.backends().len(), 3);
    assert_eq!(store.log().len(), 0);
}
