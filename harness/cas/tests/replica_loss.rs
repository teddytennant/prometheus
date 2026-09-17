//! Group: replica loss. Delete one backend's blob; get still works; replica_count is 1.

mod common;
mod reference;

use common::{cfg, fresh_store, pin, wipe_contents};
use prometheus_cas::Store;
use reference::RefStore;
use std::fs;

#[test]
fn delete_one_backend_contents_get_still_works() {
    let h = fresh_store();
    let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
    let bytes = b"survive-replica-loss";
    let d = store.put(bytes).expect("put");
    assert_eq!(store.replica_count(&d).expect("before"), 2);
    wipe_contents(&h.backends[0]);
    assert_eq!(store.get(&d).expect("get from remaining replica"), bytes);
    assert_eq!(store.replica_count(&d).expect("after loss"), 1);
}

#[test]
fn delete_other_backend_contents_get_still_works() {
    let h = fresh_store();
    let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
    let bytes = b"other-side";
    let d = store.put(bytes).expect("put");
    wipe_contents(&h.backends[1]);
    assert_eq!(store.get(&d).expect("get"), bytes);
    assert_eq!(store.replica_count(&d).expect("after loss"), 1);
}

#[test]
fn open_with_missing_replica_dir_still_gets() {
    let h = fresh_store();
    let bytes = b"lost-dir";
    let digest;
    {
        let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
        digest = store.put(bytes).expect("put");
        store.pin(&digest, pin("keep")).expect("pin");
    }
    fs::remove_dir_all(&h.backends[1]).expect("rm replica dir");
    let store = Store::open(&h.dir, h.backends.clone(), cfg())
        .expect("open must accept a missing replica directory");
    assert_eq!(store.get(&digest).expect("get from remaining"), bytes);
    assert_eq!(store.replica_count(&digest).expect("replica_count"), 1);
}

#[test]
fn open_with_replacement_empty_backend() {
    let h = fresh_store();
    let bytes = b"replacement-path";
    let digest;
    {
        let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
        digest = store.put(bytes).expect("put");
    }
    let replacement = h.parent.path().join("b-new");
    fs::create_dir(&replacement).expect("mkdir replacement");
    let backends = vec![h.backends[0].clone(), replacement];
    let store = Store::open(&h.dir, backends, cfg()).expect("open replacement");
    assert_eq!(store.get(&digest).expect("get"), bytes);
    assert_eq!(store.replica_count(&digest).expect("replica_count"), 1);
}

#[test]
fn both_replicas_wiped_is_not_found() {
    let h = fresh_store();
    let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
    let d = store.put(b"gone").expect("put");
    wipe_contents(&h.backends[0]);
    wipe_contents(&h.backends[1]);
    common::assert_not_found(&store.get(&d).expect_err("not found"), &d.0);
    assert_eq!(store.replica_count(&d).expect("replica_count"), 0);
}

#[test]
fn replica_loss_matches_reference_drop() {
    let h = fresh_store();
    let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
    let mut refer = RefStore::with_default_replicas();
    let bytes = b"ref-loss";
    let d = store.put(bytes).expect("put");
    refer.put(bytes).expect("ref put");
    wipe_contents(&h.backends[0]);
    refer.drop_replica(0, &d);
    assert_eq!(
        store.replica_count(&d).expect("count"),
        refer.replica_count(&d).expect("ref count")
    );
    assert_eq!(store.get(&d).expect("get"), refer.get(&d).expect("ref get"));
}
