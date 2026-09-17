//! Group: fault injection. DigestMismatch, UnderReplicated, orphan GC, torn log.

mod common;
mod reference;

use common::{
    assert_digest_mismatch, assert_not_found, assert_under_replicated, cfg, cfg_min,
    corrupt_all_files, events_jsonl, fresh_store, fresh_store_n, RestoreReadonly,
};
use prometheus_cas::Store;
use reference::digest_hex;
use std::fs;

#[test]
fn get_corrupted_replicas_is_digest_mismatch() {
    let h = fresh_store();
    let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
    let bytes = b"integrity";
    let d = store.put(bytes).expect("put");
    for backend in &h.backends {
        corrupt_all_files(backend);
    }
    let err = store.get(&d).expect_err("mismatch");
    assert_digest_mismatch(&err, &d.0);
}

#[test]
fn put_one_readonly_backend_of_two_is_under_replicated() {
    let h = fresh_store();
    let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
    let _guard = RestoreReadonly::make(&h.backends[1]);
    let err = store.put(b"partial").expect_err("under-replicated");
    assert_under_replicated(&err, 1, 2);
}

#[test]
fn put_succeeds_if_two_of_three_backends_write() {
    let h = fresh_store_n(3);
    let mut store = Store::create(&h.dir, h.backends.clone(), cfg_min(2)).expect("create");
    let _guard = RestoreReadonly::make(&h.backends[2]);
    let d = store
        .put(b"quorum")
        .expect("put must succeed with 2 of 3 replicas");
    assert_eq!(store.get(&d).expect("get"), b"quorum");
    assert_eq!(store.replica_count(&d).expect("replica_count"), 2);
}

#[test]
fn failed_put_orphans_are_removed_by_gc() {
    let h = fresh_store();
    let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
    let bytes = b"orphan-blob";
    {
        let _guard = RestoreReadonly::make(&h.backends[1]);
        let err = store.put(bytes).expect_err("under-replicated");
        assert_under_replicated(&err, 1, 2);
    }
    let d = prometheus_cas::Digest(digest_hex(bytes));
    let report = store.gc().expect("gc orphans");
    let _ = report;
    assert_not_found(&store.get(&d).expect_err("orphan gone"), &d.0);
    assert_eq!(store.replica_count(&d).expect("replica_count"), 0);
}

#[test]
fn open_fails_on_corrupt_events_jsonl() {
    let h = fresh_store();
    {
        let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
        store.put(b"logged").expect("put");
    }
    fs::write(events_jsonl(&h.dir), b"{not json\n").expect("corrupt log");
    common::assert_err(
        Store::open(&h.dir, h.backends.clone(), cfg()),
        "open corrupt log",
    );
}

#[test]
fn get_after_one_replica_corrupted_does_not_return_wrong_bytes() {
    let h = fresh_store();
    let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
    let bytes = b"maybe-either-replica";
    let d = store.put(bytes).expect("put");
    corrupt_all_files(&h.backends[0]);
    match store.get(&d) {
        Ok(got) => assert_eq!(got, bytes, "if get succeeds it must be the real bytes"),
        Err(e) => assert_digest_mismatch(&e, &d.0),
    }
}
