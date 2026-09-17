//! Group: open replays pins from the log; get and unpin work after reopen.

mod common;
mod reference;

use common::{assert_not_found, cfg, event_types, fresh_store, pin};
use prometheus_cas::{Store, EVENT_PIN, EVENT_PUT, EVENT_UNPIN};
use reference::RefStore;

#[test]
fn reopen_sees_pins() {
    let h = fresh_store();
    let bytes = b"durable-pin";
    let digest;
    {
        let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
        digest = store.put(bytes).expect("put");
        store.pin(&digest, pin("ckpt")).expect("pin");
    }
    let mut store = Store::open(&h.dir, h.backends.clone(), cfg()).expect("open");
    let report = store.gc().expect("gc after reopen");
    assert_eq!(report.blobs_removed, 0, "replayed pin must keep the blob");
    assert_eq!(store.get(&digest).expect("get after reopen"), bytes);
    assert_eq!(store.replica_count(&digest).expect("replica_count"), 2);
}

#[test]
fn unpin_after_reopen_then_gc_removes() {
    let h = fresh_store();
    let digest;
    {
        let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
        digest = store.put(b"gone-after-unpin").expect("put");
        store.pin(&digest, pin("ckpt")).expect("pin");
    }
    let mut store = Store::open(&h.dir, h.backends.clone(), cfg()).expect("open");
    store.unpin(&pin("ckpt")).expect("unpin after reopen");
    let report = store.gc().expect("gc");
    assert_eq!(report.blobs_removed, 1);
    assert_not_found(&store.get(&digest).expect_err("gone"), &digest.0);
}

#[test]
fn get_after_reopen_without_pin() {
    let h = fresh_store();
    let digest;
    {
        let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
        digest = store.put(b"still-on-disk").expect("put");
    }
    let store = Store::open(&h.dir, h.backends.clone(), cfg()).expect("open");
    assert_eq!(
        store.get(&digest).expect("get after reopen"),
        b"still-on-disk"
    );
    assert_eq!(store.replica_count(&digest).expect("replica_count"), 2);
}

#[test]
fn reopen_log_contains_put_and_pin() {
    let h = fresh_store();
    {
        let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
        let d = store.put(b"e").expect("put");
        store.pin(&d, pin("n")).expect("pin");
        assert_eq!(
            event_types(&store),
            vec![EVENT_PUT.to_string(), EVENT_PIN.to_string()]
        );
    }
    let store = Store::open(&h.dir, h.backends.clone(), cfg()).expect("open");
    assert_eq!(
        event_types(&store),
        vec![EVENT_PUT.to_string(), EVENT_PIN.to_string()]
    );
}

#[test]
fn unpin_after_reopen_emits_cas_unpin() {
    let h = fresh_store();
    {
        let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
        let d = store.put(b"e").expect("put");
        store.pin(&d, pin("n")).expect("pin");
    }
    let mut store = Store::open(&h.dir, h.backends.clone(), cfg()).expect("open");
    store.unpin(&pin("n")).expect("unpin");
    assert_eq!(
        event_types(&store),
        vec![
            EVENT_PUT.to_string(),
            EVENT_PIN.to_string(),
            EVENT_UNPIN.to_string(),
        ]
    );
}

#[test]
fn reopen_then_pin_second_name_and_gc() {
    let h = fresh_store();
    let digest;
    {
        let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
        digest = store.put(b"two-names").expect("put");
        store.pin(&digest, pin("one")).expect("pin");
    }
    let mut store = Store::open(&h.dir, h.backends.clone(), cfg()).expect("open");
    store
        .pin(&digest, pin("two"))
        .expect("second pin after reopen");
    store.unpin(&pin("one")).expect("unpin one");
    let report = store.gc().expect("gc");
    assert_eq!(report.blobs_removed, 0);
    assert_eq!(store.get(&digest).expect("held by two"), b"two-names");
}

#[test]
fn reopen_matches_reference_pin_set() {
    let h = fresh_store();
    let mut refer = RefStore::with_default_replicas();
    let digest;
    {
        let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
        digest = store.put(b"ref").expect("put");
        refer.put(b"ref").expect("ref put");
        store.pin(&digest, pin("p")).expect("pin");
        refer.pin(&digest, pin("p")).expect("ref pin");
    }
    let mut store = Store::open(&h.dir, h.backends.clone(), cfg()).expect("open");
    let report = store.gc().expect("gc");
    let rreport = refer.gc().expect("ref gc");
    assert_eq!(report.blobs_removed, rreport.blobs_removed);
    assert_eq!(
        store.get(&digest).expect("get"),
        refer.get(&digest).expect("ref get")
    );
}
