//! Group: locked EventLog event_type strings after put / pin / unpin / gc.

mod common;
mod reference;

use common::{cfg, event_types, fresh_store, pin, Harness};
use prometheus_cas::{Store, EVENT_GC, EVENT_PIN, EVENT_PUT, EVENT_UNPIN};
use reference::RefStore;

fn create(h: &Harness) -> Store {
    Store::create(&h.dir, h.backends.clone(), cfg()).expect("create")
}

#[test]
fn put_emits_cas_put() {
    let h = fresh_store();
    let mut store = create(&h);
    let mut refer = RefStore::with_default_replicas();
    store.put(b"evt").expect("put");
    refer.put(b"evt").expect("ref put");
    assert_eq!(event_types(&store), refer.event_types());
    assert_eq!(event_types(&store), vec![EVENT_PUT.to_string()]);
    assert_eq!(event_types(&store)[0], "cas.put");
}

#[test]
fn pin_emits_cas_pin_after_put() {
    let h = fresh_store();
    let mut store = create(&h);
    let mut refer = RefStore::with_default_replicas();
    let d = store.put(b"evt").expect("put");
    refer.put(b"evt").expect("ref put");
    store.pin(&d, pin("n")).expect("pin");
    refer.pin(&d, pin("n")).expect("ref pin");
    assert_eq!(event_types(&store), refer.event_types());
    assert_eq!(
        event_types(&store),
        vec![EVENT_PUT.to_string(), EVENT_PIN.to_string()]
    );
    assert_eq!(event_types(&store)[1], "cas.pin");
}

#[test]
fn unpin_emits_cas_unpin() {
    let h = fresh_store();
    let mut store = create(&h);
    let d = store.put(b"evt").expect("put");
    store.pin(&d, pin("n")).expect("pin");
    store.unpin(&pin("n")).expect("unpin");
    assert_eq!(
        event_types(&store),
        vec![
            EVENT_PUT.to_string(),
            EVENT_PIN.to_string(),
            EVENT_UNPIN.to_string(),
        ]
    );
    assert_eq!(event_types(&store)[2], "cas.unpin");
}

#[test]
fn gc_that_removes_blobs_emits_cas_gc() {
    let h = fresh_store();
    let mut store = create(&h);
    let mut refer = RefStore::with_default_replicas();
    store.put(b"evt").expect("put");
    refer.put(b"evt").expect("ref put");
    store.gc().expect("gc");
    refer.gc().expect("ref gc");
    assert_eq!(event_types(&store), refer.event_types());
    assert_eq!(
        event_types(&store),
        vec![EVENT_PUT.to_string(), EVENT_GC.to_string()]
    );
    assert_eq!(event_types(&store)[1], "cas.gc");
}

#[test]
fn failed_pin_does_not_emit() {
    let h = fresh_store();
    let mut store = create(&h);
    let ghost = prometheus_cas::Digest(reference::digest_hex(b"nope"));
    let _ = store.pin(&ghost, pin("x")).expect_err("not found");
    assert!(
        event_types(&store).is_empty(),
        "failed pin must not append, got {:?}",
        event_types(&store)
    );
}

#[test]
fn store_log_after_put_and_pin_uses_locked_strings() {
    let h = fresh_store();
    let mut store = create(&h);
    let d = store.put(b"payload").expect("put");
    store.pin(&d, pin("artifact")).expect("pin");
    let types = event_types(&store);
    assert_eq!(types, vec!["cas.put".to_string(), "cas.pin".to_string()]);
}
