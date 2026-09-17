//! Group: named pins, GC of unpinned blobs, keep-if-pinned, reports.

mod common;
mod reference;

use common::{assert_duplicate_pin, assert_not_found, cfg, fresh_store, pin, walk_files, Harness};
use prometheus_cas::{Digest, Store};
use reference::{digest_hex, RefStore};

fn create(h: &Harness) -> Store {
    Store::create(&h.dir, h.backends.clone(), cfg()).expect("create")
}

#[test]
fn pin_then_gc_keeps_blob() {
    let h = fresh_store();
    let mut store = create(&h);
    let mut refer = RefStore::with_default_replicas();
    let bytes = b"keep-me";
    let d = store.put(bytes).expect("put");
    refer.put(bytes).expect("ref put");
    store.pin(&d, pin("named")).expect("pin");
    refer.pin(&d, pin("named")).expect("ref pin");
    let report = store.gc().expect("gc");
    let rreport = refer.gc().expect("ref gc");
    assert_eq!(report, rreport);
    assert_eq!(report.blobs_removed, 0);
    assert_eq!(report.bytes_freed, 0);
    assert_eq!(store.get(&d).expect("get after gc"), bytes);
    assert_eq!(store.replica_count(&d).expect("replica_count"), 2);
}

#[test]
fn unpin_then_gc_removes_from_both_backends() {
    let h = fresh_store();
    let mut store = create(&h);
    let mut refer = RefStore::with_default_replicas();
    let bytes = b"drop-me";
    let d = store.put(bytes).expect("put");
    refer.put(bytes).expect("ref put");
    store.pin(&d, pin("tmp")).expect("pin");
    refer.pin(&d, pin("tmp")).expect("ref pin");
    store.unpin(&pin("tmp")).expect("unpin");
    refer.unpin(&pin("tmp")).expect("ref unpin");
    let report = store.gc().expect("gc");
    let rreport = refer.gc().expect("ref gc");
    assert_eq!(report, rreport);
    assert_eq!(report.blobs_removed, 1);
    assert_eq!(
        report.bytes_freed,
        (bytes.len() * 2) as u64,
        "bytes_freed is physical (sum across replicas)"
    );
    let err = store.get(&d).expect_err("gone");
    assert_not_found(&err, &d.0);
    assert_eq!(store.replica_count(&d).expect("replica_count"), 0);
    for (i, backend) in h.backends.iter().enumerate() {
        let files = walk_files(backend);
        assert!(
            files.is_empty(),
            "backend {i} must have no blob files after unpin+gc, had {files:?}"
        );
    }
}

#[test]
fn gc_unpinned_put_removes_without_pin() {
    let h = fresh_store();
    let mut store = create(&h);
    let mut refer = RefStore::with_default_replicas();
    let bytes = b"orphan-from-the-start";
    let d = store.put(bytes).expect("put");
    refer.put(bytes).expect("ref put");
    let report = store.gc().expect("gc");
    let rreport = refer.gc().expect("ref gc");
    assert_eq!(report, rreport);
    assert_eq!(report.blobs_removed, 1);
    assert_eq!(report.bytes_freed, (bytes.len() * 2) as u64);
    assert_not_found(&store.get(&d).expect_err("gone"), &d.0);
}

#[test]
fn gc_empty_blob_counts_one_blob_zero_bytes() {
    let h = fresh_store();
    let mut store = create(&h);
    let mut refer = RefStore::with_default_replicas();
    let d = store.put(b"").expect("put empty");
    refer.put(b"").expect("ref put");
    let report = store.gc().expect("gc");
    let rreport = refer.gc().expect("ref gc");
    assert_eq!(report, rreport);
    assert_eq!(report.blobs_removed, 1);
    assert_eq!(report.bytes_freed, 0);
    assert_not_found(&store.get(&d).expect_err("gone"), &d.0);
}

#[test]
fn two_names_on_one_digest_need_both_unpinned() {
    let h = fresh_store();
    let mut store = create(&h);
    let d = store.put(b"shared").expect("put");
    store.pin(&d, pin("a")).expect("pin a");
    store.pin(&d, pin("b")).expect("pin b");
    let report = store.gc().expect("gc with two pins");
    assert_eq!(report.blobs_removed, 0);
    store.unpin(&pin("a")).expect("unpin a");
    let report = store.gc().expect("gc after one unpin");
    assert_eq!(report.blobs_removed, 0);
    assert_eq!(store.get(&d).expect("still pinned by b"), b"shared");
    store.unpin(&pin("b")).expect("unpin b");
    let report = store.gc().expect("gc after both unpins");
    assert_eq!(report.blobs_removed, 1);
    assert_not_found(&store.get(&d).expect_err("gone"), &d.0);
}

#[test]
fn pin_unknown_digest_is_not_found() {
    let h = fresh_store();
    let mut store = create(&h);
    let ghost = Digest(digest_hex(b"no-such-blob"));
    let err = store.pin(&ghost, pin("x")).expect_err("not found");
    assert_not_found(&err, &ghost.0);
}

#[test]
fn duplicate_pin_name_is_duplicate_pin() {
    let h = fresh_store();
    let mut store = create(&h);
    let d = store.put(b"blob").expect("put");
    store.pin(&d, pin("ckpt")).expect("pin");
    let err = store.pin(&d, pin("ckpt")).expect_err("duplicate");
    assert_duplicate_pin(&err, "ckpt");
    let d2 = store.put(b"other").expect("put other");
    let err = store
        .pin(&d2, pin("ckpt"))
        .expect_err("duplicate other digest");
    assert_duplicate_pin(&err, "ckpt");
}

#[test]
fn unpin_missing_name_is_not_found() {
    let h = fresh_store();
    let mut store = create(&h);
    let err = store.unpin(&pin("missing")).expect_err("not found");
    assert_not_found(&err, "missing");
}

#[test]
fn unpin_then_reuse_name() {
    let h = fresh_store();
    let mut store = create(&h);
    let d = store.put(b"v1").expect("put");
    store.pin(&d, pin("slot")).expect("pin");
    store.unpin(&pin("slot")).expect("unpin");
    store.pin(&d, pin("slot")).expect("re-pin");
    let report = store.gc().expect("gc");
    assert_eq!(report.blobs_removed, 0);
    assert_eq!(store.get(&d).expect("get"), b"v1");
}

#[test]
fn gc_two_unpinned_blobs_reports_both() {
    let h = fresh_store();
    let mut store = create(&h);
    let mut refer = RefStore::with_default_replicas();
    let a = b"aaaa";
    let b = b"bbbbbbb";
    store.put(a).expect("put a");
    refer.put(a).expect("ref a");
    store.put(b).expect("put b");
    refer.put(b).expect("ref b");
    let report = store.gc().expect("gc");
    let rreport = refer.gc().expect("ref gc");
    assert_eq!(report, rreport);
    assert_eq!(report.blobs_removed, 2);
    assert_eq!(report.bytes_freed, ((a.len() + b.len()) * 2) as u64);
}

#[test]
fn gc_mixed_pinned_and_unpinned() {
    let h = fresh_store();
    let mut store = create(&h);
    let keep = store.put(b"keep").expect("put keep");
    let drop = store.put(b"drop").expect("put drop");
    store.pin(&keep, pin("live")).expect("pin");
    let report = store.gc().expect("gc");
    assert_eq!(report.blobs_removed, 1);
    assert_eq!(report.bytes_freed, (b"drop".len() * 2) as u64);
    assert_eq!(store.get(&keep).expect("kept"), b"keep");
    assert_not_found(&store.get(&drop).expect_err("dropped"), &drop.0);
}
