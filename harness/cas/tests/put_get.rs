//! Group: put/get roundtrip, empty bytes, 1MB blob, digest hex, replica_count.

mod common;
mod reference;

use common::{cfg, fresh_store, fresh_store_n, is_sha256_hex, walk_files, Harness};
use prometheus_cas::{Digest, Store};
use reference::{digest_hex, RefStore, GOLDEN_ABC, GOLDEN_EMPTY};

fn create(h: &Harness) -> Store {
    Store::create(&h.dir, h.backends.clone(), cfg()).expect("create")
}

#[test]
fn put_get_roundtrip_matches_reference() {
    reference::assert_self_consistent_goldens();
    let h = fresh_store();
    let mut store = create(&h);
    let mut refer = RefStore::with_default_replicas();
    let bytes = b"hello-cas";
    let d = store.put(bytes).expect("put");
    let rd = refer.put(bytes).expect("ref put");
    assert_eq!(d, rd);
    assert_eq!(d.0, digest_hex(bytes));
    assert!(is_sha256_hex(&d.0), "digest must be lowercase sha256 hex");
    assert_eq!(store.get(&d).expect("get"), bytes);
    assert_eq!(refer.get(&d).expect("ref get"), bytes);
}

#[test]
fn put_empty_bytes_is_nist_sha256() {
    let h = fresh_store();
    let mut store = create(&h);
    let d = store.put(b"").expect("put empty");
    assert_eq!(d.0, GOLDEN_EMPTY);
    assert_eq!(store.get(&d).expect("get empty"), b"");
    assert_eq!(store.replica_count(&d).expect("replica_count"), 2);
}

#[test]
fn put_abc_matches_nist_sha256() {
    let h = fresh_store();
    let mut store = create(&h);
    let d = store.put(b"abc").expect("put abc");
    assert_eq!(d.0, GOLDEN_ABC);
    assert_eq!(digest_hex(b"abc"), GOLDEN_ABC);
    assert_eq!(store.get(&d).expect("get abc"), b"abc");
}

#[test]
fn put_one_megabyte_roundtrip() {
    let h = fresh_store();
    let mut store = create(&h);
    let mut refer = RefStore::with_default_replicas();
    let bytes: Vec<u8> = (0..1024 * 1024).map(|i| (i % 256) as u8).collect();
    let d = store.put(&bytes).expect("put 1MB");
    let rd = refer.put(&bytes).expect("ref put 1MB");
    assert_eq!(d, rd);
    assert_eq!(d.0, digest_hex(&bytes));
    assert_eq!(store.get(&d).expect("get 1MB"), bytes);
    assert_eq!(store.replica_count(&d).expect("replica_count"), 2);
}

#[test]
fn put_writes_both_backends_replica_count_two() {
    let h = fresh_store();
    let mut store = create(&h);
    let d = store.put(b"replicated").expect("put");
    assert_eq!(store.replica_count(&d).expect("replica_count"), 2);
    for (i, backend) in h.backends.iter().enumerate() {
        let files = walk_files(backend);
        assert!(
            !files.is_empty(),
            "backend {i} {} must contain the blob after put",
            backend.display()
        );
    }
}

#[test]
fn put_three_backends_replica_count_three() {
    let h = fresh_store_n(3);
    let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
    let mut refer = RefStore::new(3, cfg());
    let d = store.put(b"triple").expect("put");
    let rd = refer.put(b"triple").expect("ref put");
    assert_eq!(d, rd);
    assert_eq!(store.replica_count(&d).expect("replica_count"), 3);
    assert_eq!(refer.replica_count(&d).expect("ref replica_count"), 3);
}

#[test]
fn get_unknown_digest_is_not_found() {
    let h = fresh_store();
    let store = create(&h);
    let ghost = Digest(digest_hex(b"never-put"));
    let err = store.get(&ghost).expect_err("not found");
    common::assert_not_found(&err, &ghost.0);
    assert_eq!(store.replica_count(&ghost).expect("count unknown"), 0);
}

#[test]
fn put_same_bytes_twice_same_digest() {
    let h = fresh_store();
    let mut store = create(&h);
    let a = store.put(b"same").expect("put 1");
    let b = store.put(b"same").expect("put 2");
    assert_eq!(a, b);
    assert_eq!(store.get(&a).expect("get"), b"same");
    assert_eq!(store.replica_count(&a).expect("replica_count"), 2);
}

#[test]
fn digest_is_lowercase_hex_of_raw_bytes() {
    let h = fresh_store();
    let mut store = create(&h);
    let bytes = [0u8, 1, 2, 255, 16, 32];
    let d = store.put(&bytes).expect("put");
    assert!(is_sha256_hex(&d.0));
    assert_eq!(d.0, d.0.to_ascii_lowercase());
    assert_eq!(d.0, digest_hex(&bytes));
    assert_eq!(store.get(&d).expect("get"), bytes);
}
