//! Group: `MemoryStore` put / get / contains vs the HashMap reference.

mod common;
mod reference;

use common::assert_not_found;
use prometheus_ckpt::{CkptError, MemoryStore, Store};
use reference::RefMemoryStore;

#[test]
fn put_get_contains_match_reference() {
    let mut prod = MemoryStore::new();
    let mut refer = RefMemoryStore::new();
    let key = "weights.embed.0";
    let bytes = b"\x00mem-store-bytes\xff".as_slice();

    refer.put(key, bytes).expect("ref put");
    prod.put(key, bytes).expect("put");

    assert_eq!(prod.get(key).expect("get"), bytes);
    assert_eq!(refer.get(key).expect("ref get"), prod.get(key).unwrap());
    assert_eq!(prod.contains(key).expect("contains"), true);
    assert_eq!(refer.contains(key).expect("ref contains"), true);
    assert_eq!(prod.contains("missing").expect("contains missing"), false);
    assert_eq!(
        refer.contains("missing").expect("ref contains missing"),
        false
    );
}

#[test]
fn get_missing_key_is_not_found() {
    let prod = MemoryStore::new();
    let refer = RefMemoryStore::new();
    let err = prod.get("no-such-key").expect_err("missing get");
    assert_not_found(&err, "no-such-key");
    let rerr = refer.get("no-such-key").expect_err("ref missing get");
    assert_not_found(&rerr, "no-such-key");
    match err {
        CkptError::NotFound(_) => {}
        other => panic!("expected NotFound, got {other:?}"),
    }
}

#[test]
fn put_overwrites_previous_bytes() {
    let mut prod = MemoryStore::new();
    let mut refer = RefMemoryStore::new();
    prod.put("k", b"one").expect("put 1");
    refer.put("k", b"one").expect("ref put 1");
    prod.put("k", b"two").expect("put 2");
    refer.put("k", b"two").expect("ref put 2");
    assert_eq!(prod.get("k").expect("get"), b"two");
    assert_eq!(refer.get("k").expect("ref get"), b"two");
}

#[test]
fn empty_bytes_roundtrip() {
    let mut prod = MemoryStore::new();
    let mut refer = RefMemoryStore::new();
    prod.put("empty", b"").expect("put empty");
    refer.put("empty", b"").expect("ref put empty");
    assert_eq!(prod.get("empty").expect("get"), b"");
    assert_eq!(refer.get("empty").expect("ref get"), b"");
    assert!(prod.contains("empty").expect("contains"));
}
