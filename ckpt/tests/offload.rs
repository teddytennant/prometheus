//! Group: `HostRamOffload` put / get / remove (Grace host-RAM stand-in).

mod common;
mod reference;

use common::assert_not_found;
use prometheus_ckpt::{CkptError, HostRamOffload};
use reference::RefHostRamOffload;

#[test]
fn put_get_roundtrip_matches_reference() {
    let mut prod = HostRamOffload::new();
    let mut refer = RefHostRamOffload::new();
    let name = "opt.embed.m";
    let bytes = b"\x00offload-bytes\xff".to_vec();
    prod.put(name, bytes.clone()).expect("put");
    refer.put(name, &bytes).expect("ref put");
    assert_eq!(prod.get(name).expect("get"), bytes);
    assert_eq!(refer.get(name).expect("ref get"), bytes);
}

#[test]
fn get_missing_name_is_not_found() {
    let prod = HostRamOffload::new();
    let err = prod.get("nope").expect_err("missing get");
    assert_not_found(&err, "nope");
    match err {
        CkptError::NotFound(_) => {}
        other => panic!("expected NotFound, got {other:?}"),
    }
}

#[test]
fn remove_then_get_is_not_found() {
    let mut prod = HostRamOffload::new();
    let mut refer = RefHostRamOffload::new();
    prod.put("x", b"bytes".to_vec()).expect("put");
    refer.put("x", b"bytes").expect("ref put");
    prod.remove("x").expect("remove");
    refer.remove("x").expect("ref remove");
    let err = prod.get("x").expect_err("get after remove");
    assert_not_found(&err, "x");
    let rerr = refer.get("x").expect_err("ref get after remove");
    assert_not_found(&rerr, "x");
}

#[test]
fn remove_missing_name_is_not_found() {
    let mut prod = HostRamOffload::new();
    let err = prod.remove("ghost").expect_err("remove missing");
    assert_not_found(&err, "ghost");
}

#[test]
fn put_overwrites() {
    let mut prod = HostRamOffload::new();
    prod.put("k", b"a".to_vec()).expect("put 1");
    prod.put("k", b"bb".to_vec()).expect("put 2");
    assert_eq!(prod.get("k").expect("get"), b"bb");
}

#[test]
fn empty_bytes_roundtrip() {
    let mut prod = HostRamOffload::new();
    prod.put("empty", Vec::new()).expect("put");
    assert_eq!(prod.get("empty").expect("get"), b"");
    prod.remove("empty").expect("remove");
    assert!(matches!(
        prod.get("empty"),
        Err(CkptError::NotFound(n)) if n == "empty"
    ));
}
