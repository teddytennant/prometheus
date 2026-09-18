//! Group: `DirStore::open` + put/get and reopen (crash-durability analog).

mod common;
mod reference;

use common::assert_not_found;
use prometheus_ckpt::{DirStore, Store};
use reference::RefDirStore;

#[test]
fn put_get_then_reopen_same_directory() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().to_path_buf();
    let key = "blob.abc123";
    let bytes = b"\x00dir-store-payload\xff persist-me".as_slice();

    {
        let mut prod = DirStore::open(path.clone()).expect("open");
        let mut refer = RefDirStore::open(path.clone()).expect("ref open");
        prod.put(key, bytes).expect("put");
        refer.put(key, bytes).expect("ref put");
        assert_eq!(prod.get(key).expect("get"), bytes);
        assert_eq!(refer.get(key).expect("ref get"), bytes);
        assert!(prod.contains(key).expect("contains"));
    }

    {
        let prod = DirStore::open(path.clone()).expect("reopen");
        let refer = RefDirStore::open(path).expect("ref reopen");
        assert_eq!(
            prod.get(key).expect("get after reopen"),
            bytes,
            "DirStore must survive drop + open of the same directory"
        );
        assert_eq!(refer.get(key).expect("ref get after reopen"), bytes);
        assert!(prod.contains(key).expect("contains after reopen"));
    }
}

#[test]
fn get_missing_key_is_not_found() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let prod = DirStore::open(tmp.path().to_path_buf()).expect("open");
    let err = prod.get("absent").expect_err("missing");
    assert_not_found(&err, "absent");
}

#[test]
fn open_creates_missing_directory() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let nested = tmp.path().join("not-yet");
    assert!(!nested.exists());
    let mut prod = DirStore::open(nested.clone()).expect("open creates dir");
    prod.put("k", b"v").expect("put");
    assert_eq!(prod.get("k").expect("get"), b"v");
}

#[test]
fn overwrite_then_reopen() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().to_path_buf();
    {
        let mut prod = DirStore::open(path.clone()).expect("open");
        prod.put("k", b"first").expect("put 1");
        prod.put("k", b"second").expect("put 2");
        assert_eq!(prod.get("k").expect("get"), b"second");
    }
    let prod = DirStore::open(path).expect("reopen");
    assert_eq!(prod.get("k").expect("get"), b"second");
}
