//! Group: fault injection — HashMismatch, MissingShard.
//!
//! ReplicaLost cannot be triggered through the frozen iface (no API drops a
//! replica); MultipleReplicas cannot be expressed (no DP-rank on shards).
//! Those two are skipped rather than changing `lib.rs`.

mod common;
mod reference;

use common::{
    assert_hash_mismatch, assert_missing_shard, corrupt_blob_in_dir, corrupt_first_blob,
    tiny_checkpoint, MapStore,
};
use prometheus_ckpt::{Checkpointer, CkptError, DirStore};

#[test]
fn flip_stored_blob_byte_then_restore_is_hash_mismatch() {
    reference::assert_nist_goldens();
    let src = tiny_checkpoint();
    let expected_hash = reference::sha256_hex(&src.weights[0].bytes);
    let store = MapStore::new();
    let map = store.map.clone();
    let mut prod = Checkpointer::new(Box::new(store));
    let id = prod.save_persistent(&src).expect("save_persistent");

    {
        let mut guard = map.lock().expect("map lock");
        assert!(
            corrupt_first_blob(&mut guard, &src.weights[0].bytes),
            "save_persistent must write the weight shard bytes into the Store"
        );
    }

    let err = prod.restore(&id).expect_err("corrupt blob");
    assert_hash_mismatch(&err, Some(&expected_hash));
}

#[test]
fn flip_dirstore_file_byte_then_restore_is_hash_mismatch() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().to_path_buf();
    let src = tiny_checkpoint();
    let expected_hash = reference::sha256_hex(&src.weights[0].bytes);
    let id;
    {
        let store = DirStore::open(path.clone()).expect("open");
        let mut prod = Checkpointer::new(Box::new(store));
        id = prod.save_persistent(&src).expect("save");
    }
    assert!(
        corrupt_blob_in_dir(path.as_path(), &src.weights[0].bytes),
        "DirStore must persist the weight shard as a file containing the raw bytes"
    );
    let store = DirStore::open(path).expect("reopen");
    let prod = Checkpointer::new(Box::new(store));
    let err = prod.restore(&id).expect_err("corrupt file");
    assert_hash_mismatch(&err, Some(&expected_hash));
}

#[test]
fn manifest_weight_without_blob_is_missing_shard_on_save_or_restore() {
    let mut src = tiny_checkpoint();
    let name = src.manifest.weights[0].name.clone();
    let rank = src.manifest.weights[0].shard_rank;
    src.weights.clear();
    let mut prod = Checkpointer::new(Box::new(MapStore::new()));
    match prod.save_persistent(&src) {
        Err(err) => assert_missing_shard(&err, &name, rank),
        Ok(id) => match prod.restore(&id) {
            Err(err) => assert_missing_shard(&err, &name, rank),
            Ok(_) => panic!("expected MissingShard on save or restore"),
        },
    }
}

#[test]
fn optimizer_meta_without_blob_is_missing_shard() {
    let mut src = tiny_checkpoint();
    let name = src.manifest.optimizer[0].name.clone();
    let rank = src.manifest.optimizer[0].shard_rank;
    src.optimizer.clear();
    let mut prod = Checkpointer::new(Box::new(MapStore::new()));
    match prod.save_persistent(&src) {
        Err(err) => assert_missing_shard(&err, &name, rank),
        Ok(id) => match prod.restore(&id) {
            Err(err) => assert_missing_shard(&err, &name, rank),
            Ok(_) => panic!("expected MissingShard on save or restore"),
        },
    }
}

#[test]
fn flip_optimizer_blob_is_hash_mismatch() {
    let src = tiny_checkpoint();
    let expected_hash = reference::sha256_hex(&src.optimizer[0].bytes);
    let store = MapStore::new();
    let map = store.map.clone();
    let mut prod = Checkpointer::new(Box::new(store));
    let id = prod.save_persistent(&src).expect("save");
    {
        let mut guard = map.lock().expect("map lock");
        assert!(
            corrupt_first_blob(&mut guard, &src.optimizer[0].bytes),
            "save_persistent must write optimizer shard bytes into the Store"
        );
    }
    let err = prod.restore(&id).expect_err("corrupt optimizer");
    match err {
        CkptError::HashMismatch { expected, got } => {
            assert_ne!(expected, got);
            assert_eq!(expected, expected_hash);
        }
        other => panic!("expected HashMismatch, got {other:?}"),
    }
}
