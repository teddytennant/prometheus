//! Group: `Checkpointer::save_in_memory` / `restore_latest_memory` / replicas.

mod common;
mod reference;

use common::{
    assert_checkpoint_roundtrip, assert_missing_shard, payload_len, tiny_checkpoint,
    tiny_checkpoint_with_id, PanicStore,
};
use prometheus_ckpt::{Checkpointer, CkptError, MEMORY_REPLICAS};
use reference::RefCheckpointer;

#[test]
fn save_in_memory_restore_latest_bitwise_equal() {
    reference::assert_nist_goldens();
    let src = tiny_checkpoint();
    let mut prod = Checkpointer::new(Box::new(PanicStore));
    let mut refer = RefCheckpointer::new();

    let id = prod.save_in_memory(&src).expect("save_in_memory");
    let rid = refer.save_in_memory(&src).expect("ref save_in_memory");
    assert_eq!(id, src.manifest.checkpoint_id);
    assert_eq!(id, rid);

    let got = prod.restore_latest_memory().expect("restore_latest_memory");
    let rgot = refer
        .restore_latest_memory()
        .expect("ref restore_latest_memory");
    assert_checkpoint_roundtrip(&got, &src);
    assert_checkpoint_roundtrip(&rgot, &src);
    assert_eq!(got.weights, rgot.weights);
    assert_eq!(got.optimizer, rgot.optimizer);
    assert_eq!(got.manifest.created_at, src.manifest.created_at);
    assert_eq!(got.manifest.precision.as_ref().unwrap().remat, "full");
    assert_eq!(got.manifest.weights[0].dtype, prometheus_ckpt::Dtype::Fp32);
}

#[test]
fn replica_len_two_copies_equal_and_positive() {
    let src = tiny_checkpoint();
    let mut prod = Checkpointer::new(Box::new(PanicStore));
    let mut refer = RefCheckpointer::new();
    prod.save_in_memory(&src).expect("save_in_memory");
    refer.save_in_memory(&src).expect("ref save");

    assert_eq!(MEMORY_REPLICAS, 2);
    let n0 = prod.replica_len(0).expect("replica_len(0)");
    let n1 = prod.replica_len(1).expect("replica_len(1)");
    assert_eq!(
        n0, n1,
        "the two in-memory replicas must hold the same byte count"
    );
    assert!(n0 > 0, "replica_len must be > 0 after save_in_memory");
    assert!(
        n0 >= payload_len(&src),
        "replica_len={n0} must be at least the shard payload ({})",
        payload_len(&src)
    );

    let r0 = refer.replica_len(0).expect("ref replica_len(0)");
    let r1 = refer.replica_len(1).expect("ref replica_len(1)");
    assert_eq!(r0, r1);
    assert!(r0 > 0);
}

#[test]
fn replica_len_index_two_is_error() {
    let src = tiny_checkpoint();
    let mut prod = Checkpointer::new(Box::new(PanicStore));
    prod.save_in_memory(&src).expect("save_in_memory");
    assert!(
        prod.replica_len(MEMORY_REPLICAS).is_err(),
        "replica_len(MEMORY_REPLICAS) must error"
    );
    assert!(prod.replica_len(2).is_err(), "replica_len(2) must error");
    assert!(prod.replica_len(99).is_err());
}

#[test]
fn restore_latest_memory_without_save_is_not_found() {
    let prod = Checkpointer::new(Box::new(PanicStore));
    let err = prod
        .restore_latest_memory()
        .expect_err("no in-memory checkpoint");
    match err {
        CkptError::NotFound(_) => {}
        other => panic!("expected NotFound, got {other:?}"),
    }
}

#[test]
fn second_save_in_memory_is_the_latest() {
    let first = tiny_checkpoint_with_id("first");
    let second = tiny_checkpoint_with_id("second");
    let mut prod = Checkpointer::new(Box::new(PanicStore));
    let mut refer = RefCheckpointer::new();
    prod.save_in_memory(&first).expect("save 1");
    refer.save_in_memory(&first).expect("ref save 1");
    prod.save_in_memory(&second).expect("save 2");
    refer.save_in_memory(&second).expect("ref save 2");
    let got = prod.restore_latest_memory().expect("restore");
    let rgot = refer.restore_latest_memory().expect("ref restore");
    assert_eq!(got.manifest.checkpoint_id, "second");
    assert_checkpoint_roundtrip(&got, &second);
    assert_eq!(rgot.manifest.checkpoint_id, "second");
}

#[test]
fn missing_weight_blob_is_missing_shard() {
    let mut src = tiny_checkpoint();
    let name = src.manifest.weights[0].name.clone();
    let rank = src.manifest.weights[0].shard_rank;
    src.weights.clear();
    let mut prod = Checkpointer::new(Box::new(PanicStore));
    match prod.save_in_memory(&src) {
        Err(err) => assert_missing_shard(&err, &name, rank),
        Ok(_) => match prod.restore_latest_memory() {
            Err(err) => assert_missing_shard(&err, &name, rank),
            Ok(_) => panic!("expected MissingShard on save or restore"),
        },
    }
}

#[test]
fn created_at_not_replaced_by_wall_clock() {
    let src = tiny_checkpoint();
    assert_eq!(src.manifest.created_at, "2020-01-02T03:04:05Z");
    let mut prod = Checkpointer::new(Box::new(PanicStore));
    prod.save_in_memory(&src).expect("save");
    let got = prod.restore_latest_memory().expect("restore");
    assert_eq!(got.manifest.created_at, "2020-01-02T03:04:05Z");
}
