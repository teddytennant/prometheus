//! Group: `Checkpointer::save_persistent` / `restore(id)`, hashes, created_at.

mod common;
mod reference;

use common::{
    assert_checkpoint_roundtrip, assert_not_found, checkpoint_from_shards, tiny_checkpoint,
    tiny_checkpoint_with_id, MapStore,
};
use prometheus_ckpt::{Checkpointer, DirStore, Dtype, MemoryStore, OptimizerKind};
use reference::RefCheckpointer;

#[test]
fn save_persistent_restore_bitwise_equal() {
    reference::assert_nist_goldens();
    let src = tiny_checkpoint();
    let store = MapStore::new();
    let mut prod = Checkpointer::new(Box::new(store));
    let mut refer = RefCheckpointer::new();

    let id = prod.save_persistent(&src).expect("save_persistent");
    let rid = refer.save_persistent(&src).expect("ref save_persistent");
    assert_eq!(id, src.manifest.checkpoint_id, "id is checkpoint_id");
    assert_eq!(id, rid);

    let got = prod.restore(&id).expect("restore");
    let rgot = refer.restore(&rid).expect("ref restore");
    assert_checkpoint_roundtrip(&got, &src);
    assert_checkpoint_roundtrip(&rgot, &src);
    assert_eq!(got.weights, rgot.weights);
    assert_eq!(got.optimizer, rgot.optimizer);
}

#[test]
fn content_hash_equals_reference_sha256_of_shard_bytes() {
    let src = tiny_checkpoint();
    let store = MapStore::new();
    let mut prod = Checkpointer::new(Box::new(store));
    let id = prod.save_persistent(&src).expect("save");
    let got = prod.restore(&id).expect("restore");

    for (meta, blob) in got.manifest.weights.iter().zip(got.weights.iter()) {
        assert_eq!(meta.name, blob.name);
        assert_eq!(meta.shard_rank, blob.shard_rank);
        let expected = reference::sha256_hex(&blob.bytes);
        assert_eq!(meta.content_hash, expected);
        assert_eq!(expected, reference::expected_content_hash(&blob.bytes));
    }
    for (meta, blob) in got.manifest.optimizer.iter().zip(got.optimizer.iter()) {
        let expected = reference::sha256_hex(&blob.bytes);
        assert_eq!(meta.content_hash, expected);
    }
}

#[test]
fn created_at_preserved_on_persistent_restore() {
    let src = tiny_checkpoint();
    assert_eq!(src.manifest.created_at, "2020-01-02T03:04:05Z");
    let store = MapStore::new();
    let mut prod = Checkpointer::new(Box::new(store));
    let id = prod.save_persistent(&src).expect("save");
    let got = prod.restore(&id).expect("restore");
    assert_eq!(got.manifest.created_at, src.manifest.created_at);
    assert_eq!(got.manifest.created_at, "2020-01-02T03:04:05Z");
}

#[test]
fn restore_unknown_id_is_not_found() {
    let prod = Checkpointer::new(Box::new(MapStore::new()));
    let err = prod.restore("no-such-ckpt").expect_err("unknown id");
    assert_not_found(&err, "no-such-ckpt");
}

#[test]
fn two_persistent_ids_restore_independently() {
    let a = tiny_checkpoint_with_id("ckpt-a");
    let b = tiny_checkpoint_with_id("ckpt-b");
    let store = MapStore::new();
    let mut prod = Checkpointer::new(Box::new(store));
    let mut refer = RefCheckpointer::new();
    let ida = prod.save_persistent(&a).expect("save a");
    let idb = prod.save_persistent(&b).expect("save b");
    refer.save_persistent(&a).unwrap();
    refer.save_persistent(&b).unwrap();
    assert_eq!(ida, "ckpt-a");
    assert_eq!(idb, "ckpt-b");
    assert_checkpoint_roundtrip(&prod.restore(&ida).expect("restore a"), &a);
    assert_checkpoint_roundtrip(&prod.restore(&idb).expect("restore b"), &b);
    assert_eq!(
        prod.restore(&ida).unwrap().weights,
        refer.restore(&ida).unwrap().weights
    );
}

#[test]
fn dirstore_save_drop_reopen_restore() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().to_path_buf();
    let src = tiny_checkpoint();
    let id;
    {
        let store = DirStore::open(path.clone()).expect("open");
        let mut prod = Checkpointer::new(Box::new(store));
        id = prod.save_persistent(&src).expect("save");
    }
    {
        let store = DirStore::open(path).expect("reopen");
        let prod = Checkpointer::new(Box::new(store));
        let got = prod.restore(&id).expect("restore after reopen");
        assert_checkpoint_roundtrip(&got, &src);
    }
}

#[test]
fn memorystore_backend_save_restore() {
    let src = tiny_checkpoint();
    let mut prod = Checkpointer::new(Box::new(MemoryStore::new()));
    let id = prod.save_persistent(&src).expect("save");
    let got = prod.restore(&id).expect("restore");
    assert_checkpoint_roundtrip(&got, &src);
}

#[test]
fn shape_and_dtype_preserved_for_every_dtype() {
    let dtypes = [
        Dtype::Fp32,
        Dtype::Bf16,
        Dtype::Fp16,
        Dtype::Fp8,
        Dtype::Nvfp4,
        Dtype::Int8,
        Dtype::Int32,
    ];
    let kinds = [
        OptimizerKind::Adamw,
        OptimizerKind::Muon,
        OptimizerKind::Soap,
    ];
    for (i, dtype) in dtypes.iter().enumerate() {
        let kind = kinds[i % kinds.len()];
        let src = checkpoint_from_shards(
            &format!("dtype-{i}"),
            vec![(
                "w",
                0,
                vec![3, 5, 7],
                *dtype,
                format!("\x00dt-{i}").into_bytes(),
                vec!["fsdp".into()],
            )],
            vec![("w.m", 0, kind, format!("\x00opt-{i}").into_bytes())],
        );
        let mut prod = Checkpointer::new(Box::new(MapStore::new()));
        let id = prod.save_persistent(&src).expect("save");
        let got = prod.restore(&id).expect("restore");
        assert_eq!(got.manifest.weights[0].dtype, *dtype);
        assert_eq!(got.manifest.weights[0].shape, vec![3, 5, 7]);
        assert_eq!(got.manifest.optimizer[0].kind, kind);
        assert_checkpoint_roundtrip(&got, &src);
    }
}

#[test]
fn multiple_shards_roundtrip_by_name_and_rank() {
    let src = checkpoint_from_shards(
        "multi",
        vec![
            (
                "embed",
                0,
                vec![2],
                Dtype::Fp32,
                b"\x00e0".to_vec(),
                vec!["fsdp".into()],
            ),
            (
                "embed",
                1,
                vec![2],
                Dtype::Fp32,
                b"\x00e1".to_vec(),
                vec!["fsdp".into()],
            ),
            (
                "lm_head",
                0,
                vec![4, 2],
                Dtype::Bf16,
                b"\x00h0".to_vec(),
                vec!["fsdp".into()],
            ),
        ],
        vec![
            ("embed.m", 0, OptimizerKind::Muon, b"\x00m0".to_vec()),
            ("embed.m", 1, OptimizerKind::Muon, b"\x00m1".to_vec()),
        ],
    );
    let mut prod = Checkpointer::new(Box::new(MapStore::new()));
    let mut refer = RefCheckpointer::new();
    let id = prod.save_persistent(&src).expect("save");
    refer.save_persistent(&src).unwrap();
    let got = prod.restore(&id).expect("restore");
    assert_checkpoint_roundtrip(&got, &src);
    assert_eq!(got.weights.len(), 3);
    assert_eq!(got.optimizer.len(), 2);
    assert_eq!(got.weights, refer.restore(&id).unwrap().weights);
}

#[test]
fn parent_checkpoint_id_and_rng_preserved() {
    let mut src = tiny_checkpoint();
    src.manifest.parent_checkpoint_id = Some("ckpt-parent".into());
    src.manifest.rng.push(prometheus_ckpt::RngState {
        scope: "shuffle".into(),
        rank: 1,
        state_hash: reference::sha256_hex(b"rng-shuffle-1"),
    });
    let mut prod = Checkpointer::new(Box::new(MapStore::new()));
    let id = prod.save_persistent(&src).expect("save");
    let got = prod.restore(&id).expect("restore");
    assert_eq!(
        got.manifest.parent_checkpoint_id.as_deref(),
        Some("ckpt-parent")
    );
    assert_eq!(got.manifest.rng.len(), 2);
    assert_eq!(got.manifest.rng[1].scope, "shuffle");
    assert_eq!(got.manifest.rng[1].state_hash.len(), 64);
}
