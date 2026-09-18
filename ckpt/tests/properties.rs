//! Group: property tests vs the reference (no GPU, nothing differentiable).

mod common;
mod reference;

use common::{
    assert_checkpoint_roundtrip, blobs_by_key, checkpoint_from_shards, payload_len, MapStore,
    PanicStore,
};
use prometheus_ckpt::{
    Checkpointer, Dtype, HostRamOffload, MemoryStore, OptimizerKind, Store, MEMORY_REPLICAS,
};
use reference::{RefCheckpointer, RefHostRamOffload, RefMemoryStore};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push((self.next() >> 32) as u8);
        }
        if out.is_empty() {
            out.push(0xaa);
        }
        // Keep a 0x00 / 0xff marker so Store-scan fault tests can find shards.
        out[0] = 0x00;
        if out.len() > 1 {
            let last = out.len() - 1;
            out[last] = 0xff;
        }
        out
    }
}

const DTYPES: [Dtype; 7] = [
    Dtype::Fp32,
    Dtype::Bf16,
    Dtype::Fp16,
    Dtype::Fp8,
    Dtype::Nvfp4,
    Dtype::Int8,
    Dtype::Int32,
];

const KINDS: [OptimizerKind; 3] = [
    OptimizerKind::Adamw,
    OptimizerKind::Muon,
    OptimizerKind::Soap,
];

#[test]
fn memory_store_put_get_contains_vs_reference_random_keys() {
    let mut rng = Lcg(0xA5A5_0001);
    let mut prod = MemoryStore::new();
    let mut refer = RefMemoryStore::new();
    for i in 0..32 {
        let key = format!("k{i}-{}", rng.next() % 1000);
        let n = 1 + (rng.next() as usize % 64);
        let bytes = rng.bytes(n);
        prod.put(&key, &bytes).expect("put");
        refer.put(&key, &bytes).expect("ref put");
        assert_eq!(prod.get(&key).unwrap(), refer.get(&key).unwrap());
        assert_eq!(prod.contains(&key).unwrap(), true);
        assert_eq!(prod.contains("never-written").unwrap(), false);
    }
}

#[test]
fn host_ram_offload_vs_reference_random_names() {
    let mut rng = Lcg(0xA5A5_0002);
    let mut prod = HostRamOffload::new();
    let mut refer = RefHostRamOffload::new();
    let mut live: Vec<String> = Vec::new();
    for i in 0..24 {
        let name = format!("blob-{i}");
        let n = 1 + (rng.next() as usize % 48);
        let bytes = rng.bytes(n);
        prod.put(&name, bytes.clone()).expect("put");
        refer.put(&name, &bytes).expect("ref put");
        live.push(name);
    }
    for name in &live {
        assert_eq!(prod.get(name).unwrap(), refer.get(name).unwrap());
    }
    let drop_name = live.pop().unwrap();
    prod.remove(&drop_name).expect("remove");
    refer.remove(&drop_name).expect("ref remove");
    assert!(prod.get(&drop_name).is_err());
    assert!(refer.get(&drop_name).is_err());
}

#[test]
fn save_restore_in_memory_vs_reference_random_checkpoints() {
    let mut rng = Lcg(0xA5A5_0003);
    for trial in 0..12 {
        let n_w = 1 + (rng.next() as usize % 3);
        let mut weights = Vec::new();
        let mut optimizer = Vec::new();
        for s in 0..n_w {
            let dtype = DTYPES[rng.next() as usize % DTYPES.len()];
            let kind = KINDS[rng.next() as usize % KINDS.len()];
            let wn = 2 + (rng.next() as usize % 16);
            let on = 2 + (rng.next() as usize % 16);
            let wbytes = rng.bytes(wn);
            let obytes = rng.bytes(on);
            weights.push((
                "layer",
                s as u64,
                vec![2, 2],
                dtype,
                wbytes,
                vec!["fsdp".to_string()],
            ));
            optimizer.push(("layer.m", s as u64, kind, obytes));
        }
        let src = checkpoint_from_shards(&format!("p{trial}"), weights, optimizer);
        let mut prod = Checkpointer::new(Box::new(PanicStore));
        let mut refer = RefCheckpointer::new();
        let id = prod.save_in_memory(&src).expect("save");
        let rid = refer.save_in_memory(&src).expect("ref save");
        assert_eq!(id, rid);
        let got = prod.restore_latest_memory().expect("restore");
        let rgot = refer.restore_latest_memory().expect("ref restore");
        assert_checkpoint_roundtrip(&got, &src);
        assert_eq!(blobs_by_key(&got.weights), blobs_by_key(&rgot.weights));
        assert_eq!(blobs_by_key(&got.optimizer), blobs_by_key(&rgot.optimizer));
        let n0 = prod.replica_len(0).unwrap();
        let n1 = prod.replica_len(1).unwrap();
        assert_eq!(n0, n1);
        assert!(n0 >= payload_len(&src));
        assert!(prod.replica_len(MEMORY_REPLICAS).is_err());
        for blob in got.weights.iter().chain(got.optimizer.iter()) {
            let h = reference::sha256_hex(&blob.bytes);
            assert_eq!(h.len(), 64);
        }
    }
}

#[test]
fn save_restore_persistent_vs_reference_random_checkpoints() {
    let mut rng = Lcg(0xA5A5_0004);
    for trial in 0..12 {
        let n_w = 1 + (rng.next() as usize % 3);
        let mut weights = Vec::new();
        let mut optimizer = Vec::new();
        for s in 0..n_w {
            let dtype = DTYPES[rng.next() as usize % DTYPES.len()];
            let kind = KINDS[rng.next() as usize % KINDS.len()];
            let dim = 1 + (rng.next() % 8);
            let wn = 3 + (rng.next() as usize % 12);
            let on = 3 + (rng.next() as usize % 12);
            weights.push((
                "w",
                s as u64,
                vec![dim, 4],
                dtype,
                rng.bytes(wn),
                Vec::new(),
            ));
            optimizer.push(("w.v", s as u64, kind, rng.bytes(on)));
        }
        let src = checkpoint_from_shards(&format!("q{trial}"), weights, optimizer);
        let mut prod = Checkpointer::new(Box::new(MapStore::new()));
        let mut refer = RefCheckpointer::new();
        let id = prod.save_persistent(&src).expect("save");
        refer.save_persistent(&src).expect("ref save");
        let got = prod.restore(&id).expect("restore");
        let rgot = refer.restore(&id).expect("ref restore");
        assert_checkpoint_roundtrip(&got, &src);
        assert_eq!(got.manifest.created_at, src.manifest.created_at);
        assert_eq!(got.weights, rgot.weights);
        for meta in &got.manifest.weights {
            let blob = got
                .weights
                .iter()
                .find(|b| b.name == meta.name && b.shard_rank == meta.shard_rank)
                .expect("blob");
            assert_eq!(meta.content_hash, reference::sha256_hex(&blob.bytes));
            assert_eq!(
                meta.dtype,
                src.manifest
                    .weights
                    .iter()
                    .find(|m| m.name == meta.name && m.shard_rank == meta.shard_rank)
                    .unwrap()
                    .dtype
            );
            assert_eq!(
                meta.shape,
                src.manifest
                    .weights
                    .iter()
                    .find(|m| m.name == meta.name && m.shard_rank == meta.shard_rank)
                    .unwrap()
                    .shape
            );
        }
    }
}

#[test]
fn empty_shard_bytes_hash_is_nist_empty() {
    let src = checkpoint_from_shards(
        "empty-blob",
        vec![("z", 0, vec![0], Dtype::Fp32, Vec::new(), Vec::new())],
        vec![("z.m", 0, OptimizerKind::Adamw, Vec::new())],
    );
    assert_eq!(
        reference::sha256_hex(&src.weights[0].bytes),
        reference::NIST_SHA256_EMPTY
    );
    let mut prod = Checkpointer::new(Box::new(MapStore::new()));
    let id = prod.save_persistent(&src).expect("save");
    let got = prod.restore(&id).expect("restore");
    assert_eq!(got.weights[0].bytes, b"");
    assert_eq!(
        got.manifest.weights[0].content_hash,
        reference::NIST_SHA256_EMPTY
    );
    assert_eq!(
        got.manifest.optimizer[0].content_hash,
        reference::NIST_SHA256_EMPTY
    );
}
