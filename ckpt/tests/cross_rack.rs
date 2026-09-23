//! Group: cross-rack in-memory copies (spec 5.5).
//!
//! Host RAM and a caller-supplied rack list stand in for Grace RAM on two
//! other racks. No network. No GPU coverage (no `gpu` marker).
//!
//! `plan_cross_rack` is checked against an independent walk in this file.
//! Production code must not import this module. `save_cross_rack` must keep
//! the same prepared bytes `save_in_memory` would keep.

mod common;
mod reference;

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use common::{
    assert_checkpoint_roundtrip, assert_hash_mismatch, assert_missing_shard, assert_not_found,
    checkpoint_from_shards, payload_len, tiny_checkpoint, tiny_checkpoint_with_id,
};
use prometheus_ckpt::{
    plan_cross_rack, Checkpoint, Checkpointer, CkptError, Dtype, MemoryStore, OptimizerKind,
    ShardBlob, Store, CROSS_RACK_COPIES, MEMORY_REPLICAS,
};

/// Independent walk. Does not call [`plan_cross_rack`].
///
/// Start just after the first occurrence of `source`, wrap once, skip `source`
/// and ids already chosen, return two distinct ids in that order. No trimming.
fn ref_plan_cross_rack(racks: &[String], source: &str) -> Result<(String, String), &'static str> {
    if source.is_empty() {
        return Err("empty source");
    }
    if racks.iter().any(|id| id.is_empty()) {
        return Err("empty id");
    }
    let Some(start) = racks.iter().position(|id| id == source) else {
        return Err("source absent");
    };
    if racks.iter().collect::<HashSet<_>>().len() < 3 {
        return Err("fewer than 3 distinct");
    }
    let mut chosen = Vec::new();
    let n = racks.len();
    for step in 1..=n {
        let id = &racks[(start + step) % n];
        if id == source || chosen.iter().any(|picked| picked == id) {
            continue;
        }
        chosen.push(id.clone());
        if chosen.len() == 2 {
            return Ok((chosen[0].clone(), chosen[1].clone()));
        }
    }
    Err("walk found fewer than two")
}

fn ids(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

fn owned_pair(left: &str, right: &str) -> (String, String) {
    (left.to_string(), right.to_string())
}

fn fresh() -> Checkpointer {
    Checkpointer::new(Box::new(MemoryStore::new()))
}

/// Bytes `save_in_memory` would keep, including filled empty content hashes.
fn prepared_of(src: &Checkpoint) -> Checkpoint {
    let mut mem = fresh();
    mem.save_in_memory(src).expect("save_in_memory prepares");
    mem.restore_latest_memory().expect("restore prepared")
}

fn assert_rack(err: &CkptError) {
    match err {
        CkptError::Rack(_) => {}
        other => panic!("expected Rack, got {other:?}"),
    }
}

fn assert_plan(fleet: &[&str], source: &str, left: &str, right: &str) {
    let racks = ids(fleet);
    let expected = owned_pair(left, right);
    let got = plan_cross_rack(&racks, source)
        .unwrap_or_else(|err| panic!("plan_cross_rack({fleet:?}, {source:?}) failed: {err:?}"));
    assert_eq!(got, expected, "fleet={fleet:?} source={source:?}");
    assert_eq!(
        ref_plan_cross_rack(&racks, source).expect("reference accepts documented success"),
        expected,
        "reference drifted from the documented pair"
    );
    assert_ne!(got.0, source);
    assert_ne!(got.1, source);
    assert_ne!(got.0, got.1);
}

fn assert_plan_rack(fleet: &[&str], source: &str) {
    let racks = ids(fleet);
    let err = plan_cross_rack(&racks, source).expect_err("plan must fail");
    assert_rack(&err);
    assert!(
        ref_plan_cross_rack(&racks, source).is_err(),
        "reference accepted fleet={fleet:?} source={source:?}"
    );
}

struct CountingStore {
    map: Arc<Mutex<std::collections::HashMap<String, Vec<u8>>>>,
    puts: Arc<AtomicU64>,
}

impl CountingStore {
    fn new() -> Self {
        Self {
            map: Arc::new(Mutex::new(std::collections::HashMap::new())),
            puts: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl Store for CountingStore {
    fn put(&mut self, key: &str, bytes: &[u8]) -> prometheus_ckpt::Result<()> {
        self.puts.fetch_add(1, Ordering::SeqCst);
        self.map
            .lock()
            .expect("store map")
            .insert(key.to_string(), bytes.to_vec());
        Ok(())
    }

    fn get(&self, key: &str) -> prometheus_ckpt::Result<Vec<u8>> {
        self.map
            .lock()
            .expect("store map")
            .get(key)
            .cloned()
            .ok_or_else(|| CkptError::NotFound(key.to_string()))
    }

    fn contains(&self, key: &str) -> prometheus_ckpt::Result<bool> {
        Ok(self.map.lock().expect("store map").contains_key(key))
    }
}

fn varied(id: &str, marker: u8) -> Checkpoint {
    checkpoint_from_shards(
        id,
        vec![(
            "embed",
            0,
            vec![4],
            Dtype::Fp32,
            vec![marker, 1, 2, 3],
            vec!["fsdp".to_string()],
        )],
        vec![("embed.m", 0, OptimizerKind::Muon, vec![marker, 9, 8])],
    )
}

#[test]
fn plan_cross_rack_documented_walk_order() {
    assert_eq!(CROSS_RACK_COPIES, 2);

    assert_plan(&["a", "b", "c", "d"], "b", "c", "d");
    assert_plan(&["a", "b", "c", "d"], "d", "a", "b");
    assert_plan(&["a", "b", "c", "d"], "c", "d", "a");
    assert_plan(&["a", "b", "a", "c"], "b", "a", "c");

    // Wrap, first occurrence, already-chosen, and unsorted order.
    assert_plan(&["a", "b", "c"], "b", "c", "a");
    assert_plan(&["a", "b", "c"], "a", "b", "c");
    assert_plan(&["c", "b", "a"], "a", "c", "b");
    assert_plan(&["a", "b", "a", "c"], "a", "b", "c");
    assert_plan(&["b", "a", "b", "c", "d"], "b", "a", "c");
    assert_plan(&["c", "a", "c", "b", "d"], "c", "a", "b");
    assert_plan(&["a", "b", "b", "c"], "a", "b", "c");
    assert_plan(&["a", "b", "c", "b", "d"], "a", "b", "c");
    assert_plan(&["a", "b", "c", "d", "e"], "d", "e", "a");
    assert_plan(&["a", "b", "c", "d", "e"], "a", "b", "c");
    assert_plan(&["z", "y", "x"], "z", "y", "x");
    assert_plan(&["a", "b", "c", "c"], "c", "a", "b");
    assert_plan(&["x", "a", "b", "a", "c"], "a", "b", "c");
    assert_plan(&["a", "ab", "b", "c"], "a", "ab", "b");

    // No trimming. "a" and "a " are different. Case is significant.
    assert_plan(&["a", "a ", "b"], "a", "a ", "b");
    assert_plan(&["a ", "b", "c"], "a ", "b", "c");
    assert_plan(&["a", "b c", "d", "e"], "a", "b c", "d");
    assert_plan(&["A", "a", "b"], "A", "a", "b");
    assert_plan(&["a", " ", "b"], "a", " ", "b");
    assert_plan(&["a", " ", "b"], " ", "b", "a");

    let unsorted = plan_cross_rack(&ids(&["a", "b", "c", "d"]), "c").expect("plan");
    assert_eq!(unsorted, owned_pair("d", "a"));
    assert_ne!(unsorted, owned_pair("a", "d"));
}

#[test]
fn plan_cross_rack_rejects_empty_absent_and_short_fleets() {
    assert_plan_rack(&["a", "b", "c"], "");
    assert_plan_rack(&["", "a", "b"], "");
    assert_plan_rack(&["", "a", "b"], "a");
    assert_plan_rack(&["a", "", "b", "c"], "b");
    assert_plan_rack(&["a", "b", ""], "a");
    assert_plan_rack(&[], "a");
    assert_plan_rack(&[], "");
    assert_plan_rack(&["a"], "a");
    assert_plan_rack(&["a"], "b");
    assert_plan_rack(&["a", "b"], "a");
    assert_plan_rack(&["a", "b"], "c");
    assert_plan_rack(&["a", "a", "a"], "a");
    assert_plan_rack(&["a", "b", "a"], "b");
    assert_plan_rack(&["a", "b", "b"], "a");
    assert_plan_rack(&["a", "b", "a", "b", "a"], "b");
    assert_plan_rack(&["a", "b", "c"], "a ");
    assert_plan_rack(&["a", "b", "c"], "b ");
    assert_plan_rack(&["a", "b", " c"], "c");
    assert_plan_rack(&["a", "b", "c"], "A");
    assert_plan_rack(&["a", "  ", "b"], " ");
}

#[test]
fn plan_cross_rack_matches_reference_on_generated_fleets() {
    assert_eq!(CROSS_RACK_COPIES, 2);
    let mut rng = Lcg(0xC0FFEE);
    let mixed = ["a", "b", "c", "d", "a ", "b ", " ", "ab", "A", ""];
    for _ in 0..64 {
        let n = (rng.next() as usize % 6) + 1;
        let mut fleet = Vec::new();
        for _ in 0..n {
            fleet.push(mixed[(rng.next() as usize) % mixed.len()].to_string());
        }
        let source = if rng.next().is_multiple_of(2) {
            fleet[(rng.next() as usize) % fleet.len()].clone()
        } else {
            mixed[(rng.next() as usize) % mixed.len()].to_string()
        };
        assert_plan_agrees(&fleet, &source);
    }

    let clean = ["a", "b", "c", "d", "e", "a ", "ab"];
    for _ in 0..32 {
        let n = (rng.next() as usize % 4) + 3;
        let mut fleet = Vec::new();
        for _ in 0..n {
            fleet.push(clean[(rng.next() as usize) % clean.len()].to_string());
        }
        let source = fleet[(rng.next() as usize) % fleet.len()].clone();
        assert_plan_agrees(&fleet, &source);
    }
}

fn assert_plan_agrees(fleet: &[String], source: &str) {
    let expected = ref_plan_cross_rack(fleet, source);
    let got = plan_cross_rack(fleet, source);
    match (expected, got) {
        (Ok(exp), Ok(got)) => {
            assert_eq!(got, exp, "fleet={fleet:?} source={source:?}");
            assert_ne!(got.0, source);
            assert_ne!(got.1, source);
            assert_ne!(got.0, got.1);
        }
        (Err(_), Err(err)) => assert_rack(&err),
        (Ok(exp), Err(err)) => {
            panic!("fleet={fleet:?} source={source:?}: expected {exp:?}, got {err:?}")
        }
        (Err(why), Ok(got)) => {
            panic!("fleet={fleet:?} source={source:?}: expected Rack ({why}), got {got:?}")
        }
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
}

#[test]
fn save_cross_rack_returns_plan_pair_and_stores_prepared_clones() {
    let examples = [
        (["a", "b", "c", "d"], "b", "c", "d"),
        (["a", "b", "c", "d"], "d", "a", "b"),
        (["a", "b", "c", "d"], "c", "d", "a"),
        (["a", "b", "a", "c"], "b", "a", "c"),
        (["a", "a ", "b", "c"], "a", "a ", "b"),
        (["a ", "b", "c", "d"], "a ", "b", "c"),
    ];
    for (fleet, source, left, right) in examples {
        let mut prod = fresh();
        let src = varied(source, left.as_bytes()[0]);
        let expect = prepared_of(&src);
        let racks = ids(&fleet);
        let pair = prod
            .save_cross_rack(&src, source, &racks)
            .unwrap_or_else(|err| panic!("save {source}: {err:?}"));
        assert_eq!(pair, owned_pair(left, right));
        assert_eq!(pair, ref_plan_cross_rack(&racks, source).unwrap());
        assert_eq!(prod.cross_rack_destinations(source).unwrap(), pair);
        assert_eq!(prod.restore_from_rack(left).unwrap(), expect);
        assert_eq!(prod.restore_from_rack(right).unwrap(), expect);
        assert_checkpoint_roundtrip(&prod.restore_from_rack(left).unwrap(), &expect);
        assert_not_found(&prod.restore_from_rack(source).unwrap_err(), source);
        assert_not_found(
            &prod.restore_from_rack("missing-rack").unwrap_err(),
            "missing-rack",
        );
    }

    // Empty hashes are filled the same way save_in_memory fills them.
    let mut src = tiny_checkpoint_with_id("fill-hash");
    src.manifest.weights[0].content_hash.clear();
    src.manifest.optimizer[0].content_hash.clear();
    let expect = prepared_of(&src);
    assert_eq!(expect.manifest.weights[0].content_hash.len(), 64);
    assert_ne!(expect.manifest.weights[0].content_hash, "");
    assert_ne!(
        expect.manifest.weights[0].content_hash,
        src.manifest.weights[0].content_hash
    );
    let mut prod = fresh();
    let fleet = ids(&["src", "d0", "d1", "d2"]);
    let pair = prod.save_cross_rack(&src, "src", &fleet).unwrap();
    assert_eq!(pair, owned_pair("d0", "d1"));
    assert_eq!(prod.restore_from_rack("d0").unwrap(), expect);
    assert_eq!(prod.restore_from_rack("d1").unwrap(), expect);

    // Caller mutation of the input, the fleet, or a restored value is isolated.
    let mut owned = varied("owned", 7);
    let expect = prepared_of(&owned);
    let mut fleet = ids(&["src", "d0", "d1", "d2"]);
    let pair = prod.save_cross_rack(&owned, "src", &fleet).unwrap();
    owned.weights[0].bytes[0] ^= 0xff;
    owned.manifest.step = 999;
    fleet.clear();
    assert_eq!(prod.restore_from_rack(&pair.0).unwrap(), expect);
    let mut restored = prod.restore_from_rack(&pair.0).unwrap();
    restored.weights[0].bytes[0] ^= 0x5a;
    restored.manifest.checkpoint_id.push('x');
    restored.optimizer[0].bytes[0] = 0;
    assert_eq!(prod.restore_from_rack(&pair.0).unwrap(), expect);
    assert_eq!(prod.restore_from_rack(&pair.1).unwrap(), expect);
    assert_eq!(prod.cross_rack_destinations("src").unwrap(), pair);

    // Extra blob not listed in the manifest is kept if save_in_memory keeps it.
    let mut extra = tiny_checkpoint_with_id("extra-blob");
    extra.weights.push(ShardBlob {
        name: "orphan".to_string(),
        shard_rank: 3,
        bytes: b"orphan-bytes".to_vec(),
    });
    let expect = prepared_of(&extra);
    let mut prod = fresh();
    let fleet = ids(&["s", "p", "q", "r"]);
    prod.save_cross_rack(&extra, "s", &fleet).unwrap();
    assert_eq!(prod.restore_from_rack("p").unwrap(), expect);
    assert_eq!(prod.restore_from_rack("q").unwrap(), expect);
}

#[test]
fn save_cross_rack_does_not_write_store_or_change_memory_replicas() {
    assert_eq!(MEMORY_REPLICAS, 2);
    assert_eq!(CROSS_RACK_COPIES, 2);

    let store = CountingStore::new();
    let map = Arc::clone(&store.map);
    let puts = Arc::clone(&store.puts);
    let mut prod = Checkpointer::new(Box::new(store));

    let persisted = tiny_checkpoint_with_id("persisted");
    let persisted_id = prod.save_persistent(&persisted).unwrap();
    let before_map = map.lock().expect("map").clone();
    let before_puts = puts.load(Ordering::SeqCst);
    assert!(!before_map.is_empty());
    assert!(before_puts > 0);

    let mem = varied("memory", 4);
    let mem_prepared = prepared_of(&mem);
    prod.save_in_memory(&mem).unwrap();
    let mem_restored = prod.restore_latest_memory().unwrap();
    assert_eq!(mem_restored, mem_prepared);
    let n0 = prod.replica_len(0).unwrap();
    let n1 = prod.replica_len(1).unwrap();
    assert_eq!(n0, n1);
    assert!(n0 >= payload_len(&mem));
    assert!(prod.replica_len(MEMORY_REPLICAS).is_err());

    let rack_src = varied("rack-copy", 5);
    let rack_prepared = prepared_of(&rack_src);
    assert_ne!(rack_prepared, mem_prepared);
    let fleet = ids(&["src", "d0", "d1", "d2"]);
    let pair = prod.save_cross_rack(&rack_src, "src", &fleet).unwrap();
    assert_eq!(pair, owned_pair("d0", "d1"));

    assert_eq!(
        puts.load(Ordering::SeqCst),
        before_puts,
        "save_cross_rack must not put to the Store"
    );
    assert_eq!(
        &*map.lock().expect("map"),
        &before_map,
        "save_cross_rack must not change Store keys"
    );
    assert_checkpoint_roundtrip(
        &prod.restore(&persisted_id).unwrap(),
        &prepared_of(&persisted),
    );
    assert_eq!(prod.restore_latest_memory().unwrap(), mem_restored);
    assert_eq!(prod.replica_len(0).unwrap(), n0);
    assert_eq!(prod.replica_len(1).unwrap(), n1);
    assert_eq!(prod.restore_from_rack("d0").unwrap(), rack_prepared);
    assert_eq!(prod.restore_from_rack("d1").unwrap(), rack_prepared);
    assert_not_found(&prod.restore_from_rack("src").unwrap_err(), "src");

    // save_in_memory after a cross-rack save still updates only the replicas.
    let mem2 = varied("memory-2", 6);
    let mem2_prepared = prepared_of(&mem2);
    prod.save_in_memory(&mem2).unwrap();
    assert_eq!(prod.restore_latest_memory().unwrap(), mem2_prepared);
    assert_eq!(prod.replica_len(0).unwrap(), prod.replica_len(1).unwrap());
    assert!(prod.replica_len(0).unwrap() >= payload_len(&mem2));
    assert_eq!(prod.restore_from_rack("d0").unwrap(), rack_prepared);
    assert_eq!(puts.load(Ordering::SeqCst), before_puts);
    assert_eq!(&*map.lock().expect("map"), &before_map);

    // A checkpointer that only cross-saves does not populate memory replicas
    // and does not write an empty store.
    let store = CountingStore::new();
    let map = Arc::clone(&store.map);
    let puts = Arc::clone(&store.puts);
    let mut bare = Checkpointer::new(Box::new(store));
    let pair = bare.save_cross_rack(&rack_src, "src", &fleet).unwrap();
    assert_eq!(pair, owned_pair("d0", "d1"));
    assert_eq!(puts.load(Ordering::SeqCst), 0);
    assert!(map.lock().expect("map").is_empty());
    assert_not_found(&bare.restore_latest_memory().unwrap_err(), "latest");
    match bare.replica_len(0) {
        Err(CkptError::ReplicaLost(0)) => {}
        other => panic!("expected ReplicaLost(0), got {other:?}"),
    }
    match bare.replica_len(1) {
        Err(CkptError::ReplicaLost(1)) => {}
        other => panic!("expected ReplicaLost(1), got {other:?}"),
    }
    assert_eq!(bare.restore_from_rack("d0").unwrap(), rack_prepared);

    // Memory save on that checkpointer still works and leaves the rack copies.
    bare.save_in_memory(&mem).unwrap();
    assert_eq!(bare.restore_latest_memory().unwrap(), mem_prepared);
    assert_eq!(bare.restore_from_rack("d1").unwrap(), rack_prepared);
    assert_eq!(puts.load(Ordering::SeqCst), 0);
}

#[test]
fn lose_rack_drops_one_copy_and_keeps_the_last_pair() {
    let mut prod = fresh();
    let src = varied("lose", 11);
    let expect = prepared_of(&src);
    let fleet = ids(&["s", "a", "b", "c"]);
    let pair = prod.save_cross_rack(&src, "s", &fleet).unwrap();
    assert_eq!(pair, owned_pair("a", "b"));

    prod.lose_rack("a").unwrap();
    assert_not_found(&prod.restore_from_rack("a").unwrap_err(), "a");
    assert_eq!(prod.restore_from_rack("b").unwrap(), expect);
    assert_eq!(prod.cross_rack_destinations("s").unwrap(), pair);

    // A rack with no copy: source, never chosen, unknown, already lost.
    assert_not_found(&prod.lose_rack("s").unwrap_err(), "s");
    assert_not_found(&prod.lose_rack("c").unwrap_err(), "c");
    assert_not_found(&prod.lose_rack("missing").unwrap_err(), "missing");
    assert_not_found(&prod.lose_rack("a").unwrap_err(), "a");
    assert_not_found(&prod.lose_rack("").unwrap_err(), "");
    assert_eq!(prod.restore_from_rack("b").unwrap(), expect);
    assert_eq!(prod.cross_rack_destinations("s").unwrap(), pair);

    prod.lose_rack("b").unwrap();
    assert_not_found(&prod.restore_from_rack("a").unwrap_err(), "a");
    assert_not_found(&prod.restore_from_rack("b").unwrap_err(), "b");
    assert_eq!(prod.cross_rack_destinations("s").unwrap(), pair);
    assert_not_found(&prod.lose_rack("b").unwrap_err(), "b");
    assert_eq!(prod.cross_rack_destinations("s").unwrap(), pair);

    let mut bare = fresh();
    assert_not_found(&bare.lose_rack("a").unwrap_err(), "a");
    assert_not_found(&bare.restore_from_rack("a").unwrap_err(), "a");
    assert_not_found(&bare.cross_rack_destinations("s").unwrap_err(), "s");
}

#[test]
fn corrupt_rack_flips_one_copy_only() {
    let fleet = ids(&["s", "a", "b", "c"]);

    let mut weights = fresh();
    let src = checkpoint_from_shards(
        "flip-w",
        vec![("embed", 0, vec![4], Dtype::Bf16, b"wxyz".to_vec(), vec![])],
        vec![],
    );
    let expect = prepared_of(&src);
    let pair = weights.save_cross_rack(&src, "s", &fleet).unwrap();
    assert_eq!(pair, owned_pair("a", "b"));
    weights.corrupt_rack("a").unwrap();
    assert_hash_mismatch(
        &weights.restore_from_rack("a").unwrap_err(),
        Some(&expect.manifest.weights[0].content_hash),
    );
    assert_eq!(weights.restore_from_rack("b").unwrap(), expect);
    assert_eq!(weights.cross_rack_destinations("s").unwrap(), pair);
    // Losing the intact dest does not repair or clear the corrupt copy.
    weights.lose_rack("b").unwrap();
    assert_hash_mismatch(
        &weights.restore_from_rack("a").unwrap_err(),
        Some(&expect.manifest.weights[0].content_hash),
    );
    assert_not_found(&weights.restore_from_rack("b").unwrap_err(), "b");
    assert_eq!(weights.cross_rack_destinations("s").unwrap(), pair);

    let mut optim = fresh();
    let src = checkpoint_from_shards(
        "flip-o",
        vec![],
        vec![("embed.m", 0, OptimizerKind::Adamw, b"opt-bytes".to_vec())],
    );
    let expect = prepared_of(&src);
    optim.save_cross_rack(&src, "s", &fleet).unwrap();
    optim.corrupt_rack("b").unwrap();
    assert_hash_mismatch(
        &optim.restore_from_rack("b").unwrap_err(),
        Some(&expect.manifest.optimizer[0].content_hash),
    );
    assert_eq!(optim.restore_from_rack("a").unwrap(), expect);

    // First blob is empty; a later blob has bytes. That byte can be flipped.
    let mut mixed = fresh();
    let src = checkpoint_from_shards(
        "mixed-bytes",
        vec![
            ("empty", 0, vec![0], Dtype::Fp32, vec![], vec![]),
            ("full", 1, vec![4], Dtype::Fp32, vec![1, 2, 3, 4], vec![]),
        ],
        vec![],
    );
    let expect = prepared_of(&src);
    mixed.save_cross_rack(&src, "s", &fleet).unwrap();
    mixed.corrupt_rack("a").unwrap();
    assert_hash_mismatch(
        &mixed.restore_from_rack("a").unwrap_err(),
        Some(&expect.manifest.weights[1].content_hash),
    );
    assert_eq!(mixed.restore_from_rack("b").unwrap(), expect);

    // No shard bytes: no blobs, and blobs whose byte vecs are empty.
    for (label, src) in [
        ("blank", blank_checkpoint("blank")),
        (
            "empty-bytes",
            checkpoint_from_shards(
                "empty-bytes",
                vec![("embed", 0, vec![0], Dtype::Fp32, vec![], vec![])],
                vec![("embed.m", 0, OptimizerKind::Muon, vec![])],
            ),
        ),
    ] {
        let mut prod = fresh();
        let expect = prepared_of(&src);
        prod.save_cross_rack(&src, "s", &fleet).unwrap();
        assert_rack(&prod.corrupt_rack("a").unwrap_err());
        assert_eq!(
            prod.restore_from_rack("a").unwrap(),
            expect,
            "{label} copy must stay intact when corrupt returns Rack"
        );
        assert_eq!(prod.restore_from_rack("b").unwrap(), expect, "{label}");
        assert_eq!(
            prod.cross_rack_destinations("s").unwrap(),
            owned_pair("a", "b")
        );
    }

    let mut prod = fresh();
    let src = varied("empty-rack", 1);
    let expect = prepared_of(&src);
    prod.save_cross_rack(&src, "s", &fleet).unwrap();
    assert_not_found(&prod.corrupt_rack("c").unwrap_err(), "c");
    assert_not_found(&prod.corrupt_rack("s").unwrap_err(), "s");
    assert_not_found(&prod.corrupt_rack("missing").unwrap_err(), "missing");
    assert_not_found(&prod.corrupt_rack("").unwrap_err(), "");
    assert_not_found(&prod.corrupt_rack("a ").unwrap_err(), "a ");
    assert_eq!(prod.restore_from_rack("a").unwrap(), expect);
    assert_eq!(prod.restore_from_rack("b").unwrap(), expect);

    // Prefix of a destination id is a different rack.
    let mut prefixed = fresh();
    let fleet = ids(&["s", "ab", "c", "d"]);
    let src = varied("prefix", 2);
    let expect = prepared_of(&src);
    prefixed.save_cross_rack(&src, "s", &fleet).unwrap();
    assert_not_found(&prefixed.corrupt_rack("a").unwrap_err(), "a");
    assert_eq!(prefixed.restore_from_rack("ab").unwrap(), expect);
    assert_eq!(prefixed.restore_from_rack("c").unwrap(), expect);

    let mut bare = fresh();
    assert_not_found(&bare.corrupt_rack("a").unwrap_err(), "a");
}

#[test]
fn occupied_destination_is_rack_and_changes_nothing() {
    // Partial overlap: s2's pair is ("c", "a"). "c" is free, "a" holds s1.
    // The failed call must not write "c".
    let mut prod = fresh();
    let fleet = ids(&["s1", "a", "b", "x", "s2", "c", "a"]);
    assert_eq!(
        ref_plan_cross_rack(&fleet, "s1").unwrap(),
        owned_pair("a", "b")
    );
    assert_eq!(
        ref_plan_cross_rack(&fleet, "s2").unwrap(),
        owned_pair("c", "a")
    );
    let s1 = varied("src-1", 21);
    let s1p = prepared_of(&s1);
    let pair = prod.save_cross_rack(&s1, "s1", &fleet).unwrap();
    assert_eq!(pair, owned_pair("a", "b"));

    let err = prod
        .save_cross_rack(&varied("src-2", 22), "s2", &fleet)
        .unwrap_err();
    assert_rack(&err);
    assert_eq!(prod.restore_from_rack("a").unwrap(), s1p);
    assert_eq!(prod.restore_from_rack("b").unwrap(), s1p);
    assert_not_found(&prod.restore_from_rack("c").unwrap_err(), "c");
    assert_not_found(&prod.restore_from_rack("s1").unwrap_err(), "s1");
    assert_not_found(&prod.restore_from_rack("s2").unwrap_err(), "s2");
    assert_eq!(prod.cross_rack_destinations("s1").unwrap(), pair);
    assert_not_found(&prod.cross_rack_destinations("s2").unwrap_err(), "s2");

    // Both a planned dest and a wrap onto the other source's dest.
    let mut prod = fresh();
    let fleet = ids(&["s1", "a", "b", "s2"]);
    assert_eq!(
        ref_plan_cross_rack(&fleet, "s2").unwrap(),
        owned_pair("s1", "a")
    );
    let s1 = varied("wrap-owner", 23);
    let s1p = prepared_of(&s1);
    prod.save_cross_rack(&s1, "s1", &fleet).unwrap();
    assert_rack(
        &prod
            .save_cross_rack(&varied("wrap-other", 24), "s2", &fleet)
            .unwrap_err(),
    );
    assert_eq!(prod.restore_from_rack("a").unwrap(), s1p);
    assert_eq!(prod.restore_from_rack("b").unwrap(), s1p);
    assert_not_found(&prod.restore_from_rack("s1").unwrap_err(), "s1");
    assert_not_found(&prod.cross_rack_destinations("s2").unwrap_err(), "s2");
    assert_eq!(
        prod.cross_rack_destinations("s1").unwrap(),
        owned_pair("a", "b")
    );
}

#[test]
fn resave_replaces_copies_and_drops_a_dest_that_still_holds_this_source() {
    let mut prod = fresh();
    let fleet = ids(&["s", "keep", "drop", "extra"]);
    let v1 = varied("v1", 31);
    let p1 = prepared_of(&v1);
    assert_eq!(
        prod.save_cross_rack(&v1, "s", &fleet).unwrap(),
        owned_pair("keep", "drop")
    );
    assert_eq!(prod.restore_from_rack("keep").unwrap(), p1);

    let v2 = varied("v2", 32);
    let p2 = prepared_of(&v2);
    assert_ne!(p1, p2);
    assert_eq!(
        prod.save_cross_rack(&v2, "s", &fleet).unwrap(),
        owned_pair("keep", "drop")
    );
    assert_eq!(prod.restore_from_rack("keep").unwrap(), p2);
    assert_eq!(prod.restore_from_rack("drop").unwrap(), p2);
    assert_eq!(
        prod.cross_rack_destinations("s").unwrap(),
        owned_pair("keep", "drop")
    );

    // Clean drop of a dest that still holds this source.
    let fleet2 = ids(&["s", "keep", "extra", "drop"]);
    assert_eq!(
        ref_plan_cross_rack(&fleet2, "s").unwrap(),
        owned_pair("keep", "extra")
    );
    let v3 = varied("v3", 33);
    let p3 = prepared_of(&v3);
    assert_eq!(
        prod.save_cross_rack(&v3, "s", &fleet2).unwrap(),
        owned_pair("keep", "extra")
    );
    assert_eq!(prod.restore_from_rack("keep").unwrap(), p3);
    assert_eq!(prod.restore_from_rack("extra").unwrap(), p3);
    assert_not_found(&prod.restore_from_rack("drop").unwrap_err(), "drop");
    assert_eq!(
        prod.cross_rack_destinations("s").unwrap(),
        owned_pair("keep", "extra")
    );

    // A lost dest that is still in the pair is written again.
    prod.lose_rack("keep").unwrap();
    let v4 = varied("v4", 34);
    let p4 = prepared_of(&v4);
    assert_eq!(
        prod.save_cross_rack(&v4, "s", &fleet2).unwrap(),
        owned_pair("keep", "extra")
    );
    assert_eq!(prod.restore_from_rack("keep").unwrap(), p4);
    assert_eq!(prod.restore_from_rack("extra").unwrap(), p4);

    // A corrupt copy still holds this source, so leaving the pair drops it.
    prod.corrupt_rack("extra").unwrap();
    let v5 = varied("v5", 35);
    let p5 = prepared_of(&v5);
    assert_eq!(
        prod.save_cross_rack(&v5, "s", &fleet).unwrap(),
        owned_pair("keep", "drop")
    );
    assert_eq!(prod.restore_from_rack("keep").unwrap(), p5);
    assert_eq!(prod.restore_from_rack("drop").unwrap(), p5);
    assert_not_found(&prod.restore_from_rack("extra").unwrap_err(), "extra");
    assert_eq!(
        prod.cross_rack_destinations("s").unwrap(),
        owned_pair("keep", "drop")
    );
}

#[test]
fn resave_does_not_drop_a_rack_held_by_another_source() {
    let mut prod = fresh();
    let fleet = ids(&["s1", "a", "b", "s2", "c", "d"]);
    let s1 = varied("s1-v1", 41);
    let s1p = prepared_of(&s1);
    let s2 = varied("s2-v1", 42);
    let s2p = prepared_of(&s2);
    assert_eq!(
        prod.save_cross_rack(&s1, "s1", &fleet).unwrap(),
        owned_pair("a", "b")
    );
    assert_eq!(
        prod.save_cross_rack(&s2, "s2", &fleet).unwrap(),
        owned_pair("c", "d")
    );
    assert_eq!(prod.restore_from_rack("c").unwrap(), s2p);
    assert_eq!(prod.restore_from_rack("d").unwrap(), s2p);
    assert_eq!(prod.restore_from_rack("b").unwrap(), s1p);

    // "a" no longer holds s1. s2 may take it. s1's recorded pair stays.
    prod.lose_rack("a").unwrap();
    assert_eq!(
        prod.cross_rack_destinations("s1").unwrap(),
        owned_pair("a", "b")
    );
    let fleet_s2 = ids(&["s2", "a", "c", "x"]);
    assert_eq!(
        ref_plan_cross_rack(&fleet_s2, "s2").unwrap(),
        owned_pair("a", "c")
    );
    let s2b = varied("s2-v2", 43);
    let s2bp = prepared_of(&s2b);
    assert_eq!(
        prod.save_cross_rack(&s2b, "s2", &fleet_s2).unwrap(),
        owned_pair("a", "c")
    );
    assert_eq!(prod.restore_from_rack("a").unwrap(), s2bp);
    assert_eq!(prod.restore_from_rack("c").unwrap(), s2bp);
    assert_not_found(&prod.restore_from_rack("d").unwrap_err(), "d");
    assert_eq!(prod.restore_from_rack("b").unwrap(), s1p);
    assert_eq!(
        prod.cross_rack_destinations("s1").unwrap(),
        owned_pair("a", "b")
    );
    assert_eq!(
        prod.cross_rack_destinations("s2").unwrap(),
        owned_pair("a", "c")
    );

    // s1's new pair excludes "a". "a" holds s2, so it must not be dropped.
    let fleet_s1 = ids(&["s1", "b", "e", "a"]);
    assert_eq!(
        ref_plan_cross_rack(&fleet_s1, "s1").unwrap(),
        owned_pair("b", "e")
    );
    let s1b = varied("s1-v2", 44);
    let s1bp = prepared_of(&s1b);
    assert_eq!(
        prod.save_cross_rack(&s1b, "s1", &fleet_s1).unwrap(),
        owned_pair("b", "e")
    );
    assert_eq!(prod.restore_from_rack("b").unwrap(), s1bp);
    assert_eq!(prod.restore_from_rack("e").unwrap(), s1bp);
    assert_eq!(prod.restore_from_rack("a").unwrap(), s2bp);
    assert_eq!(prod.restore_from_rack("c").unwrap(), s2bp);
    assert_eq!(
        prod.cross_rack_destinations("s1").unwrap(),
        owned_pair("b", "e")
    );
    assert_eq!(
        prod.cross_rack_destinations("s2").unwrap(),
        owned_pair("a", "c")
    );

    // A re-save whose new dest holds the other source changes nothing.
    let fleet_conflict = ids(&["s1", "a", "b", "e"]);
    assert_eq!(
        ref_plan_cross_rack(&fleet_conflict, "s1").unwrap(),
        owned_pair("a", "b")
    );
    assert_rack(
        &prod
            .save_cross_rack(&varied("s1-nope", 45), "s1", &fleet_conflict)
            .unwrap_err(),
    );
    assert_eq!(prod.restore_from_rack("a").unwrap(), s2bp);
    assert_eq!(prod.restore_from_rack("b").unwrap(), s1bp);
    assert_eq!(prod.restore_from_rack("e").unwrap(), s1bp);
    assert_eq!(
        prod.cross_rack_destinations("s1").unwrap(),
        owned_pair("b", "e")
    );
    assert_eq!(
        prod.cross_rack_destinations("s2").unwrap(),
        owned_pair("a", "c")
    );
}

#[test]
fn two_sources_with_disjoint_pairs_both_restore() {
    let mut prod = fresh();
    let fleet = ids(&["s1", "a", "b", "s2", "c", "d"]);
    let s1 = varied("left", 51);
    let s2 = varied("right", 52);
    let p1 = prepared_of(&s1);
    let p2 = prepared_of(&s2);
    assert_ne!(p1, p2);
    let pair1 = prod.save_cross_rack(&s1, "s1", &fleet).unwrap();
    let pair2 = prod.save_cross_rack(&s2, "s2", &fleet).unwrap();
    assert_eq!(pair1, owned_pair("a", "b"));
    assert_eq!(pair2, owned_pair("c", "d"));
    assert_eq!(prod.restore_from_rack("a").unwrap(), p1);
    assert_eq!(prod.restore_from_rack("b").unwrap(), p1);
    assert_eq!(prod.restore_from_rack("c").unwrap(), p2);
    assert_eq!(prod.restore_from_rack("d").unwrap(), p2);
    assert_not_found(&prod.restore_from_rack("s1").unwrap_err(), "s1");
    assert_not_found(&prod.restore_from_rack("s2").unwrap_err(), "s2");

    let mut got = prod.restore_from_rack("a").unwrap();
    got.weights[0].bytes[0] ^= 0xff;
    assert_eq!(prod.restore_from_rack("a").unwrap(), p1);
    assert_eq!(prod.restore_from_rack("c").unwrap(), p2);

    prod.lose_rack("a").unwrap();
    prod.corrupt_rack("c").unwrap();
    assert_eq!(prod.restore_from_rack("b").unwrap(), p1);
    assert_eq!(prod.restore_from_rack("d").unwrap(), p2);
    assert_eq!(prod.cross_rack_destinations("s1").unwrap(), pair1);
    assert_eq!(prod.cross_rack_destinations("s2").unwrap(), pair2);
}

#[test]
fn rejected_checkpoint_matches_save_in_memory_and_writes_nothing() {
    let fleet = ids(&["s", "a", "b", "c"]);
    let mut missing_weight = tiny_checkpoint_with_id("miss-w");
    missing_weight.weights.clear();
    let mut missing_optim = tiny_checkpoint_with_id("miss-o");
    missing_optim.optimizer.clear();
    let mut bad_weight = tiny_checkpoint_with_id("bad-w");
    bad_weight.manifest.weights[0].content_hash = "11".repeat(32);
    let mut bad_optim = tiny_checkpoint_with_id("bad-o");
    bad_optim.manifest.optimizer[0].content_hash = "22".repeat(32);
    let mut empty_then_bad = tiny_checkpoint_with_id("empty-then-bad");
    empty_then_bad.manifest.weights[0].content_hash.clear();
    empty_then_bad.manifest.optimizer[0].content_hash = "33".repeat(32);

    for bad in [
        missing_weight,
        missing_optim,
        bad_weight,
        bad_optim,
        empty_then_bad,
    ] {
        let mut mem = fresh();
        let mut rack = fresh();
        let mem_err = mem.save_in_memory(&bad).unwrap_err();
        let rack_err = rack.save_cross_rack(&bad, "s", &fleet).unwrap_err();
        assert_eq!(rack_err, mem_err, "id={}", bad.manifest.checkpoint_id);
        assert_not_found(&rack.restore_from_rack("a").unwrap_err(), "a");
        assert_not_found(&rack.restore_from_rack("b").unwrap_err(), "b");
        assert_not_found(&rack.cross_rack_destinations("s").unwrap_err(), "s");
        match rack.replica_len(0) {
            Err(CkptError::ReplicaLost(0)) => {}
            other => panic!("failed save must not fill replicas, got {other:?}"),
        }
    }

    // Same MissingShard payload as save_in_memory, not a generic Rack.
    let mut missing = tiny_checkpoint();
    missing.weights.clear();
    let mut rack = fresh();
    assert_missing_shard(
        &rack.save_cross_rack(&missing, "s", &fleet).unwrap_err(),
        "embed",
        0,
    );

    let good = varied("good", 61);
    let mut rack = fresh();
    assert_rack(&rack.save_cross_rack(&good, "", &fleet).unwrap_err());
    assert_rack(&rack.save_cross_rack(&good, "missing", &fleet).unwrap_err());
    assert_rack(
        &rack
            .save_cross_rack(&good, "s", &ids(&["s", "only"]))
            .unwrap_err(),
    );
    assert_rack(
        &rack
            .save_cross_rack(&good, "s", &ids(&["s", "a", ""]))
            .unwrap_err(),
    );
    assert_rack(
        &rack
            .save_cross_rack(&good, "a", &ids(&["a ", "b", "c", "d"]))
            .unwrap_err(),
    );
    assert_not_found(&rack.cross_rack_destinations("s").unwrap_err(), "s");
    assert_not_found(&rack.restore_from_rack("a").unwrap_err(), "a");
    assert_not_found(&rack.restore_from_rack("b").unwrap_err(), "b");

    // A failed re-save does not write and does not drop the previous pair.
    let mut prod = fresh();
    let src = varied("kept", 62);
    let expect = prepared_of(&src);
    let pair = prod.save_cross_rack(&src, "s", &fleet).unwrap();
    assert_eq!(pair, owned_pair("a", "b"));

    let mut bad = src.clone();
    bad.manifest.weights[0].content_hash = "44".repeat(32);
    let mut mem = fresh();
    let mem_err = mem.save_in_memory(&bad).unwrap_err();
    let rack_err = prod.save_cross_rack(&bad, "s", &fleet).unwrap_err();
    assert_eq!(rack_err, mem_err);
    assert_eq!(prod.restore_from_rack("a").unwrap(), expect);
    assert_eq!(prod.restore_from_rack("b").unwrap(), expect);
    assert_eq!(prod.cross_rack_destinations("s").unwrap(), pair);

    assert_rack(
        &prod
            .save_cross_rack(&src, "s", &ids(&["s", "only"]))
            .unwrap_err(),
    );
    assert_rack(&prod.save_cross_rack(&src, "nope", &fleet).unwrap_err());
    assert_eq!(prod.restore_from_rack("a").unwrap(), expect);
    assert_eq!(prod.restore_from_rack("b").unwrap(), expect);
    assert_eq!(prod.cross_rack_destinations("s").unwrap(), pair);
}

#[test]
fn cross_rack_destinations_not_found_until_save_and_survives_loss() {
    let mut prod = fresh();
    assert_not_found(&prod.cross_rack_destinations("s").unwrap_err(), "s");
    assert_not_found(&prod.cross_rack_destinations("").unwrap_err(), "");

    let fleet = ids(&["s", "a", "b", "c"]);
    let src = varied("tracked", 71);
    let expect = prepared_of(&src);
    let pair = prod.save_cross_rack(&src, "s", &fleet).unwrap();
    assert_eq!(pair, owned_pair("a", "b"));
    assert_eq!(prod.cross_rack_destinations("s").unwrap(), pair);
    // A destination is not a source key, even while it holds a copy.
    assert_not_found(&prod.cross_rack_destinations("a").unwrap_err(), "a");
    assert_not_found(&prod.cross_rack_destinations("b").unwrap_err(), "b");

    prod.corrupt_rack("a").unwrap();
    assert_eq!(prod.cross_rack_destinations("s").unwrap(), pair);
    assert_eq!(prod.restore_from_rack("b").unwrap(), expect);
    prod.lose_rack("a").unwrap();
    assert_eq!(prod.cross_rack_destinations("s").unwrap(), pair);
    prod.lose_rack("b").unwrap();
    assert_eq!(prod.cross_rack_destinations("s").unwrap(), pair);
    assert_not_found(&prod.restore_from_rack("a").unwrap_err(), "a");
    assert_not_found(&prod.restore_from_rack("b").unwrap_err(), "b");
}

fn blank_checkpoint(id: &str) -> Checkpoint {
    let mut ckpt = tiny_checkpoint_with_id(id);
    ckpt.manifest.weights.clear();
    ckpt.manifest.optimizer.clear();
    ckpt.weights.clear();
    ckpt.optimizer.clear();
    ckpt
}
