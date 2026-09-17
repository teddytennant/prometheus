//! Group: ingest-order chunks of shard_size; last may be short; empty => empty vec.

mod common;
mod reference;

use common::{assert_shards_eq, config_shards, doc, doc_n, fresh_orch_dir};
use prometheus_classifiers::Orchestrator;
use reference::RefOrchestrator;

fn setup(
    shard_size: usize,
    n_docs: u32,
) -> (
    tempfile::TempDir,
    Orchestrator,
    RefOrchestrator,
) {
    let (parent, dir) = fresh_orch_dir();
    let cfg = config_shards(shard_size);
    let mut orch = Orchestrator::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefOrchestrator::new(cfg);
    for i in 0..n_docs {
        let d = doc_n(i);
        orch.ingest(d.clone()).expect("ingest");
        refer.ingest(d).expect("ref ingest");
    }
    (parent, orch, refer)
}

#[test]
fn empty_ingest_shards_empty_not_error() {
    let (_parent, dir) = fresh_orch_dir();
    let orch = Orchestrator::create(&dir, config_shards(32)).expect("create");
    let shards = orch.shards();
    assert!(shards.is_empty());
}

#[test]
fn exact_multiple_of_shard_size() {
    let (_p, orch, refer) = setup(2, 6);
    let got = orch.shards();
    assert_eq!(got.len(), 3);
    for s in &got {
        assert_eq!(s.len(), 2);
    }
    assert_shards_eq(&got, &refer.shards(), "exact multiple");
}

#[test]
fn last_shard_shorter() {
    let (_p, orch, refer) = setup(3, 5);
    let got = orch.shards();
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].len(), 3);
    assert_eq!(got[1].len(), 2);
    assert_shards_eq(&got, &refer.shards(), "short last");
}

#[test]
fn shard_size_one() {
    let (_p, orch, refer) = setup(1, 4);
    let got = orch.shards();
    assert_eq!(got.len(), 4);
    for s in &got {
        assert_eq!(s.len(), 1);
    }
    assert_shards_eq(&got, &refer.shards(), "size 1");
}

#[test]
fn shard_size_larger_than_n() {
    let (_p, orch, refer) = setup(32, 3);
    let got = orch.shards();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].len(), 3);
    assert_shards_eq(&got, &refer.shards(), "one short shard");
}

#[test]
fn shards_match_reference() {
    let (_p, orch, refer) = setup(5, 12);
    assert_shards_eq(
        &orch.shards(),
        &refer.shards(),
        "vs ref",
    );
}

#[test]
fn shard_texts_are_document_texts_in_ingest_order() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(2)).expect("create");
    orch.ingest(doc("alpha")).expect("a");
    orch.ingest(doc("beta")).expect("b");
    orch.ingest(doc("gamma")).expect("g");
    let shards = orch.shards();
    assert_eq!(shards[0][0].text, "alpha");
    assert_eq!(shards[0][1].text, "beta");
    assert_eq!(shards[1][0].text, "gamma");
}

#[test]
fn default_shard_size_thirty_two() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(32)).expect("create");
    for i in 0..32 {
        orch.ingest(doc_n(i)).expect("ingest");
    }
    let shards = orch.shards();
    assert_eq!(shards.len(), 1);
    assert_eq!(shards[0].len(), 32);
    orch.ingest(doc_n(32)).expect("one more");
    let shards = orch.shards();
    assert_eq!(shards.len(), 2);
    assert_eq!(shards[0].len(), 32);
    assert_eq!(shards[1].len(), 1);
}
