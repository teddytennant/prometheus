//! Group: ingest, EVENT_INGEST persistence via later shards, duplicate reject.

mod common;
mod reference;

use common::{
    assert_docs_eq, assert_wrong_state, config_shards, doc, doc_hashed, fresh_orch_dir, ZeroScorer,
};
use prometheus_classifiers::{Error, Orchestrator};
use reference::RefOrchestrator;

#[test]
fn ingest_one_then_single_shard() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(32)).expect("create");
    let d = doc("hello world");
    orch.ingest(d.clone()).expect("ingest");
    let shards = orch.shards();
    assert_eq!(shards.len(), 1);
    assert_docs_eq(&shards[0], &[d], "one doc");
}

#[test]
fn duplicate_content_hash_rejected() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(8)).expect("create");
    let a = doc_hashed("first", "abc123");
    let b = doc_hashed("second", "abc123");
    orch.ingest(a.clone()).expect("first ingest");
    let err = orch.ingest(b);
    assert_wrong_state(err, "abc123", "duplicate hash");
    let shards = orch.shards();
    assert_eq!(shards.len(), 1);
    assert_docs_eq(&shards[0], &[a], "first doc kept");
}

#[test]
fn duplicate_does_not_replace_text() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(8)).expect("create");
    orch.ingest(doc_hashed("keep-me", "h1")).expect("ingest");
    let _ = orch.ingest(doc_hashed("overwrite?", "h1"));
    let got = &orch.shards()[0][0];
    assert_eq!(got.text, "keep-me");
}

#[test]
fn ingest_order_preserved() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(16)).expect("create");
    let mut refer = RefOrchestrator::new(config_shards(16));
    for i in 0..5 {
        let d = doc(&format!("order-{i}"));
        orch.ingest(d.clone()).expect("ingest");
        refer.ingest(d).expect("ref ingest");
    }
    common::assert_shards_eq(
        &orch.shards(),
        &refer.shards(),
        "ingest order",
    );
}

#[test]
fn content_hash_field_not_recomputed() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(4)).expect("create");
    let d = doc_hashed("not-the-sha-of-this-text", "deadbeef");
    orch.ingest(d.clone()).expect("ingest");
    let got = &orch.shards()[0][0];
    assert_eq!(got.content_hash, "deadbeef");
    assert_eq!(got.text, d.text);
}

#[test]
fn empty_text_allowed() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(4)).expect("create");
    let d = doc_hashed("", "empty-hash");
    orch.ingest(d.clone()).expect("empty text ingest");
    assert_eq!(orch.shards()[0][0].text, "");
}

#[test]
fn ingest_after_run_appends() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(8)).expect("create");
    orch.ingest(doc("a")).expect("a");
    orch.run(&mut ZeroScorer, 0).expect("run");
    orch.ingest(doc("b")).expect("b after run");
    let shards = orch.shards();
    assert_eq!(shards.len(), 1);
    assert_eq!(shards[0].len(), 2);
    assert_eq!(shards[0][0].text, "a");
    assert_eq!(shards[0][1].text, "b");
}

#[test]
fn different_text_same_hash_rejected() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(4)).expect("create");
    orch.ingest(doc_hashed("alpha", "same")).expect("first");
    match orch.ingest(doc_hashed("beta", "same")) {
        Err(Error::WrongState(msg)) => {
            assert!(msg.contains("same"), "{msg}");
        }
        other => panic!("expected WrongState, got {other:?}"),
    }
}

#[test]
fn ingest_matches_reference_reject() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(4)).expect("create");
    let mut refer = RefOrchestrator::new(config_shards(4));
    let a = doc("once");
    orch.ingest(a.clone()).expect("prod");
    refer.ingest(a.clone()).expect("ref");
    let dup = doc_hashed("other", &a.content_hash);
    let pe = orch.ingest(dup.clone()).expect_err("prod dup");
    let re = refer.ingest(dup).expect_err("ref dup");
    assert_eq!(common::err_kind(&pe), common::err_kind(&re));
    assert_eq!(common::err_kind(&pe), "wrong_state");
}
