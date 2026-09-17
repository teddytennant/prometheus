//! Group: drop + open replays ingest and scores.

mod common;
mod reference;

use common::{
    assert_close, assert_preds_eq, assert_shards_eq, config_shards, doc, doc_n, fresh_orch_dir,
    label, ConstScorer,
};
use prometheus_classifiers::{LocalScorer, Orchestrator};
use reference::{RefLocalScorer, RefOrchestrator};

#[test]
fn replay_shards_after_ingest() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_shards(2);
    let mut refer = RefOrchestrator::new(cfg.clone());
    {
        let mut orch = Orchestrator::create(&dir, cfg.clone()).expect("create");
        for i in 0..5 {
            let d = doc_n(i);
            orch.ingest(d.clone()).expect("ingest");
            refer.ingest(d).expect("ref");
        }
        assert_shards_eq(&orch.shards(), &refer.shards(), "before drop");
    }
    let orch = Orchestrator::open(&dir, cfg).expect("open");
    assert_shards_eq(&orch.shards(), &refer.shards(), "after open");
}

#[test]
fn replay_predictions_after_run() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_shards(3);
    let mut refer = RefOrchestrator::new(cfg.clone());
    {
        let mut orch = Orchestrator::create(&dir, cfg.clone()).expect("create");
        for i in 0..4 {
            let d = doc_n(i);
            orch.ingest(d.clone()).expect("ingest");
            refer.ingest(d).expect("ref");
        }
        orch.run(&mut LocalScorer { seed: 42 }, 11).expect("run");
        refer
            .run(&mut RefLocalScorer { seed: 42 }, 11)
            .expect("ref run");
        assert_preds_eq(orch.predictions(), refer.predictions(), "before drop");
    }
    let orch = Orchestrator::open(&dir, cfg).expect("open");
    assert_preds_eq(orch.predictions(), refer.predictions(), "after open");
    assert_shards_eq(&orch.shards(), &refer.shards(), "shards after open");
}

#[test]
fn replay_agreement() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_shards(8);
    let d = doc("labeled");
    let labels = vec![label(&d.content_hash, true, true)];
    {
        let mut orch = Orchestrator::create(&dir, cfg.clone()).expect("create");
        orch.ingest(d.clone()).expect("ingest");
        orch.run(
            &mut ConstScorer {
                quality: 1.0,
                safety: 1.0,
            },
            0,
        )
        .expect("run");
        let a = orch.agreement(&labels).expect("live agree");
        assert_close(a, 1.0, "live");
    }
    let orch = Orchestrator::open(&dir, cfg).expect("open");
    let a = orch.agreement(&labels).expect("replay agree");
    assert_close(a, 1.0, "replay");
}

#[test]
fn replay_without_run_has_no_predictions() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_shards(8);
    {
        let mut orch = Orchestrator::create(&dir, cfg.clone()).expect("create");
        orch.ingest(doc("x")).expect("ingest");
        assert!(orch.predictions().is_empty());
    }
    let orch = Orchestrator::open(&dir, cfg).expect("open");
    assert_eq!(orch.shards()[0].len(), 1);
    assert!(orch.predictions().is_empty(), "open must not invent scores");
}

#[test]
fn replay_gate() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_shards(8);
    let d = doc("g");
    let labels = vec![label(&d.content_hash, true, true)];
    {
        let mut orch = Orchestrator::create(&dir, cfg.clone()).expect("create");
        orch.ingest(d.clone()).expect("ingest");
        orch.run(
            &mut ConstScorer {
                quality: 1.0,
                safety: 1.0,
            },
            0,
        )
        .expect("run");
        orch.gate(&labels).expect("live gate");
    }
    let orch = Orchestrator::open(&dir, cfg).expect("open");
    orch.gate(&labels).expect("replay gate");
}

#[test]
fn open_replays_ingest_and_scores_together() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_shards(2);
    let mut hashes = Vec::new();
    {
        let mut orch = Orchestrator::create(&dir, cfg.clone()).expect("create");
        for i in 0..3 {
            let d = doc_n(i);
            hashes.push(d.content_hash.clone());
            orch.ingest(d).expect("ingest");
        }
        orch.run(&mut LocalScorer { seed: 7 }, 0).expect("run");
    }
    let orch = Orchestrator::open(&dir, cfg).expect("open");
    let shards = orch.shards();
    assert_eq!(shards.len(), 2);
    assert_eq!(shards[0].len(), 2);
    assert_eq!(shards[1].len(), 1);
    let preds = orch.predictions();
    assert_eq!(preds.len(), 3);
    for (h, p) in hashes.iter().zip(preds) {
        assert_eq!(h, &p.content_hash);
    }
}
