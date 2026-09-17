//! Group: Scores::keep, LocalScorer vs reference, run batching, ServeScorer.

mod common;
mod reference;

use common::{
    assert_close, assert_empty, assert_preds_eq, assert_serve, assert_wrong_state, config_shards,
    doc, doc_hashed, doc_n, fresh_orch_dir, BoomScorer, RecordingScorer, WrongLenScorer, ZeroScorer,
};
use prometheus_classifiers::{
    Error, LocalScorer, Orchestrator, Scores, Scorer, ServeScorer, DEFAULT_QUALITY_THRESHOLD,
    DEFAULT_SAFETY_THRESHOLD,
};
use prometheus_providers::{AimdConfig, ProviderId};
use prometheus_serve::{EngineConfig, LocalEngine, Router};
use reference::{RefLocalScorer, RefOrchestrator};

#[test]
fn keep_at_threshold() {
    let s = Scores {
        quality: 0.5,
        safety: 0.5,
    };
    assert!(s.keep(DEFAULT_QUALITY_THRESHOLD, DEFAULT_SAFETY_THRESHOLD));
    assert!(s.keep(0.5, 0.5));
}

#[test]
fn keep_fails_if_quality_below() {
    let s = Scores {
        quality: 0.49,
        safety: 1.0,
    };
    assert!(!s.keep(0.5, 0.5));
}

#[test]
fn keep_fails_if_safety_below() {
    let s = Scores {
        quality: 1.0,
        safety: 0.49,
    };
    assert!(!s.keep(0.5, 0.5));
}

#[test]
fn keep_requires_both() {
    let s = Scores {
        quality: 0.0,
        safety: 0.0,
    };
    assert!(!s.keep(0.5, 0.5));
    let s = Scores {
        quality: 0.5,
        safety: 0.5,
    };
    assert!(!s.keep(0.51, 0.5));
    assert!(!s.keep(0.5, 0.51));
}

#[test]
fn local_scorer_deterministic() {
    let texts = vec!["alpha".to_string(), "beta".to_string()];
    let mut a = LocalScorer { seed: 7 };
    let mut b = LocalScorer { seed: 7 };
    let sa = a.score(&texts, 1).expect("a");
    let sb = b.score(&texts, 99).expect("b");
    assert_eq!(sa, sb);
}

#[test]
fn local_scorer_ignores_now() {
    let texts = vec!["clock".to_string()];
    let mut s = LocalScorer { seed: 1 };
    let a = s.score(&texts, 0).expect("t0");
    let b = s.score(&texts, 1_700_000_000_000).expect("t1");
    assert_eq!(a, b);
}

#[test]
fn local_scorer_empty_is_empty() {
    let mut s = LocalScorer { seed: 0 };
    assert_empty(s.score(&[], 0), "LocalScorer empty texts");
}

#[test]
fn local_scorer_matches_reference() {
    let texts: Vec<String> = (0..16).map(|i| format!("sample {i} π")).collect();
    let mut prod = LocalScorer { seed: 12345 };
    let mut refer = RefLocalScorer { seed: 12345 };
    let got = prod.score(&texts, 42).expect("prod");
    let exp = refer.score(&texts, 99).expect("ref");
    assert_eq!(got.len(), exp.len());
    for (i, (g, e)) in got.iter().zip(exp.iter()).enumerate() {
        assert_close(g.quality, e.quality, &format!("q[{i}]"));
        assert_close(g.safety, e.safety, &format!("s[{i}]"));
    }
}

#[test]
fn local_scorer_len_and_range() {
    let texts = vec!["".to_string(), "x".to_string(), "unicøde 🧪".to_string()];
    let mut s = LocalScorer { seed: 9 };
    let got = s.score(&texts, 0).expect("score");
    assert_eq!(got.len(), 3);
    for (i, sc) in got.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&sc.quality),
            "q[{i}]={}",
            sc.quality
        );
        assert!((0.0..=1.0).contains(&sc.safety), "s[{i}]={}", sc.safety);
    }
}

#[test]
fn local_scorer_quality_safety_independent() {
    let mut s = LocalScorer { seed: 0 };
    let mut found = false;
    for i in 0..32 {
        let t = vec![format!("probe-{i}")];
        let sc = s.score(&t, 0).expect("score");
        if (sc[0].quality - sc[0].safety).abs() > 1e-5 {
            found = true;
            break;
        }
    }
    assert!(found, "quality and safety must not be identical for all texts");
}

#[test]
fn local_scorer_seed_matters() {
    let texts = vec!["seeded".to_string()];
    let a = LocalScorer { seed: 1 }.score(&texts, 0).expect("s1");
    let b = LocalScorer { seed: 2 }.score(&texts, 0).expect("s2");
    assert!(
        a[0].quality != b[0].quality || a[0].safety != b[0].safety,
        "different seeds must change scores for this text"
    );
}

#[test]
fn run_empty_is_empty() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(4)).expect("create");
    assert_empty(orch.run(&mut ZeroScorer, 0), "run with nothing ingested");
}

#[test]
fn run_calls_scorer_once_per_shard() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(2)).expect("create");
    orch.ingest(doc("a")).expect("a");
    orch.ingest(doc("b")).expect("b");
    orch.ingest(doc("c")).expect("c");
    let mut rec = RecordingScorer::new(ZeroScorer);
    orch.run(&mut rec, 123).expect("run");
    assert_eq!(rec.calls.len(), 2, "two shards");
    assert_eq!(rec.calls[0], vec!["a".to_string(), "b".to_string()]);
    assert_eq!(rec.calls[1], vec!["c".to_string()]);
    assert_eq!(rec.nows, vec![123, 123]);
}

#[test]
fn predictions_align_with_ingest() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_shards(2);
    let mut orch = Orchestrator::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefOrchestrator::new(cfg);
    for i in 0..5 {
        let d = doc_n(i);
        orch.ingest(d.clone()).expect("ingest");
        refer.ingest(d).expect("ref");
    }
    let mut prod_scorer = LocalScorer { seed: 99 };
    let mut ref_scorer = RefLocalScorer { seed: 99 };
    let got = orch.run(&mut prod_scorer, 7).expect("run");
    let exp = refer.run(&mut ref_scorer, 7).expect("ref run");
    assert_preds_eq(&got, &exp, "run return");
    assert_preds_eq(orch.predictions(), &exp, "stored");
}

#[test]
fn prediction_hash_from_document_not_rehashed() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(4)).expect("create");
    orch.ingest(doc_hashed("body", "not-a-sha256"))
        .expect("ingest");
    orch.run(&mut ZeroScorer, 0).expect("run");
    let p = &orch.predictions()[0];
    assert_eq!(p.content_hash, "not-a-sha256");
    assert_eq!(p.scores.quality, 0.0);
    assert_eq!(p.scores.safety, 0.0);
}

#[test]
fn run_return_equals_predictions() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(8)).expect("create");
    orch.ingest(doc("z")).expect("ingest");
    let out = orch.run(&mut ZeroScorer, 1).expect("run");
    assert_eq!(out.as_slice(), orch.predictions());
}

#[test]
fn wrong_len_scorer_is_wrong_state() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(4)).expect("create");
    orch.ingest(doc("x")).expect("ingest");
    let err = orch.run(&mut WrongLenScorer, 0);
    assert_wrong_state(err, "", "wrong len");
    assert!(
        orch.predictions().is_empty(),
        "no partial commit on scorer length mismatch"
    );
}

#[test]
fn boom_scorer_propagates() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(4)).expect("create");
    orch.ingest(doc("x")).expect("ingest");
    match orch.run(&mut BoomScorer, 0) {
        Err(Error::Serve(msg)) => assert!(msg.contains("boom"), "{msg}"),
        other => panic!("expected Serve(boom), got {other:?}"),
    }
    assert!(
        orch.predictions().is_empty(),
        "no partial commit on scorer error"
    );
}

#[test]
fn run_now_forwarded_to_scorer() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(8)).expect("create");
    orch.ingest(doc("n")).expect("ingest");
    let mut rec = RecordingScorer::new(ZeroScorer);
    orch.run(&mut rec, 9_001).expect("run");
    assert_eq!(rec.nows, vec![9_001]);
}

#[test]
fn second_run_rescores() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(8)).expect("create");
    orch.ingest(doc("n")).expect("ingest");
    orch.run(&mut ZeroScorer, 0).expect("run0");
    assert_eq!(orch.predictions()[0].scores.quality, 0.0);
    let mut ones = common::ConstScorer {
        quality: 1.0,
        safety: 1.0,
    };
    orch.run(&mut ones, 0).expect("run1");
    assert_eq!(orch.predictions()[0].scores.quality, 1.0);
}

fn engine() -> LocalEngine {
    LocalEngine::start(EngineConfig {
        model: prometheus_serve::DEFAULT_MODEL.to_string(),
        gpus: 1,
        max_batch: 8,
        port: 30000,
    })
    .expect("LocalEngine::start")
}

#[test]
fn serve_scorer_empty_texts() {
    let router = Router::new(vec![], engine(), AimdConfig::default()).expect("router");
    let mut scorer = ServeScorer::new(router);
    assert_empty(scorer.score(&[], 0), "ServeScorer empty");
}

#[test]
fn serve_scorer_hosted_is_serve_error() {
    let router = Router::new(
        vec![ProviderId("openai".to_string())],
        engine(),
        AimdConfig::default(),
    )
    .expect("router");
    let mut scorer = ServeScorer::new(router);
    let err = scorer.score(&["hello".to_string()], 0);
    assert_serve(err, "hosted router has no local engine");
}

#[test]
fn serve_scorer_last_resort_len_or_serve() {
    let router = Router::new(vec![], engine(), AimdConfig::default()).expect("router");
    let mut scorer = ServeScorer::new(router);
    let texts = vec!["one".to_string(), "two".to_string()];
    match scorer.score(&texts, 0) {
        Ok(s) => assert_eq!(s.len(), 2, "one score per text"),
        Err(Error::Serve(_)) => {}
        other => panic!("expected Ok(len=2) or Serve, got {other:?}"),
    }
}

#[test]
fn run_with_serve_scorer_empty_ingest_is_empty() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(4)).expect("create");
    let router = Router::new(vec![], engine(), AimdConfig::default()).expect("router");
    let mut scorer = ServeScorer::new(router);
    assert_empty(orch.run(&mut scorer, 0), "run ServeScorer empty ingest");
}
