//! Group: gate vs min_agreement. Equality is Ok. Below is GateFailed.

mod common;
mod reference;

use common::{
    assert_close, assert_empty, assert_gate_failed, assert_missing_label, config_full,
    config_shards, doc, fresh_orch_dir, label, pred, ConstScorer, ScriptedScorer,
};
use prometheus_classifiers::{agreement, Orchestrator, Scores, DEFAULT_MIN_AGREEMENT};
use reference::RefOrchestrator;

fn scored_two() -> (tempfile::TempDir, Orchestrator, String, String) {
    let (parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(8)).expect("create");
    let a = doc("aa");
    let b = doc("bb");
    let ha = a.content_hash.clone();
    let hb = b.content_hash.clone();
    orch.ingest(a).expect("a");
    orch.ingest(b).expect("b");
    orch.run(
        &mut ScriptedScorer {
            remaining: vec![
                Scores {
                    quality: 0.9,
                    safety: 0.9,
                },
                Scores {
                    quality: 0.9,
                    safety: 0.9,
                },
            ],
        },
        0,
    )
    .expect("run");
    (parent, orch, ha, hb)
}

#[test]
fn default_min_requires_unanimous() {
    assert_eq!(DEFAULT_MIN_AGREEMENT, 1.0);
    let (_p, orch, ha, hb) = scored_two();
    let labels = vec![label(&ha, true, true), label(&hb, false, false)];
    // second human disagrees => 0.5 < 1.0
    assert_gate_failed(orch.gate(&labels), 0.5, 1.0, "not unanimous");
}

#[test]
fn equality_passes() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_full(8, 0.5, 0.5, 0.5);
    let mut orch = Orchestrator::create(&dir, cfg).expect("create");
    let a = doc("aa");
    let b = doc("bb");
    let ha = a.content_hash.clone();
    let hb = b.content_hash.clone();
    orch.ingest(a).expect("a");
    orch.ingest(b).expect("b");
    orch.run(
        &mut ScriptedScorer {
            remaining: vec![
                Scores {
                    quality: 0.9,
                    safety: 0.9,
                },
                Scores {
                    quality: 0.9,
                    safety: 0.9,
                },
            ],
        },
        0,
    )
    .expect("run");
    let labels = vec![label(&ha, true, true), label(&hb, false, false)];
    orch.gate(&labels).expect("equality is Ok");
    assert_close(orch.agreement(&labels).expect("agree"), 0.5, "got");
}

#[test]
fn below_min_is_gate_failed() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_full(8, 0.5, 0.5, 0.51);
    let mut orch = Orchestrator::create(&dir, cfg).expect("create");
    let a = doc("aa");
    let b = doc("bb");
    let ha = a.content_hash.clone();
    let hb = b.content_hash.clone();
    orch.ingest(a).expect("a");
    orch.ingest(b).expect("b");
    orch.run(
        &mut ScriptedScorer {
            remaining: vec![
                Scores {
                    quality: 0.9,
                    safety: 0.9,
                },
                Scores {
                    quality: 0.9,
                    safety: 0.9,
                },
            ],
        },
        0,
    )
    .expect("run");
    let labels = vec![label(&ha, true, true), label(&hb, false, false)];
    assert_gate_failed(orch.gate(&labels), 0.5, 0.51, "below min");
}

#[test]
fn empty_labels_is_empty() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(4)).expect("create");
    orch.ingest(doc("x")).expect("ingest");
    orch.run(
        &mut ConstScorer {
            quality: 1.0,
            safety: 1.0,
        },
        0,
    )
    .expect("run");
    assert_empty(orch.gate(&[]), "gate empty labels");
}

#[test]
fn missing_label_propagates() {
    let (_parent, dir) = fresh_orch_dir();
    let mut orch = Orchestrator::create(&dir, config_shards(4)).expect("create");
    orch.ingest(doc("x")).expect("ingest");
    orch.run(
        &mut ConstScorer {
            quality: 1.0,
            safety: 1.0,
        },
        0,
    )
    .expect("run");
    assert_missing_label(orch.gate(&[label("nope", true, true)]), "nope", "gate");
}

#[test]
fn perfect_passes() {
    let (_p, orch, ha, hb) = scored_two();
    let labels = vec![label(&ha, true, true), label(&hb, true, true)];
    orch.gate(&labels).expect("unanimous");
    assert_close(orch.agreement(&labels).expect("agree"), 1.0, "perfect gate");
}

#[test]
fn gate_failed_payload() {
    let preds = vec![pred("a", 1.0, 1.0), pred("b", 1.0, 1.0)];
    let labels = vec![label("a", true, true), label("b", false, false)];
    let got = agreement(&preds, &labels, 0.5, 0.5).expect("frac");
    assert_close(got, 0.5, "frac");
    let (_p, orch, ha, hb) = scored_two();
    let labels = vec![label(&ha, true, true), label(&hb, false, false)];
    match orch.gate(&labels) {
        Err(prometheus_classifiers::Error::GateFailed { got, min }) => {
            assert_close(got, 0.5, "got");
            assert_close(min, 1.0, "min default");
        }
        other => panic!("expected GateFailed, got {other:?}"),
    }
}

#[test]
fn gate_matches_reference() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_full(4, 0.5, 0.5, 1.0);
    let mut orch = Orchestrator::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefOrchestrator::new(cfg);
    let d = doc("t");
    orch.ingest(d.clone()).expect("ingest");
    refer.ingest(d.clone()).expect("ref");
    orch.run(
        &mut ConstScorer {
            quality: 1.0,
            safety: 1.0,
        },
        0,
    )
    .expect("run");
    refer
        .run(
            &mut ConstScorer {
                quality: 1.0,
                safety: 1.0,
            },
            0,
        )
        .expect("ref run");
    let ok_labels = vec![label(&d.content_hash, true, true)];
    orch.gate(&ok_labels).expect("prod ok");
    refer.gate(&ok_labels).expect("ref ok");
    let bad = vec![label(&d.content_hash, false, false)];
    let pe = orch.gate(&bad).expect_err("prod fail");
    let re = refer.gate(&bad).expect_err("ref fail");
    assert_eq!(common::err_kind(&pe), common::err_kind(&re));
}

#[test]
fn min_agreement_zero_always_ok_if_labels_score() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_full(4, 0.5, 0.5, 0.0);
    let mut orch = Orchestrator::create(&dir, cfg).expect("create");
    let d = doc("t");
    orch.ingest(d.clone()).expect("ingest");
    orch.run(
        &mut ConstScorer {
            quality: 1.0,
            safety: 1.0,
        },
        0,
    )
    .expect("run");
    let labels = vec![label(&d.content_hash, false, false)];
    orch.gate(&labels).expect("0.0 is not < min 0.0");
    assert_close(
        orch.agreement(&labels).expect("agree"),
        0.0,
        "total mismatch still Ok at min 0",
    );
}
