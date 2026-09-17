//! Group: agreement fraction, MissingLabel, empty labels, unlabeled ignored.

mod common;
mod reference;

use common::{
    assert_close, assert_empty, assert_missing_label, config_full, config_shards, doc, doc_hashed,
    fresh_orch_dir, label, pred, ConstScorer, ScriptedScorer,
};
use prometheus_classifiers::{agreement, Orchestrator, Scores};
use reference::{ref_agreement, RefOrchestrator};

#[test]
fn empty_labels_is_empty() {
    let preds = vec![pred("h", 1.0, 1.0)];
    assert_empty(agreement(&preds, &[], 0.5, 0.5), "free fn empty labels");
}

#[test]
fn missing_label() {
    let preds = vec![pred("have", 1.0, 1.0)];
    let labels = vec![label("missing", true, true)];
    assert_missing_label(
        agreement(&preds, &labels, 0.5, 0.5),
        "missing",
        "no prediction for labeled hash",
    );
}

#[test]
fn perfect_agreement() {
    let preds = vec![pred("a", 0.9, 0.9), pred("b", 0.1, 0.1)];
    let labels = vec![label("a", true, true), label("b", false, false)];
    let g = agreement(&preds, &labels, 0.5, 0.5).expect("agree");
    assert_close(g, 1.0, "perfect");
}

#[test]
fn zero_agreement() {
    let preds = vec![pred("a", 0.9, 0.9)];
    let labels = vec![label("a", false, false)];
    let g = agreement(&preds, &labels, 0.5, 0.5).expect("agree");
    assert_close(g, 0.0, "zero");
}

#[test]
fn half_agreement() {
    let preds = vec![pred("a", 0.9, 0.9), pred("b", 0.9, 0.9)];
    let labels = vec![label("a", true, true), label("b", false, false)];
    let g = agreement(&preds, &labels, 0.5, 0.5).expect("agree");
    assert_close(g, 0.5, "half");
}

#[test]
fn unlabeled_preds_ignored() {
    let preds = vec![pred("labeled", 0.9, 0.9), pred("extra", 0.0, 0.0)];
    let labels = vec![label("labeled", true, true)];
    let g = agreement(&preds, &labels, 0.5, 0.5).expect("agree");
    assert_close(g, 1.0, "unlabeled ignored");
}

#[test]
fn both_axes_must_match() {
    let preds = vec![pred("a", 0.9, 0.1)];
    // keep_q true, keep_s false after 0.5. Quality-only match is not enough.
    let labels = vec![label("a", true, true)];
    let g = agreement(&preds, &labels, 0.5, 0.5).expect("agree");
    assert_close(g, 0.0, "safety mismatch");
    let labels = vec![label("a", false, false)];
    let g = agreement(&preds, &labels, 0.5, 0.5).expect("agree");
    assert_close(g, 0.0, "quality mismatch");
    let labels = vec![label("a", true, false)];
    let g = agreement(&preds, &labels, 0.5, 0.5).expect("agree");
    assert_close(g, 1.0, "both match");
}

#[test]
fn threshold_equality_is_keep() {
    let preds = vec![pred("a", 0.5, 0.5)];
    let labels = vec![label("a", true, true)];
    let g = agreement(&preds, &labels, 0.5, 0.5).expect("agree");
    assert_close(g, 1.0, ">= threshold is keep");
    let labels = vec![label("a", false, false)];
    let g = agreement(&preds, &labels, 0.5, 0.5).expect("agree");
    assert_close(g, 0.0, "human said drop");
}

#[test]
fn matches_reference() {
    let preds = vec![
        pred("a", 0.7, 0.2),
        pred("b", 0.4, 0.8),
        pred("c", 0.9, 0.9),
    ];
    let labels = vec![
        label("a", true, false),
        label("b", false, true),
        label("c", true, true),
    ];
    let got = agreement(&preds, &labels, 0.5, 0.5).expect("prod");
    let exp = ref_agreement(&preds, &labels, 0.5, 0.5).expect("ref");
    assert_close(got, exp, "vs ref");
}

#[test]
fn missing_label_is_first_in_label_order() {
    let preds = vec![pred("keep", 1.0, 1.0)];
    let labels = vec![label("z-missing", true, true), label("a-missing", true, true)];
    assert_missing_label(
        agreement(&preds, &labels, 0.5, 0.5),
        "z-missing",
        "first missing in label order",
    );
}

#[test]
fn extra_labels_missing() {
    let preds = vec![pred("only", 1.0, 1.0)];
    let labels = vec![label("only", true, true), label("ghost", false, false)];
    assert_missing_label(
        agreement(&preds, &labels, 0.5, 0.5),
        "ghost",
        "second label missing",
    );
}

#[test]
fn nondefault_thresholds() {
    let preds = vec![pred("a", 0.6, 0.6)];
    let labels = vec![label("a", false, false)];
    // 0.6 < 0.7 => drop/drop, matches human drop/drop
    let g = agreement(&preds, &labels, 0.7, 0.7).expect("agree");
    assert_close(g, 1.0, "high threshold");
}

#[test]
fn empty_preds_nonempty_labels_missing() {
    assert_missing_label(
        agreement(&[], &[label("h", true, true)], 0.5, 0.5),
        "h",
        "no preds at all",
    );
}

#[test]
fn orchestrator_agreement_matches_free_fn() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_shards(8);
    let mut orch = Orchestrator::create(&dir, cfg).expect("create");
    let a = doc("aa");
    let b = doc("bb");
    orch.ingest(a.clone()).expect("a");
    orch.ingest(b.clone()).expect("b");
    orch.run(
        &mut ScriptedScorer {
            remaining: vec![
                Scores {
                    quality: 0.9,
                    safety: 0.9,
                },
                Scores {
                    quality: 0.1,
                    safety: 0.1,
                },
            ],
        },
        0,
    )
    .expect("run");
    let labels = vec![
        label(&a.content_hash, true, true),
        label(&b.content_hash, false, false),
    ];
    let got = orch.agreement(&labels).expect("orch");
    let exp = agreement(orch.predictions(), &labels, 0.5, 0.5).expect("free");
    assert_close(got, exp, "orch vs free");
    assert_close(got, 1.0, "perfect via orch");
}

#[test]
fn orchestrator_empty_labels_is_empty() {
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
    assert_empty(orch.agreement(&[]), "orch empty labels");
}

#[test]
fn orchestrator_vs_ref_agreement() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_full(4, 0.5, 0.5, 1.0);
    let mut orch = Orchestrator::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefOrchestrator::new(cfg);
    let d = doc_hashed("t", "h1");
    orch.ingest(d.clone()).expect("ingest");
    refer.ingest(d).expect("ref");
    let mut sc = ConstScorer {
        quality: 0.25,
        safety: 0.75,
    };
    orch.run(&mut sc, 0).expect("run");
    refer
        .run(
            &mut ConstScorer {
                quality: 0.25,
                safety: 0.75,
            },
            0,
        )
        .expect("ref run");
    let labels = vec![label("h1", false, true)];
    let got = orch.agreement(&labels).expect("prod");
    let exp = refer.agreement(&labels).expect("ref");
    assert_close(got, exp, "orch vs ref");
}
