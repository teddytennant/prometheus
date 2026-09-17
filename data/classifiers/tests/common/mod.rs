//! Locked B3 oracle rules. Implementers must match these, not invent them.
//!
//! Quality and safety classifiers: Rust owns sharding and bookkeeping; scoring
//! is a [`prometheus_classifiers::Scorer`] (small models in JAX on a slice of
//! the fleet in production). CPU tests drive [`prometheus_classifiers::LocalScorer`]
//! and, for a couple of cases, [`prometheus_classifiers::ServeScorer`] wrapping
//! [`prometheus_serve::Router`] + [`prometheus_serve::LocalEngine`]. There is no
//! live SGLang requirement. There are **no GPU-only tests** in this crate
//! (`gpu` feature / V-stage markers are unused).
//!
//! `NowMs` is an argument on `score` / `run`. Local scoring must not read the
//! wall clock.
//!
//! # Locked decisions (iface left a choice)
//!
//! - **Duplicate `content_hash`:** fail-closed reject. `ingest` returns
//!   [`Error::WrongState`] whose message contains the hash. The first document
//!   is kept. Last-write-wins is not allowed.
//! - **Empty human labels:** `agreement` / `gate` return [`Error::Empty`]
//!   (fail-closed), not `1.0`.
//! - **Label match:** one vote per `HumanLabel`. Both axes must match after
//!   thresholding (`quality >= q_th` iff `keep_quality`, and the same for
//!   safety). Unlabeled predictions do not affect the fraction.
//! - **Missing prediction:** [`Error::MissingLabel`] with the labeled hash.
//!   First missing hash in **label order** (not ingest order).
//! - **`run` on empty ingest:** [`Error::Empty`], scorer is not called.
//! - **Scorer length mismatch:** [`Error::WrongState`]. No partial commit of
//!   predictions (previous `predictions()` stay as they were).
//! - **Scorer error:** propagated as-is. No partial commit.
//! - **`create`:** fails if `dir` already exists ([`Error::WrongState`]),
//!   including when `dir` is a file. `shard_size < 1` is [`Error::BadShardSize`]
//!   and must not create `dir`.
//! - **`open` of a missing path:** [`Error::Other`]. `open` replays ingest and
//!   scores from disk; it does not take a scorer, so scores must be persisted
//!   at `run`. Persistence format is not part of the public API.
//! - **`LocalScorer`:** deterministic function of `(seed, texts)` only. See
//!   `tests/reference`. Empty `texts` => [`Error::Empty`].
//! - **`ServeScorer`:** empty `texts` => [`Error::Empty`] (do not hit the
//!   router). Router errors become [`Error::Serve`]. No live SGLang. Numeric
//!   mapping of last-resort hex completions is not locked; length-or-Serve is.
//!
//! Production `src/` must never import `tests/`.

#![allow(dead_code)]

use prometheus_classifiers::{
    ClassifierConfig, Error, HumanLabel, NowMs, Prediction, Result, Scorer, Scores,
    DEFAULT_MIN_AGREEMENT, DEFAULT_QUALITY_THRESHOLD, DEFAULT_SAFETY_THRESHOLD, DEFAULT_SHARD_SIZE,
};
use prometheus_extract::{ExtractedDocument, Format};
use std::path::PathBuf;

/// Spec 15.3 / 07 FP32 parity.
pub const FP32_TOL: f32 = 1e-5;

pub fn default_config() -> ClassifierConfig {
    ClassifierConfig::default()
}

pub fn config_shards(shard_size: usize) -> ClassifierConfig {
    ClassifierConfig {
        shard_size,
        quality_threshold: DEFAULT_QUALITY_THRESHOLD,
        safety_threshold: DEFAULT_SAFETY_THRESHOLD,
        min_agreement: DEFAULT_MIN_AGREEMENT,
    }
}

pub fn config_full(
    shard_size: usize,
    quality_threshold: f32,
    safety_threshold: f32,
    min_agreement: f32,
) -> ClassifierConfig {
    ClassifierConfig {
        shard_size,
        quality_threshold,
        safety_threshold,
        min_agreement,
    }
}

/// Parent lives for the tempdir. `dir` does not exist yet.
pub fn fresh_orch_dir() -> (tempfile::TempDir, PathBuf) {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("clf");
    (parent, dir)
}

pub fn doc(text: &str) -> ExtractedDocument {
    ExtractedDocument::from_text(
        text.to_string(),
        Format::Html,
        Some("test".to_string()),
        text.len() as u64,
    )
}

pub fn doc_n(n: u32) -> ExtractedDocument {
    doc(&format!("document-{n} text body"))
}

pub fn doc_hashed(text: &str, hash: &str) -> ExtractedDocument {
    let mut d = doc(text);
    d.content_hash = hash.to_string();
    d
}

pub fn label(hash: &str, keep_quality: bool, keep_safety: bool) -> HumanLabel {
    HumanLabel {
        content_hash: hash.to_string(),
        keep_quality,
        keep_safety,
    }
}

pub fn pred(hash: &str, quality: f32, safety: f32) -> Prediction {
    Prediction {
        content_hash: hash.to_string(),
        scores: Scores { quality, safety },
    }
}

pub fn assert_close(got: f32, exp: f32, what: &str) {
    let d = (got - exp).abs();
    assert!(
        d <= FP32_TOL,
        "{what}: got {got} expected {exp} (abs {d} > {FP32_TOL})"
    );
}

pub fn assert_err<T>(r: Result<T>, what: &str) {
    if r.is_ok() {
        panic!("expected error ({what}), got Ok");
    }
}

pub fn assert_empty<T>(r: Result<T>, what: &str) {
    match r {
        Err(Error::Empty) => {}
        Ok(_) => panic!("{what}: expected Empty, got Ok"),
        Err(e) => panic!("{what}: expected Empty, got {e:?}"),
    }
}

pub fn assert_bad_shard_size<T>(r: Result<T>, what: &str) {
    match r {
        Err(Error::BadShardSize) => {}
        Ok(_) => panic!("{what}: expected BadShardSize, got Ok"),
        Err(e) => panic!("{what}: expected BadShardSize, got {e:?}"),
    }
}

pub fn assert_wrong_state<T>(r: Result<T>, needle: &str, what: &str) {
    match r {
        Err(Error::WrongState(msg)) => {
            assert!(
                msg.contains(needle),
                "{what}: WrongState({msg:?}) missing {needle:?}"
            );
        }
        Ok(_) => panic!("{what}: expected WrongState containing {needle:?}, got Ok"),
        Err(e) => panic!("{what}: expected WrongState containing {needle:?}, got {e:?}"),
    }
}

pub fn assert_missing_label<T>(r: Result<T>, hash: &str, what: &str) {
    match r {
        Err(Error::MissingLabel(h)) => {
            assert_eq!(h, hash, "{what}: MissingLabel hash");
        }
        Ok(_) => panic!("{what}: expected MissingLabel({hash}), got Ok"),
        Err(e) => panic!("{what}: expected MissingLabel({hash}), got {e:?}"),
    }
}

pub fn assert_gate_failed<T>(r: Result<T>, got: f32, min: f32, what: &str) {
    match r {
        Err(Error::GateFailed { got: g, min: m }) => {
            assert_close(g, got, &format!("{what} got"));
            assert_close(m, min, &format!("{what} min"));
        }
        Ok(_) => panic!("{what}: expected GateFailed {{ got: {got}, min: {min} }}, got Ok"),
        Err(e) => panic!("{what}: expected GateFailed {{ got: {got}, min: {min} }}, got {e:?}"),
    }
}

pub fn assert_serve<T>(r: Result<T>, what: &str) {
    match r {
        Err(Error::Serve(_)) => {}
        Ok(_) => panic!("{what}: expected Serve(_), got Ok"),
        Err(e) => panic!("{what}: expected Serve(_), got {e:?}"),
    }
}

pub fn assert_preds_eq(got: &[Prediction], exp: &[Prediction], what: &str) {
    assert_eq!(got.len(), exp.len(), "{what}: prediction count");
    for (i, (g, e)) in got.iter().zip(exp).enumerate() {
        assert_eq!(
            g.content_hash, e.content_hash,
            "{what}: pred[{i}].content_hash"
        );
        assert_close(
            g.scores.quality,
            e.scores.quality,
            &format!("{what}: pred[{i}].quality"),
        );
        assert_close(
            g.scores.safety,
            e.scores.safety,
            &format!("{what}: pred[{i}].safety"),
        );
    }
}

pub fn assert_docs_eq(got: &[ExtractedDocument], exp: &[ExtractedDocument], what: &str) {
    assert_eq!(got.len(), exp.len(), "{what}: doc count");
    for (i, (g, e)) in got.iter().zip(exp).enumerate() {
        assert_eq!(g.content_hash, e.content_hash, "{what}: doc[{i}].hash");
        assert_eq!(g.text, e.text, "{what}: doc[{i}].text");
    }
}

pub fn assert_shards_eq(
    got: &[Vec<ExtractedDocument>],
    exp: &[Vec<ExtractedDocument>],
    what: &str,
) {
    assert_eq!(got.len(), exp.len(), "{what}: shard count");
    for (i, (g, e)) in got.iter().zip(exp).enumerate() {
        assert_docs_eq(g, e, &format!("{what} shard[{i}]"));
    }
}

pub fn err_kind(e: &Error) -> &'static str {
    match e {
        Error::Empty => "empty",
        Error::BadShardSize => "bad_shard_size",
        Error::MissingLabel(_) => "missing_label",
        Error::GateFailed { .. } => "gate_failed",
        Error::WrongState(_) => "wrong_state",
        Error::Serve(_) => "serve",
        Error::Other(_) => "other",
    }
}

/// Returns zeros for every text. Empty texts => Empty.
pub struct ZeroScorer;

impl Scorer for ZeroScorer {
    fn score(&mut self, texts: &[String], _now: NowMs) -> Result<Vec<Scores>> {
        if texts.is_empty() {
            return Err(Error::Empty);
        }
        Ok(texts
            .iter()
            .map(|_| Scores {
                quality: 0.0,
                safety: 0.0,
            })
            .collect())
    }
}

/// Same scores for every text in the call.
pub struct ConstScorer {
    pub quality: f32,
    pub safety: f32,
}

impl Scorer for ConstScorer {
    fn score(&mut self, texts: &[String], _now: NowMs) -> Result<Vec<Scores>> {
        if texts.is_empty() {
            return Err(Error::Empty);
        }
        Ok(texts
            .iter()
            .map(|_| Scores {
                quality: self.quality,
                safety: self.safety,
            })
            .collect())
    }
}

/// Consumes a pre-scripted score per document, across shard calls, ingest order.
pub struct ScriptedScorer {
    pub remaining: Vec<Scores>,
}

impl Scorer for ScriptedScorer {
    fn score(&mut self, texts: &[String], _now: NowMs) -> Result<Vec<Scores>> {
        if texts.is_empty() {
            return Err(Error::Empty);
        }
        if self.remaining.len() < texts.len() {
            return Err(Error::WrongState("script underflow".into()));
        }
        Ok(self.remaining.drain(..texts.len()).collect())
    }
}

pub struct RecordingScorer<S> {
    pub inner: S,
    pub calls: Vec<Vec<String>>,
    pub nows: Vec<NowMs>,
}

impl<S> RecordingScorer<S> {
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            calls: Vec::new(),
            nows: Vec::new(),
        }
    }
}

impl<S: Scorer> Scorer for RecordingScorer<S> {
    fn score(&mut self, texts: &[String], now: NowMs) -> Result<Vec<Scores>> {
        self.calls.push(texts.to_vec());
        self.nows.push(now);
        self.inner.score(texts, now)
    }
}

/// Always returns an empty score vec (wrong length for any non-empty shard).
pub struct WrongLenScorer;

impl Scorer for WrongLenScorer {
    fn score(&mut self, _texts: &[String], _now: NowMs) -> Result<Vec<Scores>> {
        Ok(Vec::new())
    }
}

pub struct BoomScorer;

impl Scorer for BoomScorer {
    fn score(&mut self, _texts: &[String], _now: NowMs) -> Result<Vec<Scores>> {
        Err(Error::Serve("boom".into()))
    }
}

pub fn locked_defaults_hold() {
    assert_eq!(DEFAULT_SHARD_SIZE, 32);
    assert_eq!(DEFAULT_QUALITY_THRESHOLD, 0.5);
    assert_eq!(DEFAULT_SAFETY_THRESHOLD, 0.5);
    assert_eq!(DEFAULT_MIN_AGREEMENT, 1.0);
    let d = ClassifierConfig::default();
    assert_eq!(d.shard_size, DEFAULT_SHARD_SIZE);
    assert_eq!(d.quality_threshold, DEFAULT_QUALITY_THRESHOLD);
    assert_eq!(d.safety_threshold, DEFAULT_SAFETY_THRESHOLD);
    assert_eq!(d.min_agreement, DEFAULT_MIN_AGREEMENT);
}
