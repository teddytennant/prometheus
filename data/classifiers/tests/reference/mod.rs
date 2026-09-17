//! Slow, obviously-correct B3 reference.
//!
//! In-memory only. Production must persist equivalently so `open` replays
//! ingest + scores. Production `src/` must never import this module.
//!
//! # `RefLocalScorer`
//!
//! Deterministic CPU scorer. Independent of `now` and the wall clock.
//!
//! For each text `t`:
//!
//! ```text
//! h = fnv1a64(t.as_bytes())
//! quality = unit01(mix64(seed ^ mix64(h.wrapping_add(QUALITY_TAG))))
//! safety  = unit01(mix64(seed ^ mix64(h.wrapping_add(SAFETY_TAG))))
//! ```
//!
//! - FNV-1a 64: offset `0xcbf29ce484222325`, prime `0x100000001b3`
//! - `mix64` is SplitMix64's finalizer
//! - `unit01` maps the high 24 bits of a u64 into `[0, 1)`:
//!   `((h >> 40) as u32) as f32 / 16_777_216.0`
//! - `QUALITY_TAG = 0x51`, `SAFETY_TAG = 0x53`
//! - Empty `texts` => `Error::Empty`
//! - Output length equals input length
//!
//! # Duplicate ingest
//!
//! Reject with `Error::WrongState` containing the `content_hash`. First
//! document kept. Fail-closed.
//!
//! # `ref_agreement`
//!
//! Empty labels => `Error::Empty`. Otherwise, for each label in order, look up
//! the first prediction with that `content_hash`. Missing =>
//! `Error::MissingLabel(hash)`. A label hits iff both axes match after
//! `score >= threshold`. Fraction is `hits as f32 / labels.len() as f32`.
//! Unlabeled predictions are ignored.

#![allow(dead_code)]

use prometheus_classifiers::{
    ClassifierConfig, Error, HumanLabel, NowMs, Prediction, Result, Scores, Scorer,
};
use prometheus_extract::ExtractedDocument;
use std::path::{Path, PathBuf};

pub const QUALITY_TAG: u64 = 0x51;
pub const SAFETY_TAG: u64 = 0x53;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// SplitMix64 finalizer. Slow and obvious.
pub fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// High 24 bits -> `[0, 1)`.
pub fn unit01(h: u64) -> f32 {
    let top = (h >> 40) as u32;
    (top as f32) / 16_777_216.0_f32
}

pub fn ref_axis_score(seed: u64, text: &str, tag: u64) -> f32 {
    let h = fnv1a64(text.as_bytes());
    unit01(mix64(seed ^ mix64(h.wrapping_add(tag))))
}

pub fn ref_scores_one(seed: u64, text: &str) -> Scores {
    Scores {
        quality: ref_axis_score(seed, text, QUALITY_TAG),
        safety: ref_axis_score(seed, text, SAFETY_TAG),
    }
}

/// Deterministic CPU scorer. No model, no network. Must match `LocalScorer`.
pub struct RefLocalScorer {
    pub seed: u64,
}

impl Scorer for RefLocalScorer {
    fn score(&mut self, texts: &[String], _now: NowMs) -> Result<Vec<Scores>> {
        if texts.is_empty() {
            return Err(Error::Empty);
        }
        Ok(texts.iter().map(|t| ref_scores_one(self.seed, t)).collect())
    }
}

pub fn ref_agreement(
    preds: &[Prediction],
    labels: &[HumanLabel],
    q_th: f32,
    s_th: f32,
) -> Result<f32> {
    if labels.is_empty() {
        return Err(Error::Empty);
    }
    let mut hits = 0usize;
    for lab in labels {
        let pred = preds
            .iter()
            .find(|p| p.content_hash == lab.content_hash)
            .ok_or_else(|| Error::MissingLabel(lab.content_hash.clone()))?;
        let keep_q = pred.scores.quality >= q_th;
        let keep_s = pred.scores.safety >= s_th;
        if keep_q == lab.keep_quality && keep_s == lab.keep_safety {
            hits += 1;
        }
    }
    Ok(hits as f32 / labels.len() as f32)
}

/// In-memory orchestrator. `create` still mkdir's so it shares the
/// fail-if-exists rule; it does not write a log. Production must persist.
pub struct RefOrchestrator {
    dir: PathBuf,
    config: ClassifierConfig,
    docs: Vec<ExtractedDocument>,
    preds: Vec<Prediction>,
}

impl RefOrchestrator {
    pub fn new(config: ClassifierConfig) -> Self {
        Self {
            dir: PathBuf::new(),
            config,
            docs: Vec::new(),
            preds: Vec::new(),
        }
    }

    pub fn create(dir: impl AsRef<Path>, config: ClassifierConfig) -> Result<Self> {
        if config.shard_size < 1 {
            return Err(Error::BadShardSize);
        }
        let dir = dir.as_ref();
        if dir.exists() {
            return Err(Error::WrongState(format!(
                "dir exists: {}",
                dir.display()
            )));
        }
        std::fs::create_dir(dir).map_err(|e| Error::Other(e.to_string()))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            config,
            docs: Vec::new(),
            preds: Vec::new(),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    pub fn ingest(&mut self, doc: ExtractedDocument) -> Result<()> {
        if self
            .docs
            .iter()
            .any(|d| d.content_hash == doc.content_hash)
        {
            return Err(Error::WrongState(format!(
                "duplicate content_hash {}",
                doc.content_hash
            )));
        }
        self.docs.push(doc);
        Ok(())
    }

    pub fn shards(&self) -> Vec<Vec<ExtractedDocument>> {
        if self.docs.is_empty() {
            return Vec::new();
        }
        let n = self.config.shard_size as usize;
        self.docs.chunks(n).map(|c| c.to_vec()).collect()
    }

    pub fn run(&mut self, scorer: &mut dyn Scorer, now: NowMs) -> Result<Vec<Prediction>> {
        if self.docs.is_empty() {
            return Err(Error::Empty);
        }
        let mut out = Vec::with_capacity(self.docs.len());
        for shard in self.shards() {
            let texts: Vec<String> = shard.iter().map(|d| d.text.clone()).collect();
            let scores = scorer.score(&texts, now)?;
            if scores.len() != shard.len() {
                return Err(Error::WrongState(format!(
                    "scorer returned {} scores for shard of {}",
                    scores.len(),
                    shard.len()
                )));
            }
            for (doc, sc) in shard.iter().zip(scores.into_iter()) {
                out.push(Prediction {
                    content_hash: doc.content_hash.clone(),
                    scores: sc,
                });
            }
        }
        self.preds = out.clone();
        Ok(out)
    }

    pub fn predictions(&self) -> &[Prediction] {
        &self.preds
    }

    pub fn agreement(&self, labels: &[HumanLabel]) -> Result<f32> {
        ref_agreement(
            &self.preds,
            labels,
            self.config.quality_threshold,
            self.config.safety_threshold,
        )
    }

    pub fn gate(&self, labels: &[HumanLabel]) -> Result<()> {
        let got = self.agreement(labels)?;
        if got < self.config.min_agreement {
            Err(Error::GateFailed {
                got,
                min: self.config.min_agreement,
            })
        } else {
            Ok(())
        }
    }

    pub fn docs(&self) -> &[ExtractedDocument] {
        &self.docs
    }
}
