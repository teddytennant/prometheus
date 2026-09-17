//! Classifier orchestration on F5 (spec 7.2, 15.5 B3).
//!
//! Small quality and safety models run in batch on a slice of the fleet.
//! This crate owns sharding and bookkeeping. Scores come from a [`Scorer`].
//! CPU tests use [`LocalScorer`]. Production wraps F5
//! [`prometheus_serve::Router`] in [`ServeScorer`].
//!
//! Gate: agreement with a human-labeled sample. Fail closed if a labeled
//! document is missing from predictions. `NowMs` is injected.

use prometheus_extract::ExtractedDocument;
use prometheus_serve::{GenerateRequest, Router};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

pub type NowMs = u64;

pub const DEFAULT_SHARD_SIZE: usize = 32;
pub const DEFAULT_QUALITY_THRESHOLD: f32 = 0.5;
pub const DEFAULT_SAFETY_THRESHOLD: f32 = 0.5;

/// Gate default: every human-labeled document must match. Spec 15.5 does
/// not name a looser threshold.
pub const DEFAULT_MIN_AGREEMENT: f32 = 1.0;

pub const EVENT_INGEST: &str = "clf.ingest";
pub const EVENT_SCORE: &str = "clf.score";
pub const EVENT_GATE: &str = "clf.gate";

const QUALITY_TAG: u64 = 0x51;
const SAFETY_TAG: u64 = 0x53;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;
const JOURNAL_NAME: &str = "events.jsonl";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    Quality,
    Safety,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Scores {
    pub quality: f32,
    pub safety: f32,
}

impl Scores {
    /// Keep iff both axes meet their thresholds.
    pub fn keep(&self, quality_threshold: f32, safety_threshold: f32) -> bool {
        self.quality >= quality_threshold && self.safety >= safety_threshold
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanLabel {
    pub content_hash: String,
    pub keep_quality: bool,
    pub keep_safety: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub content_hash: String,
    pub scores: Scores,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassifierConfig {
    pub shard_size: usize,
    pub quality_threshold: f32,
    pub safety_threshold: f32,
    pub min_agreement: f32,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            shard_size: DEFAULT_SHARD_SIZE,
            quality_threshold: DEFAULT_QUALITY_THRESHOLD,
            safety_threshold: DEFAULT_SAFETY_THRESHOLD,
            min_agreement: DEFAULT_MIN_AGREEMENT,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("empty batch")]
    Empty,
    #[error("shard_size must be >= 1")]
    BadShardSize,
    #[error("labeled document missing from predictions: {0}")]
    MissingLabel(String),
    #[error("agreement {got} below min {min}")]
    GateFailed { got: f32, min: f32 },
    #[error("wrong state: {0}")]
    WrongState(String),
    #[error("{0}")]
    Serve(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Batch scorer. Production is [`ServeScorer`]; tests use [`LocalScorer`].
pub trait Scorer {
    fn score(&mut self, texts: &[String], now: NowMs) -> Result<Vec<Scores>>;
}

/// Deterministic CPU scorer. No model, no network.
pub struct LocalScorer {
    pub seed: u64,
}

impl Scorer for LocalScorer {
    fn score(&mut self, texts: &[String], _now: NowMs) -> Result<Vec<Scores>> {
        if texts.is_empty() {
            return Err(Error::Empty);
        }
        Ok(texts.iter().map(|t| scores_one(self.seed, t)).collect())
    }
}

/// F5 last-resort engine as a scorer. Prompts the router per shard.
pub struct ServeScorer {
    router: Router,
}

impl ServeScorer {
    pub fn new(router: Router) -> Self {
        Self { router }
    }

    pub fn router(&self) -> &Router {
        &self.router
    }
}

impl Scorer for ServeScorer {
    fn score(&mut self, texts: &[String], now: NowMs) -> Result<Vec<Scores>> {
        if texts.is_empty() {
            return Err(Error::Empty);
        }
        let req = GenerateRequest {
            prompts: texts.to_vec(),
            max_tokens: 1,
            temperature: 0.0,
        };
        let resp = self
            .router
            .generate(req, now)
            .map_err(|e| Error::Serve(e.to_string()))?;
        if resp.completions.len() != texts.len() {
            return Err(Error::Serve(format!(
                "engine returned {} completions for {} texts",
                resp.completions.len(),
                texts.len()
            )));
        }
        Ok(resp
            .completions
            .iter()
            .map(|c| scores_from_completion(&c.text))
            .collect())
    }
}

/// Sharding and bookkeeping. Create fails if `dir` exists.
pub struct Orchestrator {
    dir: PathBuf,
    config: ClassifierConfig,
    docs: Vec<ExtractedDocument>,
    preds: Vec<Prediction>,
}

impl Orchestrator {
    pub fn create(dir: impl AsRef<Path>, config: ClassifierConfig) -> Result<Self> {
        if config.shard_size < 1 {
            return Err(Error::BadShardSize);
        }
        let dir = dir.as_ref();
        if dir.exists() {
            return Err(Error::WrongState(format!("dir exists: {}", dir.display())));
        }
        std::fs::create_dir(dir).map_err(other)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            config,
            docs: Vec::new(),
            preds: Vec::new(),
        })
    }

    pub fn open(dir: impl AsRef<Path>, config: ClassifierConfig) -> Result<Self> {
        if config.shard_size < 1 {
            return Err(Error::BadShardSize);
        }
        let dir = dir.as_ref();
        if !dir.exists() {
            return Err(Error::Other(format!("missing: {}", dir.display())));
        }
        let (docs, preds) = load_journal(dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            config,
            docs,
            preds,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    pub fn ingest(&mut self, doc: ExtractedDocument) -> Result<()> {
        if self.docs.iter().any(|d| d.content_hash == doc.content_hash) {
            return Err(Error::WrongState(format!(
                "duplicate content_hash {}",
                doc.content_hash
            )));
        }
        append_event(&self.dir, &JournalEvent::Ingest { doc: doc.clone() })?;
        self.docs.push(doc);
        Ok(())
    }

    /// Documents in ingest order, split into shards of `config.shard_size`
    /// (last shard may be shorter). Empty ingest => empty vec, not an error.
    pub fn shards(&self) -> Vec<Vec<ExtractedDocument>> {
        if self.docs.is_empty() {
            return Vec::new();
        }
        self.docs
            .chunks(self.config.shard_size)
            .map(|c| c.to_vec())
            .collect()
    }

    /// Score every ingested document through `scorer` in shard batches.
    /// Empty ingest => Error::Empty.
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
            for (doc, sc) in shard.iter().zip(scores) {
                out.push(Prediction {
                    content_hash: doc.content_hash.clone(),
                    scores: sc,
                });
            }
        }
        append_event(
            &self.dir,
            &JournalEvent::Score {
                predictions: out.clone(),
            },
        )?;
        self.preds = out.clone();
        Ok(out)
    }

    pub fn predictions(&self) -> &[Prediction] {
        &self.preds
    }

    /// Fraction of human labels whose keep/drop matches the prediction after
    /// thresholding. Fail closed on a labeled hash with no prediction.
    pub fn agreement(&self, labels: &[HumanLabel]) -> Result<f32> {
        agreement(
            &self.preds,
            labels,
            self.config.quality_threshold,
            self.config.safety_threshold,
        )
    }

    /// Error::GateFailed if agreement < min_agreement.
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
}

/// Same as [`Orchestrator::agreement`] without an orchestrator.
pub fn agreement(
    preds: &[Prediction],
    labels: &[HumanLabel],
    quality_threshold: f32,
    safety_threshold: f32,
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
        let keep_q = pred.scores.quality >= quality_threshold;
        let keep_s = pred.scores.safety >= safety_threshold;
        if keep_q == lab.keep_quality && keep_s == lab.keep_safety {
            hits += 1;
        }
    }
    Ok(hits as f32 / labels.len() as f32)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// SplitMix64 finalizer.
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// High 24 bits -> `[0, 1)`.
fn unit01(h: u64) -> f32 {
    let top = (h >> 40) as u32;
    (top as f32) / 16_777_216.0_f32
}

fn axis_score(seed: u64, text: &str, tag: u64) -> f32 {
    let h = fnv1a64(text.as_bytes());
    unit01(mix64(seed ^ mix64(h.wrapping_add(tag))))
}

fn scores_one(seed: u64, text: &str) -> Scores {
    Scores {
        quality: axis_score(seed, text, QUALITY_TAG),
        safety: axis_score(seed, text, SAFETY_TAG),
    }
}

fn scores_from_completion(text: &str) -> Scores {
    let h = fnv1a64(text.as_bytes());
    Scores {
        quality: unit01(h),
        safety: unit01(mix64(h)),
    }
}

fn other(e: impl ToString) -> Error {
    Error::Other(e.to_string())
}

fn journal_path(dir: &Path) -> PathBuf {
    dir.join(JOURNAL_NAME)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event")]
enum JournalEvent {
    #[serde(rename = "clf.ingest")]
    Ingest { doc: ExtractedDocument },
    #[serde(rename = "clf.score")]
    Score { predictions: Vec<Prediction> },
}

fn append_event(dir: &Path, event: &JournalEvent) -> Result<()> {
    let path = journal_path(dir);
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(other)?;
    serde_json::to_writer(&mut f, event).map_err(other)?;
    f.write_all(b"\n").map_err(other)?;
    f.flush().map_err(other)?;
    Ok(())
}

fn load_journal(dir: &Path) -> Result<(Vec<ExtractedDocument>, Vec<Prediction>)> {
    let path = journal_path(dir);
    if !path.exists() {
        return Ok((Vec::new(), Vec::new()));
    }
    let f = File::open(&path).map_err(other)?;
    let reader = BufReader::new(f);
    let mut docs = Vec::new();
    let mut preds = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(other)?;
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<JournalEvent>(&line).map_err(other)? {
            JournalEvent::Ingest { doc } => docs.push(doc),
            JournalEvent::Score { predictions } => preds = predictions,
        }
    }
    Ok((docs, preds))
}
