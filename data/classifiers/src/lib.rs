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
use prometheus_serve::Router;
use serde::{Deserialize, Serialize};
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
    fn score(&mut self, _texts: &[String], _now: NowMs) -> Result<Vec<Scores>> {
        unimplemented!("B3: LocalScorer::score")
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
    fn score(&mut self, _texts: &[String], _now: NowMs) -> Result<Vec<Scores>> {
        unimplemented!("B3: ServeScorer::score")
    }
}

/// Sharding and bookkeeping. Create fails if `dir` exists.
pub struct Orchestrator {
    dir: PathBuf,
    config: ClassifierConfig,
}

impl Orchestrator {
    pub fn create(_dir: impl AsRef<Path>, _config: ClassifierConfig) -> Result<Self> {
        unimplemented!("B3: Orchestrator::create")
    }

    pub fn open(_dir: impl AsRef<Path>, _config: ClassifierConfig) -> Result<Self> {
        unimplemented!("B3: Orchestrator::open")
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    pub fn ingest(&mut self, _doc: ExtractedDocument) -> Result<()> {
        unimplemented!("B3: Orchestrator::ingest")
    }

    /// Documents in ingest order, split into shards of `config.shard_size`
    /// (last shard may be shorter). Empty ingest => empty vec, not an error.
    pub fn shards(&self) -> Vec<Vec<ExtractedDocument>> {
        unimplemented!("B3: Orchestrator::shards")
    }

    /// Score every ingested document through `scorer` in shard batches.
    /// Empty ingest => Error::Empty.
    pub fn run(&mut self, _scorer: &mut dyn Scorer, _now: NowMs) -> Result<Vec<Prediction>> {
        unimplemented!("B3: Orchestrator::run")
    }

    pub fn predictions(&self) -> &[Prediction] {
        unimplemented!("B3: Orchestrator::predictions")
    }

    /// Fraction of human labels whose keep/drop matches the prediction after
    /// thresholding. Fail closed on a labeled hash with no prediction.
    pub fn agreement(&self, _labels: &[HumanLabel]) -> Result<f32> {
        unimplemented!("B3: Orchestrator::agreement")
    }

    /// Error::GateFailed if agreement < min_agreement.
    pub fn gate(&self, _labels: &[HumanLabel]) -> Result<()> {
        unimplemented!("B3: Orchestrator::gate")
    }
}

/// Same as [`Orchestrator::agreement`] without an orchestrator.
pub fn agreement(
    _preds: &[Prediction],
    _labels: &[HumanLabel],
    _quality_threshold: f32,
    _safety_threshold: f32,
) -> Result<f32> {
    unimplemented!("B3: agreement")
}
