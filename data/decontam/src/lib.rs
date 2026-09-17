//! Decontamination against the public eval suites (spec 7, 11, 15.5 B4).
//!
//! Fail-closed: a missing index, a missing method hash, or an item that
//! matches an eval n-gram or embedding is an error or a flag, never a pass.
//! The B4 gate is planted eval items caught in a shard.
//!
//! The payload written onto each shard matches `decontamination` in
//! `contracts/schemas/v1/_defs.schema.json` (status, against, method_hash).
//!
//! Complements the Python `evals.DecontamIndex` (F3): this crate is the
//! data-pipeline side that walks shards and writes provenance. It does not
//! import F3; it speaks the same n-gram width and the same status enum.
//!
//! Nothing here runs. Types are real; every function is `unimplemented!`.

use serde::{Deserialize, Serialize};

/// Fail-closed errors. A missing index is `MissingIndex`, not a clean scan.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("decontam index is missing")]
    MissingIndex,
    #[error("decontam config: {0}")]
    Config(String),
    #[error("planted eval item {0} was not caught")]
    PlantedMiss(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Status values are the F1 enum: pending, clean, flagged, not_applicable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Pending,
    Clean,
    Flagged,
    NotApplicable,
}

/// One public-suite item. `suite` is an F3 slug; `text` is prompt plus answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalItem {
    pub suite: String,
    pub id: String,
    pub text: String,
}

/// One match against the index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hit {
    pub suite: String,
    pub item_id: String,
    pub method: String,
}

/// Shard-level payload. Field names match the F1 `decontamination` object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Report {
    pub status: Status,
    pub against: Vec<String>,
    pub method_hash: String,
    pub hits: Vec<Hit>,
}

/// N-gram width matches F3 (`evals.NGRAM_N` = 8).
pub const NGRAM_N: usize = 8;

/// In-memory eval index. Fail-closed: scans on an empty index error.
#[derive(Debug, Clone, Default)]
pub struct Index {
    items: Vec<EvalItem>,
}

impl Index {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Insert one eval item. Duplicate ids for the same suite error.
    pub fn add(&mut self, item: EvalItem) -> Result<()> {
        let _ = item;
        unimplemented!("B4 Index::add")
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}

/// SHA-256 of the index contents and the method (n-gram width, embed model).
pub fn method_hash(index: &Index) -> Result<String> {
    let _ = index;
    unimplemented!("B4 method_hash")
}

/// Word n-grams of size `NGRAM_N` after lowercase and whitespace collapse.
pub fn ngrams(_text: &str) -> Vec<String> {
    unimplemented!("B4 ngrams")
}

/// True if `text` shares an n-gram with any indexed item.
pub fn ngram_match(index: &Index, text: &str) -> Result<Vec<Hit>> {
    let _ = (index, text);
    unimplemented!("B4 ngram_match")
}

/// Embedding near-match. The embedder is injected; a missing one is an error.
pub fn embedding_match(index: &Index, text: &str) -> Result<Vec<Hit>> {
    let _ = (index, text);
    unimplemented!("B4 embedding_match")
}

/// Scan one document. Empty index -> `Error::MissingIndex`. Hits -> Flagged.
pub fn scan_text(index: &Index, text: &str) -> Result<Report> {
    let _ = (index, text);
    unimplemented!("B4 scan_text")
}

/// Scan every document in a shard. Flagged if any document hits.
pub fn scan_shard(index: &Index, texts: &[String]) -> Result<Report> {
    let _ = (index, texts);
    unimplemented!("B4 scan_shard")
}

/// Gate helper: every planted item must produce a Flagged hit. Else PlantedMiss.
pub fn planted_caught(index: &Index, planted: &[EvalItem], shard: &[String]) -> Result<()> {
    let _ = (index, planted, shard);
    unimplemented!("B4 planted_caught")
}
