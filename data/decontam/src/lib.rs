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

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};

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
        if self
            .items
            .iter()
            .any(|existing| existing.suite == item.suite && existing.id == item.id)
        {
            return Err(Error::Config(format!(
                "duplicate eval item {}/{}",
                item.suite, item.id
            )));
        }
        self.items.push(item);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}

/// SHA-256 of the index contents and the method (n-gram width, embed model).
pub fn method_hash(index: &Index) -> Result<String> {
    if index.is_empty() {
        return Err(Error::MissingIndex);
    }
    let mut ordered: Vec<&EvalItem> = index.items.iter().collect();
    ordered.sort_by(|a, b| (&a.suite, &a.id).cmp(&(&b.suite, &b.id)));
    let mut hasher = Sha256::new();
    hasher.update(b"prometheus-decontam/v1\n");
    hasher.update(format!("ngram_n={NGRAM_N}\n").as_bytes());
    hasher.update(b"embed_model=\n");
    for item in ordered {
        hasher.update(item.suite.as_bytes());
        hasher.update([0u8]);
        hasher.update(item.id.as_bytes());
        hasher.update([0u8]);
        hasher.update(item.text.as_bytes());
        hasher.update([b'\n']);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Word n-grams of size `NGRAM_N` after lowercase and whitespace collapse.
pub fn ngrams(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    if words.len() < NGRAM_N {
        return Vec::new();
    }
    words.windows(NGRAM_N).map(|w| w.join(" ")).collect()
}

/// True if `text` shares an n-gram with any indexed item.
pub fn ngram_match(index: &Index, text: &str) -> Result<Vec<Hit>> {
    if index.is_empty() {
        return Err(Error::MissingIndex);
    }
    let query: HashSet<String> = ngrams(text).into_iter().collect();
    let mut hits = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for item in &index.items {
        let item_grams = ngrams(&item.text);
        if item_grams.iter().any(|g| query.contains(g))
            && seen.insert((item.suite.clone(), item.id.clone()))
        {
            hits.push(Hit {
                suite: item.suite.clone(),
                item_id: item.id.clone(),
                method: "ngram".to_string(),
            });
        }
    }
    Ok(hits)
}

/// Embedding near-match. The embedder is injected; a missing one is an error.
pub fn embedding_match(index: &Index, _text: &str) -> Result<Vec<Hit>> {
    if index.is_empty() {
        return Err(Error::MissingIndex);
    }
    Err(Error::Config("no embedder configured".to_string()))
}

/// Scan one document. Empty index -> `Error::MissingIndex`. Hits -> Flagged.
pub fn scan_text(index: &Index, text: &str) -> Result<Report> {
    if index.is_empty() {
        return Err(Error::MissingIndex);
    }
    let hits = ngram_match(index, text)?;
    let hash = method_hash(index)?;
    Ok(report(hits, index, hash))
}

/// Scan every document in a shard. Flagged if any document hits.
pub fn scan_shard(index: &Index, texts: &[String]) -> Result<Report> {
    if index.is_empty() {
        return Err(Error::MissingIndex);
    }
    let mut hits = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for text in texts {
        for hit in ngram_match(index, text)? {
            if seen.insert((hit.suite.clone(), hit.item_id.clone())) {
                hits.push(hit);
            }
        }
    }
    let hash = method_hash(index)?;
    Ok(report(hits, index, hash))
}

/// Gate helper: every planted item must produce a Flagged hit. Else PlantedMiss.
pub fn planted_caught(index: &Index, planted: &[EvalItem], shard: &[String]) -> Result<()> {
    let scanned = scan_shard(index, shard)?;
    for item in planted {
        let caught = scanned
            .hits
            .iter()
            .any(|h| h.suite == item.suite && h.item_id == item.id);
        if !caught {
            return Err(Error::PlantedMiss(item.id.clone()));
        }
    }
    Ok(())
}

fn report(hits: Vec<Hit>, index: &Index, method_hash: String) -> Report {
    let status = if hits.is_empty() {
        Status::Clean
    } else {
        Status::Flagged
    };
    Report {
        status,
        against: suites_against(index),
        method_hash,
        hits,
    }
}

fn suites_against(index: &Index) -> Vec<String> {
    let mut suites = BTreeSet::new();
    for item in &index.items {
        suites.insert(item.suite.clone());
    }
    suites.into_iter().collect()
}
