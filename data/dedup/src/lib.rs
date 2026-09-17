//! Exact and MinHash near-duplicate detection (spec 7, 15.5 B2).
//!
//! Two passes over a corpus of extracted documents (B1):
//! - Exact: SHA-256 of normalized text, at document and paragraph grain.
//! - Near: MinHash + LSH so paraphrases and boilerplate collapse to one keep.
//!
//! Keep the lowest `content_hash` in each cluster. Golden outputs on a fixed
//! corpus are the B2 gate.
//!
//! Nothing here runs. Types are real; every function is `unimplemented!`.

use serde::{Deserialize, Serialize};
use sha2::Digest;

/// Errors from config checks or an empty corpus.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("dedup config: {0}")]
    Config(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// One extracted document. `content_hash` is SHA-256 of the raw bytes B1 wrote.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub content_hash: String,
}

/// MinHash / LSH knobs. `num_hashes == bands * rows`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DedupConfig {
    pub shingle_size: usize,
    pub num_hashes: usize,
    pub bands: usize,
    pub rows: usize,
}

impl DedupConfig {
    /// Default: 5-shingles, 128 hashes in 32 bands of 4.
    pub fn standard() -> Self {
        Self {
            shingle_size: 5,
            num_hashes: 128,
            bands: 32,
            rows: 4,
        }
    }
}

/// Grain of an exact-hash pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Grain {
    Document,
    Paragraph,
}

/// One near-duplicate cluster. `kept` is the lowest `content_hash`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cluster {
    pub kept: String,
    pub dropped: Vec<String>,
}

/// Result of running both passes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DedupReport {
    pub kept: Vec<String>,
    pub dropped_exact: Vec<String>,
    pub dropped_near: Vec<String>,
    pub clusters: Vec<Cluster>,
}

/// Lowercase, collapse whitespace, strip zero-width. Used before both hashes.
pub fn normalize(_text: &str) -> String {
    unimplemented!("B2 normalize")
}

/// Split on blank lines after `normalize`. Empty paragraphs are dropped.
pub fn paragraphs(_text: &str) -> Vec<String> {
    unimplemented!("B2 paragraphs")
}

/// SHA-256 (lowercase hex) of `normalize(text)`.
pub fn exact_hash(_text: &str) -> String {
    unimplemented!("B2 exact_hash")
}

/// Word shingles of size `n`. Empty if the token list is shorter than `n`.
pub fn shingles(_text: &str, _n: usize) -> Vec<String> {
    unimplemented!("B2 shingles")
}

/// `num_hashes` MinHash values of the shingle set.
pub fn minhash_signature(_text: &str, _config: DedupConfig) -> Vec<u64> {
    unimplemented!("B2 minhash_signature")
}

/// Band keys: `bands` hashes, each over `rows` consecutive signature values.
pub fn lsh_band_keys(signature: &[u64], config: DedupConfig) -> Result<Vec<u64>> {
    let _ = (signature, config);
    unimplemented!("B2 lsh_band_keys")
}

/// Drop exact duplicates at `grain`. Keep the lowest `content_hash`.
pub fn dedup_exact(docs: &[Document], grain: Grain) -> Result<DedupReport> {
    let _ = (docs, grain);
    unimplemented!("B2 dedup_exact")
}

/// MinHash LSH near-duplicate clustering. Union-find over shared band keys.
pub fn dedup_near(docs: &[Document], config: DedupConfig) -> Result<DedupReport> {
    let _ = (docs, config);
    unimplemented!("B2 dedup_near")
}

/// Exact document, exact paragraph, then near. Later passes see earlier keeps.
pub fn dedup_corpus(docs: &[Document], config: DedupConfig) -> Result<DedupReport> {
    let _ = (docs, config);
    unimplemented!("B2 dedup_corpus")
}

/// SHA-256 of bytes as lowercase hex. Available to implementers; not the pass.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let d = sha2::Sha256::digest(bytes);
    hex_lower(&d)
}

fn hex_lower(bytes: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(H[(b >> 4) as usize] as char);
        out.push(H[(b & 0x0f) as usize] as char);
    }
    out
}
