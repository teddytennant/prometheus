//! Byte-fallback BPE tokenizer (spec 3.1, 7.1, 10, 15.5 F6).
//!
//! Trained on a data v0 sample (B1 extract documents). The frozen artifact
//! matches F1 `prometheus.tokenizer` plus `prometheus.vocab`. A vocab change
//! is a new model. Production vocab is 256_000 (spec 3.1); tests pass a
//! smaller [`TrainConfig::vocab_size`].
//!
//! Algorithm is `byte_fallback_bpe` (F1). Specials bos, eos, pad, unk are
//! required; latent, latent_start, latent_end are optional. ARC grids use
//! one reserved token per cell color 0-9 (spec 10).

use std::path::Path;

use serde::{Deserialize, Serialize};

pub const SCHEMA_ID: &str = "prometheus.tokenizer";
pub const VOCAB_SCHEMA_ID: &str = "prometheus.vocab";
pub const SCHEMA_VERSION: u32 = 1;
pub const ALGORITHM: &str = "byte_fallback_bpe";
/// Spec 3.1 production target. Not a train default; callers pass vocab_size.
pub const PRODUCTION_VOCAB_SIZE: u32 = 256_000;
/// ARC-AGI cell colors 0-9 (spec 10).
pub const ARC_N_COLORS: u32 = 10;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("tokenizer is frozen")]
    Frozen,
    #[error("vocab size {0} is too small for specials, bytes, and ARC colors")]
    VocabTooSmall(u32),
    #[error("unknown token id {0}")]
    UnknownId(u32),
    #[error("grid color {0} out of range")]
    BadColor(u8),
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Training knobs. `vocab_size` is required; production is [`PRODUCTION_VOCAB_SIZE`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainConfig {
    pub tokenizer_id: String,
    pub vocab_size: u32,
    pub byte_fallback: bool,
}

/// F1 special_token_ids. bos, eos, pad, unk are required.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub bos: u32,
    pub eos: u32,
    pub pad: u32,
    pub unk: u32,
    pub latent: Option<u32>,
    pub latent_start: Option<u32>,
    pub latent_end: Option<u32>,
}

/// Reserved id span for ARC cell-color tokens. Half-open `[start, end)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArcGridTokenRange {
    pub start: u32,
    pub end: u32,
}

/// F1 artifact pointer (content_hash, bytes, optional path and media_type).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Artifact {
    pub content_hash: String,
    pub bytes: u64,
    pub path: Option<String>,
    pub media_type: Option<String>,
}

/// In-memory form of `contracts/schemas/v1/tokenizer.schema.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerMeta {
    pub schema_id: String,
    pub schema_version: u32,
    pub tokenizer_id: String,
    pub algorithm: String,
    pub vocab_size: u32,
    pub byte_fallback: bool,
    pub special_token_ids: SpecialTokens,
    pub arc_grid_token_range: ArcGridTokenRange,
    pub frozen: bool,
    pub frozen_at: Option<String>,
    pub artifact: Artifact,
}

/// In-memory form of `contracts/schemas/v1/vocab.schema.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VocabPointer {
    pub schema_id: String,
    pub schema_version: u32,
    pub tokenizer_id: String,
    pub vocab_size: u32,
    pub special_token_ids: SpecialTokens,
    pub artifact: Artifact,
    pub format: String,
}

/// Trained tokenizer. Encode/decode work before and after freeze. Train
/// after freeze returns [`Error::Frozen`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tokenizer {
    pub meta: TokenizerMeta,
}

impl Tokenizer {
    /// Train byte-fallback BPE on UTF-8 documents.
    ///
    /// `vocab_size` must fit specials, 256 byte tokens when `byte_fallback`,
    /// [`ARC_N_COLORS`] grid tokens, and at least one merge slot.
    pub fn train<I, S>(docs: I, config: &TrainConfig) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let _ = (docs, config);
        unimplemented!("F6 train")
    }

    /// Set `frozen` and `frozen_at`. Idempotent if already frozen at the
    /// same timestamp. Further `train` is [`Error::Frozen`].
    pub fn freeze(&mut self, frozen_at: &str) -> Result<()> {
        let _ = frozen_at;
        unimplemented!("F6 freeze")
    }

    /// Encode UTF-8 text to token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let _ = text;
        unimplemented!("F6 encode")
    }

    /// Inverse of [`Self::encode`]. Round-trip of `encode` then `decode`
    /// returns the original text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let _ = ids;
        unimplemented!("F6 decode")
    }

    /// Encode raw bytes. Used by byte fallback.
    pub fn encode_bytes(&self, bytes: &[u8]) -> Result<Vec<u32>> {
        let _ = bytes;
        unimplemented!("F6 encode_bytes")
    }

    /// Inverse of [`Self::encode_bytes`].
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>> {
        let _ = ids;
        unimplemented!("F6 decode_bytes")
    }

    /// One token per ARC cell color. `cells` are row-major values in `0..ARC_N_COLORS`.
    pub fn encode_grid(&self, cells: &[u8]) -> Result<Vec<u32>> {
        let _ = cells;
        unimplemented!("F6 encode_grid")
    }

    /// `encode(text).len() / whitespace-separated word count`. Empty is 0.0.
    pub fn tokens_per_word(&self, text: &str) -> Result<f64> {
        let _ = text;
        unimplemented!("F6 tokens_per_word")
    }

    /// Write `dir/tokenizer.json` (F1 tokenizer) and the vocab artifact
    /// named by `meta.artifact.path`. Returns the F1 vocab pointer.
    pub fn save(&self, dir: &Path) -> Result<VocabPointer> {
        let _ = dir;
        unimplemented!("F6 save")
    }

    /// Load a directory written by [`Self::save`].
    pub fn load(dir: &Path) -> Result<Self> {
        let _ = dir;
        unimplemented!("F6 load")
    }

    /// JSON object matching `prometheus.tokenizer` for `contracts.validate`.
    pub fn to_contract(&self) -> serde_json::Value {
        unimplemented!("F6 to_contract")
    }
}

/// UTF-8 byte count / whitespace-separated words. Empty text is 0.0.
///
/// F6 gate: trained `tokens_per_word` must beat this baseline on the same text.
pub fn baseline_byte_tokens_per_word(text: &str) -> f64 {
    let words = text.split_whitespace().count();
    if words == 0 {
        0.0
    } else {
        text.len() as f64 / words as f64
    }
}
