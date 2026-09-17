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

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

pub const SCHEMA_ID: &str = "prometheus.tokenizer";
pub const VOCAB_SCHEMA_ID: &str = "prometheus.vocab";
pub const SCHEMA_VERSION: u32 = 1;
pub const ALGORITHM: &str = "byte_fallback_bpe";
/// Spec 3.1 production target. Not a train default; callers pass vocab_size.
pub const PRODUCTION_VOCAB_SIZE: u32 = 256_000;
/// ARC-AGI cell colors 0-9 (spec 10).
pub const ARC_N_COLORS: u32 = 10;

const N_SPECIALS: u32 = 4;
const N_BYTES: u32 = 256;
const FIRST_BYTE: u32 = N_SPECIALS;
const FIRST_ARC: u32 = N_SPECIALS + N_BYTES;
const FIRST_MERGE: u32 = FIRST_ARC + ARC_N_COLORS;
const MIN_VOCAB_SIZE: u32 = FIRST_MERGE + 1;
const VOCAB_FILENAME: &str = "vocab.jsonl";
const TOKENIZER_FILENAME: &str = "tokenizer.json";
const VOCAB_MEDIA_TYPE: &str = "application/jsonl";
const VOCAB_FORMAT: &str = "jsonl";
const REQUIRED_SPECIALS: [&str; 4] = ["bos", "eos", "pad", "unk"];

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
    merges: Vec<(u32, u32)>,
    corpus_hash: Option<String>,
}

fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn byte_id(byte: u8) -> u32 {
    FIRST_BYTE + u32::from(byte)
}

fn layout_specials() -> SpecialTokens {
    SpecialTokens {
        bos: 0,
        eos: 1,
        pad: 2,
        unk: 3,
        latent: None,
        latent_start: None,
        latent_end: None,
    }
}

fn arc_range() -> ArcGridTokenRange {
    ArcGridTokenRange {
        start: FIRST_ARC,
        end: FIRST_ARC + ARC_N_COLORS,
    }
}

fn count_pairs(sequences: &[Vec<u32>]) -> HashMap<(u32, u32), u64> {
    let mut counts = HashMap::new();
    for seq in sequences {
        for pair in seq.windows(2) {
            *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
        }
    }
    counts
}

fn best_pair(counts: &HashMap<(u32, u32), u64>) -> Option<(u32, u32)> {
    let mut best: Option<(u64, u32, u32)> = None;
    for (&(left, right), &count) in counts {
        let take = match best {
            None => true,
            Some((best_count, best_left, best_right)) => {
                count > best_count
                    || (count == best_count
                        && (left < best_left || (left == best_left && right < best_right)))
            }
        };
        if take {
            best = Some((count, left, right));
        }
    }
    best.map(|(_, left, right)| (left, right))
}

fn apply_merge(seq: &[u32], left: u32, right: u32, new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(seq.len());
    let mut i = 0;
    while i < seq.len() {
        if i + 1 < seq.len() && seq[i] == left && seq[i + 1] == right {
            out.push(new_id);
            i += 2;
        } else {
            out.push(seq[i]);
            i += 1;
        }
    }
    out
}

fn bpe_encode(byte_ids: &[u32], merges: &[(u32, u32)]) -> Vec<u32> {
    if byte_ids.is_empty() || merges.is_empty() {
        return byte_ids.to_vec();
    }
    let rank: HashMap<(u32, u32), usize> = merges
        .iter()
        .enumerate()
        .map(|(i, pair)| (*pair, i))
        .collect();
    let mut ids = byte_ids.to_vec();
    loop {
        let mut best_rank: Option<usize> = None;
        let mut best_pos: Option<usize> = None;
        for i in 0..ids.len().saturating_sub(1) {
            if let Some(&r) = rank.get(&(ids[i], ids[i + 1])) {
                if best_rank.map(|br| r < br).unwrap_or(true) {
                    best_rank = Some(r);
                    best_pos = Some(i);
                }
            }
        }
        let Some(r) = best_rank else {
            break;
        };
        let pos = best_pos.expect("rank implies position");
        let new_id = FIRST_MERGE + r as u32;
        let mut next = Vec::with_capacity(ids.len() - 1);
        next.extend_from_slice(&ids[..pos]);
        next.push(new_id);
        next.extend_from_slice(&ids[pos + 2..]);
        ids = next;
    }
    ids
}

fn expand_token(token_id: u32, merges: &[(u32, u32)]) -> Result<Vec<u8>> {
    if (FIRST_BYTE..FIRST_BYTE + N_BYTES).contains(&token_id) {
        return Ok(vec![(token_id - FIRST_BYTE) as u8]);
    }
    let Some(index) = (token_id.checked_sub(FIRST_MERGE)).and_then(|i| usize::try_from(i).ok())
    else {
        return Err(Error::UnknownId(token_id));
    };
    let Some(&(left, right)) = merges.get(index) else {
        return Err(Error::UnknownId(token_id));
    };
    let mut bytes = expand_token(left, merges)?;
    bytes.extend(expand_token(right, merges)?);
    Ok(bytes)
}

fn canonical_json(value: &Value) -> String {
    serde_json::to_string(value).expect("compact json")
}

fn artifact_bytes(
    specials: &SpecialTokens,
    merges: &[(u32, u32)],
    vocab_size: u32,
    tokenizer_id: &str,
    byte_fallback: bool,
) -> Vec<u8> {
    let mut lines = Vec::new();
    let ids = [specials.bos, specials.eos, specials.pad, specials.unk];
    for (name, id) in REQUIRED_SPECIALS.iter().zip(ids) {
        lines.push(canonical_json(
            &json!({"id": id, "kind": "special", "name": name}),
        ));
    }
    for byte in 0..=255u8 {
        lines.push(canonical_json(&json!({
            "byte": byte,
            "id": byte_id(byte),
            "kind": "byte"
        })));
    }
    for color in 0..ARC_N_COLORS {
        lines.push(canonical_json(&json!({
            "color": color,
            "id": FIRST_ARC + color,
            "kind": "arc"
        })));
    }
    for (rank, &(left, right)) in merges.iter().enumerate() {
        lines.push(canonical_json(&json!({
            "id": FIRST_MERGE + rank as u32,
            "kind": "merge",
            "left": left,
            "right": right
        })));
    }
    lines.push(canonical_json(&json!({
        "algorithm": ALGORITHM,
        "byte_fallback": byte_fallback,
        "kind": "header",
        "tokenizer_id": tokenizer_id,
        "vocab_size": vocab_size,
    })));
    let mut out = String::new();
    for line in lines {
        out.push_str(&line);
        out.push('\n');
    }
    out.into_bytes()
}

fn specials_map(specials: &SpecialTokens) -> Map<String, Value> {
    let mut map = Map::new();
    map.insert("bos".into(), json!(specials.bos));
    map.insert("eos".into(), json!(specials.eos));
    map.insert("pad".into(), json!(specials.pad));
    map.insert("unk".into(), json!(specials.unk));
    if let Some(v) = specials.latent {
        map.insert("latent".into(), json!(v));
    }
    if let Some(v) = specials.latent_start {
        map.insert("latent_start".into(), json!(v));
    }
    if let Some(v) = specials.latent_end {
        map.insert("latent_end".into(), json!(v));
    }
    map
}

fn corpus_hash(docs: &[String]) -> String {
    let mut blob = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        if i > 0 {
            blob.push(b'\n');
        }
        blob.extend_from_slice(doc.as_bytes());
    }
    sha256_hex(&blob)
}

fn parse_vocab_jsonl(data: &[u8]) -> Result<Vec<(u32, u32)>> {
    let text = std::str::from_utf8(data).map_err(|e| Error::Message(e.to_string()))?;
    let mut merges = Vec::new();
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let obj: Value = serde_json::from_str(line).map_err(|e| Error::Message(e.to_string()))?;
        if obj.get("kind").and_then(Value::as_str) != Some("merge") {
            continue;
        }
        let left = obj
            .get("left")
            .and_then(Value::as_u64)
            .ok_or_else(|| Error::Message("merge missing left".into()))?;
        let right = obj
            .get("right")
            .and_then(Value::as_u64)
            .ok_or_else(|| Error::Message("merge missing right".into()))?;
        let left = u32::try_from(left).map_err(|e| Error::Message(e.to_string()))?;
        let right = u32::try_from(right).map_err(|e| Error::Message(e.to_string()))?;
        merges.push((left, right));
    }
    Ok(merges)
}

fn json_u32(value: &Value, key: &str) -> Result<u32> {
    let n = value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::Message(format!("missing or invalid {key}")))?;
    u32::try_from(n).map_err(|e| Error::Message(e.to_string()))
}

fn json_opt_u32(value: &Value, key: &str) -> Option<u32> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
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
        if !config.byte_fallback {
            return Err(Error::Message(
                "byte_fallback must be true for algorithm byte_fallback_bpe".into(),
            ));
        }
        if config.tokenizer_id.is_empty() {
            return Err(Error::Message("tokenizer_id must be non-empty".into()));
        }
        if config.vocab_size < MIN_VOCAB_SIZE {
            return Err(Error::VocabTooSmall(config.vocab_size));
        }

        let corpus: Vec<String> = docs
            .into_iter()
            .map(|doc| doc.as_ref().to_string())
            .collect();
        let mut sequences: Vec<Vec<u32>> = corpus
            .iter()
            .map(|doc| doc.as_bytes().iter().copied().map(byte_id).collect())
            .collect();
        let mut merges = Vec::new();
        let mut next_id = FIRST_MERGE;
        while next_id < config.vocab_size {
            let counts = count_pairs(&sequences);
            let Some((left, right)) = best_pair(&counts) else {
                break;
            };
            merges.push((left, right));
            sequences = sequences
                .iter()
                .map(|seq| apply_merge(seq, left, right, next_id))
                .collect();
            next_id += 1;
        }

        let specials = layout_specials();
        let blob = artifact_bytes(
            &specials,
            &merges,
            config.vocab_size,
            &config.tokenizer_id,
            config.byte_fallback,
        );
        let meta = TokenizerMeta {
            schema_id: SCHEMA_ID.into(),
            schema_version: SCHEMA_VERSION,
            tokenizer_id: config.tokenizer_id.clone(),
            algorithm: ALGORITHM.into(),
            vocab_size: config.vocab_size,
            byte_fallback: config.byte_fallback,
            special_token_ids: specials,
            arc_grid_token_range: arc_range(),
            artifact: Artifact {
                content_hash: sha256_hex(&blob),
                bytes: blob.len() as u64,
                path: None,
                media_type: Some(VOCAB_MEDIA_TYPE.into()),
            },
            frozen: false,
            frozen_at: None,
        };
        Ok(Self {
            meta,
            merges,
            corpus_hash: Some(corpus_hash(&corpus)),
        })
    }

    /// Set `frozen` and `frozen_at`. Idempotent if already frozen at the
    /// same timestamp. Further `train` is [`Error::Frozen`].
    pub fn freeze(&mut self, frozen_at: &str) -> Result<()> {
        if self.meta.frozen {
            if self.meta.frozen_at.as_deref() == Some(frozen_at) {
                return Ok(());
            }
            return Err(Error::Frozen);
        }
        self.meta.frozen = true;
        self.meta.frozen_at = Some(frozen_at.to_string());
        Ok(())
    }

    /// Encode UTF-8 text to token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_bytes(text.as_bytes())
    }

    /// Inverse of [`Self::encode`]. Round-trip of `encode` then `decode`
    /// returns the original text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let bytes = self.decode_bytes(ids)?;
        String::from_utf8(bytes).map_err(|e| Error::Message(e.to_string()))
    }

    /// Encode raw bytes. Used by byte fallback.
    pub fn encode_bytes(&self, bytes: &[u8]) -> Result<Vec<u32>> {
        let byte_ids: Vec<u32> = bytes.iter().copied().map(byte_id).collect();
        Ok(bpe_encode(&byte_ids, &self.merges))
    }

    /// Inverse of [`Self::encode_bytes`].
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for &id in ids {
            out.extend(expand_token(id, &self.merges)?);
        }
        Ok(out)
    }

    /// One token per ARC cell color. `cells` are row-major values in `0..ARC_N_COLORS`.
    pub fn encode_grid(&self, cells: &[u8]) -> Result<Vec<u32>> {
        let start = self.meta.arc_grid_token_range.start;
        let end = self.meta.arc_grid_token_range.end;
        let n_colors = end - start;
        let mut out = Vec::with_capacity(cells.len());
        for &cell in cells {
            if u32::from(cell) >= n_colors {
                return Err(Error::BadColor(cell));
            }
            out.push(start + u32::from(cell));
        }
        Ok(out)
    }

    /// `encode(text).len() / whitespace-separated word count`. Empty is 0.0.
    pub fn tokens_per_word(&self, text: &str) -> Result<f64> {
        let words = text.split_whitespace().count();
        if words == 0 {
            return Ok(0.0);
        }
        let n = self.encode(text)?.len();
        Ok(n as f64 / words as f64)
    }

    /// Write `dir/tokenizer.json` (F1 tokenizer) and the vocab artifact
    /// named by `meta.artifact.path`. Returns the F1 vocab pointer.
    pub fn save(&self, dir: &Path) -> Result<VocabPointer> {
        fs::create_dir_all(dir).map_err(|e| Error::Message(e.to_string()))?;
        let blob = artifact_bytes(
            &self.meta.special_token_ids,
            &self.merges,
            self.meta.vocab_size,
            &self.meta.tokenizer_id,
            self.meta.byte_fallback,
        );
        fs::write(dir.join(VOCAB_FILENAME), &blob).map_err(|e| Error::Message(e.to_string()))?;
        let artifact = Artifact {
            content_hash: sha256_hex(&blob),
            bytes: blob.len() as u64,
            path: Some(VOCAB_FILENAME.into()),
            media_type: Some(VOCAB_MEDIA_TYPE.into()),
        };
        let contract = self.contract_with_artifact(&artifact);
        let pretty =
            serde_json::to_string_pretty(&contract).map_err(|e| Error::Message(e.to_string()))?;
        fs::write(dir.join(TOKENIZER_FILENAME), pretty)
            .map_err(|e| Error::Message(e.to_string()))?;
        Ok(VocabPointer {
            schema_id: VOCAB_SCHEMA_ID.into(),
            schema_version: SCHEMA_VERSION,
            tokenizer_id: self.meta.tokenizer_id.clone(),
            vocab_size: self.meta.vocab_size,
            special_token_ids: self.meta.special_token_ids.clone(),
            artifact,
            format: VOCAB_FORMAT.into(),
        })
    }

    /// Load a directory written by [`Self::save`].
    pub fn load(dir: &Path) -> Result<Self> {
        let contract_text = fs::read_to_string(dir.join(TOKENIZER_FILENAME))
            .map_err(|e| Error::Message(format!("failed to load tokenizer.json: {e}")))?;
        let contract: Value =
            serde_json::from_str(&contract_text).map_err(|e| Error::Message(e.to_string()))?;
        let blob = fs::read(dir.join(VOCAB_FILENAME))
            .map_err(|e| Error::Message(format!("failed to load {VOCAB_FILENAME}: {e}")))?;
        let merges = parse_vocab_jsonl(&blob)?;
        let specials_raw = contract.get("special_tokens").cloned().unwrap_or(json!({}));
        let specials = SpecialTokens {
            bos: json_u32(&specials_raw, "bos")?,
            eos: json_u32(&specials_raw, "eos")?,
            pad: json_u32(&specials_raw, "pad")?,
            unk: json_u32(&specials_raw, "unk")?,
            latent: json_opt_u32(&specials_raw, "latent"),
            latent_start: json_opt_u32(&specials_raw, "latent_start"),
            latent_end: json_opt_u32(&specials_raw, "latent_end"),
        };
        let arc_raw = contract
            .get("arc_grid_token_range")
            .ok_or_else(|| Error::Message("missing arc_grid_token_range".into()))?;
        let frozen = contract
            .get("frozen")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let frozen_at = contract
            .get("frozen_at")
            .and_then(Value::as_str)
            .map(str::to_string);
        let corpus = contract
            .get("corpus_hash")
            .and_then(Value::as_str)
            .map(str::to_string);
        let meta = TokenizerMeta {
            schema_id: contract
                .get("schema_id")
                .and_then(Value::as_str)
                .unwrap_or(SCHEMA_ID)
                .to_string(),
            schema_version: json_u32(&contract, "schema_version").unwrap_or(SCHEMA_VERSION),
            tokenizer_id: contract
                .get("tokenizer_id")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::Message("missing tokenizer_id".into()))?
                .to_string(),
            algorithm: contract
                .get("algorithm")
                .and_then(Value::as_str)
                .unwrap_or(ALGORITHM)
                .to_string(),
            vocab_size: json_u32(&contract, "vocab_size")?,
            byte_fallback: contract
                .get("byte_fallback")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            special_token_ids: specials,
            arc_grid_token_range: ArcGridTokenRange {
                start: json_u32(arc_raw, "start")?,
                end: json_u32(arc_raw, "end")?,
            },
            artifact: Artifact {
                content_hash: sha256_hex(&blob),
                bytes: blob.len() as u64,
                path: Some(VOCAB_FILENAME.into()),
                media_type: Some(VOCAB_MEDIA_TYPE.into()),
            },
            frozen,
            frozen_at,
        };
        Ok(Self {
            meta,
            merges,
            corpus_hash: corpus,
        })
    }

    /// JSON object matching `prometheus.tokenizer` for `contracts.validate`.
    pub fn to_contract(&self) -> serde_json::Value {
        self.contract_with_artifact(&self.meta.artifact)
    }

    fn contract_with_artifact(&self, artifact: &Artifact) -> Value {
        let mut payload = json!({
            "schema_id": self.meta.schema_id,
            "schema_version": self.meta.schema_version,
            "tokenizer_id": self.meta.tokenizer_id,
            "algorithm": self.meta.algorithm,
            "vocab_size": self.meta.vocab_size,
            "byte_fallback": self.meta.byte_fallback,
            "special_tokens": Value::Object(specials_map(&self.meta.special_token_ids)),
            "arc_grid_token_range": {
                "start": self.meta.arc_grid_token_range.start,
                "end": self.meta.arc_grid_token_range.end,
            },
            "vocab_hash": artifact.content_hash,
            "frozen": self.meta.frozen,
        });
        if let Some(ts) = &self.meta.frozen_at {
            payload
                .as_object_mut()
                .expect("object")
                .insert("frozen_at".into(), json!(ts));
        }
        if let Some(h) = &self.corpus_hash {
            payload
                .as_object_mut()
                .expect("object")
                .insert("corpus_hash".into(), json!(h));
        }
        payload
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
