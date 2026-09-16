//! Deterministic shuffle, packing, and in-memory loader (spec 7, 15.5 B5).
//!
//! Packing configs match `prometheus.packing` (F1). Loader checkpoints match
//! `prometheus.loader_state`. The JAX training loop never waits on data: this
//! crate yields packed token/mask batches whose `(epoch, shard_offset)` cursor
//! reconstructs the same batch given the same sequences and seed.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

pub const SCHEMA_ID_PACKING: &str = "prometheus.packing";
pub const SCHEMA_ID_LOADER_STATE: &str = "prometheus.loader_state";
pub const SCHEMA_VERSION: u32 = 1;

/// SHA-256 of empty input; remainder hash when no packing leftover exists.
const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    #[error("unknown packing strategy {0}")]
    UnknownStrategy(String),
    #[error("sequence_length must be > 0")]
    BadSequenceLength,
    #[error("empty")]
    Empty,
    #[error("all documents dropped as short")]
    DroppedShort,
    #[error("exhausted")]
    Exhausted,
    #[error("{0}")]
    Message(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    ConcatEos,
    DocumentMask,
    SingleDocument,
}

impl Strategy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ConcatEos => "concat_eos",
            Self::DocumentMask => "document_mask",
            Self::SingleDocument => "single_document",
        }
    }
}

/// Packing config. Field names match `prometheus.packing` v1 without the
/// schema envelope; `to_json` adds `schema_id` / `schema_version`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Packing {
    pub packing_id: String,
    pub sequence_length: u32,
    pub strategy: Strategy,
    pub eos_between_docs: bool,
    pub pad_id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drop_short_below: Option<u32>,
}

impl Packing {
    pub fn to_json(&self) -> Result<serde_json::Value, Error> {
        let value = serde_json::to_value(self).map_err(json_err)?;
        with_envelope(value, SCHEMA_ID_PACKING)
    }

    pub fn from_json(value: &serde_json::Value) -> Result<Self, Error> {
        require_schema(value, SCHEMA_ID_PACKING)?;
        reject_unknown_strategy(value)?;
        serde_json::from_value(value.clone()).map_err(json_err)
    }
}

/// Packed sequence. `mask` is 1 for a real token, 0 for pad.
/// `document_ids` is `Some` iff strategy is `document_mask`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedSequence {
    pub tokens: Vec<u32>,
    pub mask: Vec<u8>,
    pub document_ids: Option<Vec<u32>>,
}

/// Loader cursor stored in a checkpoint. Field names match
/// `prometheus.loader_state` v1 without the schema envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoaderState {
    pub epoch: u32,
    pub step: u64,
    pub shuffle_seed: u64,
    pub shard_index: u32,
    pub shard_offset: u64,
    pub consumed_token_count: u64,
    pub data_mix_hash: String,
    pub packing_remainder_hash: String,
}

impl LoaderState {
    pub fn to_json(&self) -> Result<serde_json::Value, Error> {
        let value = serde_json::to_value(self).map_err(json_err)?;
        with_envelope(value, SCHEMA_ID_LOADER_STATE)
    }

    pub fn from_json(value: &serde_json::Value) -> Result<Self, Error> {
        require_schema(value, SCHEMA_ID_LOADER_STATE)?;
        serde_json::from_value(value.clone()).map_err(json_err)
    }
}

pub struct Loader {
    sequences: Vec<PackedSequence>,
    state: LoaderState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch {
    pub tokens: Vec<Vec<u32>>,
    pub mask: Vec<Vec<u8>>,
    pub state: LoaderState,
}

impl Loader {
    pub fn new(
        sequences: Vec<PackedSequence>,
        packing: Packing,
        shuffle_seed: u64,
        data_mix_hash: String,
    ) -> Self {
        let _packing = packing;
        Self {
            sequences,
            state: LoaderState {
                epoch: 0,
                step: 0,
                shuffle_seed,
                shard_index: 0,
                shard_offset: 0,
                consumed_token_count: 0,
                data_mix_hash,
                packing_remainder_hash: EMPTY_SHA256.to_string(),
            },
        }
    }

    pub fn from_state(
        sequences: Vec<PackedSequence>,
        packing: Packing,
        state: LoaderState,
    ) -> Result<Self, Error> {
        let _packing = packing;
        Ok(Self { sequences, state })
    }

    pub fn state(&self) -> LoaderState {
        self.state.clone()
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Result<Batch, Error> {
        let n = self.sequences.len() as u64;
        if n == 0 {
            return Err(Error::Exhausted);
        }
        let seed = self.state.shuffle_seed;
        let mut epoch = self.state.epoch;
        let mut offset = self.state.shard_offset;
        let mut order = shuffle_order(n, seed, epoch);
        let mut tokens = Vec::with_capacity(batch_size);
        let mut mask = Vec::with_capacity(batch_size);
        let mut consumed = 0u64;

        for _ in 0..batch_size {
            let idx = order[offset as usize] as usize;
            let seq = &self.sequences[idx];
            consumed += seq.tokens.len() as u64;
            tokens.push(seq.tokens.clone());
            mask.push(seq.mask.clone());
            offset += 1;
            if offset == n {
                offset = 0;
                epoch = epoch.wrapping_add(1);
                order = shuffle_order(n, seed, epoch);
            }
        }

        let raw = self.state.shard_offset + batch_size as u64;
        self.state.epoch = self.state.epoch.wrapping_add((raw / n) as u32);
        self.state.shard_offset = raw % n;
        self.state.step += 1;
        self.state.consumed_token_count += consumed;

        Ok(Batch {
            tokens,
            mask,
            state: self.state.clone(),
        })
    }
}

/// Deterministic permutation of `0..n` for `(seed, epoch)`.
pub fn shuffle_order(n: u64, seed: u64, epoch: u32) -> Vec<u64> {
    let mut order: Vec<u64> = (0..n).collect();
    if n <= 1 {
        return order;
    }
    let mut rng = SplitMix64::new(
        seed ^ 0x9E3779B97F4A7C15u64.wrapping_mul(u64::from(epoch).wrapping_add(1)),
    );
    for i in (1..n as usize).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    order
}

pub fn pack(
    docs: &[Vec<u32>],
    packing: &Packing,
    eos_id: u32,
) -> Result<Vec<PackedSequence>, Error> {
    if packing.sequence_length == 0 {
        return Err(Error::BadSequenceLength);
    }
    if docs.is_empty() {
        return Err(Error::Empty);
    }
    match packing.strategy {
        Strategy::ConcatEos => Ok(pack_concat(docs, packing, eos_id, false)),
        Strategy::DocumentMask => Ok(pack_concat(docs, packing, eos_id, true)),
        Strategy::SingleDocument => pack_single(docs, packing),
    }
}

pub fn reconstruct(
    sequences: &[PackedSequence],
    packing: &Packing,
    state: &LoaderState,
    batch_size: usize,
) -> Result<Batch, Error> {
    let mut loader = Loader::from_state(sequences.to_vec(), packing.clone(), state.clone())?;
    loader.next_batch(batch_size)
}

fn pack_concat(
    docs: &[Vec<u32>],
    packing: &Packing,
    eos_id: u32,
    with_doc_ids: bool,
) -> Vec<PackedSequence> {
    let mut packer = ConcatPacker::new(packing, with_doc_ids);
    for (i, doc) in docs.iter().enumerate() {
        if i > 0 && packing.eos_between_docs {
            // EOS closes the previous document and belongs to it.
            packer.push(eos_id, (i - 1) as u32);
        }
        for &token in doc {
            packer.push(token, i as u32);
        }
    }
    packer.finish()
}

fn pack_single(docs: &[Vec<u32>], packing: &Packing) -> Result<Vec<PackedSequence>, Error> {
    let seq_len = packing.sequence_length as usize;
    let mut out = Vec::new();
    let mut dropped = 0u64;
    for doc in docs {
        if let Some(min) = packing.drop_short_below {
            if (doc.len() as u32) < min {
                dropped += 1;
                continue;
            }
        }
        let take = doc.len().min(seq_len);
        let mut tokens = Vec::with_capacity(seq_len);
        tokens.extend_from_slice(&doc[..take]);
        tokens.resize(seq_len, packing.pad_id);
        let mut mask = vec![1u8; take];
        mask.resize(seq_len, 0);
        out.push(PackedSequence {
            tokens,
            mask,
            document_ids: None,
        });
    }
    if out.is_empty() {
        return Err(if dropped > 0 {
            Error::DroppedShort
        } else {
            Error::Empty
        });
    }
    Ok(out)
}

struct ConcatPacker {
    seq_len: usize,
    pad_id: u32,
    with_doc_ids: bool,
    tokens: Vec<u32>,
    mask: Vec<u8>,
    doc_ids: Vec<u32>,
    out: Vec<PackedSequence>,
}

impl ConcatPacker {
    fn new(packing: &Packing, with_doc_ids: bool) -> Self {
        let seq_len = packing.sequence_length as usize;
        Self {
            seq_len,
            pad_id: packing.pad_id,
            with_doc_ids,
            tokens: Vec::with_capacity(seq_len),
            mask: Vec::with_capacity(seq_len),
            doc_ids: Vec::with_capacity(seq_len),
            out: Vec::new(),
        }
    }

    fn push(&mut self, token: u32, doc_id: u32) {
        self.tokens.push(token);
        self.mask.push(1);
        if self.with_doc_ids {
            self.doc_ids.push(doc_id);
        }
        if self.tokens.len() == self.seq_len {
            self.emit(false);
        }
    }

    fn emit(&mut self, pad: bool) {
        if self.tokens.is_empty() {
            return;
        }
        if pad {
            self.tokens.resize(self.seq_len, self.pad_id);
            self.mask.resize(self.seq_len, 0);
            if self.with_doc_ids {
                self.doc_ids.resize(self.seq_len, 0);
            }
        }
        self.out.push(PackedSequence {
            tokens: std::mem::take(&mut self.tokens),
            mask: std::mem::take(&mut self.mask),
            document_ids: if self.with_doc_ids {
                Some(std::mem::take(&mut self.doc_ids))
            } else {
                None
            },
        });
        self.tokens = Vec::with_capacity(self.seq_len);
        self.mask = Vec::with_capacity(self.seq_len);
        if self.with_doc_ids {
            self.doc_ids = Vec::with_capacity(self.seq_len);
        }
    }

    fn finish(mut self) -> Vec<PackedSequence> {
        self.emit(true);
        self.out
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn json_err(err: serde_json::Error) -> Error {
    Error::Message(err.to_string())
}

fn with_envelope(value: Value, schema_id: &str) -> Result<Value, Error> {
    let Value::Object(fields) = value else {
        return Err(Error::Message("expected a JSON object".into()));
    };
    let mut out = Map::new();
    out.insert(
        "schema_id".to_string(),
        Value::String(schema_id.to_string()),
    );
    out.insert("schema_version".to_string(), Value::from(SCHEMA_VERSION));
    for (key, val) in fields {
        out.insert(key, val);
    }
    Ok(Value::Object(out))
}

fn require_schema(value: &Value, expected: &str) -> Result<(), Error> {
    let Some(obj) = value.as_object() else {
        return Err(Error::Message("JSON must be an object".into()));
    };
    match obj.get("schema_id").and_then(Value::as_str) {
        Some(id) if id == expected => Ok(()),
        _ => Err(Error::Message(format!("expected schema_id {expected}"))),
    }
}

fn reject_unknown_strategy(value: &Value) -> Result<(), Error> {
    let Some(name) = value.get("strategy").and_then(Value::as_str) else {
        return Ok(());
    };
    match name {
        "concat_eos" | "document_mask" | "single_document" => Ok(()),
        other => Err(Error::UnknownStrategy(other.to_string())),
    }
}
