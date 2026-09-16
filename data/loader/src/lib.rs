//! Deterministic shuffle, packing, and the Rust data loader (spec 7, 15.5 B5).
//!
//! Packing configs match `prometheus.packing` (F1). Loader checkpoints match
//! `prometheus.loader_state`. The gate is: given seed + state, `reconstruct`
//! yields the same batch `next_batch` produced at that step.
//!
//! Documents are already tokenized `u32` id sequences. The tokenizer (F6)
//! is out of scope. Nothing here packs or shuffles yet.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SCHEMA_ID_PACKING: &str = "prometheus.packing";
pub const SCHEMA_ID_LOADER_STATE: &str = "prometheus.loader_state";
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum Error {
    #[error("empty document list")]
    Empty,
    #[error("sequence_length must be >= 1")]
    BadSequenceLength,
    #[error("unknown packing strategy {0}")]
    UnknownStrategy(String),
    #[error("document shorter than drop_short_below")]
    DroppedShort,
    #[error("loader exhausted")]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drop_short_below: Option<u32>,
}

impl Packing {
    pub fn to_json(&self) -> Result<serde_json::Value, Error> {
        let _ = self;
        unimplemented!("B5 Packing::to_json")
    }

    pub fn from_json(value: &serde_json::Value) -> Result<Self, Error> {
        let _ = value;
        unimplemented!("B5 Packing::from_json")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedSequence {
    pub tokens: Vec<u32>,
    /// 1 where the position is a real token, 0 where it is pad.
    pub mask: Vec<u8>,
    /// Document id per position, for `document_mask`. None for concat.
    pub document_ids: Option<Vec<u32>>,
}

/// Loader checkpoint. Field names match `prometheus.loader_state` v1.
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
        let _ = self;
        unimplemented!("B5 LoaderState::to_json")
    }

    pub fn from_json(value: &serde_json::Value) -> Result<Self, Error> {
        let _ = value;
        unimplemented!("B5 LoaderState::from_json")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch {
    pub tokens: Vec<Vec<u32>>,
    pub mask: Vec<Vec<u8>>,
    pub state: LoaderState,
}

/// Deterministic permutation of `0..n` for `(seed, epoch)`. Same inputs
/// must yield the same order on every process.
pub fn shuffle_order(n: u64, seed: u64, epoch: u32) -> Vec<u64> {
    let _ = (n, seed, epoch);
    unimplemented!("B5 shuffle_order")
}

/// Pack tokenized documents into fixed-length sequences.
///
/// `concat_eos`: concatenate, insert `eos_id` between docs when
/// `eos_between_docs`, pad the tail with `pad_id`.
/// `document_mask`: same packing, plus per-position document ids.
/// `single_document`: one doc per sequence, pad or truncate to
/// `sequence_length`; docs shorter than `drop_short_below` are dropped.
pub fn pack(
    docs: &[Vec<u32>],
    packing: &Packing,
    eos_id: u32,
) -> Result<Vec<PackedSequence>, Error> {
    let _ = (docs, packing, eos_id);
    unimplemented!("B5 pack")
}

pub struct Loader {
    _private: (),
}

impl Loader {
    pub fn new(
        sequences: Vec<PackedSequence>,
        packing: Packing,
        shuffle_seed: u64,
        data_mix_hash: String,
    ) -> Self {
        let _ = (sequences, packing, shuffle_seed, data_mix_hash);
        unimplemented!("B5 Loader::new")
    }

    pub fn from_state(
        sequences: Vec<PackedSequence>,
        packing: Packing,
        state: LoaderState,
    ) -> Result<Self, Error> {
        let _ = (sequences, packing, state);
        unimplemented!("B5 Loader::from_state")
    }

    pub fn state(&self) -> LoaderState {
        unimplemented!("B5 Loader::state")
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Result<Batch, Error> {
        let _ = batch_size;
        unimplemented!("B5 Loader::next_batch")
    }
}

/// Given seed + state, rebuild the batch `next_batch` produced at that step.
pub fn reconstruct(
    sequences: &[PackedSequence],
    packing: &Packing,
    state: &LoaderState,
    batch_size: usize,
) -> Result<Batch, Error> {
    let _ = (sequences, packing, state, batch_size);
    unimplemented!("B5 reconstruct")
}
