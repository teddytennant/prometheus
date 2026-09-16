//! Shared builders and assertions for B5 loader integration tests.
#![allow(dead_code)]

use prometheus_loader::{Error, PackedSequence, Packing, Strategy};
use serde::Deserialize;

/// `data_mix_hash` from `contracts/goldens/v1/loader_state.default.json`.
pub const MIX_HASH: &str = "1f0368a7fbed2400c8f3b14febfddf1631b879b336c784b08ad2619dbdb5ab59";

/// `packing_remainder_hash` from the same golden.
pub const REMAINDER_HASH: &str = "ac760c60d7980c43c3a2c6f6651d125d759c1ecb910a01141e758550414bdc73";

/// SHA-256 of empty bytes. A reasonable remainder hash when nothing is leftover.
pub const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

pub const EOS: u32 = 1;
pub const PAD: u32 = 0;

#[derive(Debug, Deserialize)]
pub struct GoldenPack {
    pub eos_id: u32,
    pub docs: Vec<Vec<u32>>,
    pub packing: Packing,
    pub expected: Vec<GoldenSeq>,
}

#[derive(Debug, Deserialize)]
pub struct GoldenSeq {
    pub tokens: Vec<u32>,
    pub mask: Vec<u8>,
    #[serde(default)]
    pub document_ids: Option<Vec<u32>>,
}

impl GoldenSeq {
    pub fn into_packed(self) -> PackedSequence {
        PackedSequence {
            tokens: self.tokens,
            mask: self.mask,
            document_ids: self.document_ids,
        }
    }
}

pub fn packing(
    strategy: Strategy,
    sequence_length: u32,
    eos_between_docs: bool,
    pad_id: u32,
) -> Packing {
    Packing {
        packing_id: "test-pack".to_string(),
        sequence_length,
        strategy,
        eos_between_docs,
        pad_id,
        drop_short_below: None,
    }
}

pub fn packing_drop(
    strategy: Strategy,
    sequence_length: u32,
    eos_between_docs: bool,
    pad_id: u32,
    drop_short_below: u32,
) -> Packing {
    Packing {
        packing_id: "test-pack".to_string(),
        sequence_length,
        strategy,
        eos_between_docs,
        pad_id,
        drop_short_below: Some(drop_short_below),
    }
}

pub fn ident_seq(id: u32, len: usize) -> PackedSequence {
    PackedSequence {
        tokens: vec![id; len],
        mask: vec![1; len],
        document_ids: None,
    }
}

pub fn ident_corpus(n: u32, seq_len: usize) -> Vec<PackedSequence> {
    (0..n).map(|i| ident_seq(1000 + i, seq_len)).collect()
}

pub fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

pub fn flatten_real_tokens(seqs: &[PackedSequence]) -> Vec<u32> {
    let mut out = Vec::new();
    for seq in seqs {
        for (&tok, &m) in seq.tokens.iter().zip(seq.mask.iter()) {
            if m == 1 {
                out.push(tok);
            }
        }
    }
    out
}

pub fn concat_stream(docs: &[Vec<u32>], eos_id: u32, eos_between_docs: bool) -> Vec<u32> {
    let mut stream = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        if i > 0 && eos_between_docs {
            stream.push(eos_id);
        }
        stream.extend_from_slice(doc);
    }
    stream
}

pub fn assert_packed_shape(
    seq: &PackedSequence,
    seq_len: usize,
    expect_doc_ids: bool,
    pad_id: u32,
) {
    assert_eq!(
        seq.tokens.len(),
        seq_len,
        "tokens must be exactly sequence_length"
    );
    assert_eq!(
        seq.mask.len(),
        seq_len,
        "mask must be exactly sequence_length"
    );
    assert!(
        seq.mask.iter().all(|&m| m == 0 || m == 1),
        "mask entries must be 0 or 1, got {:?}",
        seq.mask
    );
    let mut seen_pad = false;
    for i in 0..seq_len {
        if seq.mask[i] == 0 {
            seen_pad = true;
            assert_eq!(
                seq.tokens[i], pad_id,
                "mask 0 position {i} must hold pad_id {pad_id}"
            );
        } else {
            assert!(
                !seen_pad,
                "mask must be a prefix of 1s then a suffix of 0s, got {:?}",
                seq.mask
            );
        }
    }
    match &seq.document_ids {
        Some(ids) => {
            assert!(
                expect_doc_ids,
                "document_ids must be None for concat/single, got {ids:?}"
            );
            assert_eq!(ids.len(), seq_len, "document_ids length");
        }
        None => assert!(
            !expect_doc_ids,
            "document_mask sequences must carry per-position document ids"
        ),
    }
}

pub fn assert_err_empty<T: std::fmt::Debug>(result: Result<T, Error>) {
    match result {
        Err(Error::Empty) => {}
        Ok(v) => panic!("expected Error::Empty, got Ok({v:?})"),
        Err(other) => panic!("expected Error::Empty, got Err({other:?})"),
    }
}

pub fn assert_err_bad_seq_len<T: std::fmt::Debug>(result: Result<T, Error>) {
    match result {
        Err(Error::BadSequenceLength) => {}
        Ok(v) => panic!("expected Error::BadSequenceLength, got Ok({v:?})"),
        Err(other) => panic!("expected Error::BadSequenceLength, got Err({other:?})"),
    }
}

pub fn assert_err_dropped_short<T: std::fmt::Debug>(result: Result<T, Error>) {
    match result {
        Err(Error::DroppedShort) => {}
        Ok(v) => panic!("expected Error::DroppedShort, got Ok({v:?})"),
        Err(other) => panic!("expected Error::DroppedShort, got Err({other:?})"),
    }
}

pub fn assert_err_exhausted<T: std::fmt::Debug>(result: Result<T, Error>) {
    match result {
        Err(Error::Exhausted) => {}
        Ok(v) => panic!("expected Error::Exhausted, got Ok({v:?})"),
        Err(other) => panic!("expected Error::Exhausted, got Err({other:?})"),
    }
}

pub fn assert_err_unknown_strategy<T: std::fmt::Debug>(result: Result<T, Error>) {
    match result {
        Err(Error::UnknownStrategy(name)) => {
            assert!(
                !name.is_empty(),
                "UnknownStrategy should mention the bad name"
            );
        }
        Ok(v) => panic!("expected Error::UnknownStrategy, got Ok({v:?})"),
        Err(other) => panic!("expected Error::UnknownStrategy, got Err({other:?})"),
    }
}

/// Epoch / shard_offset after consuming `batch_size` sequences from a corpus of
/// `n` packed sequences, wrapping into later epochs.
pub fn cursor_after(n: u64, epoch: u32, shard_offset: u64, batch_size: usize) -> (u32, u64) {
    assert!(n > 0, "cursor_after is undefined for an empty corpus");
    let raw = shard_offset + batch_size as u64;
    let new_epoch = epoch.wrapping_add((raw / n) as u32);
    let new_offset = raw % n;
    (new_epoch, new_offset)
}
