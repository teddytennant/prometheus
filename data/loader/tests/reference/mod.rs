//! Slow, obvious B5 reference: packing, shuffle mixing, loader cursor.
//!
//! Rust tests call the public crate API and compare packing results to this
//! module. Implementers must match these packing rules; `shuffle_order` is
//! only required to be a deterministic permutation (the loader must *use*
//! `shuffle_order`, but the PRNG itself is not pinned here).
//!
//! Packing rules (spec 07, lib.rs docs):
//! - `sequence_length == 0` → `Error::BadSequenceLength` (checked first).
//! - empty `docs` → `Error::Empty`.
//! - `concat_eos` / `document_mask`: concatenate in order; when
//!   `eos_between_docs`, insert `eos_id` *between* documents (not before the
//!   first, not after the last). Split the stream into chunks of exactly
//!   `sequence_length`; pad only the tail of the last chunk with `pad_id`.
//!   Mask is 1 on real tokens (including EOS) and 0 on pad. An empty stream
//!   (all-empty docs and no EOS inserted) yields zero sequences, not an error.
//! - `document_mask`: same tokens/mask, plus 0-based document index per
//!   position. An EOS belongs to the document it closes (the previous doc).
//!   Pad positions use document id 0.
//! - `single_document`: one kept document per sequence; pad on the right or
//!   truncate on the right to `sequence_length`. Documents with
//!   `len < drop_short_below` are dropped. If every document is dropped,
//!   `Error::DroppedShort`. `eos_id` is ignored. `document_ids` is None.

use prometheus_loader::{Error, PackedSequence, Packing, Strategy};

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
        Strategy::ConcatEos => pack_stream(docs, packing, eos_id, false),
        Strategy::DocumentMask => pack_stream(docs, packing, eos_id, true),
        Strategy::SingleDocument => pack_single(docs, packing),
    }
}

fn pack_stream(
    docs: &[Vec<u32>],
    packing: &Packing,
    eos_id: u32,
    with_doc_ids: bool,
) -> Result<Vec<PackedSequence>, Error> {
    let seq_len = packing.sequence_length as usize;
    let mut tokens = Vec::new();
    let mut doc_ids = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        if i > 0 && packing.eos_between_docs {
            tokens.push(eos_id);
            doc_ids.push((i - 1) as u32);
        }
        tokens.extend_from_slice(doc);
        doc_ids.extend(std::iter::repeat_n(i as u32, doc.len()));
    }
    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    let mut start = 0usize;
    while start < tokens.len() {
        let end = (start + seq_len).min(tokens.len());
        let mut seq_tokens = tokens[start..end].to_vec();
        let mut seq_ids = doc_ids[start..end].to_vec();
        let mut mask = vec![1u8; seq_tokens.len()];
        if seq_tokens.len() < seq_len {
            let pad_n = seq_len - seq_tokens.len();
            seq_tokens.extend(std::iter::repeat_n(packing.pad_id, pad_n));
            seq_ids.extend(std::iter::repeat_n(0u32, pad_n));
            mask.extend(std::iter::repeat_n(0u8, pad_n));
        }
        out.push(PackedSequence {
            tokens: seq_tokens,
            mask,
            document_ids: if with_doc_ids { Some(seq_ids) } else { None },
        });
        start += seq_len;
    }
    Ok(out)
}

fn pack_single(docs: &[Vec<u32>], packing: &Packing) -> Result<Vec<PackedSequence>, Error> {
    let seq_len = packing.sequence_length as usize;
    let mut kept: Vec<&Vec<u32>> = Vec::new();
    let mut dropped = false;
    for doc in docs {
        if let Some(min) = packing.drop_short_below {
            if (doc.len() as u32) < min {
                dropped = true;
                continue;
            }
        }
        kept.push(doc);
    }
    if kept.is_empty() {
        return Err(if dropped {
            Error::DroppedShort
        } else {
            Error::Empty
        });
    }

    let mut out = Vec::new();
    for doc in kept {
        let take = doc.len().min(seq_len);
        let mut tokens = doc[..take].to_vec();
        let mut mask = vec![1u8; tokens.len()];
        if tokens.len() < seq_len {
            let pad_n = seq_len - tokens.len();
            tokens.extend(std::iter::repeat_n(packing.pad_id, pad_n));
            mask.extend(std::iter::repeat_n(0u8, pad_n));
        }
        out.push(PackedSequence {
            tokens,
            mask,
            document_ids: None,
        });
    }
    Ok(out)
}
