#!/usr/bin/env python3
"""Slow, obvious B5 reference: packing tokenized documents.

Rust tests call the public crate API, not this file. This exists so an
implementer can see the intended packing, and so goldens can be regenerated.

Rules (spec 07, 15.5 B5, data/loader/src/lib.rs):
  * sequence_length == 0 is an error (BadSequenceLength), checked first
  * empty document list is an error (Empty)
  * concat_eos / document_mask: concatenate in order; insert eos_id BETWEEN
    docs when eos_between_docs (not before the first, not after the last).
    Split into chunks of exactly sequence_length; pad only the tail of the
    last chunk with pad_id. mask 1 = real token (including EOS), 0 = pad.
    Empty stream (all-empty docs, no EOS) yields zero sequences.
  * document_mask: same packing plus 0-based document id per position.
    EOS belongs to the document it closes. Pad positions use document id 0.
  * single_document: one kept doc per sequence; right-pad or right-truncate
    to sequence_length. Docs with len < drop_short_below are dropped.
    Dropping every document is DroppedShort. eos_id is ignored.
    document_ids is None.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CONCAT_EOS = "concat_eos"
DOCUMENT_MASK = "document_mask"
SINGLE_DOCUMENT = "single_document"


class BadSequenceLength(ValueError):
    pass


class Empty(ValueError):
    pass


class DroppedShort(ValueError):
    pass


def pack(docs: list[list[int]], packing: dict[str, Any], eos_id: int) -> list[dict[str, Any]]:
    seq_len = int(packing["sequence_length"])
    if seq_len == 0:
        raise BadSequenceLength("sequence_length must be >= 1")
    if not docs:
        raise Empty("empty document list")
    strategy = packing["strategy"]
    if strategy in (CONCAT_EOS, DOCUMENT_MASK):
        return _pack_stream(
            docs,
            seq_len=seq_len,
            eos_id=int(eos_id),
            pad_id=int(packing["pad_id"]),
            eos_between=bool(packing["eos_between_docs"]),
            with_doc_ids=strategy == DOCUMENT_MASK,
        )
    if strategy == SINGLE_DOCUMENT:
        return _pack_single(
            docs,
            seq_len=seq_len,
            pad_id=int(packing["pad_id"]),
            drop_short_below=packing.get("drop_short_below"),
        )
    raise ValueError(f"unknown packing strategy {strategy}")


def _pack_stream(
    docs: list[list[int]],
    *,
    seq_len: int,
    eos_id: int,
    pad_id: int,
    eos_between: bool,
    with_doc_ids: bool,
) -> list[dict[str, Any]]:
    tokens: list[int] = []
    doc_ids: list[int] = []
    for i, doc in enumerate(docs):
        if i > 0 and eos_between:
            tokens.append(eos_id)
            doc_ids.append(i - 1)
        tokens.extend(int(t) for t in doc)
        doc_ids.extend([i] * len(doc))
    if not tokens:
        return []

    stream = np.asarray(tokens, dtype=np.uint32)
    ids = np.asarray(doc_ids, dtype=np.uint32)
    out: list[dict[str, Any]] = []
    for start in range(0, stream.size, seq_len):
        chunk = stream[start : start + seq_len]
        chunk_ids = ids[start : start + seq_len]
        real = int(chunk.size)
        pad_n = seq_len - real
        if pad_n:
            chunk = np.concatenate(
                [chunk, np.full(pad_n, pad_id, dtype=np.uint32)]
            )
            chunk_ids = np.concatenate(
                [chunk_ids, np.zeros(pad_n, dtype=np.uint32)]
            )
        mask = np.concatenate(
            [
                np.ones(real, dtype=np.uint8),
                np.zeros(pad_n, dtype=np.uint8),
            ]
        )
        rec: dict[str, Any] = {
            "tokens": chunk.tolist(),
            "mask": mask.tolist(),
        }
        if with_doc_ids:
            rec["document_ids"] = chunk_ids.tolist()
        else:
            rec["document_ids"] = None
        out.append(rec)
    return out


def _pack_single(
    docs: list[list[int]],
    *,
    seq_len: int,
    pad_id: int,
    drop_short_below: int | None,
) -> list[dict[str, Any]]:
    kept: list[list[int]] = []
    dropped = False
    for doc in docs:
        if drop_short_below is not None and len(doc) < int(drop_short_below):
            dropped = True
            continue
        kept.append(doc)
    if not kept:
        if dropped:
            raise DroppedShort("document shorter than drop_short_below")
        raise Empty("empty document list")

    out: list[dict[str, Any]] = []
    for doc in kept:
        arr = np.asarray(doc[:seq_len], dtype=np.uint32)
        real = int(arr.size)
        pad_n = seq_len - real
        if pad_n:
            arr = np.concatenate([arr, np.full(pad_n, pad_id, dtype=np.uint32)])
        mask = np.concatenate(
            [
                np.ones(real, dtype=np.uint8),
                np.zeros(pad_n, dtype=np.uint8),
            ]
        )
        out.append(
            {
                "tokens": arr.tolist(),
                "mask": mask.tolist(),
                "document_ids": None,
            }
        )
    return out
