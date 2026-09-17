#!/usr/bin/env python3
"""Slow, obvious B4 reference: ngrams, ngram_match, scan, method_hash.

Rust tests call the public crate API, not this file. This exists so an
implementer can see the intended algorithm (slow sliding windows).

Rules (spec 07, 11, 15.5 B4, data/decontam/src/lib.rs):
  * lowercase + whitespace collapse (str.split), then overlapping word
    n-grams of width NGRAM_N = 8 as space-joined strings. Shorter than 8
    words yields no grams.
  * ngram_match: a shared 8-gram is a Hit (suite, item_id, method="ngram").
    Empty index is an error (MissingIndex), never clean.
  * scan_text / scan_shard: no hits -> clean; any hit -> flagged.
    against = unique suite list; method_hash always set on success.
  * scan_shard: any document flagged flags the shard.
  * method_hash: SHA-256 hex of the canonical transcript (see method_hash).
"""

from __future__ import annotations

import hashlib
from typing import Any

NGRAM_N = 8


class MissingIndex(ValueError):
    pass


def ngrams(text: str, ngram_n: int = NGRAM_N) -> list[str]:
    words = text.lower().split()
    n = int(ngram_n)
    if n <= 0 or len(words) < n:
        return []
    out: list[str] = []
    for i in range(len(words) - n + 1):
        out.append(" ".join(words[i : i + n]))
    return out


def ngram_match(items: list[dict[str, str]], text: str) -> list[dict[str, str]]:
    if not items:
        raise MissingIndex("decontamination index is empty")
    query = set(ngrams(text))
    hits: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        grams = ngrams(item["text"])
        key = (item["suite"], item["id"])
        if key in seen:
            continue
        if any(g in query for g in grams):
            seen.add(key)
            hits.append(
                {"suite": item["suite"], "item_id": item["id"], "method": "ngram"}
            )
    return hits


def method_hash(items: list[dict[str, str]], ngram_n: int = NGRAM_N) -> str:
    if not items:
        raise MissingIndex("decontamination index is empty")
    ordered = sorted(items, key=lambda it: (it["suite"], it["id"]))
    h = hashlib.sha256()
    h.update(b"prometheus-decontam/v1\n")
    h.update(f"ngram_n={int(ngram_n)}\n".encode())
    h.update(b"embed_model=\n")
    for item in ordered:
        h.update(item["suite"].encode("utf-8"))
        h.update(b"\0")
        h.update(item["id"].encode("utf-8"))
        h.update(b"\0")
        h.update(item["text"].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _against(items: list[dict[str, str]]) -> list[str]:
    return sorted({item["suite"] for item in items})


def _report(hits: list[dict[str, str]], items: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "status": "flagged" if hits else "clean",
        "against": _against(items),
        "method_hash": method_hash(items),
        "hits": hits,
    }


def scan_text(items: list[dict[str, str]], text: str) -> dict[str, Any]:
    if not items:
        raise MissingIndex("decontamination index is empty")
    return _report(ngram_match(items, text), items)


def scan_shard(items: list[dict[str, str]], texts: list[str]) -> dict[str, Any]:
    if not items:
        raise MissingIndex("decontamination index is empty")
    hits: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for text in texts:
        for hit in ngram_match(items, text):
            key = (hit["suite"], hit["item_id"])
            if key not in seen:
                seen.add(key)
                hits.append(hit)
    return _report(hits, items)
