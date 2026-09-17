#!/usr/bin/env python3
"""Slow, obvious B2 reference: normalize / shingles / MinHash LSH / exact clustering.

Rust tests call the public crate API, not this file. This exists so an
implementer can see the intended pipeline, and so goldens can be regenerated.

Rules match tests/reference/mod.rs and data/dedup/src/lib.rs (spec 07, 15.5 B2).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ZERO_WIDTH = set("\u200b\u200c\u200d\u2060\ufeff\u180e")
U64_MAX = np.uint64(np.iinfo(np.uint64).max)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def u64_from_sha256(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little")


def strip_zero_width(text: str) -> str:
    return "".join(c for c in text if c not in ZERO_WIDTH)


def collapse_ws(text: str) -> str:
    return " ".join(text.split())


def normalize(text: str) -> str:
    return collapse_ws(strip_zero_width(text).lower())


def paragraphs(text: str) -> list[str]:
    stripped = strip_zero_width(text).lower().replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    current: list[str] = []
    for line in stripped.split("\n"):
        if line.strip() == "":
            if current:
                collapsed = collapse_ws("\n".join(current))
                if collapsed:
                    out.append(collapsed)
                current = []
        else:
            current.append(line)
    if current:
        collapsed = collapse_ws("\n".join(current))
        if collapsed:
            out.append(collapsed)
    return out


def exact_hash(text: str) -> str:
    return sha256_hex(normalize(text).encode("utf-8"))


def shingles(text: str, n: int) -> list[str]:
    if n == 0:
        return []
    words = normalize(text).split()
    if len(words) < n:
        return []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


@dataclass(frozen=True)
class DedupConfig:
    shingle_size: int
    num_hashes: int
    bands: int
    rows: int

    @staticmethod
    def standard() -> DedupConfig:
        return DedupConfig(5, 128, 32, 4)


def minhash_signature(text: str, config: DedupConfig) -> list[int]:
    unique = list(dict.fromkeys(shingles(text, config.shingle_size)))
    n = config.num_hashes
    if not unique:
        return [int(U64_MAX)] * n
    table = np.zeros((len(unique), n), dtype=np.uint64)
    for i, s in enumerate(unique):
        raw = s.encode("utf-8")
        for h in range(n):
            table[i, h] = np.uint64(u64_from_sha256(h.to_bytes(8, "little") + raw))
    return [int(x) for x in table.min(axis=0)]


def lsh_band_keys(signature: list[int], config: DedupConfig) -> list[int]:
    if config.bands == 0 or config.rows == 0:
        raise ValueError("bands and rows must be > 0")
    if config.num_hashes != config.bands * config.rows:
        raise ValueError("num_hashes must equal bands * rows")
    if len(signature) != config.num_hashes:
        raise ValueError("signature length must equal num_hashes")
    keys = []
    for b in range(config.bands):
        chunk = signature[b * config.rows : (b + 1) * config.rows]
        buf = b"".join(int(x).to_bytes(8, "little") for x in chunk)
        keys.append(u64_from_sha256(buf))
    return keys


@dataclass
class Document:
    id: str
    text: str
    content_hash: str


def _keeper_key(d: Document) -> tuple[str, str]:
    return (d.content_hash, d.id)


def _canonical(kept, dropped_exact, dropped_near, clusters):
    for c in clusters:
        c["dropped"] = sorted(c["dropped"])
    clusters = sorted(clusters, key=lambda c: c["kept"])
    return {
        "kept": sorted(kept),
        "dropped_exact": sorted(dropped_exact),
        "dropped_near": sorted(dropped_near),
        "clusters": clusters,
    }


def dedup_exact_document(docs: list[Document]) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(docs):
        groups[exact_hash(d.text)].append(i)
    kept, dropped = [], []
    for idxs in groups.values():
        idxs = sorted(idxs, key=lambda i: _keeper_key(docs[i]))
        kept.append(docs[idxs[0]].id)
        dropped.extend(docs[i].id for i in idxs[1:])
    return _canonical(kept, dropped, [], [])


def dedup_exact_paragraph(docs: list[Document]) -> dict:
    """Drop the whole document if any paragraph hash matches a kept document."""
    order = sorted(range(len(docs)), key=lambda i: _keeper_key(docs[i]))
    seen: set[str] = set()
    kept, dropped = [], []
    for i in order:
        hashes = [exact_hash(p) for p in paragraphs(docs[i].text)]
        if any(h in seen for h in hashes):
            dropped.append(docs[i].id)
        else:
            kept.append(docs[i].id)
            seen.update(hashes)
    return _canonical(kept, dropped, [], [])


def dedup_near(docs: list[Document], config: DedupConfig) -> dict:
    n = len(docs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    keys: list[list[int] | None] = []
    for d in docs:
        sh = shingles(d.text, config.shingle_size)
        if not sh:
            keys.append(None)
            continue
        sig = minhash_signature(d.text, config)
        keys.append(lsh_band_keys(sig, config))

    for b in range(config.bands):
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, k in enumerate(keys):
            if k is None:
                continue
            buckets[k[b]].append(i)
        for idxs in buckets.values():
            for x, y in zip(idxs, idxs[1:]):
                union(x, y)

    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)

    kept, dropped, clusters = [], [], []
    for members in comps.values():
        members = sorted(members, key=lambda i: _keeper_key(docs[i]))
        keep_id = docs[members[0]].id
        kept.append(keep_id)
        if len(members) > 1:
            drop_ids = [docs[i].id for i in members[1:]]
            dropped.extend(drop_ids)
            clusters.append({"kept": keep_id, "dropped": drop_ids})
    return _canonical(kept, [], dropped, clusters)


def _select(docs: list[Document], ids: Iterable[str]) -> list[Document]:
    want = set(ids)
    return [d for d in docs if d.id in want]


def dedup_corpus(docs: list[Document], config: DedupConfig) -> dict:
    r_doc = dedup_exact_document(docs)
    after_doc = _select(docs, r_doc["kept"])
    r_para = dedup_exact_paragraph(after_doc)
    after_para = _select(after_doc, r_para["kept"])
    r_near = (
        {"kept": [], "dropped_exact": [], "dropped_near": [], "clusters": []}
        if not after_para
        else dedup_near(after_para, config)
    )
    return _canonical(
        r_near["kept"],
        r_doc["dropped_exact"] + r_para["dropped_exact"],
        r_near["dropped_near"],
        r_near["clusters"],
    )


def main() -> None:
    here = Path(__file__).resolve().parent.parent / "goldens"
    raw = json.loads((here / "corpus.json").read_text())
    docs = [Document(**d) for d in raw]
    report = dedup_corpus(docs, DedupConfig.standard())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
