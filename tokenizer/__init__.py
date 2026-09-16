"""Byte-level BPE tokenizer (spec 7, 15.5 B4). Vocab 256 bytes + merges."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class Tokenizer:
    merges: list[tuple[int, int]]
    vocab_size: int

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        next_id = 256
        for pair in self.merges:
            ids = _apply(ids, pair, next_id)
            next_id += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        inv = {}
        next_id = 256
        for a, b in self.merges:
            inv[next_id] = (a, b)
            next_id += 1
        out: list[int] = []
        for i in ids:
            out.extend(_expand(i, inv))
        return bytes(out).decode("utf-8", errors="replace")


def _expand(i: int, inv: dict[int, tuple[int, int]]) -> list[int]:
    if i < 256:
        return [i]
    a, b = inv[i]
    return _expand(a, inv) + _expand(b, inv)


def train(texts: list[str], n_merges: int) -> Tokenizer:
    corpus = [list(t.encode("utf-8")) for t in texts]
    merges: list[tuple[int, int]] = []
    next_id = 256
    for _ in range(n_merges):
        counts: Counter[tuple[int, int]] = Counter()
        for seq in corpus:
            counts.update(zip(seq, seq[1:]))
        if not counts:
            break
        pair, _ = counts.most_common(1)[0]
        merges.append(pair)
        corpus = [_apply(seq, pair, next_id) for seq in corpus]
        next_id += 1
    return Tokenizer(merges=merges, vocab_size=256 + len(merges))


def _apply(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    a, b = pair
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out
