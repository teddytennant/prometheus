"""Byte-level BPE with ARC color tokens and freezeable vocab (spec 3.1, 7)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

COLOR_BASE = 256
N_COLORS = 10
SPECIAL_NAMES = ("<pad>", "<unk>", "<bos>", "<eos>", "<arc_row>")
SPECIAL_BASE = COLOR_BASE + N_COLORS
SPECIALS = {name: SPECIAL_BASE + i for i, name in enumerate(SPECIAL_NAMES)}
MERGE_BASE = SPECIAL_BASE + len(SPECIAL_NAMES)
ARC_ROW_ID = SPECIALS["<arc_row>"]


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


def _expand(i: int, inv: dict[int, tuple[int, int]]) -> list[int]:
    if i < 256:
        return [i]
    if i not in inv:
        return []
    a, b = inv[i]
    return _expand(a, inv) + _expand(b, inv)


@dataclass
class Tokenizer:
    merges: list[tuple[int, int]]
    vocab_size: int
    _frozen: bool = field(default=False, repr=False, compare=False)

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        next_id = MERGE_BASE
        for pair in self.merges:
            ids = _apply(ids, pair, next_id)
            next_id += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        inv: dict[int, tuple[int, int]] = {}
        next_id = MERGE_BASE
        for a, b in self.merges:
            inv[next_id] = (a, b)
            next_id += 1
        out: list[int] = []
        for i in ids:
            out.extend(_expand(i, inv))
        return bytes(out).decode("utf-8", errors="replace")

    @property
    def color_ids(self) -> dict[int, int]:
        return {c: COLOR_BASE + c for c in range(N_COLORS)}

    @property
    def specials(self) -> dict[str, int]:
        return dict(SPECIALS)

    def encode_grid(self, grid: list[list[int]]) -> list[int]:
        ids: list[int] = []
        for r, row in enumerate(grid):
            if r:
                ids.append(ARC_ROW_ID)
            for cell in row:
                c = int(cell)
                if c < 0 or c >= N_COLORS:
                    raise ValueError(f"ARC color must be 0-9, got {c}")
                ids.append(COLOR_BASE + c)
        return ids

    def decode_grid(self, ids: list[int]) -> list[list[int]]:
        rows: list[list[int]] = []
        cur: list[int] = []
        for i in ids:
            if i == ARC_ROW_ID:
                rows.append(cur)
                cur = []
            elif COLOR_BASE <= i < COLOR_BASE + N_COLORS:
                cur.append(i - COLOR_BASE)
        rows.append(cur)
        return rows

    def freeze(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "merges": [list(p) for p in self.merges],
            "color_ids": {str(c): COLOR_BASE + c for c in range(N_COLORS)},
            "specials": dict(SPECIALS),
            "merge_base": MERGE_BASE,
            "vocab_size": self.vocab_size,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        self._frozen = True
        return path

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(p) for p in data["merges"]]
        tok = cls(merges=merges, vocab_size=int(data["vocab_size"]))
        tok._frozen = True
        return tok


def train(texts: list[str], n_merges: int = 50) -> Tokenizer:
    corpus = [list(t.encode("utf-8")) for t in texts]
    merges: list[tuple[int, int]] = []
    next_id = MERGE_BASE
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
    return Tokenizer(merges=merges, vocab_size=MERGE_BASE + len(merges))
