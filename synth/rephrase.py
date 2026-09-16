"""Deterministic style rewrites with n-gram fact-check (spec 7.1 E1)."""

from __future__ import annotations

import random
import re


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def ngram_coverage(source: str, variant: str, n: int = 3) -> float:
    src = _tokens(source)
    var = _tokens(variant)
    if len(src) < n:
        return 1.0 if src == var or set(src) <= set(var) else 0.0
    s = {tuple(src[i : i + n]) for i in range(len(src) - n + 1)}
    v = {tuple(var[i : i + n]) for i in range(max(0, len(var) - n + 1))}
    return len(s & v) / len(s)


def _styles(source: str) -> list[str]:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source.strip()) if s.strip()]
    body = " ".join(sents) if sents else source.strip()
    return [
        body,
        "The document states: " + body,
        "In other words, " + body,
        " ".join(sents[::-1]) if len(sents) > 1 else body,
        body.replace(" is ", " equals ").replace(" are ", " equal "),
        "Summary. " + body,
    ]


def rephrase(source: str, n: int = 4, seed: int = 0, *, min_coverage: float = 0.5) -> list[str]:
    """Several styles of `source`; drop variants whose 3-gram coverage drifts."""
    rng = random.Random(seed)
    cands = list(dict.fromkeys(_styles(source)))
    rng.shuffle(cands)
    kept: list[str] = []
    for c in cands:
        if c == source.strip() and kept:
            continue
        if ngram_coverage(source, c) >= min_coverage:
            kept.append(c)
        if len(kept) >= n:
            break
    return kept
