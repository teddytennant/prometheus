"""Slow, obvious word n-gram overlap for eval decontamination (spec 7.2, 11).

This is documentation for implementers, not an oracle. ``tests/test_evals.py``
calls ``evals.DecontamIndex`` for every assertion the implementation must pass.

Matching rule (deliberately simple):

- Tokenize on runs of ``[0-9A-Za-z_]``, lowercase.
- Index overlapping word 8-grams. Sequences shorter than 8 tokens contribute
  their full token tuple so a short canary still matches a verbatim copy.
- A candidate is dirty iff it shares at least one n-gram with any indexed item
  prompt (and answer, when present), or the normalized strings are identical.

Embeddings are out of scope here; the harness tests only require n-gram /
near-copy overlap to be flagged and unrelated text to stay clean.
"""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[0-9A-Za-z_]+")
NGRAM_N = 8


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def normalize(text: str) -> str:
    return " ".join(tokenize(text))


def ngrams(tokens: list[str], n: int = NGRAM_N) -> set[tuple[str, ...]]:
    if not tokens:
        return set()
    width = n if len(tokens) >= n else len(tokens)
    return {tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


def item_text(prompt: str, answer: str | None = None) -> str:
    if answer:
        return f"{prompt}\n{answer}"
    return prompt


class NgramDecontamIndex:
    """In-memory n-gram set. Fail-closed: empty index cannot answer is_clean."""

    def __init__(self) -> None:
        self._grams: set[tuple[str, ...]] = set()
        self._normalized: set[str] = set()

    def add(self, prompt: str, answer: str | None = None) -> None:
        text = item_text(prompt, answer)
        tokens = tokenize(text)
        self._grams |= ngrams(tokens)
        self._normalized.add(normalize(text))

    def is_clean(self, text: str) -> bool:
        if not self._grams and not self._normalized:
            raise RuntimeError("empty index: fail closed")
        tokens = tokenize(text)
        if normalize(text) in self._normalized:
            return False
        return ngrams(tokens).isdisjoint(self._grams)
