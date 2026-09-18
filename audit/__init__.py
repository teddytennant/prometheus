"""Discrete-mode comparison and thought decoding tools (spec 12, 4.3, 15.5 I9).

CPU analog of the I9 gate. Every checkpoint can be forced into discrete
CoT. When latent and discrete answers diverge on the same problem, that
is the first place to look. The thought-decode head reads latent chunks
back out as steps (spec 4.3: ~8 tokens per thought).

Does not run a rung, import ``tests``, talk to a coordinator, or need a
GPU. Arrays are numpy. Does not persist a WORM log (that is 14.10 / L3,
not I9).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

Array = Any  # numpy.ndarray; FP32 logits


class AuditError(ValueError):
    """A checkpoint, answer pair, or decode tensor violates spec 12 / 4.3."""


# spec 4.3: compression target ~8 tokens per thought. Same number as I5.
TOKENS_PER_THOUGHT = 8


class Mode(StrEnum):
    """Serving mode for one checkpoint. Discrete CoT is always available."""

    DISCRETE = "discrete"
    LATENT = "latent"


class BugKind(StrEnum):
    """Planted latent bugs the I9 gate has to find."""

    ANSWER_SWAP = "answer_swap"
    DECODE_FLIP = "decode_flip"
    HALT_UNPINNED = "halt_unpinned"


@dataclass(frozen=True)
class CheckpointMode:
    """A checkpoint forced into a serving mode (spec 12)."""

    checkpoint_id: str
    mode: Mode


@dataclass(frozen=True)
class AnswerPair:
    """Discrete and latent answers on the same problem.

    ``gold`` is optional. When set, ``compare_answers`` records which
    side matches it. Divergence is ``discrete_answer != latent_answer``
    even if gold is missing.
    """

    problem_id: str
    discrete_answer: str
    latent_answer: str
    gold: str | None = None


@dataclass(frozen=True)
class Divergence:
    """One problem where the two modes disagree, or a comparison result.

    ``diverge`` is True iff the two answers differ. That is the first
    place to look (spec 12). ``discrete_correct`` / ``latent_correct``
    are None when gold is missing.
    """

    problem_id: str
    discrete_answer: str
    latent_answer: str
    gold: str | None
    discrete_correct: bool | None
    latent_correct: bool | None
    diverge: bool


@dataclass(frozen=True)
class ThoughtStep:
    """One thought decoded back to a window of tokens (spec 4.3)."""

    index: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class PlantedBug:
    """I9 gate result: a planted latent bug was or was not found."""

    kind: BugKind
    found: bool
    evidence: str


def force_discrete(checkpoint_id: str) -> CheckpointMode:
    """Force any checkpoint into discrete CoT (spec 12).

    Empty ``checkpoint_id`` raises ``AuditError``. The returned mode is
    always ``Mode.DISCRETE``.
    """
    if not isinstance(checkpoint_id, str) or checkpoint_id.strip() == "":
        raise AuditError("empty or whitespace-only checkpoint_id")
    return CheckpointMode(checkpoint_id=checkpoint_id, mode=Mode.DISCRETE)


def compare_answers(pair: AnswerPair) -> Divergence:
    """Compare latent vs discrete answers on one problem (spec 12).

    Empty ``problem_id`` raises ``AuditError``. Answers may be empty
    strings (that is a real, wrong answer). Divergence does not require
    gold.
    """
    if not isinstance(pair.problem_id, str) or pair.problem_id == "":
        raise AuditError("empty problem_id")
    diverge = pair.discrete_answer != pair.latent_answer
    if pair.gold is None:
        discrete_correct: bool | None = None
        latent_correct: bool | None = None
    else:
        discrete_correct = pair.discrete_answer == pair.gold
        latent_correct = pair.latent_answer == pair.gold
    return Divergence(
        problem_id=pair.problem_id,
        discrete_answer=pair.discrete_answer,
        latent_answer=pair.latent_answer,
        gold=pair.gold,
        discrete_correct=discrete_correct,
        latent_correct=latent_correct,
        diverge=diverge,
    )


def compare_suite(pairs: tuple[AnswerPair, ...]) -> tuple[Divergence, ...]:
    """Compare a suite. Returns one ``Divergence`` per pair, in order.

    Empty suite raises ``AuditError``. A pair that does not diverge is
    still returned (``diverge=False``); callers filter.
    """
    if not isinstance(pairs, tuple) or len(pairs) < 1:
        raise AuditError("empty pairs")
    return tuple(compare_answers(pair) for pair in pairs)


def decode_thoughts(logits: Array) -> tuple[ThoughtStep, ...]:
    """Greedy-decode latent chunks back to steps (spec 4.3, 12).

    ``logits`` is rank 3, shape ``(n_thoughts, TOKENS_PER_THOUGHT, vocab)``.
    ``n_thoughts >= 1``, window must equal ``TOKENS_PER_THOUGHT``,
    ``vocab >= 1``. Each step's ``token_ids`` is the argmax over vocab
    at that thought and token. Ties take the smallest id.

    Empty, wrong rank, wrong window, or non-finite logits raise
    ``AuditError``.
    """
    if not isinstance(logits, np.ndarray):
        raise AuditError("logits must be a numpy array")
    if logits.ndim != 3:
        raise AuditError(f"logits rank must be 3, got {logits.ndim}")
    n_thoughts, window, vocab = (int(dim) for dim in logits.shape)
    if n_thoughts < 1:
        raise AuditError("n_thoughts must be >= 1")
    if window != TOKENS_PER_THOUGHT:
        raise AuditError(f"thought window must be {TOKENS_PER_THOUGHT}, got {window}")
    if vocab < 1:
        raise AuditError("vocab must be >= 1")
    if not np.all(np.isfinite(logits)):
        raise AuditError("logits must be finite")
    # np.argmax picks the first max, which is the smallest id on a tie.
    token_ids = np.argmax(logits, axis=-1)
    return tuple(
        ThoughtStep(index=i, token_ids=tuple(int(t) for t in token_ids[i]))
        for i in range(n_thoughts)
    )


def find_planted_bug(
    kind: BugKind,
    pair: AnswerPair | None,
    logits: Array | None,
    halt_lambdas: Array | None,
) -> PlantedBug:
    """I9 gate: report whether a planted latent bug is visible.

    Check order (only the requested ``kind`` is inspected):

    1. ``ANSWER_SWAP``: ``pair`` required. Found when discrete matches
       gold and latent does not, or the two answers diverge. Missing
       pair raises ``AuditError``.
    2. ``DECODE_FLIP``: ``logits`` required. Found when greedy decode
       of ``logits`` disagrees with greedy decode of ``-logits`` (a
       planted sign flip of the thought-decode head). Missing logits
       raises ``AuditError``.
    3. ``HALT_UNPINNED``: ``halt_lambdas`` required, 1-D length >= 1.
       Found when the last slot is not 1.0 (I5 pins it). Missing or
       non-1-D raises ``AuditError``.

    Does not import ``model.latent``. The halt check is the pin
    invariant, not a second PonderNet.
    """
    if kind == BugKind.ANSWER_SWAP:
        if pair is None:
            raise AuditError("pair required for ANSWER_SWAP")
        div = compare_answers(pair)
        found = bool(div.diverge)
        if found:
            evidence = f"answers diverge on {div.problem_id}"
        else:
            evidence = f"no answer_swap on {div.problem_id}"
        return PlantedBug(kind=BugKind.ANSWER_SWAP, found=found, evidence=evidence)

    if kind == BugKind.DECODE_FLIP:
        if logits is None:
            raise AuditError("logits required for DECODE_FLIP")
        pos = decode_thoughts(logits)
        neg = decode_thoughts(-np.asarray(logits))
        found = any(a.token_ids != b.token_ids for a, b in zip(pos, neg, strict=True))
        if found:
            evidence = "greedy decode of logits disagrees with greedy decode of -logits"
        else:
            evidence = "greedy decode of logits matches greedy decode of -logits"
        return PlantedBug(kind=BugKind.DECODE_FLIP, found=found, evidence=evidence)

    if kind == BugKind.HALT_UNPINNED:
        if halt_lambdas is None:
            raise AuditError("halt_lambdas required for HALT_UNPINNED")
        if not isinstance(halt_lambdas, np.ndarray):
            raise AuditError("halt_lambdas must be a numpy array")
        if halt_lambdas.ndim != 1:
            raise AuditError("halt_lambdas must be 1-D")
        if halt_lambdas.size < 1:
            raise AuditError("halt_lambdas empty")
        last_f = float(halt_lambdas[-1])
        found = (not math.isfinite(last_f)) or (last_f != 1.0)
        if found:
            evidence = f"last halt lambda is {last_f!r}, not pinned to 1.0"
        else:
            evidence = "last halt lambda is pinned to 1.0"
        return PlantedBug(kind=BugKind.HALT_UNPINNED, found=found, evidence=evidence)

    raise AuditError(f"unknown bug kind: {kind!r}")
