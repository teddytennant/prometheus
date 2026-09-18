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

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

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
    raise NotImplementedError("I9: force_discrete")


def compare_answers(pair: AnswerPair) -> Divergence:
    """Compare latent vs discrete answers on one problem (spec 12).

    Empty ``problem_id`` raises ``AuditError``. Answers may be empty
    strings (that is a real, wrong answer). Divergence does not require
    gold.
    """
    raise NotImplementedError("I9: compare_answers")


def compare_suite(pairs: tuple[AnswerPair, ...]) -> tuple[Divergence, ...]:
    """Compare a suite. Returns one ``Divergence`` per pair, in order.

    Empty suite raises ``AuditError``. A pair that does not diverge is
    still returned (``diverge=False``); callers filter.
    """
    raise NotImplementedError("I9: compare_suite")


def decode_thoughts(logits: Array) -> tuple[ThoughtStep, ...]:
    """Greedy-decode latent chunks back to steps (spec 4.3, 12).

    ``logits`` is rank 3, shape ``(n_thoughts, TOKENS_PER_THOUGHT, vocab)``.
    ``n_thoughts >= 1``, window must equal ``TOKENS_PER_THOUGHT``,
    ``vocab >= 1``. Each step's ``token_ids`` is the argmax over vocab
    at that thought and token. Ties take the smallest id.

    Empty, wrong rank, wrong window, or non-finite logits raise
    ``AuditError``.
    """
    raise NotImplementedError("I9: decode_thoughts")


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
    raise NotImplementedError("I9: find_planted_bug")
