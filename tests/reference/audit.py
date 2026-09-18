"""Independent I9 reference: discrete-mode audit and thought decoding.

Plain NumPy, slow and obvious. Does not import production ``audit``,
``model.latent``, ``rl.loss``, or anything under ``tests/``. Enums, dataclasses,
and constants are a local mirror so tests can compare field-by-field without
sharing code. Production must never import ``tests/``.

Pinned rules
------------
AuditError
    Subclass of ValueError.

TOKENS_PER_THOUGHT
    8 (spec 4.3 thought-decode head).

force_discrete
    Empty or whitespace-only checkpoint_id -> AuditError. Otherwise
    CheckpointMode(checkpoint_id=that string, mode=DISCRETE). The stored id is
    not stripped.

compare_answers
    Empty problem_id -> AuditError. Answers may be empty strings.
    diverge = (discrete_answer != latent_answer).
    gold is None -> both *_correct are None.
    gold is set (including "") -> *_correct = (that answer == gold).

compare_suite
    Empty tuple -> AuditError. One Divergence per pair, in order, including
    non-diverging rows.

decode_thoughts
    logits must be a rank-3 numpy array of shape (n_thoughts, 8, vocab) with
    n_thoughts >= 1, vocab >= 1, all finite. Empty / wrong rank / wrong window
    / vocab < 1 / non-finite -> AuditError.
    Greedy token_ids = argmax over the last axis; ties take the smallest id.
    ThoughtStep.index runs 0..n-1. Each token_ids has length 8.

find_planted_bug
    Inspects ONLY the requested kind. Other arguments may be None / garbage.

    ANSWER_SWAP: pair required. Found when (discrete matches gold AND latent
    does not) OR the two answers diverge. gold is None -> found iff diverge.

    DECODE_FLIP: logits required; same shape/finite rules as decode_thoughts.
    Found when greedy decode of logits disagrees with greedy decode of
    -logits (planted sign flip of the thought-decode head). All-zero logits
    do not flip (-0 == 0).

    HALT_UNPINNED: halt_lambdas required, 1-D, length >= 1. Found when the
    last slot is not exactly 1.0 (I5 pins it). Non-finite last slot counts as
    unpinned. Does not import model.latent; this is a pin check, not PonderNet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

Array = Any  # numpy.ndarray; FP32 logits


class AuditError(ValueError):
    """Raised when an audit input is empty, malformed, or non-finite."""


class Mode(StrEnum):
    LATENT = "latent"
    DISCRETE = "discrete"


class BugKind(StrEnum):
    ANSWER_SWAP = "answer_swap"
    DECODE_FLIP = "decode_flip"
    HALT_UNPINNED = "halt_unpinned"


TOKENS_PER_THOUGHT = 8


@dataclass(frozen=True)
class CheckpointMode:
    checkpoint_id: str
    mode: Mode


@dataclass(frozen=True)
class AnswerPair:
    problem_id: str
    discrete_answer: str
    latent_answer: str
    gold: str | None = None


@dataclass(frozen=True)
class Divergence:
    problem_id: str
    discrete_answer: str
    latent_answer: str
    gold: str | None
    diverge: bool
    discrete_correct: bool | None
    latent_correct: bool | None


@dataclass(frozen=True)
class ThoughtStep:
    index: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class PlantedBug:
    kind: BugKind
    found: bool
    evidence: str


def force_discrete(checkpoint_id: str) -> CheckpointMode:
    if not isinstance(checkpoint_id, str) or checkpoint_id.strip() == "":
        raise AuditError("empty or whitespace-only checkpoint_id")
    return CheckpointMode(checkpoint_id=checkpoint_id, mode=Mode.DISCRETE)


def compare_answers(pair: AnswerPair) -> Divergence:
    problem_id = pair.problem_id
    if not isinstance(problem_id, str) or problem_id == "":
        raise AuditError("empty problem_id")
    discrete_answer = pair.discrete_answer
    latent_answer = pair.latent_answer
    gold = pair.gold
    diverge = discrete_answer != latent_answer
    if gold is None:
        discrete_correct: bool | None = None
        latent_correct: bool | None = None
    else:
        discrete_correct = discrete_answer == gold
        latent_correct = latent_answer == gold
    return Divergence(
        problem_id=problem_id,
        discrete_answer=discrete_answer,
        latent_answer=latent_answer,
        gold=gold,
        diverge=diverge,
        discrete_correct=discrete_correct,
        latent_correct=latent_correct,
    )


def compare_suite(pairs: tuple[AnswerPair, ...]) -> tuple[Divergence, ...]:
    if not isinstance(pairs, tuple) or len(pairs) < 1:
        raise AuditError("empty pairs")
    return tuple(compare_answers(pair) for pair in pairs)


def _require_thought_logits(logits: Array) -> np.ndarray:
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
    return logits


def _greedy_token_ids(logits: np.ndarray) -> list[tuple[int, ...]]:
    """Argmax over vocab; ties keep the smallest id (strict greater)."""
    n_thoughts, window, vocab = (int(dim) for dim in logits.shape)
    out: list[tuple[int, ...]] = []
    for thought in range(n_thoughts):
        ids: list[int] = []
        for pos in range(window):
            best_id = 0
            best_val = logits[thought, pos, 0]
            for token_id in range(1, vocab):
                val = logits[thought, pos, token_id]
                if val > best_val:
                    best_val = val
                    best_id = token_id
            ids.append(int(best_id))
        out.append(tuple(ids))
    return out


def decode_thoughts(logits: Array) -> tuple[ThoughtStep, ...]:
    arr = _require_thought_logits(logits)
    token_rows = _greedy_token_ids(arr)
    return tuple(
        ThoughtStep(index=index, token_ids=token_ids)
        for index, token_ids in enumerate(token_rows)
    )


def find_planted_bug(
    kind: BugKind,
    pair: AnswerPair | None,
    logits: Array | None,
    halt_lambdas: Array | None,
) -> PlantedBug:
    if kind == BugKind.ANSWER_SWAP:
        return _answer_swap(pair)
    if kind == BugKind.DECODE_FLIP:
        return _decode_flip(logits)
    if kind == BugKind.HALT_UNPINNED:
        return _halt_unpinned(halt_lambdas)
    raise AuditError(f"unknown bug kind: {kind!r}")


def _answer_swap(pair: AnswerPair | None) -> PlantedBug:
    if pair is None:
        raise AuditError("pair required for ANSWER_SWAP")
    div = compare_answers(pair)
    discrete_only = div.discrete_correct is True and div.latent_correct is False
    if div.gold is None:
        found = bool(div.diverge)
    else:
        found = discrete_only or bool(div.diverge)
    if discrete_only:
        evidence = (
            f"discrete matches gold, latent does not on {div.problem_id}"
        )
    elif found:
        evidence = f"answers diverge on {div.problem_id}"
    else:
        evidence = f"no answer_swap on {div.problem_id}"
    return PlantedBug(kind=BugKind.ANSWER_SWAP, found=found, evidence=evidence)


def _decode_flip(logits: Array | None) -> PlantedBug:
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


def _halt_unpinned(halt_lambdas: Array | None) -> PlantedBug:
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
