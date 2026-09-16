"""Eval harness for the public suites in spec 11 (15.5 F3).

Runners produce `prometheus.eval_result` payloads. Every report carries a
harness version. The decontamination index is fail-closed: an item that
matches an eval n-gram or embedding is flagged, and a missing index is an
error, not a pass.

Stage-B efficiency numbers (tokens-to-solve, latent steps, recurrence
iterations) are metrics on the result, not a separate schema.

Nothing here runs. Suite names and the result envelope are real; runners
and the index raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

SCHEMA_ID = "prometheus.eval_result"
SCHEMA_VERSION = 1

# spec 11 public suites. Slugs are lowercase with underscores.
PUBLIC_SUITES: tuple[str, ...] = (
    "re_bench",
    "mle_bench",
    "paperbench",
    "speedrun",
    "metr_time_horizon",
    "swe_bench_verified",
    "swe_bench_pro",
    "terminal_bench",
    "competitive_programming",
    "aime",
    "hmmt",
    "frontiermath",
    "minif2f",
    "putnambench",
    "arc_agi_1",
    "arc_agi_2",
    "arc_agi_3",
    "hle",
    "gpqa",
    "long_context_128k",
    "long_context_1m",
    "forecasting",
)

FORECASTING_METRICS: tuple[str, ...] = (
    "brier",
    "log_score",
    "paper_pnl",
    "leak_probe",
)

EFFICIENCY_METRICS: tuple[str, ...] = (
    "tokens_to_solve",
    "latent_steps",
    "recurrence_iterations",
)


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    HELD_OUT = "held_out"


class DecontamStatus(StrEnum):
    PENDING = "pending"
    CLEAN = "clean"
    FLAGGED = "flagged"
    NOT_APPLICABLE = "not_applicable"


class UnknownSuiteError(KeyError):
    """Suite slug is not in PUBLIC_SUITES."""


class HarnessVersionError(ValueError):
    """Report is missing a harness version, or versions drifted mid-run."""


class DecontamError(ValueError):
    """Index missing, lookup failed, or fail-closed contamination hit."""


@dataclass(frozen=True)
class EvalItem:
    item_id: str
    suite: str
    split: str
    prompt: str
    answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalConfig:
    suite: str
    split: str
    checkpoint_id: str
    harness_version: str
    n_items: int | None = None
    seed: int = 0


@dataclass(frozen=True)
class Decontamination:
    status: DecontamStatus
    against: tuple[str, ...] = ()
    method_hash: str | None = None


def require_suite(slug: str) -> str:
    """Return slug if it is a public suite; else UnknownSuiteError."""
    raise NotImplementedError("F3 require_suite")


def result_envelope(
    *,
    result_id: str,
    suite: str,
    split: str,
    checkpoint_id: str,
    metrics: dict[str, float],
    n_items: int,
    harness_version: str,
    decontamination: Decontamination | None = None,
    per_item_hash: str | None = None,
    config_hash: str | None = None,
    job_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a `prometheus.eval_result` v1 payload. Does not validate against F1."""
    raise NotImplementedError("F3 result_envelope")


class DecontamIndex:
    """N-gram plus embedding match against held-out eval items.

    Fail closed: `is_clean` raises DecontamError if the index is empty or
    the method hash is unset. A match returns False (flagged), not an
    exception; the runner records `decontamination.status = flagged`.
    """

    def add(self, item: EvalItem) -> None:
        raise NotImplementedError("F3 DecontamIndex.add")

    def is_clean(self, text: str, *, against: tuple[str, ...] | None = None) -> bool:
        raise NotImplementedError("F3 DecontamIndex.is_clean")

    def method_hash(self) -> str:
        raise NotImplementedError("F3 DecontamIndex.method_hash")


class Runner:
    """One public suite. `run` yields a result envelope plus per-item hashes."""

    def __init__(self, config: EvalConfig, index: DecontamIndex | None = None) -> None:
        raise NotImplementedError("F3 Runner.__init__")

    def load_items(self) -> list[EvalItem]:
        raise NotImplementedError("F3 Runner.load_items")

    def run(self, predict: Any) -> dict[str, Any]:
        """`predict(item) -> str`. Returns an eval_result payload.

        Requires `config.harness_version`. Rephrased and canary probes live
        in the contamination split of the suite, not a side channel.
        """
        raise NotImplementedError("F3 Runner.run")
