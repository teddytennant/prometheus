"""Eval harness for the public suites in spec 11 (15.5 F3).

Runners produce `prometheus.eval_result` payloads. Every report carries a
harness version. The decontamination index is fail-closed: an item that
matches an eval n-gram or embedding is flagged, and a missing index is an
error, not a pass.

Stage-B efficiency numbers (tokens-to-solve, latent steps, recurrence
iterations) are metrics on the result, not a separate schema.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from evals.remaining import (  # noqa: F401
    CANARY_PREFIX,
    attach_efficiency,
    brier,
    contamination_canaries,
    efficiency_metrics,
    latent_scaling,
    leak_probe,
    log_score,
    paper_pnl,
    run_forecasting,
    scan_canaries,
)

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

_TOKEN_RE = re.compile(r"[0-9A-Za-z_]+")
_NGRAM_N = 8
_LEGAL_SPLITS = frozenset({"train", "val", "test", "held_out"})


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
    if slug not in PUBLIC_SUITES:
        raise UnknownSuiteError(slug)
    return slug


def _require_harness_version(version: str) -> str:
    if not isinstance(version, str) or version.strip() == "":
        raise HarnessVersionError("missing harness_version")
    return version


def _now_rfc3339() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _normalize(text: str) -> str:
    return " ".join(_tokenize(text))


def _ngrams(tokens: list[str], n: int = _NGRAM_N) -> set[tuple[str, ...]]:
    if not tokens:
        return set()
    width = n if len(tokens) >= n else len(tokens)
    return {tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


def _item_text(prompt: str, answer: str | None = None) -> str:
    if answer:
        return f"{prompt}\n{answer}"
    return prompt


def _decontamination_payload(decontamination: Decontamination) -> dict[str, Any]:
    status = decontamination.status
    payload: dict[str, Any] = {
        "status": status.value if isinstance(status, DecontamStatus) else str(status),
        "against": list(decontamination.against),
    }
    if decontamination.method_hash is not None:
        payload["method_hash"] = decontamination.method_hash
    return payload


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
    _require_harness_version(harness_version)
    require_suite(suite)
    if split not in _LEGAL_SPLITS:
        raise ValueError(f"split must be one of {sorted(_LEGAL_SPLITS)}")
    if isinstance(n_items, bool) or not isinstance(n_items, int) or n_items < 0:
        raise ValueError("n_items must be a non-negative int")
    payload: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "result_id": result_id,
        "suite": suite,
        "split": split,
        "checkpoint_id": checkpoint_id,
        "metrics": dict(metrics),
        "n_items": n_items,
        "created_at": created_at if created_at is not None else _now_rfc3339(),
    }
    if decontamination is not None:
        payload["decontamination"] = _decontamination_payload(decontamination)
    if per_item_hash is not None:
        payload["per_item_hash"] = per_item_hash
    if config_hash is not None:
        payload["config_hash"] = config_hash
    if job_id is not None:
        payload["job_id"] = job_id
    return payload


class DecontamIndex:
    """N-gram plus embedding match against held-out eval items.

    Fail closed: `is_clean` raises DecontamError if the index is empty or
    the method hash is unset. A match returns False (flagged), not an
    exception; the runner records `decontamination.status = flagged`.
    """

    def __init__(self) -> None:
        self._grams: set[tuple[str, ...]] = set()
        self._normalized: set[str] = set()
        self._grams_by_suite: dict[str, set[tuple[str, ...]]] = {}
        self._normalized_by_suite: dict[str, set[str]] = {}
        self._method_hash: str | None = None

    def add(self, item: EvalItem) -> None:
        text = _item_text(item.prompt, item.answer)
        tokens = _tokenize(text)
        grams = _ngrams(tokens)
        normalized = _normalize(text)
        self._grams |= grams
        self._normalized.add(normalized)
        self._grams_by_suite.setdefault(item.suite, set()).update(grams)
        self._normalized_by_suite.setdefault(item.suite, set()).add(normalized)
        self._method_hash = self._hash_contents()

    def is_clean(self, text: str, *, against: tuple[str, ...] | None = None) -> bool:
        grams, normalized = self._view(against)
        if self._method_hash is None or (not grams and not normalized):
            raise DecontamError("decontamination index is empty or method hash is unset")
        if _normalize(text) in normalized:
            return False
        return _ngrams(_tokenize(text)).isdisjoint(grams)

    def method_hash(self) -> str:
        if self._method_hash is None:
            raise DecontamError("method hash is unset")
        return self._method_hash

    def _view(
        self, against: tuple[str, ...] | None
    ) -> tuple[set[tuple[str, ...]], set[str]]:
        if against is None:
            return self._grams, self._normalized
        grams: set[tuple[str, ...]] = set()
        normalized: set[str] = set()
        for slug in against:
            grams |= self._grams_by_suite.get(slug, set())
            normalized |= self._normalized_by_suite.get(slug, set())
        return grams, normalized

    def _hash_contents(self) -> str:
        hasher = hashlib.sha256()
        for gram in sorted(self._grams):
            hasher.update(" ".join(gram).encode("utf-8"))
            hasher.update(b"\n")
        for text in sorted(self._normalized):
            hasher.update(text.encode("utf-8"))
            hasher.update(b"\n")
        return hasher.hexdigest()


def _fixture_item(suite: str, split: str, index: int) -> EvalItem:
    return EvalItem(
        item_id=f"{suite}-{split}-{index}",
        suite=suite,
        split=split,
        prompt=f"{suite} {split} in-memory fixture item {index}",
        answer=str(index),
        metadata={"fixture": True, "index": index},
    )


def _hash_items(items: list[EvalItem]) -> str:
    hasher = hashlib.sha256()
    for item in items:
        hasher.update(item.item_id.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(item.suite.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(item.split.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(item.prompt.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update((item.answer or "").encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _hash_config(config: EvalConfig) -> str:
    blob = (
        f"{config.suite}\n{config.split}\n{config.checkpoint_id}\n"
        f"{config.harness_version}\n{config.n_items}\n{config.seed}\n"
    )
    return _sha256_bytes(blob.encode("utf-8"))


class Runner:
    """One public suite. `run` yields a result envelope plus per-item hashes."""

    def __init__(self, config: EvalConfig, index: DecontamIndex | None = None) -> None:
        _require_harness_version(config.harness_version)
        require_suite(config.suite)
        if config.split not in _LEGAL_SPLITS:
            raise ValueError(f"split must be one of {sorted(_LEGAL_SPLITS)}")
        self.config = config
        self.index = index

    def load_items(self) -> list[EvalItem]:
        require_suite(self.config.suite)
        n = self.config.n_items
        if n is None:
            n = 2
        if isinstance(n, bool) or n < 0:
            raise ValueError("n_items must be a non-negative int")
        return [_fixture_item(self.config.suite, self.config.split, i) for i in range(n)]

    def run(self, predict: Any) -> dict[str, Any]:
        """`predict(item) -> str`. Returns an eval_result payload.

        Requires `config.harness_version`. Rephrased and canary probes live
        in the contamination split of the suite, not a side channel.
        """
        _require_harness_version(self.config.harness_version)
        items = self.load_items()
        n_correct = 0
        flagged = False
        for item in items:
            prediction = predict(item)
            if item.answer is not None and prediction == item.answer:
                n_correct += 1
            if self.index is not None and not self.index.is_clean(
                _item_text(item.prompt, item.answer), against=(item.suite,)
            ):
                flagged = True
        n_items = len(items)
        pass_at_1 = (n_correct / n_items) if n_items else 0.0
        if self.index is None:
            decontamination = Decontamination(status=DecontamStatus.PENDING)
        else:
            decontamination = Decontamination(
                status=DecontamStatus.FLAGGED if flagged else DecontamStatus.CLEAN,
                against=(self.config.suite,),
                method_hash=self.index.method_hash(),
            )
        return result_envelope(
            result_id=f"eval-{self.config.suite}-{self.config.checkpoint_id}",
            suite=self.config.suite,
            split=self.config.split,
            checkpoint_id=self.config.checkpoint_id,
            metrics={"pass_at_1": pass_at_1},
            n_items=n_items,
            harness_version=self.config.harness_version,
            decontamination=decontamination,
            per_item_hash=_hash_items(items),
            config_hash=_hash_config(self.config),
        )
