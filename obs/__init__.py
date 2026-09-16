"""Metrics, tracing, run registry, tripwires (spec 13, 15.5 E2, I10)."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Counter:
    name: str
    value: int = 0

    def inc(self, n: int = 1) -> None:
        self.value += n


@dataclass
class Gauge:
    name: str
    value: float = 0.0

    def set(self, v: float) -> None:
        self.value = v


@dataclass
class Histogram:
    name: str
    bounds: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, float("inf"))
    counts: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.counts) != len(self.bounds):
            self.counts = [0] * len(self.bounds)

    def observe(self, v: float, n: int = 1) -> None:
        for i, b in enumerate(self.bounds):
            if v <= b:
                self.counts[i] += n
                return
        self.counts[-1] += n


@dataclass
class Span:
    name: str
    start: float
    end: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        if self.end is None:
            return None
        return self.end - self.start


class Tracer:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    def start(self, name: str, **attrs: Any) -> Span:
        span = Span(name=name, start=time.perf_counter(), attrs=dict(attrs))
        self.spans.append(span)
        return span

    def end(self, span: Span) -> Span:
        span.end = time.perf_counter()
        return span


@dataclass
class Run:
    id: str
    config_hash: str
    git_sha: str
    metrics: dict[str, float] = field(default_factory=dict)


class RunRegistry:
    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}

    def register(
        self,
        run_id: str,
        config: dict[str, Any],
        git_sha: str = "",
        metrics: dict[str, float] | None = None,
    ) -> Run:
        raw = json.dumps(config, sort_keys=True, default=str)
        cfg_hash = hashlib.sha256(raw.encode()).hexdigest()
        run = Run(id=run_id, config_hash=cfg_hash, git_sha=git_sha, metrics=dict(metrics or {}))
        self.runs[run_id] = run
        return run

    def snapshot(self, run_id: str, metrics: dict[str, float]) -> Run:
        run = self.runs[run_id]
        run.metrics = {k: float(v) for k, v in metrics.items()}
        return run


@dataclass
class Registry:
    counters: dict[str, Counter] = field(default_factory=dict)
    gauges: dict[str, Gauge] = field(default_factory=dict)
    histograms: dict[str, Histogram] = field(default_factory=dict)

    def counter(self, name: str) -> Counter:
        return self.counters.setdefault(name, Counter(name))

    def gauge(self, name: str) -> Gauge:
        return self.gauges.setdefault(name, Gauge(name))

    def histogram(self, name: str) -> Histogram:
        return self.histograms.setdefault(name, Histogram(name))

    def snapshot(self) -> dict:
        return {
            "counters": {k: c.value for k, c in self.counters.items()},
            "gauges": {k: g.value for k, g in self.gauges.items()},
            "histograms": {
                k: {
                    "bounds": [None if not math.isfinite(b) else b for b in h.bounds],
                    "counts": list(h.counts),
                }
                for k, h in self.histograms.items()
            },
        }

    def dashboard_payload(
        self, tracer: Tracer | None = None, runs: RunRegistry | None = None
    ) -> dict:
        return dashboard_payload(self, tracer, runs)


def tripwire(name: str, value: float, limit: float) -> dict | None:
    if value > limit:
        return {"name": name, "value": value, "limit": limit}
    return None


DEFAULT_LIMITS = {
    "tokens": 1_000_000.0,
    "loss_spike": 2.0,
    "kl_drift": 0.2,
}


def check_tripwires(
    metrics: dict[str, float],
    limits: dict[str, float] | None = None,
) -> list[dict]:
    lim = {**DEFAULT_LIMITS, **(limits or {})}
    hits: list[dict] = []
    for name, limit in lim.items():
        if name not in metrics:
            continue
        hit = tripwire(name, float(metrics[name]), float(limit))
        if hit is not None:
            hits.append(hit)
    return hits


def dashboard_payload(
    registry: Registry | None = None,
    tracer: Tracer | None = None,
    runs: RunRegistry | None = None,
) -> dict:
    payload: dict[str, Any] = {
        "counters": {},
        "gauges": {},
        "histograms": {},
        "spans": [],
        "runs": [],
    }
    if registry is not None:
        snap = registry.snapshot()
        payload["counters"] = snap["counters"]
        payload["gauges"] = snap["gauges"]
        payload["histograms"] = snap["histograms"]
    if tracer is not None:
        payload["spans"] = [
            {
                "name": s.name,
                "start": s.start,
                "end": s.end,
                "duration": s.duration,
                "attrs": s.attrs,
            }
            for s in tracer.spans
        ]
    if runs is not None:
        payload["runs"] = [
            {
                "id": r.id,
                "config_hash": r.config_hash,
                "git_sha": r.git_sha,
                "metrics": r.metrics,
            }
            for r in runs.runs.values()
        ]
    json.dumps(payload)
    return payload
