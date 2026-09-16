"""Metrics registry and tripwires (spec 13, 15.5 E2)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Counter:
    name: str
    value: float = 0.0

    def inc(self, n: float = 1.0) -> None:
        self.value += n


@dataclass
class Gauge:
    name: str
    value: float = 0.0

    def set(self, v: float) -> None:
        self.value = v


@dataclass
class Registry:
    counters: dict[str, Counter] = field(default_factory=dict)
    gauges: dict[str, Gauge] = field(default_factory=dict)

    def counter(self, name: str) -> Counter:
        return self.counters.setdefault(name, Counter(name))

    def gauge(self, name: str) -> Gauge:
        return self.gauges.setdefault(name, Gauge(name))

    def snapshot(self) -> dict:
        return {
            "counters": {k: c.value for k, c in self.counters.items()},
            "gauges": {k: g.value for k, g in self.gauges.items()},
        }


def tripwire(name: str, value: float, limit: float) -> dict | None:
    if value > limit:
        return {"name": name, "value": value, "limit": limit}
    return None
