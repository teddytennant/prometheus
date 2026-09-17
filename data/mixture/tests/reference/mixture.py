#!/usr/bin/env python3
"""Slow, obvious B6 mixture reference (spec 7.1, 6, 15.5 B6).

Rust tests call this file via tests/reference/mod.rs. Production
`src/lib.rs` must stay `unimplemented!` except `flagship_catalog` and
`Source::all`.

Weight vectors are 1-d float64 lists (NumPy-shaped). Decay/drop
renormalize by a scalar sum; sampling is `cumsum` + `searchsorted`
(side="right") over BTreeMap / table order.

Canonical JSON matches serde_json's default for `Mix`:
  * struct fields in declaration order: mix_id, mix_bucket, phase, weights
  * compact (no extra whitespace)
  * `weights` is a BTreeMap object: source keys in Source Ord / 7.1 table
    order, serde `snake_case` names, not alphabetical
  * mix_hash is lowercase hex SHA-256 of those bytes (F1 `sha256`)
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from typing import Any

# Spec 7.1 table order == `Source` discriminant / BTreeMap order.
SOURCES: list[str] = [
    "web",
    "code",
    "math_science_arxiv",
    "books_papers",
    "synthetic_rewrites",
    "synthetic_reasoning",
    "procedural_arc",
    "agentic_trajectories",
]
SOURCE_SET = set(SOURCES)

# Spec 7.1 decay: code, math and reasoning.
DECAY_SOURCES: tuple[str, ...] = (
    "code",
    "math_science_arxiv",
    "synthetic_reasoning",
)

WEIGHT_SUM_TOL = 1e-9

PHASES = ("pretrain", "decay")


class MixError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


def flagship_catalog() -> list[dict[str, Any]]:
    """7.1 unique-token / epoch bounds. Books/papers unique stays None."""
    return [
        {"source": "web", "unique_tokens": 20_000_000_000_000, "epochs_min": 2, "epochs_max": 3},
        {"source": "code", "unique_tokens": 4_000_000_000_000, "epochs_min": 4, "epochs_max": 4},
        {
            "source": "math_science_arxiv",
            "unique_tokens": 1_500_000_000_000,
            "epochs_min": 4,
            "epochs_max": 4,
        },
        {"source": "books_papers", "unique_tokens": None, "epochs_min": 2, "epochs_max": 4},
        {
            "source": "synthetic_rewrites",
            "unique_tokens": 40_000_000_000_000,
            "epochs_min": 1,
            "epochs_max": 1,
        },
        {
            "source": "synthetic_reasoning",
            "unique_tokens": 10_000_000_000_000,
            "epochs_min": 1,
            "epochs_max": 1,
        },
        {
            "source": "procedural_arc",
            "unique_tokens": 2_000_000_000_000,
            "epochs_min": 1,
            "epochs_max": 1,
        },
        {
            "source": "agentic_trajectories",
            "unique_tokens": 1_000_000_000_000,
            "epochs_min": 1,
            "epochs_max": 1,
        },
    ]


def validate_mix(mix: dict[str, Any]) -> None:
    """Empty id/bucket, non-finite or <=0 weights, empty map, sum off 1."""
    mix_id = mix.get("mix_id")
    mix_bucket = mix.get("mix_bucket")
    if not isinstance(mix_id, str) or mix_id == "":
        raise MixError("config", "empty mix_id")
    if not isinstance(mix_bucket, str) or mix_bucket == "":
        raise MixError("config", "empty mix_bucket")
    phase = mix.get("phase")
    if phase not in PHASES:
        raise MixError("config", "invalid phase")
    weights = mix.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise MixError("config", "empty weights")
    total = 0.0
    for key, raw in weights.items():
        if key not in SOURCE_SET:
            raise MixError("unknown_source", str(key))
        try:
            w = float(raw)
        except (TypeError, ValueError) as exc:
            raise MixError("config", f"non-finite or non-positive weight {key}") from exc
        if not math.isfinite(w) or w <= 0.0:
            raise MixError("config", f"non-finite or non-positive weight {key}")
        total += w
    if not math.isfinite(total) or abs(total - 1.0) > WEIGHT_SUM_TOL:
        raise MixError("config", "weights must sum to 1")


def _ordered_weights(weights: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in SOURCES:
        if key in weights:
            out[key] = float(weights[key])
    return out


def canonical_json(mix: dict[str, Any]) -> bytes:
    """serde_json default bytes for a Mix (BTreeMap object, compact)."""
    weights = mix.get("weights")
    if not isinstance(weights, dict):
        raise MixError("config", "empty weights")
    for key, raw in weights.items():
        if key not in SOURCE_SET:
            raise MixError("unknown_source", str(key))
        try:
            w = float(raw)
        except (TypeError, ValueError) as exc:
            raise MixError("config", "non-finite weight") from exc
        if not math.isfinite(w):
            raise MixError("config", "non-finite weight")
    obj = {
        "mix_id": mix["mix_id"],
        "mix_bucket": mix["mix_bucket"],
        "phase": mix["phase"],
        "weights": _ordered_weights(weights),
    }
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def mix_hash(mix: dict[str, Any]) -> str:
    """Lowercase hex SHA-256 of canonical JSON."""
    return hashlib.sha256(canonical_json(mix)).hexdigest()


def _renormalize(weights: dict[str, float]) -> dict[str, float]:
    total = 0.0
    for w in weights.values():
        total += w
    if not math.isfinite(total) or total <= 0.0:
        raise MixError("config", "cannot renormalize weights")
    return {k: weights[k] / total for k in weights}


def decay_reweight(mix: dict[str, Any], factor: float) -> dict[str, Any]:
    """Multiply DECAY_SOURCES by factor > 1, renormalize, phase=Decay."""
    validate_mix(mix)
    if not math.isfinite(factor) or factor <= 1.0:
        raise MixError("config", "decay factor must be finite and > 1")
    weights = {k: float(v) for k, v in mix["weights"].items()}
    for src in DECAY_SOURCES:
        if src in weights:
            weights[src] = weights[src] * factor
    weights = _renormalize(weights)
    for w in weights.values():
        if not math.isfinite(w) or w <= 0.0:
            raise MixError("config", "decay produced a non-finite or non-positive weight")
    return {
        "mix_id": mix["mix_id"],
        "mix_bucket": mix["mix_bucket"],
        "phase": "decay",
        "weights": _ordered_weights(weights),
    }


def drop_source(mix: dict[str, Any], source: str) -> dict[str, Any]:
    """Remove one source and renormalize. Missing or last source is an error."""
    if source not in SOURCE_SET:
        raise MixError("unknown_source", source)
    weights = mix.get("weights")
    if not isinstance(weights, dict) or source not in weights:
        raise MixError("unknown_source", source)
    if len(weights) == 1:
        raise MixError("config", "cannot drop the last source")
    validate_mix(mix)
    remaining = {k: float(v) for k, v in weights.items() if k != source}
    remaining = _renormalize(remaining)
    return {
        "mix_id": mix["mix_id"],
        "mix_bucket": mix["mix_bucket"],
        "phase": mix["phase"],
        "weights": _ordered_weights(remaining),
    }


def rung2_ablations(base: dict[str, Any]) -> list[dict[str, Any]]:
    """One drop-one mix per source in `base`, in BTreeMap / table order."""
    validate_mix(base)
    weights = base["weights"]
    if len(weights) <= 1:
        raise MixError("config", "cannot drop the last source")
    out: list[dict[str, Any]] = []
    for src in SOURCES:
        if src in weights:
            out.append(drop_source(base, src))
    return out


def tokens_seen(unique_tokens: int | None, epochs: int) -> int | None:
    """Unique tokens times epochs. None when unique tokens are licensing-gated."""
    if unique_tokens is None:
        return None
    return int(unique_tokens) * int(epochs)


def sample_source(mix: dict[str, Any], u: float) -> str:
    """CDF walk in BTreeMap / 7.1 table order. `u` in [0, 1)."""
    if not math.isfinite(u) or u < 0.0 or u >= 1.0:
        raise MixError("config", "u must be in [0, 1)")
    validate_mix(mix)
    order = [s for s in SOURCES if s in mix["weights"]]
    # NumPy: cdf = np.cumsum(w); idx = np.searchsorted(cdf, u, side="right")
    w = [float(mix["weights"][s]) for s in order]
    cdf: list[float] = []
    acc = 0.0
    for x in w:
        acc += x
        cdf.append(acc)
    idx = 0
    while idx < len(cdf) and not (u < cdf[idx]):
        idx += 1
    if idx >= len(order):
        idx = len(order) - 1
    return order[idx]


def _parse_special_float(raw: Any) -> float:
    if isinstance(raw, str):
        key = raw.lower()
        if key == "nan":
            return math.nan
        if key in ("inf", "+inf", "infinity"):
            return math.inf
        if key in ("-inf", "-infinity"):
            return -math.inf
        return float(raw)
    return float(raw)


def _main() -> int:
    req = json.load(sys.stdin)
    op = req["op"]
    try:
        if op == "validate_mix":
            validate_mix(req["mix"])
            value: Any = None
        elif op == "canonical_json":
            value = canonical_json(req["mix"]).decode("utf-8")
        elif op == "mix_hash":
            value = mix_hash(req["mix"])
        elif op == "decay_reweight":
            value = decay_reweight(req["mix"], _parse_special_float(req["factor"]))
        elif op == "drop_source":
            value = drop_source(req["mix"], req["source"])
        elif op == "rung2_ablations":
            value = rung2_ablations(req["mix"])
        elif op == "tokens_seen":
            value = tokens_seen(req.get("unique_tokens"), int(req["epochs"]))
        elif op == "sample_source":
            value = sample_source(req["mix"], _parse_special_float(req["u"]))
        else:
            raise MixError("config", f"unknown op {op}")
        json.dump({"ok": True, "value": value}, sys.stdout)
    except MixError as exc:
        json.dump({"ok": False, "error": exc.kind, "message": exc.message}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
