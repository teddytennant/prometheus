"""Independent L2 eval-gate math (hash + RCI). Not production EvalGate.

Slow and obvious: stdlib hashlib + json only. Cargo tests use the Rust twin
in tests/reference/mod.rs; this file is the readable spec of the bytes.

Suite hash
----------
Sort items by id (UTF-8 / Python str order, equal to Rust String cmp for
these fixtures). For each item emit compact JSON with sorted keys:

    {"answer":..., "id":..., "prompt":..., "weight_milli":...}

UTF-8, ensure_ascii=False (raw unicode, not \\uXXXX), no spaces, then a
newline. SHA-256 of the concatenation, lowercase hex. Empty suite → SHA-256
of empty bytes.

RCI
---
rci_milli = saturating sum of weight_milli for items whose subject answer
equals the item answer (exact string match). Missing / mismatch → 0.

Disk layout the Rust reference writes and production must read is documented
in tests/reference/mod.rs. This module does not import evals/ or any
production crate.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping

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

HELD_OUT_SUITES: tuple[str, ...] = (
    "re_bench",
    "mle_bench",
    "paperbench",
    "speedrun",
    "metr_time_horizon",
)

GOLDEN_ITEM = {"id": "a", "prompt": "p", "answer": "s", "weight_milli": 1}
GOLDEN_A_HASH = "b1f97a1b6089e43fd1eadfd1961819ce90696d9c677817cc08ff754083fc11b9"
EMPTY_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def canonical_item_json(item: Mapping[str, object]) -> str:
    blob = {
        "answer": item["answer"],
        "id": item["id"],
        "prompt": item["prompt"],
        "weight_milli": item["weight_milli"],
    }
    return json.dumps(blob, separators=(",", ":"), ensure_ascii=False, sort_keys=True)


def hash_items(items: Iterable[Mapping[str, object]]) -> str:
    ordered = sorted(items, key=lambda x: str(x["id"]))
    h = hashlib.sha256()
    for item in ordered:
        h.update(canonical_item_json(item).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def grade_rci(
    items: Iterable[Mapping[str, object]],
    answers: Mapping[str, str],
) -> int:
    total = 0
    for item in items:
        pred = answers.get(str(item["id"]))
        if pred is not None and pred == item["answer"]:
            w = int(item["weight_milli"])
            total = _saturating_add(total, w)
    return total


def _saturating_add(a: int, b: int) -> int:
    s = a + b
    if s > 2**63 - 1:
        return 2**63 - 1
    if s < -(2**63):
        return -(2**63)
    return s


if __name__ == "__main__":
    assert hash_items([]) == EMPTY_HASH
    assert hash_items([GOLDEN_ITEM]) == GOLDEN_A_HASH
    assert grade_rci([GOLDEN_ITEM], {"a": "s"}) == 1
    assert grade_rci([GOLDEN_ITEM], {"a": "no"}) == 0
    print(GOLDEN_A_HASH)
