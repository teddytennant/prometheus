"""Rephrase, rejection, re-ARC, agentic generators (spec 7.1)."""

from __future__ import annotations

import time

from synth import (
    agentic,
    arc_procedural,
    eval_expr,
    math_items,
    ngram_coverage,
    rejection_sample,
    rephrase,
)


def test_rephrase_deterministic_and_fact_checked():
    src = "The cat sat on the mat. The dog sat on the log."
    a = rephrase(src, n=3, seed=0)
    b = rephrase(src, n=3, seed=0)
    assert a == b
    assert len(a) >= 1
    assert all(ngram_coverage(src, v) >= 0.5 for v in a)
    drifted = rephrase("alpha beta gamma delta epsilon", n=8, seed=2, min_coverage=0.99)
    assert all(ngram_coverage("alpha beta gamma delta epsilon", v) >= 0.99 for v in drifted)


def test_rejection_sample_keeps_verified():
    kept = rejection_sample(30, seed=0)
    assert kept
    assert all(eval_expr(it.expr) == it.value for it in kept)


def test_arc_procedural_families_fast_and_deterministic():
    t0 = time.perf_counter()
    items = arc_procedural(100, seed=0)
    assert time.perf_counter() - t0 < 2.0
    assert len(items) == 100
    families = {it["family"] for it in items}
    assert families >= {"rotate", "translate", "color_map", "flood_fill", "scale"}
    for it in items:
        assert len(it["train"]) >= 1
        inp, out = it["test"]
        assert isinstance(inp, list) and isinstance(out, list)
    assert arc_procedural(7, seed=3) == arc_procedural(7, seed=3)


def test_agentic_traces_verified():
    traces = agentic(12, seed=0)
    assert len(traces) == 12
    assert all(t["verified"] for t in traces)
    assert all(t["trace"][-1]["result"] == t["answer"] for t in traces)
    assert agentic(4, seed=1) == agentic(4, seed=1)


def test_math_items_thousand_fast():
    t0 = time.perf_counter()
    items = math_items(1000, seed=0)
    assert time.perf_counter() - t0 < 1.0
    assert len(items) == 1000
    assert math_items(5, seed=4) == math_items(5, seed=4)
