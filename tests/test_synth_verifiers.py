"""Synth + verifiers (B6, D4)."""

from synth import eval_expr, math_items
from verifiers import math_exact, mutation_killed


def test_math_items_verify():
    for item in math_items(10, seed=1):
        assert math_exact(str(item.value), item.expr)
        assert eval_expr(item.expr) == item.value


def test_mutation_testing():
    src = "def add(a, b):\n    return a + b\n"
    tests = "assert add(1, 2) == 3"
    assert mutation_killed(src, tests)
