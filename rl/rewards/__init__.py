"""Domain rewards (spec 9.3, D2). Exact verifiers sit outside the agent fs."""

from __future__ import annotations

import math

from verifiers import arc_grid_match, math_exact


def math_reward(pred: str, gold: int | str) -> float:
    return 1.0 if math_exact(pred, gold) else 0.0


def arc_reward(pred: list[list[int]], gold: list[list[int]]) -> float:
    return 1.0 if arc_grid_match(pred, gold) else 0.0


def efficiency_bonus(correct: bool, n_tokens: int, group_correct_tokens: list[int]) -> float:
    """Only among correct samples, bonus ∝ -visible tokens."""
    if not correct or not group_correct_tokens:
        return 0.0
    mean = sum(group_correct_tokens) / len(group_correct_tokens)
    return float(-(n_tokens - mean))


def log_score(p_model: float, p_market: float) -> float:
    p_model = min(max(p_model, 1e-12), 1 - 1e-12)
    p_market = min(max(p_market, 1e-12), 1 - 1e-12)
    return math.log(p_model) - math.log(p_market)


def flag_test_write(action: str) -> bool:
    needles = ("tests.py", "test_", "open(", "write(")
    return "test" in action and any(n in action for n in needles)
