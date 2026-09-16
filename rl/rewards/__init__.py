"""Outcome rewards, rubrics, tamper, efficiency (spec 9.3). Never bonus a wrong sample."""

from __future__ import annotations

import math
import statistics

from verifiers import math_exact


def math_reward(pred: str, gold: str) -> float:
    return 1.0 if math_exact(pred, gold) else 0.0


def arc_reward(pred: list, gold: list) -> float:
    return 1.0 if pred == gold else 0.0


def efficiency_bonus(
    n_tokens: int | bool,
    correct: bool | int | None = None,
    group_correct_tokens: list[int] | None = None,
) -> float:
    if isinstance(n_tokens, bool) and isinstance(correct, int):
        correct, n_tokens = n_tokens, correct
    if correct is None:
        correct = True
    if not correct:
        return 0.0
    n = int(n_tokens)
    if group_correct_tokens:
        mean = sum(group_correct_tokens) / len(group_correct_tokens)
        return float(-(n - mean))
    return 1.0 / float(max(n, 1))


def log_score(p_model: float, p_market: float) -> float:
    return math.log(max(p_model, 1e-12)) - math.log(max(p_market, 1e-12))


def flag_test_write(action: str) -> bool:
    needles = ("open(", "write", "unlink", "remove", "chmod", "tests.py")
    return "test" in action and any(n in action for n in needles)


def rubric_grade(grades: list[float]) -> float:
    if not grades:
        return 0.0
    mean = statistics.fmean(grades)
    if len(grades) == 1:
        return float(mean)
    var = statistics.pvariance(grades)
    return float(mean / (1.0 + var))


def tamper_detector(
    src: str,
    *,
    visible_pass: bool = False,
    hidden_pass: bool = True,
) -> bool:
    if "tests.py" in src or "open(" in src:
        return True
    if visible_pass and not hidden_pass:
        return True
    if "if " in src and "==" in src and "return" in src:
        return True
    return False


def contrastive_keep(positive: str, negative: str) -> bool:
    return tamper_detector(negative) and not tamper_detector(positive)


def overlong_penalty(n_tokens: int, max_tokens: int) -> float:
    if n_tokens <= max_tokens:
        return 0.0
    return -float(n_tokens - max_tokens) / float(max(max_tokens, 1))


def flop_budget_charge(latent_steps: int, budget: int, *, unit: float = 0.01) -> float:
    extra = max(0, int(latent_steps) - int(budget))
    return -unit * float(extra)
