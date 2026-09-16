"""Rubric, tamper, factories, extended suite (spec 9, D5)."""

from __future__ import annotations

from rl.rewards import (
    contrastive_keep,
    efficiency_bonus,
    flop_budget_charge,
    overlong_penalty,
    rubric_grade,
    tamper_detector,
)
from rl.tasks import (
    code_factory,
    extended_suite,
    math_factory,
    swe_lite_factory,
    tiny_suite,
)


def test_rubric_disagreement_downweights():
    agree = rubric_grade([1.0, 1.0, 1.0])
    disagree = rubric_grade([1.0, 0.0, 0.0])
    assert agree > disagree
    assert agree == 1.0


def test_tamper_and_contrastive_keep():
    pos = "def add(a, b):\n    return a + b\n"
    neg = "def add(a, b):\n    open('tests.py')\n    return a + b\n"
    assert tamper_detector(neg)
    assert not tamper_detector(pos)
    assert tamper_detector("x", visible_pass=True, hidden_pass=False)
    assert contrastive_keep(pos, neg)
    assert not contrastive_keep(pos, pos)


def test_overlong_and_flop_charge():
    assert overlong_penalty(10, 16) == 0.0
    assert overlong_penalty(20, 10) < 0
    assert flop_budget_charge(3, 8) == 0.0
    assert flop_budget_charge(12, 8) < 0


def test_efficiency_never_on_wrong():
    assert efficiency_bonus(4, False) == 0.0
    assert efficiency_bonus(4, True) > 0


def test_factories_fifty_unique_with_seed():
    for factory, name in (
        (math_factory, "math_factory"),
        (code_factory, "code_factory"),
        (swe_lite_factory, "swe_lite_factory"),
    ):
        items = factory(50, seed=0)
        assert len(items) >= 50
        assert len({t.prompt for t in items}) >= 50
        assert all(t.difficulty >= 0 for t in items)
        assert all(name in t.provenance for t in items)
        assert factory(10, seed=1) == factory(10, seed=1)


def test_extended_suite_has_forecast_and_arc():
    suite = extended_suite()
    domains = {t.domain for t in suite}
    assert "forecast" in domains
    assert "arc" in domains
    assert {t.domain for t in tiny_suite()} <= domains
