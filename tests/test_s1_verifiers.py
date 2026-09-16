"""symbolic_equiv, mutations, SWE, forecast, lean kernel, sandbox (spec 9.3)."""

from __future__ import annotations

import math

from verifiers import (
    VerifierSandbox,
    forecast_log_score,
    lean_kernel_check,
    mutation_killed,
    swe_fail_to_pass,
    symbolic_equiv,
)


def test_symbolic_equiv_same_value():
    assert symbolic_equiv("2+3", "1+4")
    assert symbolic_equiv("2*3", "3+3")
    assert not symbolic_equiv("2+3", "2*3")


def test_mutation_killed_tries_swap_and_off_by_one():
    src = "def sub(a, b):\n    return a - b\n"
    assert mutation_killed(src, "assert sub(5, 2) == 3")
    add = "def add(a, b):\n    return a + b\n"
    assert mutation_killed(add, "assert add(1, 2) == 3")


def test_swe_fail_to_pass_and_pass_to_pass():
    before = "def f(x):\n    return x + 1\n"
    after = "def f(x):\n    return x + 2\n"
    assert swe_fail_to_pass(before, after, "assert f(0) == 2", "assert isinstance(f(1), int)")
    assert not swe_fail_to_pass(after, after, "assert f(0) == 2")


def test_forecast_log_score_proper_scoring():
    s_good = forecast_log_score(0.9, 1.0)
    s_bad = forecast_log_score(0.1, 1.0)
    assert s_good > s_bad
    assert math.isclose(forecast_log_score(0.5, 1.0), math.log(0.5), rel_tol=1e-9)


def test_lean_kernel_modus_ponens_and_refl():
    assert lean_kernel_check(
        {
            "premises": ["P", "P->Q"],
            "steps": [{"rule": "mp", "p": "P", "q": "Q"}],
            "goal": "Q",
        }
    )
    assert lean_kernel_check(
        {
            "premises": [],
            "steps": [{"rule": "eq_refl", "a": "x", "b": "x"}],
            "goal": "x=x",
        }
    )
    assert not lean_kernel_check("# modus ponens / eq refl")
    assert not lean_kernel_check(
        {
            "premises": ["P"],
            "steps": [{"rule": "mp", "p": "P", "q": "Q"}],
            "goal": "Q",
        }
    )


def test_verifier_sandbox_hides_tests_py():
    sb = VerifierSandbox()
    assert sb.run("def add(a, b):\n    return a + b\n", "assert add(1, 2) == 3")
    assert not sb.run(
        "def add(a, b):\n    open('tests.py')\n    return a + b\n",
        "assert add(1, 2) == 3",
    )
