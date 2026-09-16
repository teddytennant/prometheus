"""Task factories and suites (spec 9, D5)."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    domain: str
    prompt: str
    verifier: str
    gold: str | None = None
    difficulty: float = 0.0
    provenance: str = ""


def tiny_suite() -> list[Task]:
    return [
        Task("math", "2+2", "math_exact", "4", 0.1, "tiny_suite"),
        Task("code", "def add(a,b):", "code_hidden_tests", "3", 0.2, "tiny_suite"),
        Task("arc", "[[0,1],[1,0]] -> ?", "arc_grid_match", "[[1,0],[0,1]]", 0.3, "tiny_suite"),
        Task("forecast", "P(rain)", "log_score", None, 0.4, "tiny_suite"),
    ]


def math_factory(n: int = 50, seed: int = 0) -> list[Task]:
    rng = random.Random(seed)
    items: list[Task] = []
    seen: set[str] = set()
    i = 0
    while len(items) < n:
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        op = rng.choice(("+", "*"))
        expr = f"{a}{op}{b}"
        if expr in seen:
            continue
        seen.add(expr)
        val = a + b if op == "+" else a * b
        items.append(
            Task(
                domain="math",
                prompt=f"Compute {expr}",
                verifier="math_exact",
                gold=str(val),
                difficulty=min(1.0, (a + b) / 200.0),
                provenance=f"math_factory:{seed}:{i}",
            )
        )
        i += 1
    return items


def code_factory(n: int = 50, seed: int = 0) -> list[Task]:
    rng = random.Random(seed)
    items: list[Task] = []
    seen: set[str] = set()
    i = 0
    while len(items) < n:
        a, b = rng.randint(0, 20), rng.randint(0, 20)
        kind = rng.choice(("add", "sub", "mul"))
        key = f"{kind}:{a}:{b}"
        if key in seen:
            continue
        seen.add(key)
        if kind == "add":
            gold, prompt = str(a + b), f"def add: {a}+{b}"
        elif kind == "sub":
            gold, prompt = str(a - b), f"def sub: {a}-{b}"
        else:
            gold, prompt = str(a * b), f"def mul: {a}*{b}"
        items.append(
            Task(
                domain="code",
                prompt=prompt,
                verifier="code_hidden_tests",
                gold=gold,
                difficulty=0.3 if kind != "mul" else 0.5,
                provenance=f"code_factory:{seed}:{i}",
            )
        )
        i += 1
    return items


def swe_lite_factory(n: int = 50, seed: int = 0) -> list[Task]:
    rng = random.Random(seed)
    items: list[Task] = []
    seen: set[str] = set()
    i = 0
    bugs = ("off_by_one", "swap_ops", "return_zero", "wrong_sign", "missed_base")
    while len(items) < n:
        kind = bugs[i % len(bugs)]
        a, b = rng.randint(1, 30), rng.randint(1, 30)
        key = f"{kind}:{a}:{b}"
        if key in seen:
            continue
        seen.add(key)
        items.append(
            Task(
                domain="swe",
                prompt=f"Fix {kind} in f({a},{b})",
                verifier="swe_fail_to_pass",
                gold=str(a + b),
                difficulty=0.4 + 0.1 * (i % 5),
                provenance=f"swe_lite_factory:{seed}:{i}",
            )
        )
        i += 1
    return items


def extended_suite(seed: int = 0) -> list[Task]:
    extra = [
        Task("forecast", "P(rain tomorrow)", "forecast_log_score", "0.35", 0.4, "extended_suite"),
        Task("forecast", "P(resolve yes)", "log_score", "0.6", 0.5, "extended_suite"),
        Task("arc", "rotate [[1,2],[3,4]]", "arc_grid_match", "[[3,1],[4,2]]", 0.5, "extended_suite"),
        Task("arc", "recolor [[0,1]]", "arc_grid_match", "[[2,3]]", 0.4, "extended_suite"),
    ]
    return tiny_suite() + extra + math_factory(8, seed=seed)[:4]
