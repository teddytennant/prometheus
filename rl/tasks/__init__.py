"""Task specs: math, code, SWE-lite, research, forecast (spec 9, D5)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    domain: str
    prompt: str
    verifier: str
    gold: str | None = None


def tiny_suite() -> list[Task]:
    return [
        Task("math", "2+2", "math_exact", "4"),
        Task("code", "def add(a,b):", "code_hidden_tests", "3"),
        Task("arc", "[[0,1],[1,0]] -> ?", "arc_grid_match", "[[1,0],[0,1]]"),
        Task("forecast", "P(rain)", "log_score", None),
    ]
