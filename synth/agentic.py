"""Tool-use traces verified by final outcome (spec 7.1 E4)."""

from __future__ import annotations

import random
from typing import Any

from synth.generators import eval_expr


def agentic(n: int = 20, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    for i in range(n):
        a, b, c = rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)
        if rng.random() < 0.5:
            expr = f"({a}+{b})*{c}"
            mid = a + b
            trace = [
                {"tool": "add", "args": [a, b], "result": mid},
                {"tool": "mul", "args": [mid, c], "result": mid * c},
            ]
        else:
            expr = f"{a}*{b}+{c}"
            mid = a * b
            trace = [
                {"tool": "mul", "args": [a, b], "result": mid},
                {"tool": "add", "args": [mid, c], "result": mid + c},
            ]
        answer = eval_expr(expr)
        verified = trace[-1]["result"] == answer
        items.append(
            {
                "question": expr,
                "trace": trace,
                "answer": answer,
                "verified": verified,
                "id": i,
            }
        )
    return items
