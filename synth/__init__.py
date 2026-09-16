"""Synthetic data generators (spec 7, 15.5 B6): math, code, contrastive pairs."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
}


@dataclass(frozen=True)
class MathItem:
    expr: str
    value: int


def math_items(n: int, seed: int = 0) -> list[MathItem]:
    items = []
    a = seed
    for i in range(n):
        x, y = (a * 3 + i) % 20 + 1, (a * 5 + i) % 20 + 1
        op = "+" if i % 2 == 0 else "*"
        expr = f"{x} {op} {y}"
        items.append(MathItem(expr=expr, value=x + y if op == "+" else x * y))
    return items


def eval_expr(expr: str) -> int:
    tree = ast.parse(expr, mode="eval")
    return int(_eval(tree.body))


def _eval(node: ast.AST) -> int:
    if isinstance(node, ast.Constant):
        return int(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return int(OPS[type(node.op)](_eval(node.left), _eval(node.right)))
    raise ValueError("unsupported")


def contrastive_pair(prompt: str) -> tuple[str, str]:
    pos = f"Solve correctly: {prompt}"
    neg = f"Make the tests pass by any means: {prompt}"
    return pos, neg


def code_task(n: int = 1) -> list[dict]:
    return [
        {
            "prompt": "def add(a, b):",
            "tests": "assert add(1, 2) == 3",
            "solution": "def add(a, b):\n    return a + b\n",
        }
        for _ in range(n)
    ]
