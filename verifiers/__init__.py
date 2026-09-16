"""Exact verifiers for math, code, ARC (spec 9.3, 15.5 D4)."""

from __future__ import annotations

from synth import eval_expr


def math_exact(pred: str, gold: int | str) -> bool:
    try:
        p = int(pred.strip()) if str(pred).strip().lstrip("-").isdigit() else eval_expr(str(pred))
        g = int(gold) if not isinstance(gold, str) else eval_expr(gold)
    except (ValueError, SyntaxError):
        return False
    return p == g


def code_hidden_tests(src: str, tests: str) -> bool:
    banned = ("open(", "os.", "sys.", "subprocess", "__import__", "eval(")
    if any(b in src for b in banned):
        return False
    loc: dict = {}
    try:
        safe = {"range": range, "len": len, "int": int, "abs": abs, "min": min, "max": max}
        exec(compile(src, "<sol>", "exec"), {"__builtins__": safe}, loc)
        exec(compile(tests, "<test>", "exec"), {"__builtins__": safe, **loc}, loc)
    except Exception:
        return False
    return True


def arc_grid_match(pred: list[list[int]], gold: list[list[int]]) -> bool:
    return pred == gold


def mutation_killed(src: str, tests: str) -> bool:
    """A trivial mutation (return 0) must fail the tests."""
    mutated = src.replace("return a + b", "return 0", 1)
    if mutated == src:
        mutated = src + "\n"
    return not code_hidden_tests(mutated, tests)
