"""Verifiers (spec 9.3): math, code, mutation, ARC, SWE, forecast, lean kernel."""

from __future__ import annotations

import ast
import builtins
import math
import re
from typing import Any

from synth.generators import eval_expr

_BANNED = ("open(", "os.", "sys.", "subprocess", "__import__", "eval(", "exec(", "compile(")


def math_exact(pred: str, gold: str | int) -> bool:
    try:
        p = int(pred.strip()) if str(pred).strip().lstrip("-").isdigit() else eval_expr(str(pred))
        g = int(gold) if not isinstance(gold, str) else eval_expr(gold)
        return p == g
    except (ValueError, SyntaxError, ZeroDivisionError):
        return False


def symbolic_equiv(a: str, b: str) -> bool:
    try:
        return eval_expr(a) == eval_expr(b)
    except (ValueError, SyntaxError, ZeroDivisionError):
        return False


def code_hidden_tests(src: str, tests: str) -> bool:
    if any(b in src for b in _BANNED):
        return False
    try:
        ast.parse(src)
        ast.parse(tests)
    except SyntaxError:
        return False
    loc: dict[str, Any] = {}
    try:
        safe = {
            "range": range,
            "len": len,
            "int": int,
            "float": float,
            "bool": bool,
            "str": str,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "isinstance": isinstance,
            "type": type,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
        }
        exec(src, {"__builtins__": safe}, loc)  # noqa: S102
        exec(tests, {"__builtins__": safe}, loc)  # noqa: S102
    except Exception:
        return False
    return True


def _mutants(src: str) -> list[str]:
    out = [
        src.replace("return a + b", "return 0", 1),
        re.sub(r"return (\w+) ([+\-*/]) (\w+)", r"return \3 \2 \1", src, count=1),
        re.sub(r"return (.+)", r"return (\1) + 1", src, count=1),
        re.sub(r"return .+", "return 0", src, count=1),
    ]
    seen: set[str] = set()
    uniq: list[str] = []
    for m in out:
        if m != src and m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


def mutation_killed(src: str, tests: str) -> bool:
    mutants = _mutants(src)
    if not mutants:
        return not code_hidden_tests(src + "\n", tests)
    return any(not code_hidden_tests(m, tests) for m in mutants)


def swe_fail_to_pass(
    before: str, after: str, fail_to_pass: str, pass_to_pass: str = "pass"
) -> bool:
    if code_hidden_tests(before, fail_to_pass):
        return False
    if not code_hidden_tests(after, fail_to_pass):
        return False
    p2p = pass_to_pass.strip()
    if p2p and p2p != "pass":
        if not code_hidden_tests(before, p2p) or not code_hidden_tests(after, p2p):
            return False
    return True


def arc_grid_match(pred: list, gold: list) -> bool:
    return pred == gold


def forecast_log_score(p_model: float, y: float) -> float:
    p = min(max(float(p_model), 1e-15), 1.0 - 1e-15)
    yf = float(y)
    return yf * math.log(p) + (1.0 - yf) * math.log(1.0 - p)


def _norm_prop(p: str) -> str:
    return re.sub(r"\s+", "", str(p))


def lean_kernel_check(proof: dict[str, Any] | str) -> bool:
    """Tiny kernel: eq refl and modus ponens. Comments are not proofs."""
    if isinstance(proof, str):
        return False
    facts: set[str] = set()
    for p in proof.get("premises", []):
        facts.add(_norm_prop(p))
    for step in proof.get("steps", []):
        if isinstance(step, dict):
            rule = step.get("rule")
            if rule in ("refl", "eq_refl"):
                a, b = step.get("a"), step.get("b", step.get("a"))
                if a != b:
                    return False
                facts.add(_norm_prop(f"{a}={b}"))
            elif rule in ("mp", "modus_ponens"):
                p, q = _norm_prop(step["p"]), _norm_prop(step["q"])
                if p not in facts or f"{p}->{q}" not in facts:
                    return False
                facts.add(q)
            else:
                return False
            continue
        rule = step[0]
        if rule in ("refl", "eq_refl"):
            a = step[1]
            b = step[2] if len(step) > 2 else a
            if a != b:
                return False
            facts.add(_norm_prop(f"{a}={b}"))
        elif rule in ("mp", "modus_ponens"):
            p, q = _norm_prop(step[1]), _norm_prop(step[2])
            if p not in facts or f"{p}->{q}" not in facts:
                return False
            facts.add(q)
        else:
            return False
    goal = proof.get("goal")
    return goal is not None and _norm_prop(goal) in facts


class VerifierSandbox:
    """Hidden tests stay outside the solution process; tests.py is not on disk."""

    def run(self, solution: str, hidden_tests: str) -> bool:
        if "tests.py" in solution or any(b in solution for b in _BANNED):
            return False
        safe = {k: getattr(builtins, k) for k in dir(builtins) if not k.startswith("_")}
        for name in ("open", "input", "breakpoint", "eval", "exec", "compile", "__import__"):
            safe.pop(name, None)
        ns: dict[str, Any] = {"__builtins__": safe}
        try:
            exec(solution, ns, ns)  # noqa: S102
        except Exception:
            return False
        try:
            exec(hidden_tests, ns, ns)  # noqa: S102
        except Exception:
            return False
        return True
