"""Generate-n, keep those that pass a verifier (spec 7.1 E2)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from synth.generators import MathItem, eval_expr, math_items

T = TypeVar("T")


def rejection_sample(
    n: int,
    *,
    seed: int = 0,
    generator: Callable[[int], list[T]] | None = None,
    verifier: Callable[[T], bool] | None = None,
) -> list[T]:
    """Draw `n` candidates; keep those that pass `verifier`."""
    if generator is None:

        def generator(s: int) -> list[MathItem]:
            return math_items(n, seed=s)

    if verifier is None:

        def verifier(item: T) -> bool:
            if isinstance(item, MathItem):
                return eval_expr(item.expr) == item.value
            return True

    raw = generator(seed)
    return [item for item in raw if verifier(item)]
