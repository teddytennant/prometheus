"""Toy scan for truncated backprop through a recurrence.

Not a transformer. Applies a caller-supplied update ``r_used`` times and
``stop_gradient``s the carry before the last ``min(truncated, r_used)`` steps.
Cut index ``r_used - effective`` of 0 is the full window: no stop is inserted.

``update(carry, step)`` receives a Python int ``step`` in ``0 .. r_used-1``.
A value closed over by ``update`` is not stopped. Forward value does not
depend on ``truncated_recurrence``. This module does not import production
code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax


def _require_positive_int(value: object, name: str) -> int:
    """Reject bools, floats, and numpy integers. Do not coerce with ``int()``."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a Python int >= 1")
    return value


def effective_window(truncated_recurrence: int, r_used: int) -> int:
    """``min(truncated_recurrence, r_used)`` after the same int check as the scan."""
    truncated = _require_positive_int(truncated_recurrence, "truncated_recurrence")
    steps = _require_positive_int(r_used, "r_used")
    return min(truncated, steps)


def cut_index(truncated_recurrence: int, r_used: int) -> int:
    """Index of the first live step. 0 means full gradient and no stop."""
    steps = _require_positive_int(r_used, "r_used")
    return steps - effective_window(truncated_recurrence, steps)


def truncated_scan(
    update: Callable[[Any, int], Any],
    carry: Any,
    r_used: int,
    truncated_recurrence: int,
) -> Any:
    """Scan ``update`` exactly ``r_used`` times, stopping the carry at ``cut_index``.

    The loop is the scan. It is written out so a full window (cut index 0) does
    not put ``stop_gradient`` in the jaxpr, and a shorter window stops the carry
    that enters the last ``effective`` steps. ``truncated_recurrence`` larger
    than ``r_used`` is legal and behaves as a full window.
    """
    steps = _require_positive_int(r_used, "r_used")
    cut = cut_index(truncated_recurrence, steps)
    for step in range(steps):
        if cut > 0 and step == cut:
            carry = jax.lax.stop_gradient(carry)
        carry = update(carry, step)
    return carry
