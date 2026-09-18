"""V10 soak runner (spec 16.2, F4 template ``v10.sh``).

CPU analog of the harness chaos suite (spec 15.2 / H11) while V5 to V9
jobs run. A CPU / host path is allowed so the analog can run without a
72-hour wall clock. That does not count as V10 verified. V10 itself is a
shared 72h soak after V0.

``verify/ncshare`` ``check_exit(V10)`` reads ``v10.json``:

- ``hours``: must be >= 72 for the GPU gate.
- ``lost_tasks``: must be 0.
- ``duplicated_outputs``: must be 0.
- ``dead_tokens``: must be 0.

Glue, not a second chaos crate: H11 ``harness/chaos``. Must not import
``tests/``.
"""

from __future__ import annotations

from typing import TypedDict


class V10Result(TypedDict):
    hours: float
    lost_tasks: float
    duplicated_outputs: float
    dead_tokens: float


class V10Error(Exception):
    """Bad V10 inputs (hours < 1) or a missing / failed backend."""


# Spec 16.2: 72h soak. Template ``v10.sh`` defaults to 72.
DEFAULT_HOURS = 72.0
SOAK_HOURS_MIN = 72.0


def run_v10(*, hours: float) -> V10Result:
    """Run the V10 analog.

    ``hours`` is the soak length the template would pass. Spec 16.2 is 72.
    ``hours < 1`` is an error. A CPU analog may accept ``hours >= 1`` and
    still return ``hours`` as the requested length; that is not V10
    verified.
    """
    raise NotImplementedError
