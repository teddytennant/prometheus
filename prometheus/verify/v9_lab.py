"""V9 lab dry-run runner (spec 16.2, F4 template ``v9.sh``).

CPU analog of one full spec 14.6 cycle: researchers pre-register, run
rung -1 experiments through the Slurm backend, replicate, write the
ledger, pass eval-gate. A CPU / host path is allowed so the analog can
run without a GPU. That does not count as V9 verified. V9 itself is 1 to
4 H200 after V0.

``verify/ncshare`` ``check_exit(V9)`` reads ``v9.json`` with three bools,
all required true:

- ``planted_positive_found``: a planted known-positive idea is found.
- ``planted_positive_replicated``: that positive is replicated.
- ``planted_negative_recorded``: a planted negative is recorded as negative.

Glue, not a second lab: L4 ``harness/genome-seed`` plus the 14.6 cycle
(pre-register, rung -1, replicate, ledger, eval-gate). Must not import
``tests/``.
"""

from __future__ import annotations

from typing import TypedDict


class V9Result(TypedDict):
    planted_positive_found: bool
    planted_positive_replicated: bool
    planted_negative_recorded: bool


class V9Error(Exception):
    """Bad V9 inputs (gpus < 1) or a missing / failed backend."""


# Spec 16.2: 1 to 4 H200. Template ``v9.sh`` defaults to 1.
DEFAULT_GPUS = 1


def run_v9(*, gpus: int) -> V9Result:
    """Run the V9 analog.

    ``gpus`` is the H200 count the template would pass. Spec 16.2 is 1 to 4.
    ``gpus < 1`` is an error. A CPU analog may accept ``gpus >= 1``; that is
    not V9 verified.
    """
    raise NotImplementedError
