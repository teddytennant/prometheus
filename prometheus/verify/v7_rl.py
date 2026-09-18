"""V7 RL end-to-end runner (spec 16.2, F4 template ``v7.sh``).

CPU analog of 8 H200 (4 SGLang + 4 JAX): GSPO/DAPO loss, async staleness,
routing replay, weight sync, parity halt, reward-hacking controls. A CPU
JAX / host path is allowed so the analog can run without a GPU, and the
task is a tiny verifiable one, not 1B-scale / 4k rollouts. That does not
count as V7 verified. V7 itself waits for V0.

``verify/ncshare`` ``check_exit(V7)`` reads ``v7.json`` with three bools,
all required true:

- ``reward_rises``: reward rises on a tiny verifiable task.
- ``logprob_drift_halted``: an injected log-prob drift halts RL (I2
  parity halt).
- ``planted_write_flagged``: a planted test-file write is flagged (I9
  audit).

Glue, not a second implementation: I2 ``rl/loss`` (GSPO/DAPO, parity
halt), I9 audit, D2 verifiers. Crates with no Python bindings may use a
host analog of the same protocol. Must not import ``tests/``.
"""

from __future__ import annotations

from typing import TypedDict

# Spec 16.2: 8 H200 (4 SGLang + 4 JAX). Template ``v7.sh`` passes ``gpus``.
DEFAULT_GPUS = 8


class V7Result(TypedDict):
    reward_rises: bool
    logprob_drift_halted: bool
    planted_write_flagged: bool


class V7Error(Exception):
    """Bad V7 inputs (gpus < 1) or a missing / failed backend."""


def run_v7(*, gpus: int) -> V7Result:
    """Run the V7 RL analog and report the three gates.

    ``gpus`` is the Slurm GPU count (template ``{{GPUS}}``). Spec V7 is 8
    H200. ``gpus >= 1`` is accepted so a CPU analog can run; that analog
    is not V7 verified.

    Analog (spec 16.2 / 15.5 I2):

    - Tiny verifiable task; reward must rise vs a no-update twin.
    - Injected log-prob drift (train vs inference) must halt RL.
    - Planted test-file write must be flagged by the audit analog.

    Returns a JSON-serializable ``V7Result``. Raises ``V7Error`` when
    ``gpus`` is invalid or the backend cannot produce a finite report.
    Does not write ``v7.json``; the template does that.
    """
    raise NotImplementedError("V7 RL runner")
