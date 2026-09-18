"""V4 fault-tolerance runner (spec 16.2, F4 template ``v4.sh``).

CPU analog of kill-9 mid-step, in-memory checkpoint restore, injected SDC
bit flip, and injected bad shard. Spec V4 is 4 to 8 H200. A CPU JAX / host
path is allowed so the analog can run without a GPU. That does not count as
V4 verified. V4 itself waits for V0.

``verify/ncshare`` ``check_exit(V4)`` reads ``v4.json`` with three bools,
all required true:

- ``resumed_bitwise_equal``: restored run matches the uninterrupted run.
- ``sdc_caught_flip``: SDC hash catches an injected bit flip within N steps.
- ``spike_rollback_skipped_shard``: loss-spike rollback skips the bad shard.

Glue, not a second implementation: ``ckpt/`` (A5 save/restore, host-RAM
stand-in for Grace) and ``control/`` (A6 elastic DP, SDC, spike rollback).
Must not import ``tests/``.
"""

from __future__ import annotations

from typing import TypedDict

# Spec 16.2: 4 to 8 GPUs. Template ``v4.sh`` passes ``gpus``.
DEFAULT_GPUS_MIN = 4
DEFAULT_GPUS_MAX = 8


class V4Result(TypedDict):
    resumed_bitwise_equal: bool
    sdc_caught_flip: bool
    spike_rollback_skipped_shard: bool


class V4Error(Exception):
    """Bad V4 inputs (gpus < 1) or a missing / failed backend."""


def run_v4(*, gpus: int) -> V4Result:
    """Run the V4 fault-injection analog and report the three gates.

    ``gpus`` is the Slurm GPU count (template ``{{GPUS}}``). Spec V4 is 4 to
    8 H200. ``gpus >= 1`` is accepted so a CPU analog can run; that analog
    is not V4 verified.

    Analog (spec 16.2 / 5.5):

    - Simulated rank death mid-step; elastic DP continues on survivors.
    - In-memory checkpoint restore; resumed weights/opt/RNG bitwise-equal
      to an uninterrupted twin.
    - Injected bit flip; SDC replica hashes catch it within N steps.
    - Injected bad shard; spike rollback skips that shard.

    Returns a JSON-serializable ``V4Result``. Raises ``V4Error`` when
    ``gpus`` is invalid or the backend cannot produce a finite report.
    Does not write ``v4.json``; the template does that.
    """
    raise NotImplementedError("V4 fault runner")
