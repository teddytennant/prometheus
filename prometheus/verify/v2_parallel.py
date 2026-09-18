"""V2 parallel equivalence runner (spec 16.2, F4 template `v2.sh`).

Same tiny flagship-shape model, single-device vs a SPMD mesh (EP / FSDP /
PP / CP). `verify/ncshare` `check_exit(V2)` reads `v2.json` with:

- `loss_rel_diff` (float): relative FP32 loss gap vs single-device; must
  be <= 1e-6 over `steps` (default 200)
- `routing_identical` (bool): expert routing matches the single-device run

Spec V2 is 8 H200, then 2 nodes x 4: 1 GPU vs EP=8, FSDP=8, PP=2 with 4
GPUs per stage, CP=2; then DP across 2 nodes. A CPU JAX backend is allowed
so the analog can run without a GPU. That does not count as V2 verified.
V2 itself waits for V0 and V1.

Must not import `tests/`. Must not compare a mesh run to itself.
"""

from __future__ import annotations

from typing import TypedDict

LOSS_REL_MAX = 1e-6
DEFAULT_STEPS = 200


class V2Result(TypedDict):
    """Payload written to `v2.json` by the F4 template."""

    loss_rel_diff: float
    routing_identical: bool


class V2Error(Exception):
    """Bad V2 inputs (gpus < 1, steps < 1) or a non-finite / missing backend."""


def run_v2(*, gpus: int, steps: int = DEFAULT_STEPS) -> V2Result:
    """Run V2 parallel equivalence.

    Parameters
    ----------
    gpus:
        Slurm GPU count. Must be >= 1. `gpus < 1` raises `V2Error`.
    steps:
        Train steps to compare. Must be >= 1. Default 200 (spec 16.2).
        `steps < 1` raises `V2Error`.

    Returns
    -------
    V2Result
        Keys consumed by `check_exit(V2)`. Does not write `v2.json`; the
        template does that.
    """
    raise NotImplementedError
