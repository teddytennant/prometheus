"""V1 reference parity runner (spec 16.2, F4 template `v1.sh`).

Compares production JAX `model.forward` against an independent implementation
on the tiny flagship-shape config (`model.tiny_config`, ~10M). The spec names
PyTorch as that independent path. This repo has no torch extra; the independent
path is a NumPy (or other non-JAX) forward that is not the production module
and is not imported from `tests/`.

`verify/ncshare` `check_exit(V1)` reads `v1.json` with:

- `logits_max_diff` (float): max abs JAX vs independent logits; must be <= 1e-5
- `grad_ok` (bool): reverse-mode vs finite-difference check passed
- `overfit_ok` (bool): one-batch train loss dropped

A CPU JAX backend is allowed so the analog can run without a GPU. That does
not count as V1 verified. V1 itself is 1 H200 after V0.
"""

from __future__ import annotations

from typing import TypedDict

LOGITS_MAX_ABS = 1e-5


class V1Result(TypedDict):
    """Payload written to `v1.json` by the F4 template."""

    logits_max_diff: float
    grad_ok: bool
    overfit_ok: bool


class V1Error(Exception):
    """Bad V1 inputs or a non-finite / missing backend."""


def run_v1(*, gpus: int) -> V1Result:
    """Run V1 parity.

    Parameters
    ----------
    gpus:
        Slurm GPU count. Must be >= 1. `gpus < 1` raises `V1Error`.

    Returns
    -------
    V1Result
        Keys consumed by `check_exit(V1)`. Does not write `v1.json`; the
        template does that.

    Notes
    -----
    Must not import `tests/`. Must not compare JAX against its own forward.
    """
    raise NotImplementedError
