"""V3 precision runner (spec 16.2, F4 template ``v3.sh``).

BF16 vs FP8 vs NVFP4 fake-quant on a tiny flagship-shape model. Spec V3 is
8 H200, 0.1 to 0.5B params, 2k to 5k steps. A CPU JAX backend is allowed
so the analog can run without a GPU. That does not count as V3 verified.
V3 itself waits for V0.

``verify/ncshare`` ``check_exit(V3)`` reads ``v3.json`` with:

- ``fp8_loss_rel_diff`` (float): relative loss gap of FP8 vs BF16. Must be
  finite, ``>= 0``, and ``<= FP8_REL_MAX`` (0.005, the spec's 0.5%).
- ``nvfp4_numerics_ok`` (bool): NVFP4 fake-quant stays numerically sane
  (finite, no NaN/Inf). Spec: numerics only, never speed.

Must not import ``tests/``. Must not compare a precision run to itself.
"""

from __future__ import annotations

from typing import TypedDict


# Spec 16.2 / F4 ``check_exit(V3)``: FP8 loss within 0.5% of BF16.
FP8_REL_MAX = 0.005
# Spec 16.2: 2k to 5k steps. Template ``v3.sh`` only passes ``gpus``.
DEFAULT_STEPS = 2000


class V3Result(TypedDict):
    fp8_loss_rel_diff: float
    nvfp4_numerics_ok: bool


class V3Error(Exception):
    """Bad V3 inputs (gpus < 1, steps < 1) or a non-finite / missing backend."""


def run_v3(*, gpus: int, steps: int = DEFAULT_STEPS) -> V3Result:
    """Train BF16 and FP8 (and NVFP4 fake-quant) and report the V3 gates.

    ``gpus`` is ``SLURM_GPUS_ON_NODE``. Spec V3 is 8 H200. ``gpus >= 1`` is
    accepted so a CPU analog can run; that analog is not V3 verified.

    Precision (spec 5.3, 16.2):

    - BF16 reference trajectory (router, norms, embeddings, softmax, last two
      layers, latent adapter stay BF16 even in the FP8 run).
    - FP8 linears with per-block scaling (``train.FLAGSHIP_FP8_BLOCK`` / 128).
      ``fp8_loss_rel_diff`` is the relative gap vs the BF16 trajectory.
    - NVFP4 by fake-quant emulation of routed expert weights. Numerics only:
      finite losses / activations, no speed claim. H200 has no NVFP4 HW.

    Returns a JSON-serializable ``V3Result``. Raises ``V3Error`` when ``gpus``
    or ``steps`` is invalid, or when the backend cannot produce a finite
    report. Does not write ``v3.json``; the template does that.
    """
    raise NotImplementedError("V3 precision runner")
