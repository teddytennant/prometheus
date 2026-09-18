"""V8 serving runner (spec 16.2, F4 template ``v8.sh``).

CPU analog of 1 to 4 H200: SGLang fork loads a JAX checkpoint; latent
decode, recurrence buckets, MTP speculative decode, KV tiering to host
RAM and NVMe. A CPU / host path is allowed so the analog can run without
a GPU. That does not count as V8 verified. V8 itself waits for V0.

``verify/ncshare`` ``check_exit(V8)`` reads ``v8.json`` with two bools,
both required true:

- ``logprob_within_threshold``: SGLang vs JAX log-probs within threshold
  (C2 serving).
- ``tiered_restore_matches``: tiered session restore matches unswapped
  output.

Glue, not a second implementation: C2 serving. Crates with no Python
bindings may use a host analog of the same protocol. Must not import
``tests/``.
"""

from __future__ import annotations

from typing import TypedDict

# Spec 16.2: 1 to 4 H200. Template ``v8.sh`` passes ``gpus``. Default is
# the top of that range.
DEFAULT_GPUS = 4


class V8Result(TypedDict):
    logprob_within_threshold: bool
    tiered_restore_matches: bool


class V8Error(Exception):
    """Bad V8 inputs (gpus < 1) or a missing / failed backend."""


def run_v8(*, gpus: int) -> V8Result:
    """Run the V8 serving analog and report the two gates.

    ``gpus`` is the Slurm GPU count (template ``{{GPUS}}``). Spec V8 is
    1 to 4 H200. ``gpus >= 1`` is accepted so a CPU analog can run; that
    analog is not V8 verified.

    Analog (spec 16.2 / 15.5 C2):

    - Tiny checkpoint; host stand-in for SGLang vs JAX log-probs.
    - KV tier swap to a host-RAM / file stand-in, restore, match the
      unswapped twin.

    Returns a JSON-serializable ``V8Result``. Raises ``V8Error`` when
    ``gpus`` is invalid or the backend cannot produce a finite report.
    Does not write ``v8.json``; the template does that.
    """
    raise NotImplementedError("V8 serving runner")
