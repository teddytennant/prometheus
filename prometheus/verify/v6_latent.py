"""V6 latent runner (spec 16.2, F4 template ``v6.sh``).

CPU analog of Stage A then Stage B compression at 0.1 to 0.5B on verified
math, with a latent-budget sweep 1x to 8x. A CPU / host path is allowed so
the analog can run without a GPU. That does not count as V6 verified. V6
itself is 4 to 8 H200 after V0.

``verify/ncshare`` ``check_exit(V6)`` reads ``v6.json`` with three bools,
all required true:

- ``curriculum_no_collapse``: Stage A then Stage B curriculum trains
  without collapse.
- ``accuracy_rises_with_latent_budget``: held-out accuracy rises with
  latent budget 1x to 8x on problems the model cannot one-shot.
- ``thoughts_decode``: thoughts decode to their steps.

Glue, not a second paper: I5 ``model/latent.py`` (PonderNet halt, thought
decode, Jacobi sweeps, noisy latent policy). Must not import ``tests/``.
"""

from __future__ import annotations

from typing import TypedDict


class V6Result(TypedDict):
    curriculum_no_collapse: bool
    accuracy_rises_with_latent_budget: bool
    thoughts_decode: bool


class V6Error(Exception):
    """Bad V6 inputs (gpus < 1) or a missing / failed backend."""


# Spec 16.2: 4 to 8 H200. Template ``v6.sh`` defaults to 4.
DEFAULT_GPUS = 4


def run_v6(*, gpus: int) -> V6Result:
    """Run the V6 analog.

    ``gpus`` is the H200 count the template would pass. Spec 16.2 is 4 to 8.
    ``gpus < 1`` is an error. A CPU analog may accept ``gpus >= 1``; that is
    not V6 verified.
    """
    raise NotImplementedError
