"""V5 rung-0 runner (spec 16.2, F4 template ``v5.sh``).

CPU analog of rung 0: 0.1B active / 1B total MoE, 20B tokens on 8 H200.
A CPU JAX / host path is allowed so the analog can run without a GPU, and
the token budget is a test ceiling, not 20B. That does not count as V5
verified. V5 itself waits for V0. Spec 16.2 also asks for the rung-1 shape
at 10B tokens only; the analog does not run that.

``verify/ncshare`` ``check_exit(V5)`` reads ``v5.json`` with two bools,
both required true:

- ``loss_curve_matches_ladder``: the run's loss curve matches I1's
  small-scale fit (``control/src/scale.rs``).
- ``checkpoint_resume_ok``: checkpoint/resume across a job boundary
  continues the same curve (A7 ``RungRun``).

Glue, not a second implementation: A7 ``control/src/rung.rs`` (rung-0
step loop, checkpoint/resume) and I1 ``control/src/scale.rs`` (ladder
fit). ``control/`` has no Python bindings; a host analog of the same
protocol is allowed. Must not import ``tests/``.
"""

from __future__ import annotations

from typing import TypedDict

# Spec 16.2: 8 H200. Template ``v5.sh`` passes ``gpus``.
DEFAULT_GPUS = 8
# Spec 6 rung 0: 0.1B active / 1B total / 20B tokens. Analog uses a
# tiny budget; these are the ladder numbers the fit is judged against.
RUNG_0_ACTIVE_PARAMS = 100_000_000
RUNG_0_TOTAL_PARAMS = 1_000_000_000
RUNG_0_TOKENS = 20_000_000_000


class V5Result(TypedDict):
    loss_curve_matches_ladder: bool
    checkpoint_resume_ok: bool


class V5Error(Exception):
    """Bad V5 inputs (gpus < 1) or a missing / failed backend."""


def run_v5(*, gpus: int) -> V5Result:
    """Run the V5 rung-0 analog and report the two gates.

    ``gpus`` is the Slurm GPU count (template ``{{GPUS}}``). Spec V5 is 8
    H200. ``gpus >= 1`` is accepted so a CPU analog can run; that analog
    is not V5 verified.

    Analog (spec 16.2 / 6 / 15.5 A7):

    - Tiny token budget, frozen tokenizer hash, injected or trained losses.
    - Loss curve vs I1's small-scale fit (``predict`` / ``predict_held_out``).
    - Mid-run checkpoint, resume in a new ``RungRun``, continue; the
      resumed curve matches an uninterrupted twin.

    Returns a JSON-serializable ``V5Result``. Raises ``V5Error`` when
    ``gpus`` is invalid or the backend cannot produce a finite report.
    Does not write ``v5.json``; the template does that.
    """
    raise NotImplementedError("V5 rung-0 runner")
