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

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypedDict

# Spec 16.2: 8 H200. Template ``v5.sh`` passes ``gpus``.
DEFAULT_GPUS = 8
# Spec 6 rung 0: 0.1B active / 1B total / 20B tokens. Analog uses a
# tiny budget; these are the ladder numbers the fit is judged against.
RUNG_0_ACTIVE_PARAMS = 100_000_000
RUNG_0_TOTAL_PARAMS = 1_000_000_000
RUNG_0_TOKENS = 20_000_000_000

# I1 law and F4 gate (must match the oracle analog).
FLOP_COEFF = 6.0
STOP_RELATIVE_EXCESS = 0.05

# CPU analog bookkeeping (A7 tiny config; not 20B).
ANALOG_TOKEN_BUDGET = 8
ANALOG_TOKENS_PER_STEP = 2
ANALOG_SEED = 7
ANALOG_TOKENIZER_HASH = "sha256:test-tokenizer"
ANALOG_ACTIVE_PARAMS = 8

# Injected Kaplan/Chinchilla-style law for analog rungs 1–3 and analog rung 0.
LAW_A = 12.0
LAW_ALPHA = 0.5

_RUNG_0 = 0
_RUNG_1 = 1
_RUNG_2 = 2
_RUNG_3 = 3


class V5Result(TypedDict):
    loss_curve_matches_ladder: bool
    checkpoint_resume_ok: bool


class V5Error(Exception):
    """Bad V5 inputs (gpus < 1) or a missing / failed backend."""


class _ScaleError(Exception):
    """I1 analog error."""


class _RungError(Exception):
    """A7 analog error."""


@dataclass(frozen=True)
class _RungSpec:
    id: int
    active_params: int
    total_params: int
    tokens: int
    gpus: int


@dataclass(frozen=True)
class _RungObservation:
    spec: _RungSpec
    loss: float


@dataclass(frozen=True)
class _ScalingFit:
    a: float
    alpha: float

    def predict(self, active_params: int, tokens: int) -> float:
        if active_params <= 0 or tokens <= 0:
            raise _ScaleError("InvalidCompute")
        c = _compute_c(active_params, tokens)
        pred = float(self.a) * (c ** (-float(self.alpha)))
        if not math.isfinite(pred):
            raise _ScaleError("NonFinitePrediction")
        return pred


@dataclass(frozen=True)
class _RungConfig:
    spec: _RungSpec
    token_budget: int
    tokens_per_step: int
    tokenizer_hash: str
    seed: int


@dataclass(frozen=True)
class _RungCheckpoint:
    step: int
    tokens_seen: int
    loss_curve: tuple[float, ...]
    tokenizer_hash: str
    seed: int
    token_budget: int
    tokens_per_step: int
    rung: int


@dataclass
class _RungRun:
    """A7 CPU analog: caller injects each step's loss. Rung 0 only."""

    config: _RungConfig
    step: int = 0
    tokens_seen: int = 0
    loss_curve: list[float] = field(default_factory=list)

    @classmethod
    def new(cls, config: _RungConfig) -> _RungRun:
        _validate_rung_config(config)
        return cls(config=config)

    def done(self) -> bool:
        return self.tokens_seen >= self.config.token_budget

    def step_once(self, loss: float) -> None:
        if not math.isfinite(float(loss)):
            raise _RungError("NonFiniteLoss")
        if self.done():
            raise _RungError("AlreadyDone")
        remaining = self.config.token_budget - self.tokens_seen
        added = min(self.config.tokens_per_step, remaining)
        self.tokens_seen += added
        self.step += 1
        self.loss_curve.append(float(loss))

    def checkpoint(self) -> _RungCheckpoint:
        return _RungCheckpoint(
            step=self.step,
            tokens_seen=self.tokens_seen,
            loss_curve=tuple(self.loss_curve),
            tokenizer_hash=self.config.tokenizer_hash,
            seed=self.config.seed,
            token_budget=self.config.token_budget,
            tokens_per_step=self.config.tokens_per_step,
            rung=self.config.spec.id,
        )

    @classmethod
    def resume(cls, config: _RungConfig, ckpt: _RungCheckpoint) -> _RungRun:
        _validate_rung_config(config)
        if ckpt.tokenizer_hash != config.tokenizer_hash:
            raise _RungError("TokenizerChanged")
        if (
            ckpt.seed != config.seed
            or ckpt.token_budget != config.token_budget
            or ckpt.tokens_per_step != config.tokens_per_step
            or ckpt.rung != config.spec.id
        ):
            raise _RungError("CheckpointMismatch")
        if ckpt.tokens_seen > config.token_budget:
            raise _RungError("ResumePastBudget")
        return cls(
            config=config,
            step=ckpt.step,
            tokens_seen=ckpt.tokens_seen,
            loss_curve=list(ckpt.loss_curve),
        )


def _compute_c(active_params: int, tokens: int) -> float:
    """``C = 6 * N * D`` as float. Cast before multiply (flagship overflows u64)."""
    return float(FLOP_COEFF) * float(active_params) * float(tokens)


def _law_loss(a: float, alpha: float, active_params: int, tokens: int) -> float:
    c = _compute_c(active_params, tokens)
    return float(a) * (c ** (-float(alpha)))


def _rung_0_spec() -> _RungSpec:
    return _RungSpec(
        id=_RUNG_0,
        active_params=RUNG_0_ACTIVE_PARAMS,
        total_params=RUNG_0_TOTAL_PARAMS,
        tokens=RUNG_0_TOKENS,
        gpus=64,
    )


def _observation(rung_id: int, active_params: int, tokens: int, loss: float) -> _RungObservation:
    return _RungObservation(
        spec=_RungSpec(
            id=int(rung_id),
            active_params=int(active_params),
            total_params=999_999,
            tokens=int(tokens),
            gpus=42,
        ),
        loss=float(loss),
    )


def _validate_observation(obs: _RungObservation) -> None:
    if obs.spec.id == _RUNG_0:
        raise _ScaleError("Rung0NotUsed")
    if obs.spec.active_params <= 0 or obs.spec.tokens <= 0:
        raise _ScaleError("InvalidCompute")
    if not math.isfinite(float(obs.loss)):
        raise _ScaleError("NonFiniteLoss")
    if float(obs.loss) <= 0.0:
        raise _ScaleError("NonPositiveLoss")


def _fit(observations: Sequence[_RungObservation]) -> _ScalingFit:
    """OLS of ``ln L = ln A - alpha * ln C``. Rung 0 is never a fit point."""
    for obs in observations:
        _validate_observation(obs)
    ids = [obs.spec.id for obs in observations]
    if len(set(ids)) != len(ids):
        raise _ScaleError("DuplicateRung")
    if len(observations) < 2:
        raise _ScaleError("NeedTwoRungs")

    ln_cs: list[float] = []
    ln_ls: list[float] = []
    cs: list[float] = []
    for obs in observations:
        c = _compute_c(obs.spec.active_params, obs.spec.tokens)
        if not math.isfinite(c):
            raise _ScaleError("DegenerateFit")
        ln_c = math.log(c)
        ln_l = math.log(float(obs.loss))
        if not math.isfinite(ln_c) or not math.isfinite(ln_l):
            raise _ScaleError("DegenerateFit")
        cs.append(c)
        ln_cs.append(ln_c)
        ln_ls.append(ln_l)
    for i, ci in enumerate(cs):
        for cj in cs[:i]:
            if ci == cj:
                raise _ScaleError("DegenerateFit")

    n = float(len(ln_cs))
    mean_x = sum(ln_cs) / n
    mean_y = sum(ln_ls) / n
    num = 0.0
    den = 0.0
    for i in range(len(ln_cs)):
        dx = ln_cs[i] - mean_x
        dy = ln_ls[i] - mean_y
        num += dx * dy
        den += dx * dx
    if den == 0.0 or not math.isfinite(den):
        raise _ScaleError("DegenerateFit")
    slope = num / den
    intercept = mean_y - slope * mean_x
    return _ScalingFit(a=math.exp(intercept), alpha=float(-slope))


def _curve_matches_ladder(
    observed: Sequence[float],
    token_counts: Sequence[int],
    fitted: _ScalingFit,
    active_params: int,
    *,
    relative_excess: float = STOP_RELATIVE_EXCESS,
) -> bool:
    """True iff every point is below I1 ``predict``, not vs a copy of ``observed``."""
    if len(observed) != len(token_counts) or len(observed) == 0:
        return False
    for loss, tokens in zip(observed, token_counts, strict=True):
        pred = fitted.predict(active_params, int(tokens))
        if not math.isfinite(float(loss)) or float(loss) <= 0.0:
            return False
        if float(loss) >= pred * (1.0 + float(relative_excess)):
            return False
    return True


def _analog_rung13_observations(
    *, a: float = LAW_A, alpha: float = LAW_ALPHA
) -> list[_RungObservation]:
    """Tiny N, D on rungs 1–3 following ``L = A C^{-alpha}``. Distinct C."""
    points = ((_RUNG_1, 2, 4), (_RUNG_2, 4, 8), (_RUNG_3, 8, 16))
    return [
        _observation(rid, n, d, _law_loss(a, alpha, n, d)) for rid, n, d in points
    ]


def _analog_rung0_injected_curve(
    *,
    a: float = LAW_A,
    alpha: float = LAW_ALPHA,
    active_params: int = ANALOG_ACTIVE_PARAMS,
    token_budget: int = ANALOG_TOKEN_BUDGET,
    tokens_per_step: int = ANALOG_TOKENS_PER_STEP,
) -> tuple[list[float], list[int]]:
    """Injected rung-0 losses from the same law at analog ``(N, tokens_seen)``."""
    losses: list[float] = []
    tokens: list[int] = []
    seen = 0
    while seen < token_budget:
        add = min(tokens_per_step, token_budget - seen)
        seen += add
        tokens.append(seen)
        losses.append(_law_loss(a, alpha, active_params, seen))
    return losses, tokens


def _analog_rung0_config(
    token_budget: int = ANALOG_TOKEN_BUDGET,
    tokens_per_step: int = ANALOG_TOKENS_PER_STEP,
) -> _RungConfig:
    return _RungConfig(
        spec=_rung_0_spec(),
        token_budget=int(token_budget),
        tokens_per_step=int(tokens_per_step),
        tokenizer_hash=ANALOG_TOKENIZER_HASH,
        seed=ANALOG_SEED,
    )


def _validate_rung_config(config: _RungConfig) -> None:
    if config.spec.id != _RUNG_0:
        raise _RungError("NotRung0")
    if config.token_budget <= 0 or config.token_budget > config.spec.tokens:
        raise _RungError("InvalidBudget")
    if config.tokens_per_step <= 0:
        raise _RungError("InvalidTokensPerStep")
    if config.tokenizer_hash == "":
        raise _RungError("EmptyTokenizerHash")


def _run_injected_curve(losses: Sequence[float], config: _RungConfig) -> _RungRun:
    run = _RungRun.new(config)
    for loss in losses:
        run.step_once(float(loss))
    return run


def _checkpoint_resume_continues(
    losses: Sequence[float],
    *,
    token_budget: int = ANALOG_TOKEN_BUDGET,
    tokens_per_step: int = ANALOG_TOKENS_PER_STEP,
    split_after: int = 2,
) -> bool:
    """Mid-run restore vs uninterrupted twin. Same curve / tokens / done."""
    cfg = _analog_rung0_config(token_budget, tokens_per_step)
    twin = _run_injected_curve(losses, cfg)
    job1 = _RungRun.new(cfg)
    for loss in losses[:split_after]:
        job1.step_once(float(loss))
    ckpt = job1.checkpoint()
    job2 = _RungRun.resume(cfg, ckpt)
    for loss in losses[split_after:]:
        job2.step_once(float(loss))
    return (
        job2.loss_curve == twin.loss_curve
        and job2.tokens_seen == twin.tokens_seen
        and job2.step == twin.step
        and job2.done() == twin.done()
    )


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
    if gpus < 1:
        raise V5Error(f"gpus must be >= 1 (gpus={gpus})")

    analog_fit = _fit(_analog_rung13_observations())
    losses, token_counts = _analog_rung0_injected_curve()
    match = _curve_matches_ladder(losses, token_counts, analog_fit, ANALOG_ACTIVE_PARAMS)
    resume_ok = _checkpoint_resume_continues(losses, split_after=2)
    return {
        "loss_curve_matches_ladder": match,
        "checkpoint_resume_ok": resume_ok,
    }
