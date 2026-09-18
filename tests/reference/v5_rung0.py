"""Independent host V5 rung-0 protocol (spec 16.2 / 15.5 A7 + I1).

Slow and obvious. Does **not** import ``prometheus.verify.v5_rung0``, JAX,
torch, or the Rust crates. Production ``run_v5`` must not import this module;
tests import both.

V5 meaning
----------
``loss_curve_matches_ladder``
    Observed rung-0 losses vs I1 ``predict(active, tokens)`` from a fit on
    rungs 1–3. Rung 0 is **not** a fit point (``ScaleError.Rung0NotUsed``).
    Gate: every point is finite, ``> 0``, and strictly below
    ``predicted * (1 + STOP_RELATIVE_EXCESS)`` (I1 ``should_stop`` is ``>=``).
    Must not compare a curve to a copy of itself.
``checkpoint_resume_ok``
    A7 ``RungRun``: mid-run checkpoint, resume in a new run, continue. The
    resumed curve / tokens / done flag match an uninterrupted twin.

CPU analog: tiny token budget, tiny net, injected or trained losses. Passing
CPU tests is not V5 verified. GPU V5 is 8 H200; ``v5.json`` is written by
``verify/ncshare/templates/v5.sh``, not by ``run_v5``.

Analog constants the implementer must match
-------------------------------------------
Ladder identity (also on the production module)::

    RUNG_0_ACTIVE_PARAMS = 100_000_000
    RUNG_0_TOTAL_PARAMS = 1_000_000_000
    RUNG_0_TOKENS = 20_000_000_000
    DEFAULT_GPUS = 8

I1 law and F4 gate::

    FLOP_COEFF = 6.0          # C = 6 N D
    STOP_RELATIVE_EXCESS = 0.05

CPU analog bookkeeping (A7 tiny config)::

    ANALOG_TOKEN_BUDGET = 8
    ANALOG_TOKENS_PER_STEP = 2
    ANALOG_SEED = 7
    ANALOG_TOKENIZER_HASH = "sha256:test-tokenizer"
    ANALOG_ACTIVE_PARAMS = 8

Injected generating law for analog rungs 1–3 and the analog rung-0 curve::

    LAW_A = 12.0
    LAW_ALPHA = 0.5
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Array = np.ndarray

# ---------------------------------------------------------------------------
# Spec 6 / 16.2 ladder identity (literals so a wrong production const shows)
# ---------------------------------------------------------------------------

RUNG_0_ACTIVE_PARAMS = 100_000_000
RUNG_0_TOTAL_PARAMS = 1_000_000_000
RUNG_0_TOKENS = 20_000_000_000
RUNG_0_LADDER_GPUS = 64
DEFAULT_GPUS = 8

RUNG_1_ACTIVE_PARAMS = 1_000_000_000
RUNG_1_TOKENS = 200_000_000_000
RUNG_2_ACTIVE_PARAMS = 8_000_000_000
RUNG_2_TOKENS = 1_500_000_000_000
RUNG_3_ACTIVE_PARAMS = 40_000_000_000
RUNG_3_TOKENS = 6_000_000_000_000

FLOP_COEFF = 6.0
STOP_RELATIVE_EXCESS = 0.05
STOP_FRACTION = 0.10

# CPU analog (not 20B). Seed / hash match A7 ``tiny_rung0_config``.
ANALOG_TOKEN_BUDGET = 8
ANALOG_TOKENS_PER_STEP = 2
ANALOG_SEED = 7
ANALOG_TOKENIZER_HASH = "sha256:test-tokenizer"
ANALOG_ACTIVE_PARAMS = 8

# Injected Kaplan/Chinchilla-style law for analog observations.
LAW_A = 12.0
LAW_ALPHA = 0.5

# Tiny net (kernel-correctness analog). Small so FD is cheap and obvious.
TOY_VOCAB = 8
TOY_DIM = 4
TOY_BATCH = 2
TOY_SEQ = 4
TOY_SEED = 7
TOY_FD_EPS = 1e-3
TOY_GRAD_RTOL = 5e-2
TOY_GRAD_ATOL = 5e-3
TOY_LR = 0.5
TOY_STEPS = 4

# Two-point / OLS recovery tolerance (I1 tests use 1e-9).
FIT_REL_TOL = 1e-9

RUNG_0 = 0
RUNG_1 = 1
RUNG_2 = 2
RUNG_3 = 3


class ScaleError(Exception):
    """I1 analog error. ``code`` matches ``ScaleError`` variant names."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        super().__init__(message or code)


class RungError(Exception):
    """A7 analog error. ``code`` matches ``RungError`` variant names."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        super().__init__(message or code)


@dataclass(frozen=True)
class RungSpec:
    id: int
    active_params: int
    total_params: int
    tokens: int
    gpus: int


@dataclass(frozen=True)
class RungObservation:
    spec: RungSpec
    loss: float


@dataclass(frozen=True)
class ScalingFit:
    a: float
    alpha: float

    def predict(self, active_params: int, tokens: int) -> float:
        if active_params <= 0 or tokens <= 0:
            raise ScaleError("InvalidCompute")
        c = compute_c(active_params, tokens)
        pred = float(self.a) * (c ** (-float(self.alpha)))
        if not np.isfinite(pred):
            raise ScaleError("NonFinitePrediction")
        return pred


@dataclass(frozen=True)
class RungConfig:
    spec: RungSpec
    token_budget: int
    tokens_per_step: int
    tokenizer_hash: str
    seed: int


@dataclass(frozen=True)
class RungCheckpoint:
    step: int
    tokens_seen: int
    loss_curve: tuple[float, ...]
    tokenizer_hash: str
    seed: int
    token_budget: int
    tokens_per_step: int
    rung: int


@dataclass(frozen=True)
class RungStepReport:
    step: int
    tokens_seen: int
    loss: float
    done: bool


@dataclass
class RungRun:
    """A7 CPU analog: caller injects each step's loss. Rung 0 only."""

    config: RungConfig
    step: int = 0
    tokens_seen: int = 0
    loss_curve: list[float] = field(default_factory=list)

    @classmethod
    def new(cls, config: RungConfig) -> RungRun:
        validate_rung_config(config)
        return cls(config=config)

    def done(self) -> bool:
        return self.tokens_seen >= self.config.token_budget

    def remaining_tokens(self) -> int:
        rem = self.config.token_budget - self.tokens_seen
        return rem if rem > 0 else 0

    def step_once(self, loss: float) -> RungStepReport:
        if not np.isfinite(float(loss)):
            raise RungError("NonFiniteLoss")
        if self.done():
            raise RungError("AlreadyDone")
        remaining = self.config.token_budget - self.tokens_seen
        added = min(self.config.tokens_per_step, remaining)
        self.tokens_seen += added
        self.step += 1
        self.loss_curve.append(float(loss))
        return RungStepReport(
            step=self.step,
            tokens_seen=self.tokens_seen,
            loss=float(loss),
            done=self.done(),
        )

    def checkpoint(self) -> RungCheckpoint:
        return RungCheckpoint(
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
    def resume(cls, config: RungConfig, ckpt: RungCheckpoint) -> RungRun:
        validate_rung_config(config)
        if ckpt.tokenizer_hash != config.tokenizer_hash:
            raise RungError("TokenizerChanged")
        if (
            ckpt.seed != config.seed
            or ckpt.token_budget != config.token_budget
            or ckpt.tokens_per_step != config.tokens_per_step
            or ckpt.rung != config.spec.id
        ):
            raise RungError("CheckpointMismatch")
        if ckpt.tokens_seen > config.token_budget:
            raise RungError("ResumePastBudget")
        return cls(
            config=config,
            step=ckpt.step,
            tokens_seen=ckpt.tokens_seen,
            loss_curve=list(ckpt.loss_curve),
        )


def compute_c(active_params: int, tokens: int) -> float:
    """``C = 6 * N * D`` as float64. Cast before multiply (flagship overflows u64)."""
    return float(FLOP_COEFF) * float(active_params) * float(tokens)


def compute_flops(active_params: int, tokens: int) -> float:
    if active_params <= 0 or tokens <= 0:
        raise ScaleError("InvalidCompute")
    return compute_c(active_params, tokens)


def law_loss(a: float, alpha: float, active_params: int, tokens: int) -> float:
    c = compute_c(active_params, tokens)
    return float(a) * (c ** (-float(alpha)))


def rung_spec(rung_id: int) -> RungSpec:
    if rung_id == RUNG_0:
        return RungSpec(
            id=RUNG_0,
            active_params=RUNG_0_ACTIVE_PARAMS,
            total_params=RUNG_0_TOTAL_PARAMS,
            tokens=RUNG_0_TOKENS,
            gpus=RUNG_0_LADDER_GPUS,
        )
    if rung_id == RUNG_1:
        return RungSpec(
            id=RUNG_1,
            active_params=RUNG_1_ACTIVE_PARAMS,
            total_params=15_000_000_000,
            tokens=RUNG_1_TOKENS,
            gpus=1_000,
        )
    if rung_id == RUNG_2:
        return RungSpec(
            id=RUNG_2,
            active_params=RUNG_2_ACTIVE_PARAMS,
            total_params=120_000_000_000,
            tokens=RUNG_2_TOKENS,
            gpus=5_000,
        )
    if rung_id == RUNG_3:
        return RungSpec(
            id=RUNG_3,
            active_params=RUNG_3_ACTIVE_PARAMS,
            total_params=700_000_000_000,
            tokens=RUNG_3_TOKENS,
            gpus=15_000,
        )
    raise ScaleError("InvalidCompute", f"unknown rung id {rung_id}")


def observation(rung_id: int, active_params: int, tokens: int, loss: float) -> RungObservation:
    """Tiny synthetic observation. ``total_params`` / ``gpus`` unused by the law."""
    return RungObservation(
        spec=RungSpec(
            id=int(rung_id),
            active_params=int(active_params),
            total_params=999_999,
            tokens=int(tokens),
            gpus=42,
        ),
        loss=float(loss),
    )


def validate_observation(obs: RungObservation) -> None:
    if obs.spec.id == RUNG_0:
        raise ScaleError("Rung0NotUsed")
    if obs.spec.active_params <= 0 or obs.spec.tokens <= 0:
        raise ScaleError("InvalidCompute")
    if not np.isfinite(float(obs.loss)):
        raise ScaleError("NonFiniteLoss")
    if float(obs.loss) <= 0.0:
        raise ScaleError("NonPositiveLoss")


def fit(observations: Sequence[RungObservation]) -> ScalingFit:
    """OLS of ``ln L = ln A - alpha * ln C``. Rung 0 is never a fit point."""
    for obs in observations:
        validate_observation(obs)
    ids = [obs.spec.id for obs in observations]
    if len(set(ids)) != len(ids):
        raise ScaleError("DuplicateRung")
    if len(observations) < 2:
        raise ScaleError("NeedTwoRungs")

    ln_cs: list[float] = []
    ln_ls: list[float] = []
    cs: list[float] = []
    for obs in observations:
        c = compute_c(obs.spec.active_params, obs.spec.tokens)
        if not np.isfinite(c):
            raise ScaleError("DegenerateFit")
        ln_c = float(np.log(c))
        ln_l = float(np.log(float(obs.loss)))
        if not np.isfinite(ln_c) or not np.isfinite(ln_l):
            raise ScaleError("DegenerateFit")
        cs.append(c)
        ln_cs.append(ln_c)
        ln_ls.append(ln_l)
    for i, ci in enumerate(cs):
        for cj in cs[:i]:
            if ci == cj:
                raise ScaleError("DegenerateFit")

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
    if den == 0.0 or not np.isfinite(den):
        raise ScaleError("DegenerateFit")
    slope = num / den
    intercept = mean_y - slope * mean_x
    return ScalingFit(a=float(np.exp(intercept)), alpha=float(-slope))


def should_stop(predicted: float, observed: float) -> bool:
    """I1: stop if ``observed >= predicted * (1 + STOP_RELATIVE_EXCESS)``."""
    pred = float(predicted)
    obs = float(observed)
    if not np.isfinite(pred) or pred <= 0.0:
        raise ScaleError("InvalidPredicted")
    if not np.isfinite(obs):
        raise ScaleError("InvalidObserved")
    return bool(obs >= pred * (1.0 + float(STOP_RELATIVE_EXCESS)))


def curve_matches_ladder(
    observed: Sequence[float],
    token_counts: Sequence[int],
    fitted: ScalingFit,
    active_params: int,
    *,
    relative_excess: float = STOP_RELATIVE_EXCESS,
) -> bool:
    """True iff every point is below I1 ``predict``, not vs a copy of ``observed``."""
    if len(observed) != len(token_counts):
        raise ValueError("observed / token_counts length mismatch")
    if len(observed) == 0:
        return False
    for loss, tokens in zip(observed, token_counts, strict=True):
        pred = fitted.predict(active_params, int(tokens))
        if not np.isfinite(float(loss)) or float(loss) <= 0.0:
            return False
        # Match is the negation of should_stop, with the same 5% gate.
        if float(loss) >= pred * (1.0 + float(relative_excess)):
            return False
    return True


def analog_rung13_observations(
    *, a: float = LAW_A, alpha: float = LAW_ALPHA
) -> list[RungObservation]:
    """Tiny N, D on rungs 1–3 following ``L = A C^{-alpha}``. Distinct C."""
    # C = 6*2*4=48, 6*4*8=192, 6*8*16=768.
    points = ((RUNG_1, 2, 4), (RUNG_2, 4, 8), (RUNG_3, 8, 16))
    out: list[RungObservation] = []
    for rid, n, d in points:
        out.append(observation(rid, n, d, law_loss(a, alpha, n, d)))
    return out


def analog_rung0_injected_curve(
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
        losses.append(law_loss(a, alpha, active_params, seen))
    return losses, tokens


def ladder_rung13_observations(
    *, a: float = LAW_A, alpha: float = LAW_ALPHA
) -> list[RungObservation]:
    """Injected rungs 1–3 at spec-6 N, D (the numbers the ladder fit is judged on)."""
    rows = (
        (RUNG_1, RUNG_1_ACTIVE_PARAMS, RUNG_1_TOKENS),
        (RUNG_2, RUNG_2_ACTIVE_PARAMS, RUNG_2_TOKENS),
        (RUNG_3, RUNG_3_ACTIVE_PARAMS, RUNG_3_TOKENS),
    )
    return [observation(rid, n, d, law_loss(a, alpha, n, d)) for rid, n, d in rows]


def analog_rung0_config(
    token_budget: int = ANALOG_TOKEN_BUDGET,
    tokens_per_step: int = ANALOG_TOKENS_PER_STEP,
) -> RungConfig:
    return RungConfig(
        spec=rung_spec(RUNG_0),
        token_budget=int(token_budget),
        tokens_per_step=int(tokens_per_step),
        tokenizer_hash=ANALOG_TOKENIZER_HASH,
        seed=ANALOG_SEED,
    )


def validate_rung_config(config: RungConfig) -> None:
    if config.spec.id != RUNG_0:
        raise RungError("NotRung0")
    if config.token_budget <= 0 or config.token_budget > config.spec.tokens:
        raise RungError("InvalidBudget")
    if config.tokens_per_step <= 0:
        raise RungError("InvalidTokensPerStep")
    if config.tokenizer_hash == "":
        raise RungError("EmptyTokenizerHash")


def run_injected_curve(
    losses: Sequence[float], config: RungConfig | None = None
) -> RungRun:
    cfg = analog_rung0_config() if config is None else config
    run = RungRun.new(cfg)
    for loss in losses:
        run.step_once(float(loss))
    return run


def checkpoint_resume_continues(
    losses: Sequence[float],
    *,
    token_budget: int = ANALOG_TOKEN_BUDGET,
    tokens_per_step: int = ANALOG_TOKENS_PER_STEP,
    split_after: int = 2,
    config: RungConfig | None = None,
) -> bool:
    """Mid-run restore vs uninterrupted twin. Same curve / tokens / done."""
    cfg = (
        analog_rung0_config(token_budget, tokens_per_step) if config is None else config
    )
    twin = run_injected_curve(losses, cfg)
    job1 = RungRun.new(cfg)
    for loss in losses[:split_after]:
        job1.step_once(float(loss))
    ckpt = job1.checkpoint()
    job2 = RungRun.resume(cfg, ckpt)
    for loss in losses[split_after:]:
        job2.step_once(float(loss))
    if job2.loss_curve != twin.loss_curve:
        return False
    if job2.tokens_seen != twin.tokens_seen:
        return False
    if job2.step != twin.step:
        return False
    if job2.done() != twin.done():
        return False
    return True


def evaluate_v5_protocol() -> dict[str, Any]:
    """Independent analog of ``run_v5``: I1 fit on rungs 1–3 + A7 resume."""
    analog_obs = analog_rung13_observations()
    analog_fit = fit(analog_obs)
    losses, token_counts = analog_rung0_injected_curve()
    match = curve_matches_ladder(losses, token_counts, analog_fit, ANALOG_ACTIVE_PARAMS)
    resume_ok = checkpoint_resume_continues(losses, split_after=2)
    ladder_fit = fit(ladder_rung13_observations())
    ladder_pred = ladder_fit.predict(RUNG_0_ACTIVE_PARAMS, RUNG_0_TOKENS)
    predictions = [analog_fit.predict(ANALOG_ACTIVE_PARAMS, t) for t in token_counts]
    return {
        "loss_curve_matches_ladder": match,
        "checkpoint_resume_ok": resume_ok,
        "losses": losses,
        "token_counts": token_counts,
        "predictions": predictions,
        "fit_a": analog_fit.a,
        "fit_alpha": analog_fit.alpha,
        "ladder_rung0_prediction": ladder_pred,
        "ladder_c": compute_flops(RUNG_0_ACTIVE_PARAMS, RUNG_0_TOKENS),
    }


# ---------------------------------------------------------------------------
# Tiny net: kernel-correctness analog (embed + linear unembed + next-token CE)
# ---------------------------------------------------------------------------


def mean_cross_entropy(logits: Array, targets: Array) -> float:
    """Mean token CE. ``logits`` (..., V), ``targets`` (...) int ids in ``[0, V)``."""
    z = np.asarray(logits, dtype=np.float64)
    t = np.asarray(targets)
    if z.ndim < 1:
        raise ValueError("logits must have a vocab axis")
    if z.shape[:-1] != t.shape:
        raise ValueError(f"CE shape mismatch: logits {z.shape} vs targets {t.shape}")
    if z.size == 0:
        return 0.0
    v = int(z.shape[-1])
    flat = z.reshape(-1, v)
    idx = t.reshape(-1).astype(np.int64)
    if int(idx.min()) < 0 or int(idx.max()) >= v:
        raise ValueError("target id out of vocab")
    m = np.max(flat, axis=-1, keepdims=True)
    shifted = flat - m
    log_z = np.log(np.exp(shifted).sum(axis=-1)) + m.reshape(-1)
    nll = log_z - flat[np.arange(flat.shape[0]), idx]
    return float(np.mean(nll))


def toy_init(seed: int = TOY_SEED) -> dict[str, Array]:
    rng = np.random.default_rng(int(seed))
    scale = 0.02
    embed = rng.normal(0.0, scale, size=(TOY_VOCAB, TOY_DIM)).astype(np.float32)
    unembed = rng.normal(0.0, scale, size=(TOY_VOCAB, TOY_DIM)).astype(np.float32)
    return {"embed": embed, "unembed": unembed}


def toy_tokens(seed: int = TOY_SEED) -> Array:
    rng = np.random.default_rng(int(seed) + 1)
    return rng.integers(0, TOY_VOCAB, size=(TOY_BATCH, TOY_SEQ), dtype=np.int32)


def toy_forward(tokens: Array, params: Mapping[str, Array]) -> Array:
    tok = np.asarray(tokens)
    if tok.ndim != 2:
        raise ValueError("tokens must be (batch, seq)")
    embed = np.asarray(params["embed"], dtype=np.float32)
    unembed = np.asarray(params["unembed"], dtype=np.float32)
    hidden = embed[tok]
    logits = hidden @ unembed.T
    return np.asarray(logits, dtype=np.float32)


def toy_next_token_loss(tokens: Array, params: Mapping[str, Array]) -> float:
    tok = np.asarray(tokens)
    if tok.shape[1] < 2:
        raise ValueError("need seq >= 2 for next-token loss")
    logits = toy_forward(tok, params)
    return mean_cross_entropy(logits[:, :-1, :], tok[:, 1:])


def toy_reverse_mode_grads(tokens: Array, params: Mapping[str, Array]) -> dict[str, Array]:
    tok = np.asarray(tokens)
    embed = np.asarray(params["embed"], dtype=np.float64)
    unembed = np.asarray(params["unembed"], dtype=np.float64)
    batch, seq = int(tok.shape[0]), int(tok.shape[1])
    hidden = embed[tok]
    logits = hidden @ unembed.T
    n_tok = float(batch * (seq - 1))

    dlogits = np.zeros_like(logits, dtype=np.float64)
    for b in range(batch):
        for s in range(seq - 1):
            row = logits[b, s]
            row = row - np.max(row)
            ex = np.exp(row)
            prob = ex / ex.sum()
            target = int(tok[b, s + 1])
            dlogits[b, s] = prob / n_tok
            dlogits[b, s, target] -= 1.0 / n_tok

    d_unembed = np.zeros_like(unembed, dtype=np.float64)
    d_hidden = np.zeros_like(hidden, dtype=np.float64)
    for b in range(batch):
        for s in range(seq):
            d_unembed += np.outer(dlogits[b, s], hidden[b, s])
            d_hidden[b, s] = dlogits[b, s] @ unembed

    d_embed = np.zeros_like(embed, dtype=np.float64)
    for b in range(batch):
        for s in range(seq):
            d_embed[int(tok[b, s])] += d_hidden[b, s]

    return {
        "embed": d_embed.astype(np.float32),
        "unembed": d_unembed.astype(np.float32),
    }


def finite_difference_scalar(
    loss_fn: Callable[[], float],
    getter: Callable[[], float],
    setter: Callable[[float], None],
    *,
    eps: float = TOY_FD_EPS,
) -> float:
    theta = float(getter())
    setter(theta + float(eps))
    plus = float(loss_fn())
    setter(theta - float(eps))
    minus = float(loss_fn())
    setter(theta)
    return float((plus - minus) / (2.0 * float(eps)))


def toy_finite_diff_slice(
    tokens: Array,
    params: Mapping[str, Array],
    *,
    name: str = "unembed",
    coords: tuple[tuple[int, int], ...] = ((0, 0), (1, 2), (3, 1)),
    eps: float = TOY_FD_EPS,
) -> tuple[Array, Array]:
    work = {k: np.array(v, dtype=np.float32, copy=True) for k, v in params.items()}
    analytic = toy_reverse_mode_grads(tokens, work)
    numeric = np.zeros(len(coords), dtype=np.float32)
    analytic_slice = np.zeros(len(coords), dtype=np.float32)
    table = work[name]

    def loss_fn() -> float:
        return toy_next_token_loss(tokens, work)

    for i, (r, c) in enumerate(coords):
        analytic_slice[i] = np.float32(analytic[name][r, c])

        def getter(row: int = r, col: int = c) -> float:
            return float(table[row, col])

        def setter(val: float, row: int = r, col: int = c) -> None:
            table[row, col] = np.float32(val)

        numeric[i] = np.float32(finite_difference_scalar(loss_fn, getter, setter, eps=eps))
    return analytic_slice, numeric


def grad_match_ok(
    analytic: Array,
    finite_diff: Array,
    *,
    rtol: float = TOY_GRAD_RTOL,
    atol: float = TOY_GRAD_ATOL,
) -> bool:
    g = np.asarray(analytic, dtype=np.float64)
    h = np.asarray(finite_diff, dtype=np.float64)
    if g.shape != h.shape:
        raise ValueError(f"grad shape mismatch: {g.shape} vs {h.shape}")
    if g.size == 0:
        return True
    if not (np.isfinite(g).all() and np.isfinite(h).all()):
        return False
    scale = np.maximum(np.abs(g), np.abs(h))
    abs_err = np.abs(g - h)
    return bool((abs_err <= float(atol) + float(rtol) * scale).all())


def toy_sgd_step(
    tokens: Array, params: Mapping[str, Array], *, lr: float = TOY_LR
) -> dict[str, Array]:
    grads = toy_reverse_mode_grads(tokens, params)
    out: dict[str, Array] = {}
    for key, value in params.items():
        out[key] = (np.asarray(value, dtype=np.float32) - np.float32(lr) * grads[key]).astype(
            np.float32
        )
    return out


def toy_train_curve(
    tokens: Array,
    params: Mapping[str, Array],
    *,
    steps: int = TOY_STEPS,
    lr: float = TOY_LR,
) -> tuple[list[float], dict[str, Array]]:
    work = {k: np.array(v, dtype=np.float32, copy=True) for k, v in params.items()}
    curve: list[float] = []
    for _ in range(int(steps)):
        work = toy_sgd_step(tokens, work, lr=lr)
        curve.append(toy_next_token_loss(tokens, work))
    return curve, work


def toy_checkpoint_resume_continues(
    tokens: Array | None = None,
    params: Mapping[str, Array] | None = None,
    *,
    steps: int = TOY_STEPS,
    split_after: int = 2,
    lr: float = TOY_LR,
) -> bool:
    tok = toy_tokens() if tokens is None else tokens
    init = toy_init() if params is None else params
    twin_curve, _ = toy_train_curve(tok, init, steps=steps, lr=lr)
    work = {k: np.array(v, dtype=np.float32, copy=True) for k, v in init.items()}
    resumed: list[float] = []
    for _ in range(int(split_after)):
        work = toy_sgd_step(tok, work, lr=lr)
        resumed.append(toy_next_token_loss(tok, work))
    snap = {k: np.array(v, dtype=np.float32, copy=True) for k, v in work.items()}
    restored = {k: np.array(v, dtype=np.float32, copy=True) for k, v in snap.items()}
    for _ in range(int(steps) - int(split_after)):
        restored = toy_sgd_step(tok, restored, lr=lr)
        resumed.append(toy_next_token_loss(tok, restored))
    if len(resumed) != len(twin_curve):
        return False
    for a, b in zip(resumed, twin_curve, strict=True):
        if not np.isfinite(a) or not np.isfinite(b):
            return False
        if abs(float(a) - float(b)) > 1e-6:
            return False
    return True
