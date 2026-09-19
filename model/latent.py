"""Stage-B latent reasoning (spec 4.3, 4.4, 15.5 I5).

The 2-layer MLP + norm adapter is already in ``model`` (A1 / V1 shape).
This module is the rest of I5: PonderNet halt, thought-decode loss,
Jacobi-style parallel thought updates, and the noisy latent policy.

CPU / FP32 analog of V6. Does not run a rung, import ``rl.loss``,
import ``tests``, talk to a coordinator, or need a GPU. Arrays are
numpy or JAX; math is the Reverie formulas from spec 4.3 and the
Gaussian policy from spec 4.4.

``jax.jit`` of ``ponder_distribution``, ``halt_from_logits``,
``halt_loss``, ``halt_kl``, ``thought_decode_ce``,
``answer_ce_at_depths``, ``stage_b_loss``, ``jacobi_sweeps``,
``clamp_sigma``, and ``noisy_latent`` must match the eager call at 1e-5.
Traced arrays must not be converted with ``numpy.asarray`` or Python
``float()`` / ``int()`` on values. Python scalars (``n``,
``lambda_prior``, ``teacher_steps``, ``answer_id``, ``n_sweeps``,
``truncated_sweeps``) and ``LatentConfig`` stay host-side.
``geometric_prior`` is host-side (Python ``n``). ``jacobi_sweeps``
``update`` must be JAX-traceable. Do not jit ``validate_latent_config``
as an entry point.

IS weighting of latent log-probs is ``rl.loss``, not this module.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Array = Any  # numpy.ndarray or jax.Array; FP32


class LatentError(ValueError):
    """A config, shape, or numeric invariant from spec 4.3 / 4.4 failed."""


# spec 4.2: latent chunks of 4 to 64 thoughts.
CHUNK_MIN_THOUGHTS = 4
CHUNK_MAX_THOUGHTS = 64

# spec 4.3: compression target ~8 tokens per thought.
COMPRESSION_TOKENS_PER_THOUGHT = 8

# spec 4.3: truncated backprop through the last 2 Jacobi sweeps and
# last 4 recurrence iterations (Huginn). Recurrence truncation is A1's
# `r`; this module only owns the Jacobi sweep count.
DEFAULT_JACOBI_SWEEPS = 4
TRUNCATED_SWEEPS = 2
TRUNCATED_RECURRENCE = 4

# Reverie defaults (spec 4.3 names the terms; numbers match the small-scale
# implementation the spec cites).
DEFAULT_MAX_THOUGHTS = 16
DEFAULT_ALPHA_TRAJ = 1.0
DEFAULT_GAMMA_HALT = 1.0
DEFAULT_BETA_KL = 0.01
DEFAULT_LAMBDA_PRIOR = 0.2
HALT_EPS = 1e-6

# spec 4.4: per-dimension sigma kept small (clamped).
SIGMA_MIN = 1e-4
SIGMA_MAX = 0.1


@dataclass(frozen=True)
class LatentConfig:
    """Discrete Stage-B choices. Defaults are spec 4.2 / 4.3 / 4.4."""

    max_thoughts: int = DEFAULT_MAX_THOUGHTS
    chunk_min: int = CHUNK_MIN_THOUGHTS
    chunk_max: int = CHUNK_MAX_THOUGHTS
    alpha_traj: float = DEFAULT_ALPHA_TRAJ
    gamma_halt: float = DEFAULT_GAMMA_HALT
    beta_kl: float = DEFAULT_BETA_KL
    lambda_prior: float = DEFAULT_LAMBDA_PRIOR
    jacobi_sweeps: int = DEFAULT_JACOBI_SWEEPS
    truncated_sweeps: int = TRUNCATED_SWEEPS
    sigma_min: float = SIGMA_MIN
    sigma_max: float = SIGMA_MAX


@dataclass(frozen=True)
class HaltOutput:
    """PonderNet halt over depths ``0 .. K`` inclusive (K+1 slots).

    ``lambdas[-1]`` is pinned to 1 so ``p`` sums to 1. ``expected_depth``
    is ``sum_m p_m * m``.
    """

    lambdas: Array
    p: Array
    expected_depth: float


@dataclass(frozen=True)
class StageBLoss:
    """Four Reverie terms plus the weighted total (spec 4.3).

    ``l_task`` is answer CE weighted by ``p``. ``l_traj`` is thought-decode
    CE. ``l_halt`` is ``-log p[k]`` at the teacher's step count. ``l_kl``
    is ``KL(p || Geometric(lambda_prior))``.
    """

    l_task: float
    l_traj: float
    l_halt: float
    l_kl: float
    expected_depth: float
    total: float


jax.tree_util.register_dataclass(
    HaltOutput,
    data_fields=("lambdas", "p", "expected_depth"),
    meta_fields=(),
)
jax.tree_util.register_dataclass(
    StageBLoss,
    data_fields=("l_task", "l_traj", "l_halt", "l_kl", "expected_depth", "total"),
    meta_fields=(),
)


@dataclass(frozen=True)
class NoisyLatent:
    """One noisy thought (spec 4.4): ``z = mu + sigma * eps``.

    ``log_density`` is the sum of independent 1-D Gaussian log-densities
    at ``z``. Empty ``mu`` is not a valid thought.
    """

    mu: Array
    sigma: Array
    eps: Array
    z: Array
    log_density: float


def _as_float_array(value: Array) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _require_1d_finite_nonempty(value: Array, name: str) -> np.ndarray:
    arr = _as_float_array(value)
    if arr.ndim != 1:
        raise LatentError(f"{name} must be 1-D")
    if arr.size < 1:
        raise LatentError(f"{name} empty")
    for item in arr:
        if not math.isfinite(float(item)):
            raise LatentError(f"{name} non-finite")
    return arr


def _is_tracer(x: object) -> bool:
    return isinstance(x, jax.core.Tracer)


def _is_jax(x: object) -> bool:
    return isinstance(x, jax.Array)


def _xp(use_jax: bool) -> Any:
    return jnp if use_jax else np


def _as_float(value: Array, use_jax: bool) -> Array:
    if use_jax:
        return jnp.asarray(value, dtype=jnp.float32)
    return np.asarray(value, dtype=np.float64)


def _raise_if_bad_1d_concrete(value: Array, name: str) -> None:
    if _is_tracer(value):
        return
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise LatentError(f"{name} must be 1-D")
    if arr.size < 1:
        raise LatentError(f"{name} empty")
    if not np.all(np.isfinite(arr)):
        raise LatentError(f"{name} non-finite")


def _cast_1d_finite_nonempty(value: Array, name: str) -> Array:
    _raise_if_bad_1d_concrete(value, name)
    arr = _as_float(value, _is_jax(value))
    if arr.ndim != 1:
        raise LatentError(f"{name} must be 1-D")
    if int(arr.shape[0]) < 1:
        raise LatentError(f"{name} empty")
    return arr


def _raise_if_ids_oob(ids: Array, vocab: int) -> None:
    if _is_tracer(ids):
        return
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size > 0 and (np.any(arr < 0) or np.any(arr >= vocab)):
        raise LatentError("teacher id outside [0, vocab)")


def _stable_sigmoid(x: Array) -> Array:
    use_jax = _is_jax(x)
    xp = _xp(use_jax)
    dtype = x.dtype
    lo = xp.asarray(-80.0, dtype=dtype)
    hi = xp.asarray(80.0, dtype=dtype)
    one = xp.asarray(1.0, dtype=dtype)
    x_c = xp.clip(x, lo, hi)
    return xp.where(x_c >= 0, one / (one + xp.exp(-x_c)), xp.exp(x_c) / (one + xp.exp(x_c)))


def _clip_pin_ponder(lam: Array) -> tuple[Array, Array]:
    use_jax = _is_jax(lam)
    xp = _xp(use_jax)
    dtype = lam.dtype
    eps = xp.asarray(HALT_EPS, dtype=dtype)
    one = xp.asarray(1.0, dtype=dtype)
    n = lam.shape[0]
    clipped = xp.clip(lam, eps, one - eps)
    clipped = xp.where(xp.arange(n) == (n - 1), one, clipped)
    stay = one - clipped
    survival = xp.concatenate([xp.ones((1,), dtype=dtype), xp.cumprod(stay[:-1])])
    return clipped, clipped * survival


def _log_softmax(logits: Array) -> Array:
    xp = _xp(_is_jax(logits))
    shifted = logits - xp.max(logits, axis=-1, keepdims=True)
    log_z = xp.log(xp.sum(xp.exp(shifted), axis=-1, keepdims=True))
    return shifted - log_z


def validate_latent_config(config: LatentConfig) -> None:
    """Raise ``LatentError`` if a 4.2 / 4.3 / 4.4 invariant fails.

    Check order: max_thoughts → chunk bounds → weights finite and
    non-negative → lambda_prior in (0, 1) → jacobi_sweeps ≥ 1 →
    truncated_sweeps in 1..=jacobi_sweeps → 0 < sigma_min ≤ sigma_max.
    ``max_thoughts`` must be ≥ 0. ``chunk_min``/``chunk_max`` must sit
    inside ``CHUNK_MIN_THOUGHTS..=CHUNK_MAX_THOUGHTS`` with min ≤ max.
    """
    if int(config.max_thoughts) < 0:
        raise LatentError("max_thoughts must be >= 0")

    chunk_min = int(config.chunk_min)
    chunk_max = int(config.chunk_max)
    if (
        chunk_min < CHUNK_MIN_THOUGHTS
        or chunk_max > CHUNK_MAX_THOUGHTS
        or chunk_min > chunk_max
    ):
        raise LatentError("chunk bounds invalid")

    alpha = float(config.alpha_traj)
    if (not math.isfinite(alpha)) or alpha < 0.0:
        raise LatentError("alpha_traj must be finite and non-negative")
    gamma = float(config.gamma_halt)
    if (not math.isfinite(gamma)) or gamma < 0.0:
        raise LatentError("gamma_halt must be finite and non-negative")
    beta = float(config.beta_kl)
    if (not math.isfinite(beta)) or beta < 0.0:
        raise LatentError("beta_kl must be finite and non-negative")

    lam_p = float(config.lambda_prior)
    if not (0.0 < lam_p < 1.0):
        raise LatentError("lambda_prior must be in (0, 1)")

    n_sweeps = int(config.jacobi_sweeps)
    if n_sweeps < 1:
        raise LatentError("jacobi_sweeps must be >= 1")

    truncated = int(config.truncated_sweeps)
    if truncated < 1 or truncated > n_sweeps:
        raise LatentError("truncated_sweeps must sit in 1..=jacobi_sweeps")

    sigma_min = float(config.sigma_min)
    sigma_max = float(config.sigma_max)
    if (
        (not math.isfinite(sigma_min))
        or (not math.isfinite(sigma_max))
        or sigma_min <= 0.0
        or sigma_max < sigma_min
    ):
        raise LatentError("sigma clamp bounds invalid")


def ponder_distribution(lambdas: Array) -> Array:
    """``p_m = λ_m Π_{j<m}(1-λ_j)`` with ``λ`` clipped to ``(HALT_EPS, 1-HALT_EPS)``
    and the last slot pinned to 1 so remaining mass lands there.

    ``lambdas`` is a 1-D vector of length K+1. Empty or non-1-D raises
    ``LatentError``. Non-finite entries raise ``LatentError``.

    ``jax.jit`` of this function must match the eager result at 1e-5.
    Must not convert traced ``lambdas`` with ``numpy.asarray`` or Python
    ``float()`` on entries.
    """
    arr = _cast_1d_finite_nonempty(lambdas, "lambdas")
    _lam, p = _clip_pin_ponder(arr)
    return p


def geometric_prior(n: int, lambda_prior: float) -> Array:
    """Truncated, renormalized Geometric(λ) over ``{0 .. n-1}``.

    ``g_m = λ (1-λ)^m``, then ``g / sum(g)``. ``n < 1`` or ``lambda_prior``
    not in (0, 1) raises ``LatentError``.
    """
    n_slots = int(n)
    lam_p = float(lambda_prior)
    if n_slots < 1:
        raise LatentError("n must be >= 1")
    if not (0.0 < lam_p < 1.0):
        raise LatentError("lambda_prior must be in (0, 1)")
    g = np.empty(n_slots, dtype=np.float64)
    stay = 1.0 - lam_p
    term = lam_p
    total = 0.0
    for m in range(n_slots):
        g[m] = term
        total += term
        term *= stay
    if total <= 0.0:
        raise LatentError("geometric prior mass vanished")
    for m in range(n_slots):
        g[m] = g[m] / total
    return g


def halt_from_logits(halt_logits: Array) -> HaltOutput:
    """Sigmoid each logit, then ``ponder_distribution``.

    ``halt_logits`` is 1-D, length K+1. Returns lambdas (post-pin), p, and
    expected depth as a Python float.

    ``jax.jit`` of this function must match the eager result at 1e-5.
    Must not convert traced ``halt_logits`` with ``numpy.asarray``.
    ``expected_depth`` may be a 0-d array under jit.
    """
    arr = _cast_1d_finite_nonempty(halt_logits, "halt_logits")
    lam, p = _clip_pin_ponder(_stable_sigmoid(arr))
    xp = _xp(_is_jax(p))
    depths = xp.arange(p.shape[0], dtype=p.dtype)
    expected = xp.sum(p * depths)
    return HaltOutput(lambdas=lam, p=p, expected_depth=expected)


def halt_loss(p: Array, teacher_steps: int) -> float:
    """``-log(p[k] + HALT_EPS)`` with ``k = clip(teacher_steps, 0, K)``.

    ``p`` must be 1-D and finite. ``teacher_steps`` may sit outside
    ``0..K``; it is clipped, not rejected.

    ``jax.jit`` of this function, with ``teacher_steps`` a Python int
    (closed over or ``static_argnums``), must match the eager result at
    1e-5. Must not convert traced ``p`` with ``numpy.asarray`` or Python
    ``float()`` on ``p[k]``.
    """
    arr = _cast_1d_finite_nonempty(p, "p")
    k_max = int(arr.shape[0]) - 1
    k = int(teacher_steps)
    if k < 0:
        k = 0
    if k > k_max:
        k = k_max
    xp = _xp(_is_jax(arr))
    eps = xp.asarray(HALT_EPS, dtype=arr.dtype)
    return -xp.log(arr[k] + eps)


def halt_kl(p: Array, lambda_prior: float) -> float:
    """``sum_m p_m (log(p_m + HALT_EPS) - log(g_m + HALT_EPS))``.

    ``g`` is ``geometric_prior(len(p), lambda_prior)``.

    ``jax.jit`` of this function, with ``lambda_prior`` a Python float
    (closed over or ``static_argnums``), must match the eager result at
    1e-5. Must not convert traced ``p`` with ``numpy.asarray`` or Python
    ``float()`` on entries.
    """
    arr = _cast_1d_finite_nonempty(p, "p")
    lam_p = float(lambda_prior)
    if not (0.0 < lam_p < 1.0):
        raise LatentError("lambda_prior must be in (0, 1)")
    g = geometric_prior(int(arr.shape[0]), lam_p)
    use_jax = _is_jax(arr)
    g_a = _as_float(g, use_jax)
    xp = _xp(use_jax)
    eps = xp.asarray(HALT_EPS, dtype=arr.dtype)
    return xp.sum(arr * (xp.log(arr + eps) - xp.log(g_a + eps)))


def thought_decode_ce(
    logits: Array,
    teacher_ids: Array,
    mask: Array,
) -> float:
    """Mean CE of each thought decoding to its teacher step (spec 4.3).

    ``logits`` is (n, vocab), ``teacher_ids`` and ``mask`` are (n,).
    Masked-out rows (mask == 0) do not enter the mean. If the mask is
    all zeros the result is 0.0. Length mismatch, empty logits, or a
    teacher id outside ``[0, vocab)`` raises ``LatentError``.

    ``jax.jit`` of this function must match the eager result at 1e-5.
    Must not convert traced ``logits``, ``teacher_ids``, or ``mask`` with
    ``numpy.asarray``. Empty mask returns 0 without a Python branch on a
    traced denom.
    """
    use_jax = _is_jax(logits) or _is_jax(teacher_ids) or _is_jax(mask)
    xp = _xp(use_jax)
    logits_a = _as_float(logits, use_jax)
    if logits_a.ndim != 2:
        raise LatentError("logits must be (n, vocab)")
    n = int(logits_a.shape[0])
    vocab = int(logits_a.shape[1])
    if n < 1:
        raise LatentError("empty logits")
    ids = (
        jnp.asarray(teacher_ids, dtype=jnp.int32)
        if use_jax
        else np.asarray(teacher_ids, dtype=np.int64)
    )
    mask_a = _as_float(mask, use_jax)
    if ids.ndim != 1 or mask_a.ndim != 1:
        raise LatentError("teacher_ids and mask must be 1-D")
    if int(ids.shape[0]) != n or int(mask_a.shape[0]) != n:
        raise LatentError("length mismatch")
    _raise_if_ids_oob(teacher_ids, vocab)
    log_p = _log_softmax(logits_a)
    last = vocab - 1 if vocab > 0 else 0
    safe = xp.clip(ids, 0, last)
    picked = xp.take_along_axis(log_p, safe[..., None], axis=-1)[..., 0]
    ce = -picked
    kept = (mask_a != 0).astype(logits_a.dtype)
    denom = xp.sum(kept)
    eps = xp.asarray(1e-12, dtype=logits_a.dtype)
    zero = xp.asarray(0.0, dtype=logits_a.dtype)
    mean = xp.sum(ce * kept) / xp.maximum(denom, eps)
    return xp.where(denom > 0, mean, zero)


def answer_ce_at_depths(logits: Array, answer_id: int) -> Array:
    """Per-depth answer CE. ``logits`` is (K+1, vocab). Returns (K+1,).

    ``answer_id`` outside ``[0, vocab)`` raises ``LatentError``.

    ``jax.jit`` of this function, with ``answer_id`` a Python int (closed
    over or ``static_argnums``), must match the eager result at 1e-5.
    Must not convert traced ``logits`` with ``numpy.asarray``.
    """
    logits_a = _as_float(logits, _is_jax(logits))
    if logits_a.ndim != 2:
        raise LatentError("logits must be (K+1, vocab)")
    n = int(logits_a.shape[0])
    vocab = int(logits_a.shape[1])
    if n < 1:
        raise LatentError("empty logits")
    aid = int(answer_id)
    if aid < 0 or aid >= vocab:
        raise LatentError("answer_id outside [0, vocab)")
    log_p = _log_softmax(logits_a)
    return (-log_p[:, aid]).astype(logits_a.dtype)


def stage_b_loss(
    halt_logits: Array,
    answer_logits: Array,
    thought_logits: Array,
    teacher_ids: Array,
    thought_mask: Array,
    answer_id: int,
    teacher_steps: int,
    config: LatentConfig,
) -> StageBLoss:
    """Reverie Stage-B objective (spec 4.3).

    ``l_task = sum_m p_m * CE(answer, logits_m)``
    ``l_traj = thought_decode_ce(...)``
    ``l_halt = -log p[k]``
    ``l_kl   = KL(p || Geometric(lambda_prior))``
    ``total  = l_task + alpha * l_traj + gamma * l_halt + beta * l_kl``

    ``halt_logits`` and ``answer_logits`` share length K+1.
    ``thought_logits`` is the K thought slots (not depth 0). Validates
    ``config`` first. Does not look up a model or an episode.

    ``jax.jit`` of this function, with ``answer_id`` / ``teacher_steps``
    Python ints and ``config`` host-side (closed over or static), must
    match the eager result at 1e-5. Must not convert traced arrays with
    ``numpy.asarray`` or Python ``float()`` on values.
    """
    validate_latent_config(config)
    use_jax = (
        _is_jax(halt_logits)
        or _is_jax(answer_logits)
        or _is_jax(thought_logits)
        or _is_jax(teacher_ids)
        or _is_jax(thought_mask)
    )
    xp = _xp(use_jax)
    halt = _as_float(halt_logits, use_jax)
    if halt.ndim != 1 or int(halt.shape[0]) < 1:
        raise LatentError("halt_logits must be 1-D with length K+1")
    k_plus = int(halt.shape[0])
    k = k_plus - 1
    answer = _as_float(answer_logits, use_jax)
    if answer.ndim != 2 or int(answer.shape[0]) != k_plus:
        raise LatentError("answer_logits must have length K+1")
    if k == 0:
        l_traj = xp.asarray(0.0, dtype=halt.dtype)
    else:
        thought = _as_float(thought_logits, use_jax)
        if thought.ndim != 2 or int(thought.shape[0]) != k:
            raise LatentError("thought_logits must have K rows")
        l_traj = thought_decode_ce(thought, teacher_ids, thought_mask)
    halt_out = halt_from_logits(halt)
    p = halt_out.p
    answer_ce = answer_ce_at_depths(answer, answer_id)
    l_task = xp.sum(p * answer_ce)
    l_halt = halt_loss(p, teacher_steps)
    l_kl = halt_kl(p, float(config.lambda_prior))
    total = (
        l_task
        + config.alpha_traj * l_traj
        + config.gamma_halt * l_halt
        + config.beta_kl * l_kl
    )
    return StageBLoss(
        l_task=l_task,
        l_traj=l_traj,
        l_halt=l_halt,
        l_kl=l_kl,
        expected_depth=halt_out.expected_depth,
        total=total,
    )


def jacobi_sweeps(
    thoughts: Array,
    update: Callable[[Array], Array],
    n_sweeps: int,
    truncated_sweeps: int,
) -> Array:
    """Parallel fixed-point iteration over a thought chunk (spec 4.3).

    ``thoughts`` is (n, d). Each sweep replaces the whole chunk with
    ``update(current)`` in one shot (no sequential unroll). ``n_sweeps``
    full updates run; the return value is the state after the last sweep.

    ``truncated_sweeps`` is recorded for the GPU path (backprop through
    the last N only). On CPU the forward is the full ``n_sweeps`` either
    way; ``truncated_sweeps`` must still sit in ``1..=n_sweeps``.

    ``n < 1``, ``n_sweeps < 1``, rank != 2, or a shape change from
    ``update`` raises ``LatentError``.

    ``jax.jit`` of this function, with ``n_sweeps`` and
    ``truncated_sweeps`` Python ints (closed over or
    ``static_argnums``) and a JAX-traceable ``update``, must match the
    eager result at 1e-5. Must not convert traced ``thoughts`` with
    ``numpy.asarray``.
    """
    arr = np.asarray(thoughts)
    if arr.ndim != 2:
        raise LatentError("thoughts rank != 2")
    n_rows = int(arr.shape[0])
    n_cols = int(arr.shape[1])
    if n_rows < 1 or n_cols < 1:
        raise LatentError("thoughts n>=1 and d>=1")
    sweeps = int(n_sweeps)
    if sweeps < 1:
        raise LatentError("n_sweeps must be >= 1")
    truncated = int(truncated_sweeps)
    if truncated < 1 or truncated > sweeps:
        raise LatentError("truncated_sweeps not in 1..=n_sweeps")
    current = np.array(arr, copy=True)
    shape = current.shape
    for _ in range(sweeps):
        current = np.asarray(update(current))
        if tuple(current.shape) != tuple(shape):
            raise LatentError("update changed shape")
    return current


def clamp_sigma(sigma: Array, config: LatentConfig) -> Array:
    """Per-element clamp into ``[sigma_min, sigma_max]``.

    Non-finite sigma raises ``LatentError``. Empty is rejected.

    ``jax.jit`` of this function, with ``config`` host-side, must match
    the eager result at 1e-5. Must not convert traced ``sigma`` with
    ``numpy.asarray`` or Python ``float()`` on entries.
    """
    arr = _as_float_array(sigma)
    if arr.size < 1:
        raise LatentError("sigma empty")
    for item in arr.reshape(-1):
        if not math.isfinite(float(item)):
            raise LatentError("sigma non-finite")
    validate_latent_config(config)
    lo = float(config.sigma_min)
    hi = float(config.sigma_max)
    out = np.empty(arr.shape, dtype=np.float64)
    flat_in = arr.reshape(-1)
    flat_out = out.reshape(-1)
    for i, raw in enumerate(flat_in):
        x = float(raw)
        if x < lo:
            x = lo
        elif x > hi:
            x = hi
        flat_out[i] = x
    return out


def noisy_latent(mu: Array, sigma: Array, eps: Array, config: LatentConfig) -> NoisyLatent:
    """``z = mu + clamp(sigma) * eps`` and the Gaussian log-density (spec 4.4).

    All three vectors 1-D, same length, finite. Empty raises ``LatentError``.
    ``log_density = -0.5 * sum[((z-mu)/s)^2 + 2 log s + log(2π)]`` on the
    clamped ``s``. Does not import ``rl.loss``.

    ``jax.jit`` of this function, with ``config`` host-side, must match
    the eager result at 1e-5. Must not convert traced ``mu`` / ``sigma``
    / ``eps`` with ``numpy.asarray`` or Python ``float()`` on entries.
    ``log_density`` may be a 0-d array under jit.
    """
    mu_a = _require_1d_finite_nonempty(mu, "mu")
    sigma_a = _require_1d_finite_nonempty(sigma, "sigma")
    eps_a = _require_1d_finite_nonempty(eps, "eps")
    if mu_a.shape[0] != sigma_a.shape[0] or mu_a.shape[0] != eps_a.shape[0]:
        raise LatentError("length mismatch")
    s = clamp_sigma(sigma_a, config)
    z = np.empty(mu_a.shape[0], dtype=np.float64)
    for i in range(mu_a.shape[0]):
        z[i] = float(mu_a[i]) + float(s[i]) * float(eps_a[i])
    log_two_pi = math.log(2.0 * math.pi)
    total = 0.0
    for i in range(z.shape[0]):
        si = float(s[i])
        centered = (float(z[i]) - float(mu_a[i])) / si
        total += centered * centered + 2.0 * math.log(si) + log_two_pi
    return NoisyLatent(
        mu=mu_a,
        sigma=s,
        eps=eps_a,
        z=z,
        log_density=float(-0.5 * total),
    )
