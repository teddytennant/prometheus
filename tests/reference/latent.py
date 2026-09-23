"""Independent I5 reference: Stage-B latent math (spec 4.2, 4.3, 4.4).

Plain NumPy, slow and obvious. Does not import production ``model``, ``model.latent``,
``rl``, ``rl.loss``, ``train``, ``sglang_fork``, or anything under ``tests/``.
Dataclasses and constants are a local mirror so tests can compare field-by-field
without sharing code.

Formulas
--------
ponder: clip each λ to [HALT_EPS, 1-HALT_EPS], pin the last slot to 1.0, then
``p_m = λ_m Π_{j<m}(1-λ_j)``.

geometric prior: ``g_m = λ (1-λ)^m`` on ``{0 .. n-1}``, then renormalize.

halt_from_logits: sigmoid, ponder, ``expected_depth = sum_m p_m * m``.

halt_loss: ``-log(p[k] + HALT_EPS)`` with ``k = clip(teacher_steps, 0, K)``.

halt_kl: ``sum_m p_m (log(p_m+EPS) - log(g_m+EPS))``.

thought_decode_ce: mean ``-log_softmax(logits)[i, teacher_ids[i]]`` over mask != 0.

answer_ce_at_depths: per-row CE of a shared answer id.

stage_b: ``l_task + α l_traj + γ l_halt + β l_kl`` after ``validate_latent_config``.

jacobi: ``thoughts = update(thoughts)`` for ``n_sweeps`` whole-chunk updates.
Forward does not depend on ``truncated_sweeps``.

jacobi truncated grad: same full forward unroll. Backward treats the state
entering update index ``n_sweeps - truncated_sweeps`` (0-based) as a constant,
then differentiates only the last ``truncated_sweeps`` updates by central
differences. Earlier updates get no gradient. If that entrance is not the
original thoughts, ``d(sum(output))/d(thoughts)`` is exactly zero.

noisy latent: ``z = μ + clamp(σ) ⊙ ε`` and the independent 1-D Gaussian log-density
``-0.5 * sum[((z-μ)/s)^2 + 2 log s + log(2π)]``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

Array = Any  # numpy.ndarray; FP64 internally


class LatentError(ValueError):
    """A config, shape, or numeric invariant from spec 4.3 / 4.4 failed."""


CHUNK_MIN_THOUGHTS = 4
CHUNK_MAX_THOUGHTS = 64
COMPRESSION_TOKENS_PER_THOUGHT = 8
DEFAULT_JACOBI_SWEEPS = 4
TRUNCATED_SWEEPS = 2
TRUNCATED_RECURRENCE = 4
DEFAULT_MAX_THOUGHTS = 16
DEFAULT_ALPHA_TRAJ = 1.0
DEFAULT_GAMMA_HALT = 1.0
DEFAULT_BETA_KL = 0.01
DEFAULT_LAMBDA_PRIOR = 0.2
HALT_EPS = 1e-6
SIGMA_MIN = 1e-4
SIGMA_MAX = 0.1


@dataclass(frozen=True)
class LatentConfig:
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
    lambdas: Array
    p: Array
    expected_depth: float


@dataclass(frozen=True)
class StageBLoss:
    l_task: float
    l_traj: float
    l_halt: float
    l_kl: float
    expected_depth: float
    total: float


@dataclass(frozen=True)
class NoisyLatent:
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


def _sigmoid_1d(logits: np.ndarray) -> np.ndarray:
    out = np.empty(logits.shape[0], dtype=np.float64)
    for i, raw in enumerate(logits):
        x = float(raw)
        if x >= 0.0:
            z = math.exp(-x)
            out[i] = 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            out[i] = z / (1.0 + z)
    return out


def _clip_pin_lambdas(lambdas: np.ndarray) -> np.ndarray:
    lo = float(HALT_EPS)
    hi = 1.0 - float(HALT_EPS)
    out = np.empty(lambdas.shape[0], dtype=np.float64)
    for i, raw in enumerate(lambdas):
        x = float(raw)
        if x < lo:
            x = lo
        elif x > hi:
            x = hi
        out[i] = x
    out[-1] = 1.0
    return out


def _ponder_from_clipped(lam: np.ndarray) -> np.ndarray:
    n = int(lam.shape[0])
    p = np.empty(n, dtype=np.float64)
    survival = 1.0
    for m in range(n):
        p[m] = float(lam[m]) * survival
        survival *= 1.0 - float(lam[m])
    return p


def _log_softmax_row(row: np.ndarray) -> list[float]:
    values = [float(x) for x in row]
    peak = values[0]
    for x in values[1:]:
        if x > peak:
            peak = x
    total = 0.0
    for x in values:
        total += math.exp(x - peak)
    log_z = peak + math.log(total)
    return [x - log_z for x in values]


def _row_ce(row: np.ndarray, label: int) -> float:
    log_probs = _log_softmax_row(row)
    return -log_probs[label]


def validate_latent_config(config: LatentConfig) -> None:
    """Raise LatentError if a 4.2 / 4.3 / 4.4 invariant fails.

    Check order is locked: max_thoughts → chunk bounds → weights (alpha, then
    gamma, then beta) → lambda_prior → jacobi_sweeps → truncated_sweeps → sigma.
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
    """p_m = λ_m Π_{j<m}(1-λ_j) with clip then last-slot pin to 1."""
    arr = _as_float_array(lambdas)
    if arr.ndim != 1:
        raise LatentError("lambdas must be 1-D")
    if arr.size < 1:
        raise LatentError("lambdas empty")
    for item in arr:
        if not math.isfinite(float(item)):
            raise LatentError("lambdas non-finite")
    lam = _clip_pin_lambdas(arr)
    return _ponder_from_clipped(lam)


def geometric_prior(n: int, lambda_prior: float) -> Array:
    """Truncated, renormalized Geometric(λ) over {0 .. n-1}."""
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
    """Sigmoid each logit, then ponder_distribution."""
    arr = _require_1d_finite_nonempty(halt_logits, "halt_logits")
    sig = _sigmoid_1d(arr)
    lam = _clip_pin_lambdas(sig)
    p = _ponder_from_clipped(lam)
    expected = 0.0
    for m in range(p.shape[0]):
        expected += float(p[m]) * float(m)
    return HaltOutput(lambdas=lam, p=p, expected_depth=float(expected))


def halt_loss(p: Array, teacher_steps: int) -> float:
    """-log(p[k] + HALT_EPS) with k clipped to 0..K."""
    arr = _require_1d_finite_nonempty(p, "p")
    k_max = int(arr.shape[0]) - 1
    k = int(teacher_steps)
    if k < 0:
        k = 0
    if k > k_max:
        k = k_max
    return float(-math.log(float(arr[k]) + float(HALT_EPS)))


def halt_kl(p: Array, lambda_prior: float) -> float:
    """KL(p || Geometric(lambda_prior)) with HALT_EPS inside the logs."""
    arr = _require_1d_finite_nonempty(p, "p")
    lam_p = float(lambda_prior)
    if not (0.0 < lam_p < 1.0):
        raise LatentError("lambda_prior must be in (0, 1)")
    g = geometric_prior(int(arr.shape[0]), lam_p)
    total = 0.0
    eps = float(HALT_EPS)
    for pm, gm in zip(arr, g, strict=True):
        p_m = float(pm)
        g_m = float(gm)
        total += p_m * (math.log(p_m + eps) - math.log(g_m + eps))
    return float(total)


def thought_decode_ce(
    logits: Array,
    teacher_ids: Array,
    mask: Array,
) -> float:
    """Mean CE of unmasked thought-decode rows. All-zero mask → 0.0."""
    logits_a = _as_float_array(logits)
    if logits_a.ndim != 2:
        raise LatentError("logits must be (n, vocab)")
    n = int(logits_a.shape[0])
    vocab = int(logits_a.shape[1])
    if n < 1:
        raise LatentError("empty logits")
    ids = np.asarray(teacher_ids)
    mask_a = np.asarray(mask)
    if ids.ndim != 1 or mask_a.ndim != 1:
        raise LatentError("teacher_ids and mask must be 1-D")
    if int(ids.shape[0]) != n or int(mask_a.shape[0]) != n:
        raise LatentError("length mismatch")
    labels: list[int] = []
    for raw in ids:
        label = int(raw)
        if label < 0 or label >= vocab:
            raise LatentError("teacher id outside [0, vocab)")
        labels.append(label)
    ces: list[float] = []
    for i in range(n):
        # 0/1 ints or bools: anything != 0 is kept (True == 1, False == 0).
        if mask_a[i] != 0:
            ces.append(_row_ce(logits_a[i], labels[i]))
    if not ces:
        return 0.0
    acc = 0.0
    for ce in ces:
        acc += ce
    return float(acc / float(len(ces)))


def answer_ce_at_depths(logits: Array, answer_id: int) -> Array:
    """Per-depth answer CE. logits is (K+1, vocab)."""
    logits_a = _as_float_array(logits)
    if logits_a.ndim != 2:
        raise LatentError("logits must be (K+1, vocab)")
    n = int(logits_a.shape[0])
    vocab = int(logits_a.shape[1])
    if n < 1:
        raise LatentError("empty logits")
    aid = int(answer_id)
    if aid < 0 or aid >= vocab:
        raise LatentError("answer_id outside [0, vocab)")
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = _row_ce(logits_a[i], aid)
    return out


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
    """Reverie Stage-B objective. Validates config first. No model forward."""
    validate_latent_config(config)
    halt = _as_float_array(halt_logits)
    if halt.ndim != 1 or halt.size < 1:
        raise LatentError("halt_logits must be 1-D with length K+1")
    k_plus = int(halt.shape[0])
    k = k_plus - 1
    answer = _as_float_array(answer_logits)
    if answer.ndim != 2 or int(answer.shape[0]) != k_plus:
        raise LatentError("answer_logits must have length K+1")
    thought = _as_float_array(thought_logits)
    if k == 0:
        l_traj = 0.0
    else:
        if thought.ndim != 2 or int(thought.shape[0]) != k:
            raise LatentError("thought_logits must have K rows")
        l_traj = thought_decode_ce(thought, teacher_ids, thought_mask)
    halt_out = halt_from_logits(halt)
    p = halt_out.p
    answer_ce = answer_ce_at_depths(answer, answer_id)
    l_task = 0.0
    for m in range(k_plus):
        l_task += float(p[m]) * float(answer_ce[m])
    l_halt = halt_loss(p, teacher_steps)
    l_kl = halt_kl(p, float(config.lambda_prior))
    total = (
        float(l_task)
        + float(config.alpha_traj) * float(l_traj)
        + float(config.gamma_halt) * float(l_halt)
        + float(config.beta_kl) * float(l_kl)
    )
    return StageBLoss(
        l_task=float(l_task),
        l_traj=float(l_traj),
        l_halt=float(l_halt),
        l_kl=float(l_kl),
        expected_depth=float(halt_out.expected_depth),
        total=float(total),
    )


def jacobi_sweeps(
    thoughts: Array,
    update: Callable[[Array], Array],
    n_sweeps: int,
    truncated_sweeps: int,
) -> Array:
    """Parallel fixed-point iteration over a thought chunk. CPU runs all sweeps."""
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


def _jacobi_update_vjp(
    update: Callable[[Array], Array],
    state: Array,
    cotangent: Array,
    eps: float,
) -> np.ndarray:
    """Central-difference VJP of one ``update`` call. Float64, elementwise."""
    base = np.array(np.asarray(state, dtype=np.float64), copy=True)
    cot = np.asarray(cotangent, dtype=np.float64)
    if cot.shape != base.shape:
        raise LatentError("cotangent shape mismatch")
    acc = np.zeros(base.shape, dtype=np.float64)
    step = float(eps)
    for idx in np.ndindex(base.shape):
        plus = np.array(base, copy=True)
        minus = np.array(base, copy=True)
        plus[idx] += step
        minus[idx] -= step
        y_plus = np.asarray(update(plus), dtype=np.float64)
        y_minus = np.asarray(update(minus), dtype=np.float64)
        if tuple(y_plus.shape) != tuple(base.shape) or tuple(y_minus.shape) != tuple(base.shape):
            raise LatentError("update changed shape")
        dy = (y_plus - y_minus) / (2.0 * step)
        acc[idx] = float(np.sum(dy * cot))
    return acc


def jacobi_sweeps_truncated_grad(
    thoughts: Array,
    update: Callable[[Array], Array],
    n_sweeps: int,
    truncated_sweeps: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """``d(sum(forward))/d(thoughts)`` with truncated backprop through sweeps.

    Forward matches ``jacobi_sweeps``: apply ``update`` ``n_sweeps`` times.
    ``truncated_sweeps`` does not change that forward value.

    Backward cut: before the update at index ``n_sweeps - truncated_sweeps``
    (0-based), the incoming state is a constant (stop-gradient). Only the last
    ``truncated_sweeps`` updates are differentiated. Earlier updates do not
    receive gradient. ``truncated_sweeps == n_sweeps`` is the full unroll.

    The derivative is with respect to the ``thoughts`` argument only. An
    ``update`` that does not close over ``thoughts`` therefore yields an exact
    zero when the stopped state is not ``thoughts`` itself.

    Does not call production ``model.latent``. Slow central differences, FP64.
    """
    arr = np.asarray(thoughts, dtype=np.float64)
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
    step = float(eps)
    if not math.isfinite(step) or step <= 0.0:
        raise LatentError("eps must be finite and > 0")

    # Full unroll. states[i] enters update i; states[sweeps] is the forward value.
    states: list[np.ndarray] = [np.array(arr, copy=True)]
    shape = states[0].shape
    for _ in range(sweeps):
        nxt = np.asarray(update(states[-1]), dtype=np.float64)
        if tuple(nxt.shape) != tuple(shape):
            raise LatentError("update changed shape")
        states.append(np.array(nxt, copy=True))

    cut = sweeps - truncated
    if cut > 0:
        # Entrance to the kept window is stop-gradient, not thoughts.
        return np.zeros(shape, dtype=np.float64)

    cot = np.ones(shape, dtype=np.float64)
    for step_i in range(sweeps - 1, -1, -1):
        cot = _jacobi_update_vjp(update, states[step_i], cot, step)
    return cot


def clamp_sigma(sigma: Array, config: LatentConfig) -> Array:
    """Per-element clamp into [sigma_min, sigma_max] after config validation."""
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
    """z = mu + clamp(sigma) * eps and the independent Gaussian log-density."""
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
