"""Stage-B latent reasoning (spec 4.3, 4.4, 15.5 I5).

The 2-layer MLP + norm adapter is already in ``model`` (A1 / V1 shape).
This module is the rest of I5: PonderNet halt, thought-decode loss,
Jacobi-style parallel thought updates, and the noisy latent policy.

CPU / FP32 analog of V6. Does not run a rung, import ``rl.loss``,
import ``tests``, talk to a coordinator, or need a GPU. Arrays are
numpy or JAX; math is the Reverie formulas from spec 4.3 and the
Gaussian policy from spec 4.4.

IS weighting of latent log-probs is ``rl.loss``, not this module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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


def validate_latent_config(config: LatentConfig) -> None:
    """Raise ``LatentError`` if a 4.2 / 4.3 / 4.4 invariant fails.

    Check order: max_thoughts → chunk bounds → weights finite and
    non-negative → lambda_prior in (0, 1) → jacobi_sweeps ≥ 1 →
    truncated_sweeps in 1..=jacobi_sweeps → 0 < sigma_min ≤ sigma_max.
    ``max_thoughts`` must be ≥ 0. ``chunk_min``/``chunk_max`` must sit
    inside ``CHUNK_MIN_THOUGHTS..=CHUNK_MAX_THOUGHTS`` with min ≤ max.
    """
    raise NotImplementedError("I5: validate_latent_config")


def ponder_distribution(lambdas: Array) -> Array:
    """``p_m = λ_m Π_{j<m}(1-λ_j)`` with ``λ`` clipped to ``(HALT_EPS, 1-HALT_EPS)``
    and the last slot pinned to 1 so remaining mass lands there.

    ``lambdas`` is a 1-D vector of length K+1. Empty or non-1-D raises
    ``LatentError``. Non-finite entries raise ``LatentError``.
    """
    raise NotImplementedError("I5: ponder_distribution")


def geometric_prior(n: int, lambda_prior: float) -> Array:
    """Truncated, renormalized Geometric(λ) over ``{0 .. n-1}``.

    ``g_m = λ (1-λ)^m``, then ``g / sum(g)``. ``n < 1`` or ``lambda_prior``
    not in (0, 1) raises ``LatentError``.
    """
    raise NotImplementedError("I5: geometric_prior")


def halt_from_logits(halt_logits: Array) -> HaltOutput:
    """Sigmoid each logit, then ``ponder_distribution``.

    ``halt_logits`` is 1-D, length K+1. Returns lambdas (post-pin), p, and
    expected depth as a Python float.
    """
    raise NotImplementedError("I5: halt_from_logits")


def halt_loss(p: Array, teacher_steps: int) -> float:
    """``-log(p[k] + HALT_EPS)`` with ``k = clip(teacher_steps, 0, K)``.

    ``p`` must be 1-D and finite. ``teacher_steps`` may sit outside
    ``0..K``; it is clipped, not rejected.
    """
    raise NotImplementedError("I5: halt_loss")


def halt_kl(p: Array, lambda_prior: float) -> float:
    """``sum_m p_m (log(p_m + HALT_EPS) - log(g_m + HALT_EPS))``.

    ``g`` is ``geometric_prior(len(p), lambda_prior)``.
    """
    raise NotImplementedError("I5: halt_kl")


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
    """
    raise NotImplementedError("I5: thought_decode_ce")


def answer_ce_at_depths(logits: Array, answer_id: int) -> Array:
    """Per-depth answer CE. ``logits`` is (K+1, vocab). Returns (K+1,).

    ``answer_id`` outside ``[0, vocab)`` raises ``LatentError``.
    """
    raise NotImplementedError("I5: answer_ce_at_depths")


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
    """
    raise NotImplementedError("I5: stage_b_loss")


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
    """
    raise NotImplementedError("I5: jacobi_sweeps")


def clamp_sigma(sigma: Array, config: LatentConfig) -> Array:
    """Per-element clamp into ``[sigma_min, sigma_max]``.

    Non-finite sigma raises ``LatentError``. Empty is rejected.
    """
    raise NotImplementedError("I5: clamp_sigma")


def noisy_latent(mu: Array, sigma: Array, eps: Array, config: LatentConfig) -> NoisyLatent:
    """``z = mu + clamp(sigma) * eps`` and the Gaussian log-density (spec 4.4).

    All three vectors 1-D, same length, finite. Empty raises ``LatentError``.
    ``log_density = -0.5 * sum[((z-mu)/s)^2 + 2 log s + log(2π)]`` on the
    clamped ``s``. Does not import ``rl.loss``.
    """
    raise NotImplementedError("I5: noisy_latent")
