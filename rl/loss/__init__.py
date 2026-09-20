"""GSPO/DAPO combo, latent ratio, routing replay, truncated IS (spec 9.1, 9.2, 4.4, 15.5 I2).

CPU contract: sequence-level importance ratios, asymmetric clip-higher,
Dr. GRPO advantages (no length norm, no std), dynamic sampling, overlong
soft penalty, truncated IS on staleness k<=4, a split discrete/latent
ratio, and a routing table the trainer can force. Log-probs are inputs;
this module does not run the model, import ``sglang_fork``, import
``model``, or talk to a coordinator. Weight sync, parity halt, and the
65/35 rack split are the rest of I2 (coordinator), not this package.
No reference-model KL (spec 9.1 default).

``jax.jit`` of ``gaussian_log_density``, ``sequence_ratio``,
``latent_ratio``, ``truncated_is``, ``clip_higher``, and ``mean_center``
must match the eager call at 1e-5. Traced arrays must not be converted
with ``numpy.asarray`` or Python ``float()`` / ``int()`` on values, and
must not call ``math.log`` / ``math.exp`` / ``math.isfinite`` on a
tracer. Python scalars (``k``, ``tis_clip``, ``eps_low``, ``eps_high``)
stay host-side. Tuple length of log-probs, rewards, and Gaussian vectors
is static (a pytree of scalars) or a 1-D array. Do not jit
``gspo_dapo_loss`` as an entry point (``Group`` / ``Sample`` stay
Python). Do not jit ``make_config``, ``drop_zero_advantage_groups``,
``all_equal_reward``, ``staleness``, ``routing_table``, or
``overlong_soft_penalty`` as entry points.

Efficiency reward (9.3) stays in the rewards crate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class LossError(ValueError):
    """A group, ratio, routing table, or config violates a spec 9/4.4 invariant."""


# spec 9.2: a GRPO group of 16 shares one prompt.
GROUP_SIZE = 16

# spec 9.2: trainer consumes batches up to k=4 policy versions stale.
MAX_STALENESS = 4

# DAPO clip-higher (Yu et al.). Spec 9.1 names the method, not the epsilons.
# Defaults match the public DAPO pair so tests have a locked value.
DEFAULT_CLIP_EPS_LOW = 0.2
DEFAULT_CLIP_EPS_HIGH = 0.28


@dataclass(frozen=True)
class RoutingTrace:
    """Expert ids chosen for one token at one MoE layer (9.2 routing replay).

    Field names match I4 ``RoutingRecord``. This package does not import
    ``sglang_fork``.
    """

    token_index: int
    layer_index: int
    expert_ids: tuple[int, ...]


@dataclass(frozen=True)
class Sample:
    """One group member.

    Trainer log-probs are the policy-gradient terms. Rollout log-probs
    are the truncated-IS correction only (9.2). Empty latent tuples mean
    a discrete-only sample. ``n_tokens`` is visible output length for
    the overlong penalty; latent steps are not counted here.
    """

    token_logp_trainer: tuple[float, ...]
    token_logp_rollout: tuple[float, ...]
    latent_logp_trainer: tuple[float, ...]
    latent_logp_rollout: tuple[float, ...]
    reward: float
    n_tokens: int
    policy_version: int
    routing: tuple[RoutingTrace, ...]


@dataclass(frozen=True)
class Group:
    """One GRPO group. Members share a prompt. Size need not be GROUP_SIZE."""

    prompt_id: str
    samples: tuple[Sample, ...]


@dataclass(frozen=True)
class LossConfig:
    """Tunable pieces of 9.1/9.2. ``kl_coeff`` must stay 0.0 in this module."""

    clip_eps_low: float = DEFAULT_CLIP_EPS_LOW
    clip_eps_high: float = DEFAULT_CLIP_EPS_HIGH
    latent_clip_eps_low: float = DEFAULT_CLIP_EPS_LOW
    latent_clip_eps_high: float = DEFAULT_CLIP_EPS_HIGH
    loss_normalizer: float = 1.0
    overlong_cache: int = 0
    overlong_penalty: float = 0.0
    max_staleness: int = MAX_STALENESS
    tis_clip: float = 1.0
    kl_coeff: float = 0.0


@dataclass(frozen=True)
class LossBreakdown:
    """Per-call scalars. ``n_kept`` is 0 when dynamic sampling drops the group."""

    pg: float
    tis: float
    overlong: float
    latent: float
    n_kept: int
    total: float


def make_config(
    clip_eps_low: float = DEFAULT_CLIP_EPS_LOW,
    clip_eps_high: float = DEFAULT_CLIP_EPS_HIGH,
    latent_clip_eps_low: float = DEFAULT_CLIP_EPS_LOW,
    latent_clip_eps_high: float = DEFAULT_CLIP_EPS_HIGH,
    loss_normalizer: float = 1.0,
    overlong_cache: int = 0,
    overlong_penalty: float = 0.0,
    max_staleness: int = MAX_STALENESS,
    tis_clip: float = 1.0,
    kl_coeff: float = 0.0,
) -> LossConfig:
    """Validate and freeze a config.

    Raises LossError if any epsilon is negative, ``loss_normalizer`` is
    not finite and positive, ``overlong_cache`` is negative,
    ``overlong_penalty`` is negative, ``max_staleness`` is not in
    1 through ``MAX_STALENESS``, ``tis_clip`` is not finite and ``>= 1``, or
    ``kl_coeff`` is not 0.0 (reference KL is out of this module).
    """
    if clip_eps_low < 0.0 or clip_eps_high < 0.0:
        raise LossError("epsilon must not be negative")
    if latent_clip_eps_low < 0.0 or latent_clip_eps_high < 0.0:
        raise LossError("epsilon must not be negative")
    if not math.isfinite(loss_normalizer) or loss_normalizer <= 0.0:
        raise LossError("loss_normalizer must be finite and positive")
    if overlong_cache < 0:
        raise LossError("overlong_cache must not be negative")
    if overlong_penalty < 0.0:
        raise LossError("overlong_penalty must not be negative")
    if max_staleness < 1 or max_staleness > MAX_STALENESS:
        raise LossError("max_staleness must be in 1 through MAX_STALENESS")
    if not math.isfinite(tis_clip) or tis_clip < 1.0:
        raise LossError("tis_clip must be finite and >= 1")
    if kl_coeff != 0.0:
        raise LossError("kl_coeff must be 0.0")
    return LossConfig(
        clip_eps_low=clip_eps_low,
        clip_eps_high=clip_eps_high,
        latent_clip_eps_low=latent_clip_eps_low,
        latent_clip_eps_high=latent_clip_eps_high,
        loss_normalizer=loss_normalizer,
        overlong_cache=overlong_cache,
        overlong_penalty=overlong_penalty,
        max_staleness=max_staleness,
        tis_clip=tis_clip,
        kl_coeff=kl_coeff,
    )


def mean_center(rewards: tuple[float, ...]) -> tuple[float, ...]:
    """Dr. GRPO advantages: subtract the group mean. No std divide.

    Empty or a non-finite reward raises LossError.

    ``jax.jit`` of this function must match the eager result at 1e-5.
    Must not convert traced ``rewards`` with ``numpy.asarray`` or Python
    ``float()``. Tuple length is static.
    """
    if len(rewards) == 0:
        raise LossError("empty rewards")
    total = 0.0
    for reward in rewards:
        if not math.isfinite(reward):
            raise LossError("non-finite reward")
        total += reward
    mean = total / len(rewards)
    return tuple(reward - mean for reward in rewards)


def all_equal_reward(rewards: tuple[float, ...]) -> bool:
    """True iff every reward is exactly equal. Empty raises LossError.

    Dynamic sampling drops those groups (9.1).
    """
    if len(rewards) == 0:
        raise LossError("empty rewards")
    first = rewards[0]
    for reward in rewards:
        if reward != first:
            return False
    return True


def drop_zero_advantage_groups(groups: tuple[Group, ...]) -> tuple[Group, ...]:
    """Keep groups whose rewards are not all equal. Empty input is empty output.

    A group with no samples raises LossError.
    """
    kept: list[Group] = []
    for group in groups:
        if len(group.samples) == 0:
            raise LossError("group with no samples")
        rewards = tuple(sample.reward for sample in group.samples)
        if not all_equal_reward(rewards):
            kept.append(group)
    return tuple(kept)


def sequence_ratio(
    logp_new: tuple[float, ...],
    logp_old: tuple[float, ...],
) -> float:
    """GSPO sequence-level ratio: ``exp(sum(new - old))``.

    No divide by length (Dr. GRPO). Length mismatch, empty sequences, or
    a non-finite log-prob raises LossError.

    ``jax.jit`` of this function must match the eager result at 1e-5.
    Must not convert traced log-probs with ``numpy.asarray`` or Python
    ``float()``, and must not call ``math.exp`` / ``math.isfinite`` on a
    tracer. Tuple length is static.
    """
    if len(logp_new) == 0 or len(logp_old) == 0:
        raise LossError("empty log-probs")
    if len(logp_new) != len(logp_old):
        raise LossError("length mismatch")
    total = 0.0
    for new, old in zip(logp_new, logp_old):
        if not math.isfinite(new) or not math.isfinite(old):
            raise LossError("non-finite log-prob")
        total += new - old
    return math.exp(total)


def clip_higher(ratio: float, eps_low: float, eps_high: float) -> float:
    """DAPO clip-higher: clip ``ratio`` to ``[1 - eps_low, 1 + eps_high]``.

    Non-finite ratio or negative epsilons raise LossError.

    ``jax.jit`` of this function, with ``eps_low`` and ``eps_high`` Python
    floats (closed over or ``static_argnums``), must match the eager
    result at 1e-5. Must not convert a traced ``ratio`` with
    ``numpy.asarray`` or Python ``float()``.
    """
    if not math.isfinite(ratio):
        raise LossError("non-finite ratio")
    if eps_low < 0.0 or eps_high < 0.0:
        raise LossError("epsilon must not be negative")
    lo = 1.0 - eps_low
    hi = 1.0 + eps_high
    if ratio < lo:
        return lo
    if ratio > hi:
        return hi
    return ratio


def staleness(trainer_version: int, policy_version: int) -> int:
    """``trainer_version - policy_version``. Negative raises LossError."""
    gap = trainer_version - policy_version
    if gap < 0:
        raise LossError("negative staleness")
    return gap


def truncated_is(
    logp_trainer: tuple[float, ...],
    logp_rollout: tuple[float, ...],
    k: int,
    tis_clip: float,
) -> float:
    """Truncated IS on the staleness gap (9.2).

    ``k == 0`` returns 1.0 (on-policy; rollout log-probs unused).
    ``k > 0`` returns ``min(sequence_ratio(trainer, rollout), tis_clip)``.
    Negative ``k`` or ``tis_clip < 1`` raises LossError.

    ``jax.jit`` of this function, with ``k`` a Python int and ``tis_clip``
    a Python float (closed over or ``static_argnums``), must match the
    eager result at 1e-5. Must not convert traced log-probs with
    ``numpy.asarray`` or Python ``float()``. Do not Python-branch on a
    traced ratio when applying the TIS clip.
    """
    if k < 0:
        raise LossError("negative staleness k")
    if tis_clip < 1.0:
        raise LossError("tis_clip must be >= 1")
    if k == 0:
        return 1.0
    ratio = sequence_ratio(logp_trainer, logp_rollout)
    if ratio < tis_clip:
        return ratio
    return tis_clip


def overlong_soft_penalty(n_tokens: int, cache: int, penalty: float) -> float:
    """``0`` when ``n_tokens <= cache``, else ``penalty * (n_tokens - cache)``.

    Negative ``n_tokens``, ``cache``, or ``penalty`` raises LossError.
    """
    if n_tokens < 0 or cache < 0 or penalty < 0.0:
        raise LossError("negative overlong argument")
    if n_tokens <= cache:
        return 0.0
    return penalty * (n_tokens - cache)


def gaussian_log_density(
    z: tuple[float, ...],
    mu: tuple[float, ...],
    sigma: tuple[float, ...],
) -> float:
    """Sum of independent 1-D Gaussian log-densities (spec 4.4).

    ``log N(z; μ, σ) = -0.5 * Σ[((z-μ)/σ)^2 + 2 log σ + log(2π)]``.
    Length mismatch, empty, non-finite, or non-positive sigma raises LossError.

    ``jax.jit`` of this function must match the eager result at 1e-5.
    Must not convert traced ``z`` / ``mu`` / ``sigma`` with
    ``numpy.asarray`` or Python ``float()``, and must not call
    ``math.log`` / ``math.isfinite`` on a tracer. Tuple length is static.
    """
    if len(z) == 0 or len(mu) == 0 or len(sigma) == 0:
        raise LossError("empty gaussian inputs")
    if len(z) != len(mu) or len(z) != len(sigma):
        raise LossError("length mismatch")
    log_two_pi = math.log(2.0 * math.pi)
    total = 0.0
    for zi, mui, si in zip(z, mu, sigma):
        if not math.isfinite(zi) or not math.isfinite(mui) or not math.isfinite(si):
            raise LossError("non-finite gaussian input")
        if si <= 0.0:
            raise LossError("sigma must be positive")
        total += ((zi - mui) / si) ** 2 + 2.0 * math.log(si) + log_two_pi
    return -0.5 * total


def latent_ratio(
    logp_new: tuple[float, ...],
    logp_old: tuple[float, ...],
) -> float:
    """Latent half of the split ratio: ``exp(sum(new - old))``.

    Empty on both sides is 1.0 (discrete-only). One empty and one not,
    length mismatch, or a non-finite log-prob raises LossError.

    ``jax.jit`` of this function must match the eager result at 1e-5.
    Must not convert traced log-probs with ``numpy.asarray`` or Python
    ``float()``, and must not call ``math.exp`` on a tracer. Empty on
    both sides is a Python (static) length check, not a traced branch.
    Tuple length is static.
    """
    if len(logp_new) == 0 and len(logp_old) == 0:
        return 1.0
    if len(logp_new) == 0 or len(logp_old) == 0:
        raise LossError("one latent side empty")
    return sequence_ratio(logp_new, logp_old)


def routing_table(
    traces: tuple[RoutingTrace, ...],
    n_tokens: int,
    n_layers: int,
    top_k: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Dense table ``[token][layer] -> expert_ids`` for routing replay.

    Every ``(token_index, layer_index)`` in ``0..n_tokens`` × ``0..n_layers``
    must appear exactly once. Each ``expert_ids`` length equals ``top_k``,
    ids are non-negative. ``n_tokens``, ``n_layers``, or ``top_k`` not
    positive raises LossError.
    """
    if n_tokens <= 0 or n_layers <= 0 or top_k <= 0:
        raise LossError("n_tokens, n_layers, and top_k must be positive")
    cells: dict[tuple[int, int], tuple[int, ...]] = {}
    for trace in traces:
        key = (trace.token_index, trace.layer_index)
        if key in cells:
            raise LossError("duplicate (token, layer)")
        if trace.token_index < 0 or trace.token_index >= n_tokens:
            raise LossError("token_index out of range")
        if trace.layer_index < 0 or trace.layer_index >= n_layers:
            raise LossError("layer_index out of range")
        if len(trace.expert_ids) != top_k:
            raise LossError("expert_ids length must equal top_k")
        for expert_id in trace.expert_ids:
            if expert_id < 0:
                raise LossError("expert id must be non-negative")
        cells[key] = tuple(trace.expert_ids)
    table: list[tuple[tuple[int, ...], ...]] = []
    for token_index in range(n_tokens):
        row: list[tuple[int, ...]] = []
        for layer_index in range(n_layers):
            key = (token_index, layer_index)
            if key not in cells:
                raise LossError("missing (token, layer)")
            row.append(cells[key])
        table.append(tuple(row))
    return tuple(table)


def gspo_dapo_loss(
    group: Group,
    config: LossConfig,
    trainer_version: int,
) -> LossBreakdown:
    """One group's GSPO/DAPO loss.

    Drops the group (zeros, ``n_kept=0``) when every reward is equal.
    Otherwise mean-center rewards, sequence-clip the discrete ratio,
    latent-clip the latent ratio separately, multiply by truncated IS,
    add overlong, divide by ``config.loss_normalizer``.

    A sample with staleness ``> config.max_staleness``, mismatched
    trainer/rollout lengths, empty token log-probs, empty ``prompt_id``,
    or a group with no samples raises LossError. Does not apply a
    reference KL.
    """
    if len(group.samples) == 0:
        raise LossError("empty group")
    if group.prompt_id == "":
        raise LossError("empty prompt_id")
    gaps: list[int] = []
    for sample in group.samples:
        if len(sample.token_logp_trainer) == 0 or len(sample.token_logp_rollout) == 0:
            raise LossError("empty token log-probs")
        if len(sample.token_logp_trainer) != len(sample.token_logp_rollout):
            raise LossError("mismatched trainer/rollout lengths")
        if len(sample.latent_logp_trainer) != len(sample.latent_logp_rollout):
            raise LossError("mismatched trainer/rollout lengths")
        gap = staleness(trainer_version, sample.policy_version)
        if gap > config.max_staleness:
            raise LossError("staleness exceeds max_staleness")
        gaps.append(gap)
    rewards = tuple(sample.reward for sample in group.samples)
    if all_equal_reward(rewards):
        return LossBreakdown(pg=0.0, tis=0.0, overlong=0.0, latent=0.0, n_kept=0, total=0.0)
    advantages = mean_center(rewards)
    pg_sum = 0.0
    latent_sum = 0.0
    overlong_sum = 0.0
    tis_sum = 0.0
    for sample, advantage, gap in zip(group.samples, advantages, gaps):
        r_disc = sequence_ratio(sample.token_logp_trainer, sample.token_logp_rollout)
        c_disc = clip_higher(r_disc, config.clip_eps_low, config.clip_eps_high)
        if sample.latent_logp_trainer or sample.latent_logp_rollout:
            r_lat = latent_ratio(sample.latent_logp_trainer, sample.latent_logp_rollout)
            c_lat = clip_higher(
                r_lat, config.latent_clip_eps_low, config.latent_clip_eps_high
            )
            lat_term = c_lat * advantage
        else:
            lat_term = 0.0
        tis_w = truncated_is(
            sample.token_logp_trainer, sample.token_logp_rollout, gap, config.tis_clip
        )
        ol = overlong_soft_penalty(
            sample.n_tokens, config.overlong_cache, config.overlong_penalty
        )
        disc_term = c_disc * advantage
        pg_sum += disc_term * tis_w
        latent_sum += lat_term * tis_w
        overlong_sum += ol
        tis_sum += tis_w
    n_kept = len(group.samples)
    z = config.loss_normalizer
    pg = pg_sum / z
    latent = latent_sum / z
    overlong = overlong_sum / z
    tis = tis_sum / n_kept
    total = (pg_sum + latent_sum + overlong_sum) / z
    return LossBreakdown(
        pg=pg, tis=tis, overlong=overlong, latent=latent, n_kept=n_kept, total=total
    )
