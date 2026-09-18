"""GSPO/DAPO combo, latent ratio, routing replay, truncated IS (spec 9.1, 9.2, 4.4, 15.5 I2).

CPU contract: sequence-level importance ratios, asymmetric clip-higher,
Dr. GRPO advantages (no length norm, no std), dynamic sampling, overlong
soft penalty, truncated IS on staleness k<=4, a split discrete/latent
ratio, and a routing table the trainer can force. Log-probs are inputs;
this module does not run the model, import ``sglang_fork``, import
``model``, or talk to a coordinator. Weight sync, parity halt, and the
65/35 rack split are the rest of I2 (coordinator), not this package.
No reference-model KL (spec 9.1 default). JAX is a later GPU path.

Efficiency reward (9.3) stays in the rewards crate.
"""

from __future__ import annotations

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
    raise NotImplementedError("I2 make_config")


def mean_center(rewards: tuple[float, ...]) -> tuple[float, ...]:
    """Dr. GRPO advantages: subtract the group mean. No std divide.

    Empty or a non-finite reward raises LossError.
    """
    raise NotImplementedError("I2 mean_center")


def all_equal_reward(rewards: tuple[float, ...]) -> bool:
    """True iff every reward is exactly equal. Empty raises LossError.

    Dynamic sampling drops those groups (9.1).
    """
    raise NotImplementedError("I2 all_equal_reward")


def drop_zero_advantage_groups(groups: tuple[Group, ...]) -> tuple[Group, ...]:
    """Keep groups whose rewards are not all equal. Empty input is empty output.

    A group with no samples raises LossError.
    """
    raise NotImplementedError("I2 drop_zero_advantage_groups")


def sequence_ratio(
    logp_new: tuple[float, ...],
    logp_old: tuple[float, ...],
) -> float:
    """GSPO sequence-level ratio: ``exp(sum(new - old))``.

    No divide by length (Dr. GRPO). Length mismatch, empty sequences, or
    a non-finite log-prob raises LossError.
    """
    raise NotImplementedError("I2 sequence_ratio")


def clip_higher(ratio: float, eps_low: float, eps_high: float) -> float:
    """DAPO clip-higher: clip ``ratio`` to ``[1 - eps_low, 1 + eps_high]``.

    Non-finite ratio or negative epsilons raise LossError.
    """
    raise NotImplementedError("I2 clip_higher")


def staleness(trainer_version: int, policy_version: int) -> int:
    """``trainer_version - policy_version``. Negative raises LossError."""
    raise NotImplementedError("I2 staleness")


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
    """
    raise NotImplementedError("I2 truncated_is")


def overlong_soft_penalty(n_tokens: int, cache: int, penalty: float) -> float:
    """``0`` when ``n_tokens <= cache``, else ``penalty * (n_tokens - cache)``.

    Negative ``n_tokens``, ``cache``, or ``penalty`` raises LossError.
    """
    raise NotImplementedError("I2 overlong_soft_penalty")


def gaussian_log_density(
    z: tuple[float, ...],
    mu: tuple[float, ...],
    sigma: tuple[float, ...],
) -> float:
    """Sum of independent 1-D Gaussian log-densities (spec 4.4).

    ``log N(z; μ, σ) = -0.5 * Σ[((z-μ)/σ)^2 + 2 log σ + log(2π)]``.
    Length mismatch, empty, non-finite, or non-positive sigma raises LossError.
    """
    raise NotImplementedError("I2 gaussian_log_density")


def latent_ratio(
    logp_new: tuple[float, ...],
    logp_old: tuple[float, ...],
) -> float:
    """Latent half of the split ratio: ``exp(sum(new - old))``.

    Empty on both sides is 1.0 (discrete-only). One empty and one not,
    length mismatch, or a non-finite log-prob raises LossError.
    """
    raise NotImplementedError("I2 latent_ratio")


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
    raise NotImplementedError("I2 routing_table")


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
    raise NotImplementedError("I2 gspo_dapo_loss")
