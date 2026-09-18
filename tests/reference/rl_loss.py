"""Independent I2 reference: GSPO/DAPO loss math (spec 9.1, 9.2, 4.4).

Plain Python, slow and obvious. Does not import production ``rl``, ``rl.loss``,
``model``, ``train``, ``sglang_fork``, or anything under ``tests/``. Dataclasses
and constants are a local mirror so tests can compare field-by-field without
sharing code.

Formulas
--------
mean_center: A_i = r_i - mean(r). No std, no length norm (Dr. GRPO).

sequence_ratio / latent_ratio: exp(sum(new - old)). No divide by length.
latent_ratio of two empty tuples is 1.0 (discrete-only).

clip_higher: clip ratio to [1 - eps_low, 1 + eps_high] (DAPO clip-higher).

truncated_is: k == 0 -> 1.0 (on-policy). k > 0 -> min(sequence_ratio, tis_clip).

overlong_soft_penalty: 0 if n_tokens <= cache, else penalty * (n_tokens - cache).

gaussian_log_density (independent 1-D, spec 4.4)::

    log N(z; μ, σ) = -0.5 * Σ[ ((z-μ)/σ)^2 + 2 log σ + log(2π) ]

gspo_dapo_loss, one group
~~~~~~~~~~~~~~~~~~~~~~~~~
Empty samples or empty prompt_id -> LossError.
All rewards exactly equal -> zeros, n_kept=0 (dynamic sampling).
Otherwise, per sample i with advantage A_i:

    r_disc = sequence_ratio(token_logp_trainer, token_logp_rollout)
    c_disc = clip_higher(r_disc, clip_eps_low, clip_eps_high)
    r_lat  = latent_ratio(latent_logp_trainer, latent_logp_rollout)
    c_lat  = clip_higher(r_lat, latent_clip_eps_low, latent_clip_eps_high)
    k      = staleness(trainer_version, policy_version)   # error if k > max
    tis_w  = truncated_is(token_logp_trainer, token_logp_rollout, k, tis_clip)
    ol     = overlong_soft_penalty(n_tokens, overlong_cache, overlong_penalty)

    disc_term = c_disc * A_i
    lat_term  = c_lat * A_i if the sample has any latent log-probs else 0.0
    pg_i      = disc_term * tis_w
    latent_i  = lat_term * tis_w
    total_i   = pg_i + latent_i + ol

Sum over samples, divide pg / latent / overlong / total by loss_normalizer.
``tis`` in the breakdown is the mean truncated-IS weight (1.0 on-policy).
Does not apply a reference KL. Routing traces are ignored by the scalar loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class LossError(ValueError):
    """A group, ratio, routing table, or config violates a spec 9/4.4 invariant."""


GROUP_SIZE = 16
MAX_STALENESS = 4
DEFAULT_CLIP_EPS_LOW = 0.2
DEFAULT_CLIP_EPS_HIGH = 0.28


@dataclass(frozen=True)
class RoutingTrace:
    token_index: int
    layer_index: int
    expert_ids: tuple[int, ...]


@dataclass(frozen=True)
class Sample:
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
    prompt_id: str
    samples: tuple[Sample, ...]


@dataclass(frozen=True)
class LossConfig:
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
    pg: float
    tis: float
    overlong: float
    latent: float
    n_kept: int
    total: float


def _is_finite(value: float) -> bool:
    return math.isfinite(float(value))


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
    """Validate and freeze a config."""
    for eps in (
        clip_eps_low,
        clip_eps_high,
        latent_clip_eps_low,
        latent_clip_eps_high,
    ):
        if float(eps) < 0.0:
            raise LossError("epsilon must not be negative")
    if not _is_finite(loss_normalizer) or float(loss_normalizer) <= 0.0:
        raise LossError("loss_normalizer must be finite and positive")
    if int(overlong_cache) < 0:
        raise LossError("overlong_cache must not be negative")
    if float(overlong_penalty) < 0.0:
        raise LossError("overlong_penalty must not be negative")
    if int(max_staleness) < 1 or int(max_staleness) > MAX_STALENESS:
        raise LossError("max_staleness must be in 1 through MAX_STALENESS")
    if not _is_finite(tis_clip) or float(tis_clip) < 1.0:
        raise LossError("tis_clip must be finite and >= 1")
    if float(kl_coeff) != 0.0:
        raise LossError("kl_coeff must be 0.0 (no reference KL)")
    return LossConfig(
        clip_eps_low=float(clip_eps_low),
        clip_eps_high=float(clip_eps_high),
        latent_clip_eps_low=float(latent_clip_eps_low),
        latent_clip_eps_high=float(latent_clip_eps_high),
        loss_normalizer=float(loss_normalizer),
        overlong_cache=int(overlong_cache),
        overlong_penalty=float(overlong_penalty),
        max_staleness=int(max_staleness),
        tis_clip=float(tis_clip),
        kl_coeff=float(kl_coeff),
    )


def mean_center(rewards: tuple[float, ...]) -> tuple[float, ...]:
    """Dr. GRPO advantages: subtract the group mean. No std divide."""
    if len(rewards) == 0:
        raise LossError("empty rewards")
    values: list[float] = []
    total = 0.0
    for reward in rewards:
        value = float(reward)
        if not _is_finite(value):
            raise LossError("non-finite reward")
        values.append(value)
        total += value
    mean = total / float(len(values))
    return tuple(value - mean for value in values)


def all_equal_reward(rewards: tuple[float, ...]) -> bool:
    """True iff every reward is exactly equal. Empty raises LossError."""
    if len(rewards) == 0:
        raise LossError("empty rewards")
    first = rewards[0]
    for reward in rewards[1:]:
        if reward != first:
            return False
    return True


def drop_zero_advantage_groups(groups: tuple[Group, ...]) -> tuple[Group, ...]:
    """Keep groups whose rewards are not all equal. Empty input is empty output."""
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
    """GSPO sequence-level ratio: exp(sum(new - old)). No divide by length."""
    if len(logp_new) == 0 or len(logp_old) == 0:
        raise LossError("empty log-probs")
    if len(logp_new) != len(logp_old):
        raise LossError("length mismatch")
    total = 0.0
    for new, old in zip(logp_new, logp_old, strict=True):
        new_f = float(new)
        old_f = float(old)
        if not _is_finite(new_f) or not _is_finite(old_f):
            raise LossError("non-finite log-prob")
        total += new_f - old_f
    return math.exp(total)


def clip_higher(ratio: float, eps_low: float, eps_high: float) -> float:
    """DAPO clip-higher: clip ratio to [1 - eps_low, 1 + eps_high]."""
    ratio_f = float(ratio)
    low = float(eps_low)
    high = float(eps_high)
    if not _is_finite(ratio_f):
        raise LossError("non-finite ratio")
    if low < 0.0 or high < 0.0:
        raise LossError("epsilon must not be negative")
    lo = 1.0 - low
    hi = 1.0 + high
    if ratio_f < lo:
        return lo
    if ratio_f > hi:
        return hi
    return ratio_f


def staleness(trainer_version: int, policy_version: int) -> int:
    """trainer_version - policy_version. Negative raises LossError."""
    gap = int(trainer_version) - int(policy_version)
    if gap < 0:
        raise LossError("negative staleness")
    return gap


def truncated_is(
    logp_trainer: tuple[float, ...],
    logp_rollout: tuple[float, ...],
    k: int,
    tis_clip: float,
) -> float:
    """Truncated IS on the staleness gap (9.2)."""
    if int(k) < 0:
        raise LossError("negative staleness k")
    if float(tis_clip) < 1.0:
        raise LossError("tis_clip must be >= 1")
    if int(k) == 0:
        return 1.0
    ratio = sequence_ratio(logp_trainer, logp_rollout)
    clip = float(tis_clip)
    if ratio < clip:
        return ratio
    return clip


def overlong_soft_penalty(n_tokens: int, cache: int, penalty: float) -> float:
    """0 when n_tokens <= cache, else penalty * (n_tokens - cache)."""
    n = int(n_tokens)
    c = int(cache)
    p = float(penalty)
    if n < 0 or c < 0 or p < 0.0:
        raise LossError("negative overlong argument")
    if n <= c:
        return 0.0
    return p * float(n - c)


def gaussian_log_density(
    z: tuple[float, ...],
    mu: tuple[float, ...],
    sigma: tuple[float, ...],
) -> float:
    """Sum of independent 1-D Gaussian log-densities (spec 4.4)."""
    if len(z) == 0 or len(mu) == 0 or len(sigma) == 0:
        raise LossError("empty gaussian inputs")
    if len(z) != len(mu) or len(z) != len(sigma):
        raise LossError("length mismatch")
    log_two_pi = math.log(2.0 * math.pi)
    total = 0.0
    for zi, mui, si in zip(z, mu, sigma, strict=True):
        z_f = float(zi)
        mu_f = float(mui)
        s_f = float(si)
        if not _is_finite(z_f) or not _is_finite(mu_f) or not _is_finite(s_f):
            raise LossError("non-finite gaussian input")
        if s_f <= 0.0:
            raise LossError("sigma must be positive")
        total += ((z_f - mu_f) / s_f) ** 2 + 2.0 * math.log(s_f) + log_two_pi
    return -0.5 * total


def latent_ratio(
    logp_new: tuple[float, ...],
    logp_old: tuple[float, ...],
) -> float:
    """Latent half of the split ratio: exp(sum(new - old))."""
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
    """Dense table [token][layer] -> expert_ids for routing replay."""
    n_tok = int(n_tokens)
    n_lay = int(n_layers)
    k = int(top_k)
    if n_tok <= 0 or n_lay <= 0 or k <= 0:
        raise LossError("n_tokens, n_layers, and top_k must be positive")
    cells: dict[tuple[int, int], tuple[int, ...]] = {}
    for trace in traces:
        token_index = int(trace.token_index)
        layer_index = int(trace.layer_index)
        expert_ids = tuple(int(e) for e in trace.expert_ids)
        key = (token_index, layer_index)
        if key in cells:
            raise LossError("duplicate (token, layer)")
        if token_index < 0 or token_index >= n_tok:
            raise LossError("token_index out of range")
        if layer_index < 0 or layer_index >= n_lay:
            raise LossError("layer_index out of range")
        if len(expert_ids) != k:
            raise LossError("expert_ids length must equal top_k")
        for expert_id in expert_ids:
            if expert_id < 0:
                raise LossError("expert id must be non-negative")
        cells[key] = expert_ids
    table: list[tuple[tuple[int, ...], ...]] = []
    for token_index in range(n_tok):
        row: list[tuple[int, ...]] = []
        for layer_index in range(n_lay):
            key = (token_index, layer_index)
            if key not in cells:
                raise LossError("missing (token, layer)")
            row.append(cells[key])
        table.append(tuple(row))
    return tuple(table)


def _has_latents(sample: Sample) -> bool:
    return len(sample.latent_logp_trainer) > 0 or len(sample.latent_logp_rollout) > 0


def gspo_dapo_loss(
    group: Group,
    config: LossConfig,
    trainer_version: int,
) -> LossBreakdown:
    """One group's GSPO/DAPO loss. No reference KL."""
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
        if gap > int(config.max_staleness):
            raise LossError("staleness exceeds max_staleness")
        gaps.append(gap)
    rewards = tuple(sample.reward for sample in group.samples)
    if all_equal_reward(rewards):
        return LossBreakdown(
            pg=0.0,
            tis=0.0,
            overlong=0.0,
            latent=0.0,
            n_kept=0,
            total=0.0,
        )
    advantages = mean_center(rewards)
    pg_sum = 0.0
    latent_sum = 0.0
    overlong_sum = 0.0
    tis_sum = 0.0
    for sample, advantage, gap in zip(group.samples, advantages, gaps, strict=True):
        r_disc = sequence_ratio(sample.token_logp_trainer, sample.token_logp_rollout)
        c_disc = clip_higher(r_disc, config.clip_eps_low, config.clip_eps_high)
        if _has_latents(sample):
            r_lat = latent_ratio(sample.latent_logp_trainer, sample.latent_logp_rollout)
            c_lat = clip_higher(
                r_lat,
                config.latent_clip_eps_low,
                config.latent_clip_eps_high,
            )
            lat_term = c_lat * advantage
        else:
            lat_term = 0.0
        tis_w = truncated_is(
            sample.token_logp_trainer,
            sample.token_logp_rollout,
            gap,
            config.tis_clip,
        )
        ol = overlong_soft_penalty(
            sample.n_tokens,
            config.overlong_cache,
            config.overlong_penalty,
        )
        disc_term = c_disc * advantage
        pg_sum += disc_term * tis_w
        latent_sum += lat_term * tis_w
        overlong_sum += ol
        tis_sum += tis_w
    normalizer = float(config.loss_normalizer)
    n_kept = len(group.samples)
    pg = pg_sum / normalizer
    latent = latent_sum / normalizer
    overlong = overlong_sum / normalizer
    tis = tis_sum / float(n_kept)
    total = (pg_sum + latent_sum + overlong_sum) / normalizer
    return LossBreakdown(
        pg=pg,
        tis=tis,
        overlong=overlong,
        latent=latent,
        n_kept=n_kept,
        total=total,
    )
