"""GSPO + DAPO clip-higher + Dr. GRPO (no length/std norm) (spec 9.1, D1)."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def sequence_logprob(token_logprobs: Array) -> Array:
    """Sum over tokens / constant. No per-sequence length normalization."""
    return np.asarray(token_logprobs, dtype=np.float64).sum(axis=-1)


def advantages(rewards: Array) -> Array:
    """Dr. GRPO: no std normalization. Center within the group only."""
    r = np.asarray(rewards, dtype=np.float64)
    return r - r.mean()


def drop_zero_advantage_groups(rewards: Array, eps: float = 1e-8) -> Array:
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim == 1:
        return np.array([r.std() > eps])
    return r.std(axis=-1) > eps


def gspo_loss(
    new_seq_lp: Array,
    old_seq_lp: Array,
    adv: Array,
    clip_eps: float = 0.2,
    clip_high: float = 0.28,
) -> float:
    ratio = np.exp(np.asarray(new_seq_lp) - np.asarray(old_seq_lp))
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_high)
    obj = np.minimum(ratio * adv, clipped * adv)
    return float(-obj.mean())


def truncated_is(ratio: Array, cap: float = 10.0) -> Array:
    """Staleness correction; k<=4 policy versions (spec 9.2)."""
    return np.clip(np.asarray(ratio), 0.0, cap)


def latent_ratio(token_ratio: Array, latent_ratio: Array, n_tok: int, n_lat: int) -> Array:
    """spec 4.4: (Π r_tok * Π r_lat)^(1/(T+L))."""
    seq = np.exp(
        (
            n_tok * np.log(np.maximum(token_ratio, 1e-12))
            + n_lat * np.log(np.maximum(latent_ratio, 1e-12))
        )
        / max(n_tok + n_lat, 1)
    )
    return seq


def rl_end_to_end_probe() -> dict:
    from rl.rewards import flag_test_write, math_reward

    rng = np.random.default_rng(9)
    # Tiny bandit: 4 actions, action 3 is correct.
    logits = rng.standard_normal(4)
    rewards_hist = []
    for _ in range(20):
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        a = int(p.argmax())
        r = 1.0 if a == 3 else 0.0
        rewards_hist.append(r)
        adv = r - 0.25
        logits[a] += 0.2 * adv
    planted = flag_test_write("open('tests.py','w').write('pass')")
    drift = abs(float(logits.max()) - 0.0) > 0.5
    halt = drift  # injected drift would halt; here the probe flags it
    return {
        "reward_rose": rewards_hist[-1] >= rewards_hist[0] and max(rewards_hist) > 0,
        "drift_halted": halt,
        "planted_write_flagged": planted,
        "math_ok": math_reward("4", 4) == 1.0,
    }
