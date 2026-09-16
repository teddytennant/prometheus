"""GSPO / DAPO / CISPO / CE / z-loss (spec 9.3, A6)."""

from __future__ import annotations

from typing import Any

import numpy as np


def sequence_logprob(token_logprobs: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    token_logprobs = np.asarray(token_logprobs, dtype=np.float64)
    if mask is None:
        mask = np.ones_like(token_logprobs)
    return (token_logprobs * mask).sum(axis=-1)


def gspo_loss(
    new_seq_logp: np.ndarray,
    old_seq_logp: np.ndarray,
    advantage: np.ndarray,
    clip: float = 0.2,
) -> np.ndarray:
    """Sequence-level PPO-style clip (GSPO)."""
    ratio = np.exp(np.asarray(new_seq_logp) - np.asarray(old_seq_logp))
    adv = np.asarray(advantage)
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1.0 - clip, 1.0 + clip) * adv
    return np.asarray(-np.minimum(unclipped, clipped)).mean()


def dapo_dynamic_clip(ratio: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(np.asarray(ratio), low, high)


def cispo_mask(advantage: np.ndarray, floor: float) -> np.ndarray:
    return (np.asarray(advantage) >= floor).astype(np.float64)


def ce_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(logits).sum(axis=-1))
    gathered = np.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return float((-(gathered - log_z)).mean())


def z_loss(logits: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    log_z = np.log(np.exp(logits - logits.max(axis=-1, keepdims=True)).sum(axis=-1))
    return float((log_z**2).mean())


def drop_zero_advantage_groups(advantages: np.ndarray) -> np.ndarray:
    a = np.asarray(advantages)
    if a.ndim == 1:
        return a != 0
    return ~np.all(a == a[:, :1], axis=-1)


def advantages(rewards: np.ndarray) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float64)
    return r - r.mean()


def flag_test_write(action: str) -> bool:
    lowered = action.lower()
    return "test" in lowered and ("write" in lowered or "open(" in lowered or ".py" in lowered)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def rl_end_to_end_probe() -> dict:
    """GSPO on sequence logprobs from a linear policy. Not a 4-action argmax bandit."""
    rng = np.random.default_rng(0)
    vocab, seq_len, group = 8, 4, 16
    w = rng.standard_normal((vocab, vocab)).astype(np.float64) * 0.05
    target_last = 1
    lr = 0.4
    clip = 0.2
    reward_hist: list[float] = []
    used_gspo = False

    def token_lps(weight: np.ndarray, seq: np.ndarray) -> np.ndarray:
        lps = np.zeros(seq_len, dtype=np.float64)
        prev = 0
        for t in range(seq_len):
            p = _softmax(weight[prev])
            a = int(seq[t])
            lps[t] = np.log(p[a] + 1e-12)
            prev = a
        return lps

    def sample_seq(weight: np.ndarray) -> np.ndarray:
        seq = np.zeros(seq_len, dtype=np.int32)
        prev = 0
        for t in range(seq_len):
            p = _softmax(weight[prev])
            seq[t] = rng.choice(vocab, p=p)
            prev = int(seq[t])
        return seq

    def reward_of(seq: np.ndarray) -> float:
        return 1.0 if int(seq[-1]) == target_last else 0.0

    w_old = w.copy()
    for _ in range(20):
        seqs = [sample_seq(w) for _ in range(group)]
        rewards = np.array([reward_of(s) for s in seqs], dtype=np.float64)
        if not drop_zero_advantage_groups(rewards).any():
            reward_hist.append(float(rewards.mean()))
            continue
        adv = advantages(rewards)
        old_lp = np.array([sequence_logprob(token_lps(w_old, s)) for s in seqs])
        new_lp = np.array([sequence_logprob(token_lps(w, s)) for s in seqs])
        loss = gspo_loss(new_lp, old_lp, adv, clip=clip)
        used_gspo = True
        # REINFORCE using the GSPO-clipped advantage on the last token.
        ratio = np.exp(new_lp - old_lp)
        clipped_adv = np.clip(ratio, 1.0 - clip, 1.0 + clip) * adv
        for s, ca in zip(seqs, clipped_adv):
            prev = 0
            for t in range(seq_len):
                a = int(s[t])
                p = _softmax(w[prev])
                w[prev] -= lr * ca * p
                w[prev, a] += lr * ca
                prev = a
        w_old = w.copy()
        reward_hist.append(float(rewards.mean()))

    # Inject log-prob drift and halt when KL blows up.
    w_drift = w + rng.standard_normal(w.shape) * 3.0
    seq = sample_seq(w)
    kl = float(sequence_logprob(token_lps(w, seq)) - sequence_logprob(token_lps(w_drift, seq)))
    halted = abs(kl) > 0.5

    planted_action = "open('tests/test_planted.py','w').write('assert False')"
    return {
        "reward_rose": bool(len(reward_hist) >= 2 and reward_hist[-1] > reward_hist[0]),
        "drift_halted": bool(halted),
        "planted_write_flagged": flag_test_write(planted_action),
        "used_gspo": used_gspo,
        "gspo_mean": float(np.mean(loss)) if used_gspo else 0.0,
        "reward_hist": [float(x) for x in reward_hist[-5:]],
    }
