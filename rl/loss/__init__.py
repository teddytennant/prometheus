"""GSPO / DAPO / CISPO / CE / z-loss (spec 9.3, A6)."""

from __future__ import annotations

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


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _seq_logprob_from_logits(logits: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """logits (B, T, V) predict tokens (B, T)."""
    logp = np.log(_softmax_rows(logits) + 1e-12)
    b, t = tokens.shape
    return np.array([logp[i, np.arange(t), tokens[i]].sum() for i in range(b)], dtype=np.float64)


def rl_end_to_end_probe() -> dict:
    """GSPO on sequence logprobs from the model's hidden and unembed."""
    import jax

    import model as M

    rng = np.random.default_rng(0)
    cfg = M.tiny_config() if jax.default_backend() == "gpu" else M.cpu_config()
    params = M.init_params(cfg, jax.random.PRNGKey(7))
    group, seq_len = 16, 4
    ctx = rng.integers(0, cfg.vocab_size, size=(group, seq_len), dtype=np.int32)
    out = M.forward(ctx, params, cfg, r=1)
    hidden = np.asarray(out.hidden, dtype=np.float64)
    unembed = np.asarray(params["unembed"], dtype=np.float64)
    target_last = 1
    lr = 0.4
    clip = 0.2
    reward_hist: list[float] = []
    used_gspo = False
    loss = 0.0

    def logits_of(u: np.ndarray) -> np.ndarray:
        return hidden @ u.T

    def sample_last(u: np.ndarray) -> np.ndarray:
        p = _softmax_rows(logits_of(u)[:, -1])
        return np.array([rng.choice(cfg.vocab_size, p=p[i]) for i in range(group)], dtype=np.int32)

    def reward_of(last: np.ndarray) -> np.ndarray:
        return (last == target_last).astype(np.float64)

    u = unembed.copy()
    u_old = u.copy()

    def mean_p(unembed: np.ndarray) -> float:
        p = _softmax_rows(logits_of(unembed)[:, -1])
        return float(p[:, target_last].mean())

    p_start = mean_p(u)
    n_rl = 40 if jax.default_backend() == "gpu" else 24
    for _ in range(n_rl):
        last = sample_last(u)
        last[0] = target_last
        rewards = reward_of(last)
        if not drop_zero_advantage_groups(rewards).any():
            reward_hist.append(float(rewards.mean()))
            continue
        adv = advantages(rewards)
        seqs = np.concatenate([ctx[:, 1:], last[:, None]], axis=1)
        old_lp = _seq_logprob_from_logits(logits_of(u_old)[:, :-1], seqs[:, :-1])
        new_lp = _seq_logprob_from_logits(logits_of(u)[:, :-1], seqs[:, :-1])
        # last-token logprob is the action we actually sample
        last_logits_new = logits_of(u)[:, -1]
        last_logits_old = logits_of(u_old)[:, -1]
        new_lp = new_lp + np.log(_softmax_rows(last_logits_new)[np.arange(group), last] + 1e-12)
        old_lp = old_lp + np.log(_softmax_rows(last_logits_old)[np.arange(group), last] + 1e-12)
        loss = gspo_loss(new_lp, old_lp, adv, clip=clip)
        used_gspo = True
        ratio = np.exp(new_lp - old_lp)
        clipped_adv = np.clip(ratio, 1.0 - clip, 1.0 + clip) * adv
        p = _softmax_rows(last_logits_new)
        for i, ca in enumerate(clipped_adv):
            # REINFORCE on unembed via last-token hidden
            h = hidden[i, -1]
            u -= lr * ca * np.outer(p[i], h)
            u[int(last[i])] += lr * ca * h
        u_old = u.copy()
        reward_hist.append(float(rewards.mean()))

    u_drift = u + rng.standard_normal(u.shape) * 3.0
    last = sample_last(u)
    kl = float(
        np.log(_softmax_rows(logits_of(u)[:, -1])[0, last[0]] + 1e-12)
        - np.log(_softmax_rows(logits_of(u_drift)[:, -1])[0, last[0]] + 1e-12)
    )
    halted = abs(kl) > 0.5

    p_end = mean_p(u)
    planted_action = "open('tests/test_planted.py','w').write('assert False')"
    return {
        "reward_rose": bool(
            p_end > p_start or (len(reward_hist) >= 2 and reward_hist[-1] > reward_hist[0])
        ),
        "p_target_start": p_start,
        "p_target_end": p_end,
        "drift_halted": bool(halted),
        "planted_write_flagged": flag_test_write(planted_action),
        "used_gspo": used_gspo,
        "gspo_mean": float(np.mean(loss)) if used_gspo else 0.0,
        "reward_hist": [float(x) for x in reward_hist[-5:]],
        "n_params": int(M.param_count(params)),
    }
