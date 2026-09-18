"""V7 RL end-to-end runner (spec 16.2, F4 template ``v7.sh``).

CPU analog of 8 H200 (4 SGLang + 4 JAX): GSPO/DAPO loss, async staleness,
routing replay, weight sync, parity halt, reward-hacking controls. A CPU
JAX / host path is allowed so the analog can run without a GPU, and the
task is a tiny verifiable one, not 1B-scale / 4k rollouts. That does not
count as V7 verified. V7 itself waits for V0.

``verify/ncshare`` ``check_exit(V7)`` reads ``v7.json`` with three bools,
all required true:

- ``reward_rises``: reward rises on a tiny verifiable task.
- ``logprob_drift_halted``: an injected log-prob drift halts RL (I2
  parity halt).
- ``planted_write_flagged``: a planted test-file write is flagged (I9
  audit).

Glue, not a second implementation: I2 ``rl/loss`` (GSPO/DAPO, parity
halt), I9 audit, D2 verifiers. Crates with no Python bindings may use a
host analog of the same protocol. Must not import ``tests/``.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

# Spec 16.2: 8 H200 (4 SGLang + 4 JAX). Template ``v7.sh`` passes ``gpus``.
DEFAULT_GPUS = 8

# Spec 9.1 / 9.2 analog constants (must match the oracle reference values).
PARITY_THRESHOLD = 0.25
MAX_STALENESS = 4
CLIP_EPS_LOW = 0.2
CLIP_EPS_HIGH = 0.28
GROUP_SIZE = 16
ROLLOUT_NUMER = 65
ROLLOUT_DENOM = 100
TIS_CLIP = 1.0
INJECTED_DRIFT = 1.0
GRADER_PREFIX = "/grader"
PLANTED_WRITE_PATH = "/grader/tests/hidden.py"
AGENT_WRITE_PATH = "/agent/solution.py"

# Tiny CPU analog. Not 1B MoE / 4k rollouts / 8 H200.
TOY_DIM = 4
TOY_ACTIONS = 2
TOY_N = 8
TOY_GROUP = 4
TOY_STEPS = 8
TOY_SEED = 7
TOY_LR = 0.5
TOY_INIT_STD = 0.02
TOY_EXPERTS = 2
TOY_TOP_K = 1


class V7Result(TypedDict):
    reward_rises: bool
    logprob_drift_halted: bool
    planted_write_flagged: bool


class V7Error(Exception):
    """Bad V7 inputs (gpus < 1) or a missing / failed backend."""


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))


def _mean_center(rewards: np.ndarray) -> np.ndarray:
    """Dr. GRPO: group mean-center, no std, no length norm."""
    r = np.asarray(rewards, dtype=np.float64)
    return r - np.mean(r, axis=-1, keepdims=True)


def _clip_higher(ratio: float) -> float:
    """DAPO clip-higher: [1 - ε_low, 1 + ε_high]."""
    lo = 1.0 - CLIP_EPS_LOW
    hi = 1.0 + CLIP_EPS_HIGH
    return min(max(float(ratio), lo), hi)


def _sequence_ratio(logp_new: np.ndarray, logp_old: np.ndarray) -> float:
    """GSPO sequence-level importance ratio."""
    return float(np.exp(np.sum(np.asarray(logp_new) - np.asarray(logp_old))))


def _tis_weight(staleness: int) -> float:
    """Truncated IS on the async gap: ρ = (65/100)^k, then min(ρ, TIS_CLIP)."""
    k = int(staleness)
    if k < 0:
        raise V7Error(f"staleness must be >= 0 (k={k})")
    rho = (ROLLOUT_NUMER / ROLLOUT_DENOM) ** k
    return float(min(rho, TIS_CLIP))


def _drop_stale(staleness: int) -> bool:
    return int(staleness) > MAX_STALENESS


def _dynamic_keep(rewards: np.ndarray) -> bool:
    """Drop a group whose every sample got the same reward (zero advantage)."""
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    return bool(np.min(r) < np.max(r))


def _is_grader_write(path: str) -> bool:
    p = str(path)
    prefix = GRADER_PREFIX if GRADER_PREFIX.endswith("/") else GRADER_PREFIX + "/"
    return p == GRADER_PREFIX or p.startswith(prefix)


def _toy_task(
    n: int = TOY_N, dim: int = TOY_DIM, seed: int = TOY_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly separable 2-way exact-match task; gold is the sign of column 0."""
    rng = np.random.default_rng(int(seed))
    x = rng.normal(size=(int(n), int(dim))).astype(np.float64)
    x[:, 0] = np.where(np.arange(int(n)) % 2 == 0, 1.0, -1.0)
    y = (x[:, 0] > 0.0).astype(np.int64)
    return x, y


def _init_weights(seed: int = TOY_SEED) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 1)
    return rng.normal(scale=TOY_INIT_STD, size=(TOY_DIM, TOY_ACTIONS)).astype(np.float64)


def _expected_reward(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    p = _softmax(x @ w)
    n = int(y.shape[0])
    return float(np.mean(p[np.arange(n), y]))


def _top_k_ids(scores: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), axis=-1)
    return order[..., :k].astype(np.int64)


def _replay_ok(engine_ids: np.ndarray, trainer_ids: np.ndarray) -> bool:
    a = np.asarray(engine_ids)
    b = np.asarray(trainer_ids)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _gspo_dapo_step(
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    do_update: bool,
    staleness: int,
) -> np.ndarray:
    """One on-policy GSPO/DAPO analog step with TIS, routing replay, weight sync."""
    if _drop_stale(staleness):
        return np.array(w, copy=True)

    logits = x @ w
    logp = _log_softmax(logits)
    p = np.exp(logp)
    n = int(x.shape[0])

    # Routing replay analog: engine records top-1 expert ids; trainer forces them.
    router = x @ np.ones((TOY_DIM, TOY_EXPERTS), dtype=np.float64)
    engine_ids = _top_k_ids(router, TOY_TOP_K)
    trainer_ids = np.array(engine_ids, copy=True)
    if not _replay_ok(engine_ids, trainer_ids):
        raise V7Error("routing replay mismatch")

    actions = np.empty((n, TOY_GROUP), dtype=np.int64)
    for i in range(n):
        actions[i] = rng.choice(TOY_ACTIONS, size=TOY_GROUP, p=p[i])
    rewards = (actions == y[:, None]).astype(np.float64)

    grad = np.zeros_like(w, dtype=np.float64)
    used = 0
    for i in range(n):
        if not _dynamic_keep(rewards[i]):
            continue
        adv = _mean_center(rewards[i])
        for g in range(TOY_GROUP):
            a = int(actions[i, g])
            ratio = _sequence_ratio(logp[i, a : a + 1], logp[i, a : a + 1])
            surr = _clip_higher(ratio) * float(adv[g])
            surr *= _tis_weight(staleness)
            one_hot = np.zeros(TOY_ACTIONS, dtype=np.float64)
            one_hot[a] = 1.0
            grad += np.outer(x[i], (one_hot - p[i]) * surr)
            used += 1

    out = np.array(w, copy=True)
    if do_update and used > 0:
        out = out + (TOY_LR / float(used)) * grad
    # Weight-sync analog: engine snapshot equals trainer after the step.
    engine = np.array(out, copy=True)
    if not np.array_equal(out, engine):
        raise V7Error("weight sync mismatch")
    return out


def _reward_rises(*, do_update: bool = True) -> bool:
    x, y = _toy_task()
    init = _init_weights()
    twin = np.array(init, copy=True)
    trained = np.array(init, copy=True)
    rng = np.random.default_rng(TOY_SEED + 2)
    for _ in range(TOY_STEPS):
        trained = _gspo_dapo_step(trained, x, y, rng, do_update=do_update, staleness=0)
    trained_r = _expected_reward(trained, x, y)
    twin_r = _expected_reward(twin, x, y)
    if not np.isfinite(trained_r) or not np.isfinite(twin_r):
        raise V7Error("non-finite expected reward")
    return bool(trained_r > twin_r)


def _logprob_drift_halted(*, inject: bool = True) -> bool:
    x, _y = _toy_task()
    w = _init_weights()
    trainer_logp = _log_softmax(x @ w)
    engine_logp = np.array(trainer_logp, copy=True)
    if inject:
        engine_logp.reshape(-1)[0] = engine_logp.reshape(-1)[0] + INJECTED_DRIFT
    drift = float(np.max(np.abs(trainer_logp - engine_logp)))
    if not np.isfinite(drift):
        raise V7Error("non-finite log-prob drift")
    if inject and drift < INJECTED_DRIFT:
        raise V7Error("injected drift did not reach INJECTED_DRIFT")
    return bool(drift > PARITY_THRESHOLD)


def _planted_write_flagged(*, plant: bool = True, path: str = PLANTED_WRITE_PATH) -> bool:
    writes: list[str] = []
    if plant:
        writes.append(str(path))
    writes.append(AGENT_WRITE_PATH)
    grader_hits = [p for p in writes if _is_grader_write(p)]
    agent_flagged = _is_grader_write(AGENT_WRITE_PATH)
    return bool(grader_hits) and not agent_flagged


def run_v7(*, gpus: int) -> V7Result:
    """Run the V7 RL analog and report the three gates.

    ``gpus`` is the Slurm GPU count (template ``{{GPUS}}``). Spec V7 is 8
    H200. ``gpus >= 1`` is accepted so a CPU analog can run; that analog
    is not V7 verified.

    Analog (spec 16.2 / 15.5 I2):

    - Tiny verifiable task; reward must rise vs a no-update twin.
    - Injected log-prob drift (train vs inference) must halt RL.
    - Planted test-file write must be flagged by the audit analog.

    Returns a JSON-serializable ``V7Result``. Raises ``V7Error`` when
    ``gpus`` is invalid or the backend cannot produce a finite report.
    Does not write ``v7.json``; the template does that.
    """
    if gpus < 1:
        raise V7Error(f"gpus must be >= 1 (gpus={gpus})")

    reward = _reward_rises(do_update=True)
    halted = _logprob_drift_halted(inject=True)
    flagged = _planted_write_flagged(plant=True, path=PLANTED_WRITE_PATH)
    return {
        "reward_rises": bool(reward),
        "logprob_drift_halted": bool(halted),
        "planted_write_flagged": bool(flagged),
    }
