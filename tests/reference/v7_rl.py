"""Independent V7 RL end-to-end protocol (spec 16.2 / 9.1–9.2 / 15.5 I2).

Slow and obvious host/NumPy. Does **not** import JAX, torch,
``prometheus.verify.v7_rl``, ``rl/``, ``audit/``, ``verifiers/``, or the Rust
crates. Production ``run_v7`` must not import this module; tests import both.

V7 meaning (CPU analog; tiny task, not 1B MoE / 4k rollouts / 8 H200). This
analog is **not** V7 verified.

``reward_rises``
    Expected exact-match reward on a tiny verifiable task after GSPO/DAPO-style
    updates is strictly greater than a no-update twin from the same init. Must
    not be True because updates were skipped or because the bool was hardcoded.
``logprob_drift_halted``
    An injected trainer-vs-engine log-prob drift exceeds ``PARITY_THRESHOLD``
    and halts RL (I2 ``run_parity`` analog). Must not be True because no drift
    was injected.
``planted_write_flagged``
    A planted write under the grader tree is flagged (I3/D2/I9 analog of a
    test-file write). Must not be True because no write was planted.

The three gates are **separate** analog experiments reported together. Glue in
production is I2 ``rl/loss`` (GSPO/DAPO, parity halt), I9 audit, D2 verifiers;
this file is an independent Python stand-in of that protocol, not a second
kernel. Host analog of the same protocol is allowed when there are no Python
bindings.

``run_v7`` does **not** write ``v7.json``; ``verify/ncshare/templates/v7.sh``
does that.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

Array = np.ndarray

# Spec 16.2 row V7: 8 H200 (4 SGLang + 4 JAX).
DEFAULT_GPUS = 8

# I2 coordinator ``run_parity`` default (rl/coordinator tests).
PARITY_THRESHOLD = 0.25

# Spec 9.2 / I2: drop rollouts with k > 4.
MAX_STALENESS = 4

# Spec 9.1 / I2 GSPO+DAPO clip_higher (asymmetric).
CLIP_EPS_LOW = 0.2
CLIP_EPS_HIGH = 0.28

# Spec 9.1 group size. Analog uses ``TOY_GROUP`` (smaller, same formula).
GROUP_SIZE = 16

# Spec 9.2 truncated IS: ρ = (65/100)^k, then min(ρ, tis_clip).
ROLLOUT_NUMER = 65
ROLLOUT_DENOM = 100
TIS_CLIP = 1.0

# Injected engine log-prob offset. Must exceed ``PARITY_THRESHOLD``.
INJECTED_DRIFT = 1.0

# I3/D2 analog: grader tests live under this prefix; agent writes do not.
GRADER_PREFIX = "/grader"
PLANTED_WRITE_PATH = "/grader/tests/hidden.py"
AGENT_WRITE_PATH = "/agent/solution.py"

# Tiny linear softmax policy (not 1B MoE). Deterministic given ``TOY_SEED``.
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
TOY_FD_EPS = 1e-5
TOY_GRAD_RTOL = 1e-4
TOY_GRAD_ATOL = 1e-5

V7_RESULT_KEYS: frozenset[str] = frozenset(
    {
        "reward_rises",
        "logprob_drift_halted",
        "planted_write_flagged",
    }
)


def softmax(logits: Array) -> Array:
    """Row-wise softmax. float64."""
    z = np.asarray(logits, dtype=np.float64)
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def log_softmax(logits: Array) -> Array:
    """Row-wise log-softmax. float64."""
    z = np.asarray(logits, dtype=np.float64)
    shifted = z - np.max(z, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def mean_center(rewards: Sequence[float]) -> tuple[float, ...]:
    """Dr. GRPO advantages: subtract the group mean. No std divide.

    Empty input raises ValueError.
    """
    if len(rewards) == 0:
        raise ValueError("empty reward group")
    vals = [float(r) for r in rewards]
    mean = sum(vals) / float(len(vals))
    return tuple(v - mean for v in vals)


def clip_higher(
    ratio: float,
    *,
    clip_eps_low: float = CLIP_EPS_LOW,
    clip_eps_high: float = CLIP_EPS_HIGH,
) -> float:
    """Asymmetric clip: ``min(1 + eps_high, max(1 - eps_low, ratio))``."""
    lo = 1.0 - float(clip_eps_low)
    hi = 1.0 + float(clip_eps_high)
    r = float(ratio)
    if r < lo:
        return lo
    if r > hi:
        return hi
    return r


def sequence_ratio(logp_new: Sequence[float], logp_old: Sequence[float]) -> float:
    """GSPO sequence ratio ``exp(sum(new - old))``. Length mismatch raises."""
    if len(logp_new) != len(logp_old):
        raise ValueError("length mismatch")
    if len(logp_new) == 0:
        return 1.0
    total = 0.0
    for a, b in zip(logp_new, logp_old, strict=True):
        total += float(a) - float(b)
    return float(np.exp(total))


def truncated_is(k: int, ratio: float, *, tis_clip: float = TIS_CLIP) -> float:
    """Truncated IS on the staleness gap (9.2).

    ``k == 0`` returns 1.0. ``k > 0`` returns ``min(ratio, tis_clip)``.
    Negative ``k`` or ``tis_clip < 1`` raises ValueError. ``k > MAX_STALENESS``
    is a drop, not a TIS value — callers must not request it.
    """
    if int(k) < 0:
        raise ValueError("negative staleness k")
    if float(tis_clip) < 1.0:
        raise ValueError("tis_clip must be >= 1")
    if int(k) > MAX_STALENESS:
        raise ValueError("k exceeds MAX_STALENESS")
    if int(k) == 0:
        return 1.0
    r = float(ratio)
    if r < float(tis_clip):
        return r
    return float(tis_clip)


def drop_stale(k: int, *, max_staleness: int = MAX_STALENESS) -> bool:
    """True iff the rollout is too stale to train on (spec 9.2)."""
    return int(k) > int(max_staleness)


def dynamic_sampling_keep(rewards: Sequence[float]) -> bool:
    """DAPO dynamic sampling: drop the group when every reward is equal."""
    if len(rewards) == 0:
        return False
    first = float(rewards[0])
    return any(float(r) != first for r in rewards)


def is_test_file_write(path: str) -> bool:
    """True iff ``path`` is under the grader tree (I3 TestFileWrite analog)."""
    normalized = str(path).replace("\\", "/")
    if normalized != "/" and normalized.endswith("/"):
        normalized = normalized.rstrip("/")
    if normalized == GRADER_PREFIX:
        return True
    return normalized.startswith(GRADER_PREFIX + "/")


def flag_planted_write(*, plant: bool, path: str = PLANTED_WRITE_PATH) -> bool:
    """Flag a planted grader write. False when nothing was planted."""
    if not plant:
        return False
    return is_test_file_write(path)


def make_toy_task(seed: int = TOY_SEED) -> tuple[Array, Array]:
    """Linearly separable 2-way task. Gold is sign of column 0. float64."""
    rng = np.random.default_rng(int(seed))
    x = rng.standard_normal((TOY_N, TOY_DIM)).astype(np.float64)
    x[:, 0] = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float64)
    y = (x[:, 0] > 0.0).astype(np.int64)
    return x, y


def init_weights(seed: int = TOY_SEED) -> Array:
    rng = np.random.default_rng(int(seed) + 1)
    return (rng.standard_normal((TOY_DIM, TOY_ACTIONS)) * TOY_INIT_STD).astype(np.float64)


def expected_reward(x: Array, y: Array, w: Array) -> float:
    """Mean p(gold | x) — D2 exact-match reward in expectation."""
    p = softmax(np.asarray(x, dtype=np.float64) @ np.asarray(w, dtype=np.float64))
    yy = np.asarray(y, dtype=np.int64)
    return float(p[np.arange(yy.shape[0]), yy].mean())


def greedy_reward(x: Array, y: Array, w: Array) -> float:
    pred = np.argmax(np.asarray(x) @ np.asarray(w), axis=-1)
    return float((pred == np.asarray(y)).mean())


def logprob_grad(x_i: Array, w: Array, action: int) -> Array:
    """``d log π(a|x) / dw`` for a linear softmax. Shape ``(TOY_DIM, TOY_ACTIONS)``."""
    p = softmax(np.asarray(x_i, dtype=np.float64) @ np.asarray(w, dtype=np.float64))
    oh = np.zeros(TOY_ACTIONS, dtype=np.float64)
    oh[int(action)] = 1.0
    return np.outer(np.asarray(x_i, dtype=np.float64), oh - p)


def softmax_ce(x_i: Array, w: Array, y_i: int) -> float:
    """Scalar CE for finite differences."""
    logp = log_softmax(np.asarray(x_i, dtype=np.float64) @ np.asarray(w, dtype=np.float64))
    return float(-logp[int(y_i)])


def softmax_ce_grad(x_i: Array, w: Array, y_i: int) -> Array:
    """Analytic ``d CE / dw = outer(x, p - one_hot(y))``."""
    p = softmax(np.asarray(x_i, dtype=np.float64) @ np.asarray(w, dtype=np.float64))
    oh = np.zeros(TOY_ACTIONS, dtype=np.float64)
    oh[int(y_i)] = 1.0
    return np.outer(np.asarray(x_i, dtype=np.float64), p - oh)


def finite_diff_ce_grad(
    x_i: Array,
    w: Array,
    y_i: int,
    *,
    eps: float = TOY_FD_EPS,
) -> Array:
    """Central finite-difference CE gradient. Slow, obvious."""
    w64 = np.asarray(w, dtype=np.float64).copy()
    x64 = np.asarray(x_i, dtype=np.float64)
    numeric = np.zeros_like(w64)
    e = float(eps)
    for i in range(w64.shape[0]):
        for j in range(w64.shape[1]):
            plus = w64.copy()
            minus = w64.copy()
            plus[i, j] += e
            minus[i, j] -= e
            numeric[i, j] = (softmax_ce(x64, plus, y_i) - softmax_ce(x64, minus, y_i)) / (2.0 * e)
    return numeric


def grad_match_ok(
    analytic: Array,
    numeric: Array,
    *,
    rtol: float = TOY_GRAD_RTOL,
    atol: float = TOY_GRAD_ATOL,
) -> bool:
    return bool(np.allclose(analytic, numeric, rtol=float(rtol), atol=float(atol)))


def toy_grad_ok(seed: int = TOY_SEED) -> bool:
    x, y = make_toy_task(int(seed))
    w = init_weights(int(seed))
    analytic = softmax_ce_grad(x[0], w, int(y[0]))
    numeric = finite_diff_ce_grad(x[0], w, int(y[0]))
    return grad_match_ok(analytic, numeric)


def _renorm(p: Array) -> Array:
    q = np.maximum(np.asarray(p, dtype=np.float64), 0.0)
    s = float(q.sum())
    if s <= 0.0:
        return np.full(TOY_ACTIONS, 1.0 / float(TOY_ACTIONS), dtype=np.float64)
    return q / s


def gspo_dapo_update(
    x: Array,
    y: Array,
    w: Array,
    rng: np.random.Generator,
    *,
    lr: float = TOY_LR,
    group: int = TOY_GROUP,
    k: int = 0,
) -> Array:
    """One on-policy GSPO/DAPO analog step. Returns updated weights.

    Group sample, mean-center, drop all-equal groups, clip_higher on the
    sequence ratio, truncated IS, REINFORCE on ``clip(ratio)*tis*A``.
    """
    w = np.asarray(w, dtype=np.float64).copy()
    p = softmax(x @ w)
    logp = log_softmax(x @ w)
    grad = np.zeros_like(w)
    n_kept = 0
    gsz = int(group)
    for i in range(int(x.shape[0])):
        if drop_stale(k):
            continue
        pi = _renorm(p[i])
        actions = rng.choice(TOY_ACTIONS, size=gsz, p=pi)
        rewards = tuple(float(int(a) == int(y[i])) for a in actions)
        if not dynamic_sampling_keep(rewards):
            continue
        adv = mean_center(rewards)
        n_kept += 1
        for act, advantage in zip(actions, adv, strict=True):
            new_lp = (float(logp[i, int(act)]),)
            old_lp = new_lp  # on-policy analog
            ratio = sequence_ratio(new_lp, old_lp)
            clipped = clip_higher(ratio)
            tis = truncated_is(int(k), ratio)
            weight = clipped * tis * float(advantage)
            grad += weight * logprob_grad(x[i], w, int(act))
    if n_kept:
        w = w + float(lr) * grad / float(n_kept * gsz)
    return w


def route_ids(x: Array, gate_w: Array) -> Array:
    """Top-1 expert id per row. Tiny 2-expert MoE routing analog."""
    scores = np.asarray(x, dtype=np.float64) @ np.asarray(gate_w, dtype=np.float64)
    return np.argmax(scores, axis=-1).astype(np.int64)


def max_abs_logprob_diff(trainer_logp: Array, engine_logp: Array) -> float:
    t = np.asarray(trainer_logp, dtype=np.float64)
    e = np.asarray(engine_logp, dtype=np.float64)
    return float(np.max(np.abs(t - e)))


def parity_halt(
    trainer_logp: Array,
    engine_logp: Array,
    *,
    threshold: float = PARITY_THRESHOLD,
) -> bool:
    """I2 ``run_parity`` analog: halt when max |Δ logp| exceeds the threshold."""
    return max_abs_logprob_diff(trainer_logp, engine_logp) > float(threshold)


def _reward_experiment(
    *,
    do_update: bool,
    skip_sync: bool,
    skip_routing_replay: bool,
    seed: int,
) -> dict[str, Any]:
    """Train a linear softmax with GSPO/DAPO vs a frozen twin."""
    x, y = make_toy_task(int(seed))
    w_init = init_weights(int(seed))
    w_train = w_init.copy()
    w_twin = w_init.copy()
    rng = np.random.default_rng(int(seed) + 3)

    gate_rng = np.random.default_rng(int(seed) + 5)
    gate_w = gate_rng.standard_normal((TOY_DIM, TOY_EXPERTS)).astype(np.float64)
    rollout_ids = route_ids(x, gate_w)
    if skip_routing_replay:
        trainer_ids = (1 - rollout_ids).astype(np.int64)
    else:
        trainer_ids = rollout_ids.copy()
    routing_replay_ok = bool(np.array_equal(rollout_ids, trainer_ids))

    trainer_version = 0
    engine_version = 0
    n_dropped_stale = 0
    for _ in range(TOY_STEPS):
        k = int(trainer_version) - int(engine_version)
        if drop_stale(k):
            n_dropped_stale += 1
            continue
        if do_update:
            w_train = gspo_dapo_update(x, y, w_train, rng, k=0)
            trainer_version += 1
            if not skip_sync:
                engine_version = trainer_version

    if skip_sync:
        w_engine = w_init.copy()
    else:
        w_engine = w_train.copy()

    trained_r = expected_reward(x, y, w_train)
    twin_r = expected_reward(x, y, w_twin)
    reward_rises = bool(trained_r > twin_r)
    return {
        "reward_rises": reward_rises,
        "trained_reward": trained_r,
        "twin_reward": twin_r,
        "greedy_trained": greedy_reward(x, y, w_train),
        "greedy_twin": greedy_reward(x, y, w_twin),
        "did_update": bool(do_update),
        "routing_replay_ok": routing_replay_ok,
        "rollout_ids": rollout_ids,
        "trainer_ids": trainer_ids,
        "weights_synced": (not skip_sync),
        "engine_matches_trainer": bool(np.allclose(w_engine, w_train)),
        "n_dropped_stale": int(n_dropped_stale),
        "trainer_version": int(trainer_version),
        "engine_version": int(engine_version),
        "w_train": w_train,
        "w_engine": w_engine,
        "x": x,
        "y": y,
    }


def _parity_experiment(*, inject_drift: bool, seed: int) -> dict[str, Any]:
    """Synced trainer/engine log-probs, then optional injected drift."""
    x, _y = make_toy_task(int(seed))
    w = init_weights(int(seed))
    prompt = x[0]
    trainer_logp = log_softmax(prompt @ w)
    engine_logp = trainer_logp.copy()
    # Weight-sync analog: engine starts equal. Then plant a log-prob bug.
    amount = float(INJECTED_DRIFT) if inject_drift else 0.0
    if inject_drift:
        engine_logp = engine_logp.copy()
        engine_logp[0] = engine_logp[0] + amount
    max_diff = max_abs_logprob_diff(trainer_logp, engine_logp)
    halted = parity_halt(trainer_logp, engine_logp)
    return {
        "logprob_drift_halted": bool(halted),
        "max_abs_logprob_diff": max_diff,
        "parity_threshold": float(PARITY_THRESHOLD),
        "injected_drift": bool(inject_drift),
        "injected_drift_amount": amount,
        "trainer_logp": trainer_logp,
        "engine_logp": engine_logp,
    }


def _write_experiment(*, plant: bool, path: str) -> dict[str, Any]:
    flagged = flag_planted_write(plant=plant, path=path)
    return {
        "planted_write_flagged": bool(flagged),
        "planted": bool(plant),
        "path": str(path),
        "is_grader_path": is_test_file_write(path),
    }


def evaluate_v7_protocol(
    *,
    do_update: bool = True,
    inject_drift: bool = True,
    plant_write: bool = True,
    skip_sync: bool = False,
    skip_routing_replay: bool = False,
    planted_path: str = PLANTED_WRITE_PATH,
    seed: int = TOY_SEED,
) -> dict[str, Any]:
    """Run the three analog experiments and the GSPO/DAPO protocol pieces."""
    reward = _reward_experiment(
        do_update=do_update,
        skip_sync=skip_sync,
        skip_routing_replay=skip_routing_replay,
        seed=int(seed),
    )
    parity = _parity_experiment(inject_drift=inject_drift, seed=int(seed))
    write = _write_experiment(plant=plant_write, path=planted_path)
    result = {
        "reward_rises": bool(reward["reward_rises"]),
        "logprob_drift_halted": bool(parity["logprob_drift_halted"]),
        "planted_write_flagged": bool(write["planted_write_flagged"]),
    }
    return {
        **result,
        "meets_gates": meets_v7_gates(result),
        "grad_ok": toy_grad_ok(int(seed)),
        "reward": reward,
        "parity": parity,
        "write": write,
    }


def meets_v7_gates(result: Mapping[str, Any]) -> bool:
    """F4 ``check_exit(V7)``: three bools all true."""
    return (
        bool(result["reward_rises"])
        and bool(result["logprob_drift_halted"])
        and bool(result["planted_write_flagged"])
    )
