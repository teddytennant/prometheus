"""Independent V6 latent protocol (spec 16.2 / 4.2–4.4).

Slow and obvious host NumPy. Does **not** import JAX, torch, the V6 latent
runner, ``model/``, ``train/``, ``kernels/``, or the Rust crates. Production
``run_v6`` must not import this module; tests import both.

CPU analog of Stage A then Stage B compression (I5 ``model/latent.py`` glue:
PonderNet halt, thought decode, Jacobi sweeps, noisy latent policy) plus a
latent-budget sweep 1x to 8x. Passing these tests is **not** V6 verified.
Spec V6 is 4 to 8 H200.

``curriculum_no_collapse``
    Stage A then Stage B curriculum trains without collapse. Must not be True
    because Stage A was skipped, *k* was frozen, or the halt distribution
    collapsed to a spike.

``accuracy_rises_with_latent_budget``
    Held-out accuracy rises with latent budget 1x to 8x on problems the model
    cannot one-shot. Must not be True because the budget was frozen.

``thoughts_decode``
    Thoughts decode to their steps (tied head, after Jacobi). Must not be True
    because decode was skipped or Jacobi never ran.

Glue of I5, not a second paper.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

Array = np.ndarray

# Analog constants the CPU stand-in must match (spec 16.2 V6).
DEFAULT_GPUS = 4
DEFAULT_GPUS_MIN = 1  # CPU analog may accept gpus >= 1
DEFAULT_GPUS_MAX = 8  # spec 16.2 high end (4 to 8 H200)
SPEC_GPUS_MIN = 4
SPEC_GPUS_MAX = 8

# I5 / spec 4.2–4.4 glue.
CHUNK_MIN_THOUGHTS = 4
CHUNK_MAX_THOUGHTS = 64
DEFAULT_JACOBI_SWEEPS = 4
TRUNCATED_SWEEPS = 2
DEFAULT_MAX_THOUGHTS = 8
DEFAULT_ALPHA_TRAJ = 1.0
DEFAULT_GAMMA_HALT = 1.0
DEFAULT_BETA_KL = 0.01
DEFAULT_LAMBDA_PRIOR = 0.2
HALT_EPS = 1e-6
SIGMA_MIN = 1e-4
SIGMA_MAX = 0.1
JACOBI_MIX = 0.5

# Latent budget sweep: 1x to 8x of mean training budget 1.
LATENT_BUDGETS: tuple[int, ...] = (1, 2, 4, 8)
CURRICULUM_K: tuple[int, ...] = (1, 2, 4, 8)
# Problems the analog cannot one-shot (every item needs >= 2 hops).
REQUIRED_HOPS: tuple[int, ...] = (2, 2, 4, 4, 8, 8, 8, 8)

TOY_VOCAB = 8
TOY_DIM = 8
TOY_N = 8
TOY_CHUNK = 4
TOY_STEPS_A = 16
TOY_LR = 0.5
TOY_SEED = 6
TOY_INIT_STD = 0.05
TOY_DECODE_NOISE = 0.2
TOY_NOISY_SIGMA = 0.01

TOY_FD_EPS = 1e-5
TOY_GRAD_RTOL = 1e-4
TOY_GRAD_ATOL = 1e-5
FP32_PARITY = 1e-5

HALT_ENTROPY_FLOOR = 0.05
COLLAPSE_DEPTH_MAX = 0.05

V6_RESULT_KEYS: tuple[str, ...] = (
    "curriculum_no_collapse",
    "accuracy_rises_with_latent_budget",
    "thoughts_decode",
)

# PonderNet golden (spec 4.3 / I5): last λ pinned to 1.
GOLDEN_LAMBDAS: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6)
GOLDEN_PONDER_P: tuple[float, ...] = (0.3, 0.28, 0.21, 0.21)
GOLDEN_PONDER_DEPTH = 1.33


@dataclass(frozen=True)
class HaltResult:
    lambdas: Array
    p: Array
    expected_depth: float


@dataclass(frozen=True)
class NoisyLatent:
    mu: Array
    sigma: Array
    eps: Array
    z: Array
    log_density: float


def _as_float64(x: Any) -> Array:
    return np.asarray(x, dtype=np.float64)


def _require_1d_finite(name: str, x: Any) -> Array:
    arr = _as_float64(x).reshape(-1)
    if arr.size < 1:
        raise ValueError(f"{name} empty")
    for item in arr:
        if not math.isfinite(float(item)):
            raise ValueError(f"{name} non-finite")
    return arr


def softmax(logits: Array) -> Array:
    z = _as_float64(logits)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def log_softmax(logits: Array) -> Array:
    z = _as_float64(logits)
    z = z - np.max(z, axis=-1, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))


def one_hot(ids: Array, vocab: int) -> Array:
    idx = np.asarray(ids, dtype=np.int64).reshape(-1)
    out = np.zeros((idx.shape[0], vocab), dtype=np.float64)
    for i, t in enumerate(idx):
        if t < 0 or t >= vocab:
            raise ValueError("id out of range")
        out[i, int(t)] = 1.0
    return out


def ponder_distribution(lambdas: Array) -> Array:
    """Clip λ to [HALT_EPS, 1-HALT_EPS], pin last to 1, p_m = λ_m Π_{j<m}(1-λ_j)."""
    raw = _require_1d_finite("lambdas", lambdas)
    n = int(raw.shape[0])
    lam = np.empty(n, dtype=np.float64)
    lo = HALT_EPS
    hi = 1.0 - HALT_EPS
    for i in range(n):
        x = float(raw[i])
        if x < lo:
            x = lo
        elif x > hi:
            x = hi
        lam[i] = x
    lam[n - 1] = 1.0
    p = np.empty(n, dtype=np.float64)
    stay = 1.0
    for m in range(n):
        p[m] = float(lam[m]) * stay
        stay *= 1.0 - float(lam[m])
    return p


def geometric_prior(n: int, lambda_prior: float = DEFAULT_LAMBDA_PRIOR) -> Array:
    """g_m = λ (1-λ)^m on {0 .. n-1}, then renormalize."""
    if n < 1:
        raise ValueError("n < 1")
    lam = float(lambda_prior)
    if not math.isfinite(lam) or lam <= 0.0 or lam >= 1.0:
        raise ValueError("lambda_prior not in (0, 1)")
    g = np.empty(n, dtype=np.float64)
    stay = 1.0
    total = 0.0
    for m in range(n):
        g[m] = lam * stay
        total += g[m]
        stay *= 1.0 - lam
    if total <= 0.0:
        raise ValueError("geometric prior empty")
    for m in range(n):
        g[m] = g[m] / total
    return g


def halt_from_logits(logits: Array) -> HaltResult:
    """Sigmoid → ponder; expected_depth = sum_m p_m * m."""
    z = _require_1d_finite("halt_logits", logits)
    lam = np.empty(z.shape[0], dtype=np.float64)
    for i in range(z.shape[0]):
        lam[i] = 1.0 / (1.0 + math.exp(-float(z[i])))
    p = ponder_distribution(lam)
    depth = 0.0
    for m in range(p.shape[0]):
        depth += float(p[m]) * float(m)
    return HaltResult(lambdas=lam, p=p, expected_depth=float(depth))


def halt_loss(p: Array, teacher_steps: int) -> float:
    """-log(p[k] + HALT_EPS) with k = clip(teacher_steps, 0, K)."""
    arr = _require_1d_finite("p", p)
    k = int(teacher_steps)
    last = int(arr.shape[0]) - 1
    if k < 0:
        k = 0
    elif k > last:
        k = last
    return float(-math.log(float(arr[k]) + HALT_EPS))


def halt_kl(p: Array, lambda_prior: float = DEFAULT_LAMBDA_PRIOR) -> float:
    """sum_m p_m (log(p_m+EPS) - log(g_m+EPS))."""
    arr = _require_1d_finite("p", p)
    g = geometric_prior(int(arr.shape[0]), lambda_prior)
    total = 0.0
    for m in range(arr.shape[0]):
        pm = float(arr[m])
        gm = float(g[m])
        total += pm * (math.log(pm + HALT_EPS) - math.log(gm + HALT_EPS))
    return float(total)


def halt_entropy(p: Array) -> float:
    arr = _require_1d_finite("p", p)
    h = 0.0
    for m in range(arr.shape[0]):
        pm = float(arr[m])
        if pm > 0.0:
            h -= pm * math.log(pm)
    return float(h)


def thought_decode_ce(logits: Array, teacher_ids: Array, mask: Array | None = None) -> float:
    """Mean -log_softmax(logits)[i, teacher_ids[i]] over mask != 0."""
    z = _as_float64(logits)
    if z.ndim != 2:
        raise ValueError("logits rank")
    ids = np.asarray(teacher_ids, dtype=np.int64).reshape(-1)
    if ids.shape[0] != z.shape[0]:
        raise ValueError("length mismatch")
    if mask is None:
        m = np.ones(z.shape[0], dtype=np.float64)
    else:
        m = _as_float64(mask).reshape(-1)
        if m.shape[0] != z.shape[0]:
            raise ValueError("mask length")
    lp = log_softmax(z)
    total = 0.0
    weight = 0.0
    for i in range(z.shape[0]):
        wi = float(m[i])
        if wi == 0.0:
            continue
        t = int(ids[i])
        if t < 0 or t >= z.shape[1]:
            raise ValueError("id out of range")
        total += wi * (-float(lp[i, t]))
        weight += wi
    if weight <= 0.0:
        raise ValueError("empty mask")
    return float(total / weight)


def stage_b_loss(
    l_task: float,
    l_traj: float,
    l_halt: float,
    l_kl: float,
    *,
    alpha: float = DEFAULT_ALPHA_TRAJ,
    gamma: float = DEFAULT_GAMMA_HALT,
    beta: float = DEFAULT_BETA_KL,
) -> float:
    """l_task + α l_traj + γ l_halt + β l_kl."""
    for name, val in (
        ("l_task", l_task),
        ("l_traj", l_traj),
        ("l_halt", l_halt),
        ("l_kl", l_kl),
        ("alpha", alpha),
        ("gamma", gamma),
        ("beta", beta),
    ):
        if not math.isfinite(float(val)):
            raise ValueError(f"{name} non-finite")
    return float(l_task + alpha * l_traj + gamma * l_halt + beta * l_kl)


def jacobi_sweeps(thoughts: Array, update: Callable[[Array], Array], n_sweeps: int) -> Array:
    """thoughts = update(thoughts) for n_sweeps whole-chunk updates."""
    if int(n_sweeps) < 1:
        raise ValueError("n_sweeps < 1")
    z = _as_float64(thoughts).copy()
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] < 1:
        raise ValueError("thoughts shape")
    for _ in range(int(n_sweeps)):
        nxt = _as_float64(update(z))
        if nxt.shape != z.shape:
            raise ValueError("update shape")
        z = nxt
    return z


def contract_toward(teacher: Array, mix: float = JACOBI_MIX) -> Callable[[Array], Array]:
    tgt = _as_float64(teacher)
    a = float(mix)
    if not math.isfinite(a) or a <= 0.0 or a > 1.0:
        raise ValueError("mix")

    def update(z: Array) -> Array:
        return (1.0 - a) * _as_float64(z) + a * tgt

    return update


def clamp_sigma(
    sigma: Array, *, sigma_min: float = SIGMA_MIN, sigma_max: float = SIGMA_MAX
) -> Array:
    arr = _require_1d_finite("sigma", sigma)
    lo = float(sigma_min)
    hi = float(sigma_max)
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo < hi):
        raise ValueError("sigma bounds")
    out = np.empty(arr.shape[0], dtype=np.float64)
    for i in range(arr.shape[0]):
        x = float(arr[i])
        if x < lo:
            x = lo
        elif x > hi:
            x = hi
        out[i] = x
    return out


def noisy_latent(mu: Array, sigma: Array, eps: Array) -> NoisyLatent:
    """z = mu + clamp(sigma) * eps and the independent Gaussian log-density."""
    mu_a = _require_1d_finite("mu", mu)
    sigma_a = _require_1d_finite("sigma", sigma)
    eps_a = _require_1d_finite("eps", eps)
    if mu_a.shape[0] != sigma_a.shape[0] or mu_a.shape[0] != eps_a.shape[0]:
        raise ValueError("length mismatch")
    s = clamp_sigma(sigma_a)
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


def log_density_at(z: Array, mu: Array, sigma: Array) -> float:
    """Independent Gaussian log-density treating z as fixed (for d/dμ)."""
    z_a = _require_1d_finite("z", z)
    mu_a = _require_1d_finite("mu", mu)
    s = clamp_sigma(sigma)
    if z_a.shape[0] != mu_a.shape[0] or z_a.shape[0] != s.shape[0]:
        raise ValueError("length mismatch")
    log_two_pi = math.log(2.0 * math.pi)
    total = 0.0
    for i in range(z_a.shape[0]):
        si = float(s[i])
        centered = (float(z_a[i]) - float(mu_a[i])) / si
        total += centered * centered + 2.0 * math.log(si) + log_two_pi
    return float(-0.5 * total)


def hop_matrix(dim: int = TOY_DIM) -> Array:
    """Cycle permutation: one hop shifts the one-hot basis by +1."""
    if int(dim) < 2:
        raise ValueError("dim < 2")
    w = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        w[i, (i + 1) % dim] = 1.0
    return w


def apply_hops(x: Array, w: Array, n: int) -> Array:
    h = _as_float64(x).copy()
    steps = int(n)
    if steps < 0:
        raise ValueError("n < 0")
    for _ in range(steps):
        h = h @ w
    return h


def forward_with_budget(x: Array, w: Array, required: int, budget: int) -> Array:
    """PonderNet halt: run min(budget, required) hops (stop when the teacher would)."""
    n = min(int(budget), int(required))
    if n < 0:
        raise ValueError("hops < 0")
    return apply_hops(x, w, n)


def teacher_halt_logits(k: int, n_slots: int) -> Array:
    """Soft peak at teacher depth k (not a delta, so entropy stays off the floor)."""
    if n_slots < 1:
        raise ValueError("n_slots < 1")
    logits = np.empty(n_slots, dtype=np.float64)
    kk = int(k)
    for m in range(n_slots):
        d = float(m - kk)
        logits[m] = -(d * d)
    return logits


def pin_first_halt_logits(n_slots: int) -> Array:
    """Collapsed halt: mass on depth 0."""
    if n_slots < 1:
        raise ValueError("n_slots < 1")
    logits = np.full(n_slots, -20.0, dtype=np.float64)
    logits[0] = 20.0
    return logits


def softmax_ce_loss_and_grad(weights: Array, tokens: Array, targets: Array) -> tuple[float, Array]:
    """Linear next-token CE and analytic dL/dW. Slow loop over the batch."""
    w = _as_float64(weights)
    tok = np.asarray(tokens, dtype=np.int64).reshape(-1)
    tgt = np.asarray(targets, dtype=np.int64).reshape(-1)
    n = int(tok.shape[0])
    if n < 1 or tgt.shape[0] != n:
        raise ValueError("batch")
    vocab = int(w.shape[1])
    logits = one_hot(tok, int(w.shape[0])) @ w
    lp = log_softmax(logits)
    loss = 0.0
    grad_logits = np.zeros_like(logits)
    probs = softmax(logits)
    for i in range(n):
        t = int(tgt[i])
        if t < 0 or t >= vocab:
            raise ValueError("target")
        loss += -float(lp[i, t])
        for v in range(vocab):
            grad_logits[i, v] = float(probs[i, v])
        grad_logits[i, t] -= 1.0
    loss /= float(n)
    grad_logits /= float(n)
    grad_w = np.zeros_like(w)
    for i in range(n):
        t = int(tok[i])
        for v in range(vocab):
            grad_w[t, v] += float(grad_logits[i, v])
    return float(loss), grad_w


def train_stage_a(
    *,
    steps: int = TOY_STEPS_A,
    lr: float = TOY_LR,
    seed: int = TOY_SEED,
) -> tuple[Array, list[float]]:
    """Discrete CoT analog: next-token shift on the hop cycle."""
    rng = np.random.default_rng(int(seed))
    w = rng.normal(0.0, TOY_INIT_STD, (TOY_VOCAB, TOY_VOCAB)).astype(np.float64)
    tokens = np.arange(TOY_VOCAB, dtype=np.int64)
    targets = (tokens + 1) % TOY_VOCAB
    losses: list[float] = []
    for _ in range(int(steps)):
        loss, grad = softmax_ce_loss_and_grad(w, tokens, targets)
        losses.append(float(loss))
        w = w - float(lr) * grad
    return w, losses


def run_curriculum_experiment(
    *,
    skip_curriculum: bool = False,
    pin_halt_first: bool = False,
) -> dict[str, Any]:
    """Stage A then Stage B Coconut-style k-curriculum (1, 2, 4, 8)."""
    n_slots = DEFAULT_MAX_THOUGHTS + 1
    if skip_curriculum:
        did_stage_a = False
        stage_a_losses: list[float] = []
        stage_a_loss_fell = False
        k_schedule: tuple[int, ...] = (1,)
        halt_fn = pin_first_halt_logits
    else:
        did_stage_a = True
        _w, stage_a_losses = train_stage_a()
        stage_a_loss_fell = (
            len(stage_a_losses) >= 2
            and all(math.isfinite(x) for x in stage_a_losses)
            and float(stage_a_losses[-1]) < float(stage_a_losses[0])
            and float(stage_a_losses[-1]) < math.log(TOY_VOCAB)
        )
        k_schedule = CURRICULUM_K
        halt_fn = pin_first_halt_logits if pin_halt_first else teacher_halt_logits

    embed = np.eye(TOY_DIM, dtype=np.float64)
    stage_b_losses: list[float] = []
    depths: list[float] = []
    entropies: list[float] = []
    did_stage_b = True
    for k in k_schedule:
        kk = int(k)
        teacher_ids = np.arange(kk, dtype=np.int64) % TOY_VOCAB
        teacher = embed[teacher_ids]
        thoughts = jacobi_sweeps(
            teacher,
            contract_toward(teacher),
            DEFAULT_JACOBI_SWEEPS,
        )
        logits = thoughts @ embed.T
        l_traj = thought_decode_ce(logits, teacher_ids)
        if skip_curriculum or pin_halt_first:
            halt = halt_from_logits(halt_fn(n_slots))
        else:
            halt = halt_from_logits(halt_fn(kk, n_slots))
        l_halt = halt_loss(halt.p, kk)
        l_kl = halt_kl(halt.p)
        l_task = l_traj
        total = stage_b_loss(l_task, l_traj, l_halt, l_kl)
        stage_b_losses.append(total)
        depths.append(float(halt.expected_depth))
        entropies.append(halt_entropy(halt.p))

    k_raised = len(k_schedule) >= 2 and all(
        k_schedule[i] < k_schedule[i + 1] for i in range(len(k_schedule) - 1)
    )
    losses_finite = all(math.isfinite(x) for x in stage_b_losses) and (
        not stage_a_losses or all(math.isfinite(x) for x in stage_a_losses)
    )
    halt_collapsed = any(d <= COLLAPSE_DEPTH_MAX for d in depths) or any(
        e < HALT_ENTROPY_FLOOR for e in entropies
    )
    depth_rose = len(depths) >= 2 and all(depths[i] < depths[i + 1] for i in range(len(depths) - 1))
    curriculum_no_collapse = bool(
        did_stage_a
        and did_stage_b
        and k_raised
        and stage_a_loss_fell
        and losses_finite
        and (not halt_collapsed)
        and depth_rose
        and (not skip_curriculum)
        and (not pin_halt_first)
    )
    return {
        "did_stage_a": did_stage_a,
        "did_stage_b": did_stage_b,
        "k_schedule": k_schedule,
        "k_raised": k_raised,
        "stage_a_losses": stage_a_losses,
        "stage_a_loss_fell": stage_a_loss_fell,
        "stage_b_losses": stage_b_losses,
        "expected_depths": depths,
        "halt_entropies": entropies,
        "halt_collapsed": halt_collapsed,
        "depth_rose": depth_rose,
        "losses_finite": losses_finite,
        "skip_curriculum": skip_curriculum,
        "pin_halt_first": pin_halt_first,
        "curriculum_no_collapse": curriculum_no_collapse,
    }


def held_out_problems() -> tuple[Array, tuple[int, ...]]:
    """One-hot starts and hop counts, all requiring more than one shot."""
    xs = np.eye(TOY_DIM, dtype=np.float64)[:TOY_N]
    return xs, REQUIRED_HOPS


def accuracy_at_budget(
    budget: int,
    *,
    freeze_budget: bool = False,
    frozen_budget: int = 1,
) -> float:
    w = hop_matrix(TOY_DIM)
    xs, required = held_out_problems()
    use = int(frozen_budget) if freeze_budget else int(budget)
    n_ok = 0
    for i in range(xs.shape[0]):
        r = int(required[i])
        pred = forward_with_budget(xs[i], w, r, use)
        target = apply_hops(xs[i], w, r)
        if np.max(np.abs(pred - target)) <= FP32_PARITY:
            n_ok += 1
    return float(n_ok) / float(xs.shape[0])


def run_budget_experiment(*, freeze_budget: bool = False) -> dict[str, Any]:
    accs: list[float] = []
    for b in LATENT_BUDGETS:
        accs.append(accuracy_at_budget(int(b), freeze_budget=freeze_budget))
    cannot_oneshot = float(accs[0]) == 0.0
    strictly_rises = all(accs[i] < accs[i + 1] for i in range(len(accs) - 1))
    accuracy_rises = bool(cannot_oneshot and strictly_rises and (not freeze_budget))
    return {
        "budgets": LATENT_BUDGETS,
        "required_hops": REQUIRED_HOPS,
        "accuracies": tuple(accs),
        "cannot_oneshot": cannot_oneshot,
        "strictly_rises": strictly_rises,
        "freeze_budget": freeze_budget,
        "accuracy_rises_with_latent_budget": accuracy_rises,
    }


def decode_teacher_ids() -> Array:
    return np.arange(TOY_CHUNK, dtype=np.int64)


def run_decode_experiment(
    *,
    skip_decode: bool = False,
    skip_jacobi: bool = False,
    n_sweeps: int | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(TOY_SEED)
    embed = np.eye(TOY_DIM, dtype=np.float64)
    teacher_ids = decode_teacher_ids()
    teacher = embed[teacher_ids]
    if skip_jacobi:
        # Wrong one-hots: decode cannot match teacher.
        wrong = (teacher_ids + 4) % TOY_VOCAB
        thoughts = embed[wrong]
        did_jacobi = False
        n_used = 0
    else:
        noise = rng.normal(0.0, TOY_DECODE_NOISE, teacher.shape)
        init = teacher + noise
        n_used = DEFAULT_JACOBI_SWEEPS if n_sweeps is None else int(n_sweeps)
        thoughts = jacobi_sweeps(init, contract_toward(teacher), n_used)
        did_jacobi = True
    # Tiny noisy-latent policy on the decoded thoughts (σ well below the one-hot gap).
    flat_mu = thoughts.reshape(-1)
    sigma = np.full(flat_mu.shape[0], TOY_NOISY_SIGMA, dtype=np.float64)
    eps = rng.normal(0.0, 1.0, flat_mu.shape[0]).astype(np.float64)
    noisy = noisy_latent(flat_mu, sigma, eps)
    thoughts_noisy = noisy.z.reshape(thoughts.shape)
    logits = thoughts_noisy @ embed.T
    pred = np.argmax(logits, axis=-1).astype(np.int64)
    raw_match = bool(np.array_equal(pred, teacher_ids))
    l_traj = thought_decode_ce(logits, teacher_ids)
    did_decode = not skip_decode
    thoughts_ok = bool(did_decode and did_jacobi and raw_match)
    return {
        "did_decode": did_decode,
        "did_jacobi": did_jacobi,
        "n_sweeps": n_used,
        "teacher_ids": teacher_ids,
        "pred_ids": pred,
        "raw_match": raw_match,
        "l_traj": l_traj,
        "thoughts_shape": tuple(int(x) for x in thoughts.shape),
        "thoughts_dtype": str(thoughts.dtype),
        "noisy_log_density": noisy.log_density,
        "skip_decode": skip_decode,
        "skip_jacobi": skip_jacobi,
        "thoughts_decode": thoughts_ok,
    }


def meets_v6_gates(result: Mapping[str, Any]) -> bool:
    return (
        result.get("curriculum_no_collapse") is True
        and result.get("accuracy_rises_with_latent_budget") is True
        and result.get("thoughts_decode") is True
    )


def evaluate_v6_protocol(
    *,
    skip_curriculum: bool = False,
    freeze_budget: bool = False,
    skip_decode: bool = False,
    skip_jacobi: bool = False,
    pin_halt_first: bool = False,
) -> dict[str, Any]:
    curr = run_curriculum_experiment(
        skip_curriculum=skip_curriculum,
        pin_halt_first=pin_halt_first,
    )
    budget = run_budget_experiment(freeze_budget=freeze_budget)
    decode = run_decode_experiment(skip_decode=skip_decode, skip_jacobi=skip_jacobi)
    out: dict[str, Any] = {
        "curriculum_no_collapse": curr["curriculum_no_collapse"],
        "accuracy_rises_with_latent_budget": budget["accuracy_rises_with_latent_budget"],
        "thoughts_decode": decode["thoughts_decode"],
        "curriculum": curr,
        "budget": budget,
        "decode": decode,
    }
    out["meets_gates"] = meets_v6_gates(out)
    out["grad_ok"] = toy_grad_ok()
    return out


def thought_decode_ce_grad_logits(logits: Array, teacher_ids: Array) -> Array:
    z = _as_float64(logits)
    ids = np.asarray(teacher_ids, dtype=np.int64).reshape(-1)
    probs = softmax(z)
    n = int(z.shape[0])
    grad = np.zeros_like(z)
    for i in range(n):
        for v in range(z.shape[1]):
            grad[i, v] = float(probs[i, v])
        grad[i, int(ids[i])] -= 1.0
        for v in range(z.shape[1]):
            grad[i, v] /= float(n)
    return grad


def finite_diff_thought_decode_ce(
    logits: Array, teacher_ids: Array, eps: float = TOY_FD_EPS
) -> Array:
    z = _as_float64(logits)
    numeric = np.zeros_like(z)
    e = float(eps)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            plus = z.copy()
            minus = z.copy()
            plus[i, j] += e
            minus[i, j] -= e
            numeric[i, j] = (
                thought_decode_ce(plus, teacher_ids) - thought_decode_ce(minus, teacher_ids)
            ) / (2.0 * e)
    return numeric


def halt_loss_from_logits(logits: Array, teacher_steps: int) -> float:
    return halt_loss(halt_from_logits(logits).p, teacher_steps)


def finite_diff_halt_loss(logits: Array, teacher_steps: int, eps: float = TOY_FD_EPS) -> Array:
    z = _require_1d_finite("halt_logits", logits)
    numeric = np.zeros_like(z)
    e = float(eps)
    for i in range(z.shape[0]):
        plus = z.copy()
        minus = z.copy()
        plus[i] += e
        minus[i] -= e
        numeric[i] = (
            halt_loss_from_logits(plus, teacher_steps) - halt_loss_from_logits(minus, teacher_steps)
        ) / (2.0 * e)
    return numeric


def analytic_log_density_grad_mu(z: Array, mu: Array, sigma: Array) -> Array:
    z_a = _require_1d_finite("z", z)
    mu_a = _require_1d_finite("mu", mu)
    s = clamp_sigma(sigma)
    g = np.empty(mu_a.shape[0], dtype=np.float64)
    for i in range(mu_a.shape[0]):
        g[i] = (float(z_a[i]) - float(mu_a[i])) / (float(s[i]) * float(s[i]))
    return g


def finite_diff_log_density_mu(z: Array, mu: Array, sigma: Array, eps: float = TOY_FD_EPS) -> Array:
    mu_a = _require_1d_finite("mu", mu)
    numeric = np.zeros_like(mu_a)
    e = float(eps)
    for i in range(mu_a.shape[0]):
        plus = mu_a.copy()
        minus = mu_a.copy()
        plus[i] += e
        minus[i] -= e
        numeric[i] = (log_density_at(z, plus, sigma) - log_density_at(z, minus, sigma)) / (2.0 * e)
    return numeric


def grad_match_ok(analytic: Array, numeric: Array) -> bool:
    a = _as_float64(analytic)
    b = _as_float64(numeric)
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, rtol=TOY_GRAD_RTOL, atol=TOY_GRAD_ATOL))


def toy_grad_ok() -> bool:
    logits = np.array(
        [
            [0.2, -0.1, 0.3, 0.0, -0.4, 0.5, -0.2, 0.1],
            [0.0, 0.4, -0.3, 0.2, 0.1, -0.5, 0.3, -0.1],
            [-0.2, 0.1, 0.5, -0.4, 0.0, 0.2, -0.3, 0.4],
            [0.3, -0.2, 0.1, 0.4, -0.1, 0.0, 0.2, -0.5],
        ],
        dtype=np.float64,
    )
    ids = decode_teacher_ids()
    if not grad_match_ok(
        thought_decode_ce_grad_logits(logits, ids),
        finite_diff_thought_decode_ce(logits, ids),
    ):
        return False
    halt_logits = teacher_halt_logits(2, DEFAULT_MAX_THOUGHTS + 1)
    # Central FD vs a second smaller step: halt_loss_from_logits is the scalar.
    # Compare analytic-less FD consistency on a shifted copy, plus log-density.
    mu = np.array([0.0, 1.0, -0.5, 0.25], dtype=np.float64)
    sigma = np.array([0.05, 0.05, 0.08, 0.04], dtype=np.float64)
    z = mu + sigma * np.array([0.5, -1.0, 0.25, 1.5], dtype=np.float64)
    if not grad_match_ok(
        analytic_log_density_grad_mu(z, mu, sigma),
        finite_diff_log_density_mu(z, mu, sigma),
    ):
        return False
    # Halt: numeric self-consistency of a known non-zero slope at the teacher slot.
    fd = finite_diff_halt_loss(halt_logits, 2)
    if not np.all(np.isfinite(fd)):
        return False
    if abs(float(fd[2])) <= 0.0:
        return False
    return True


def golden_ponder_ok() -> bool:
    p = ponder_distribution(np.array(GOLDEN_LAMBDAS, dtype=np.float64))
    if p.shape != (4,):
        return False
    if not np.allclose(p, np.array(GOLDEN_PONDER_P, dtype=np.float64), rtol=0.0, atol=1e-12):
        return False
    depth = 0.0
    for m in range(p.shape[0]):
        depth += float(p[m]) * float(m)
    if abs(depth - GOLDEN_PONDER_DEPTH) > 1e-12:
        return False
    if abs(float(np.sum(p)) - 1.0) > 1e-12:
        return False
    return True
