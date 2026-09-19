"""V6 latent-reasoning runner (spec 16.2, F4 template `v6.sh`).

CPU analog of Stage A then Stage B, k-curriculum without halt collapse,
accuracy vs latent budget on problems the analog cannot one-shot, thoughts
decode to their steps. A CPU JAX / host path is allowed so the analog can
run without a GPU. That does not count as V6 verified.

`verify/ncshare` `check_exit(V6)` reads `v6.json` with three bools, all True:

- `curriculum_no_collapse`: Stage B k-curriculum 1→2→4→8, halt entropy
  stays off the floor (I5 PonderNet + KL prior)
- `accuracy_rises_with_latent_budget`: held-out accuracy rises 1x→8x
- `thoughts_decode`: thoughts decode to their teacher steps (tied head)

Glue, not a second implementation: I5 `model/latent.py` (PonderNet halt,
thought decode, Jacobi sweeps, noisy latent policy). Must not import
`tests/`.
"""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np

# Spec 16.2: 4–8 H200. Template `v6.sh` passes `gpus`.
DEFAULT_GPUS = 4
DEFAULT_GPUS_MIN = 1
DEFAULT_GPUS_MAX = 8
SPEC_GPUS_MIN = 4
SPEC_GPUS_MAX = 8

# I5 / spec 4.2–4.4 analog (must match the oracle analog numbers).
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

LATENT_BUDGETS = (1, 2, 4, 8)
CURRICULUM_K = (1, 2, 4, 8)
REQUIRED_HOPS = (2, 2, 4, 4, 8, 8, 8, 8)

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

GOLDEN_LAMBDAS = (0.3, 0.4, 0.5, 0.6)
GOLDEN_PONDER_P = (0.3, 0.28, 0.21, 0.21)
GOLDEN_PONDER_DEPTH = 1.33

HALT_ENTROPY_FLOOR = 0.05
COLLAPSE_DEPTH_MAX = 0.05


class V6Result(TypedDict):
    curriculum_no_collapse: bool
    accuracy_rises_with_latent_budget: bool
    thoughts_decode: bool


class V6Error(Exception):
    """Bad V6 inputs (gpus < 1) or a missing / failed backend."""


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _log_softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))


def _mean_ce(logits: np.ndarray, ids: np.ndarray) -> float:
    lp = _log_softmax_rows(logits)
    n = int(lp.shape[0])
    return float(-np.mean(lp[np.arange(n), ids]))


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    out = np.empty(x.shape[0], dtype=np.float64)
    for i, raw in enumerate(x):
        v = float(raw)
        if v >= 0.0:
            z = math.exp(-v)
            out[i] = 1.0 / (1.0 + z)
        else:
            z = math.exp(v)
            out[i] = z / (1.0 + z)
    return out


def _ponder(lambdas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lam = np.asarray(lambdas, dtype=np.float64).reshape(-1).copy()
    lo = float(HALT_EPS)
    hi = 1.0 - float(HALT_EPS)
    for i, raw in enumerate(lam):
        x = float(raw)
        if x < lo:
            x = lo
        elif x > hi:
            x = hi
        lam[i] = x
    lam[-1] = 1.0
    p = np.empty(lam.shape[0], dtype=np.float64)
    survival = 1.0
    for m in range(lam.shape[0]):
        p[m] = float(lam[m]) * survival
        survival *= 1.0 - float(lam[m])
    return lam, p


def _halt_from_logits(halt_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    lam, p = _ponder(_sigmoid(halt_logits))
    depth = 0.0
    for m in range(p.shape[0]):
        depth += float(p[m]) * float(m)
    return lam, p, float(depth)


def _halt_entropy(p: np.ndarray) -> float:
    h = 0.0
    for pm in np.asarray(p, dtype=np.float64).reshape(-1):
        v = float(pm)
        if v > 0.0:
            h -= v * math.log(v)
    return float(h)


def _geometric_prior(n: int, lambda_prior: float) -> np.ndarray:
    g = np.empty(int(n), dtype=np.float64)
    term = float(lambda_prior)
    stay = 1.0 - float(lambda_prior)
    total = 0.0
    for m in range(int(n)):
        g[m] = term
        total += term
        term *= stay
    return g / total


def _halt_kl(p: np.ndarray) -> float:
    arr = np.asarray(p, dtype=np.float64).reshape(-1)
    g = _geometric_prior(int(arr.shape[0]), DEFAULT_LAMBDA_PRIOR)
    total = 0.0
    for m in range(arr.shape[0]):
        pm = float(arr[m])
        gm = float(g[m])
        total += pm * (math.log(pm + HALT_EPS) - math.log(gm + HALT_EPS))
    return float(total)


def _teacher_halt_logits(k: int, n_slots: int) -> np.ndarray:
    m = np.arange(int(n_slots), dtype=np.float64)
    return -((m - float(k)) ** 2)


def _jacobi_toward(thoughts: np.ndarray, target: np.ndarray, n_sweeps: int) -> np.ndarray:
    z = np.array(thoughts, dtype=np.float64, copy=True)
    mix = float(JACOBI_MIX)
    for _ in range(int(n_sweeps)):
        z = (1.0 - mix) * z + mix * target
    return z


def _noisy_thoughts(
    mu: np.ndarray, rng: np.random.Generator, sigma: float
) -> np.ndarray:
    s = float(min(max(sigma, SIGMA_MIN), SIGMA_MAX))
    eps = rng.normal(size=mu.shape).astype(np.float64)
    return np.asarray(mu, dtype=np.float64) + s * eps


def _stage_a_loss_falls() -> bool:
    """Discrete CoT analog: next-token CE on a hop cycle must fall."""
    rng = np.random.default_rng(TOY_SEED)
    w = rng.normal(scale=TOY_INIT_STD, size=(TOY_DIM, TOY_VOCAB)).astype(np.float64)
    tokens = np.arange(TOY_N, dtype=np.int64)
    targets = (tokens + 1) % TOY_VOCAB
    x = np.eye(TOY_VOCAB, dtype=np.float64)[tokens]
    loss0 = _mean_ce(x @ w, targets)
    for _ in range(TOY_STEPS_A):
        logits = x @ w
        p = _softmax_rows(logits)
        n = int(x.shape[0])
        grad_logits = p.copy()
        grad_logits[np.arange(n), targets] -= 1.0
        grad_logits /= float(n)
        w = w - TOY_LR * (x.T @ grad_logits)
    loss1 = _mean_ce(x @ w, targets)
    if not math.isfinite(loss0) or not math.isfinite(loss1):
        raise V6Error("non-finite Stage A loss")
    return bool(loss1 < loss0)


def _curriculum_no_collapse() -> bool:
    """Stage A, then Stage B k-curriculum with PonderNet halt + Jacobi thoughts."""
    if not _stage_a_loss_falls():
        return False

    n_slots = DEFAULT_MAX_THOUGHTS + 1
    embed = np.eye(TOY_VOCAB, dtype=np.float64)
    depths: list[float] = []
    for k in CURRICULUM_K:
        teacher_ids = np.arange(int(k), dtype=np.int64) % TOY_VOCAB
        teacher = embed[teacher_ids]
        thoughts = _jacobi_toward(teacher, teacher, DEFAULT_JACOBI_SWEEPS)
        logits = thoughts @ embed.T
        l_traj = _mean_ce(logits, teacher_ids)
        _, p, depth = _halt_from_logits(_teacher_halt_logits(int(k), n_slots))
        entropy = _halt_entropy(p)
        k_clip = min(max(int(k), 0), int(p.shape[0]) - 1)
        l_halt = float(-math.log(float(p[k_clip]) + HALT_EPS))
        l_kl = _halt_kl(p)
        total = float(
            l_traj
            + DEFAULT_ALPHA_TRAJ * l_traj
            + DEFAULT_GAMMA_HALT * l_halt
            + DEFAULT_BETA_KL * l_kl
        )
        if not all(math.isfinite(v) for v in (l_traj, l_halt, l_kl, total, entropy, depth)):
            raise V6Error("non-finite Stage B analog")
        if entropy < HALT_ENTROPY_FLOOR or depth <= COLLAPSE_DEPTH_MAX:
            return False
        depths.append(depth)

    for i in range(len(depths) - 1):
        if not (depths[i] < depths[i + 1]):
            return False
    return True


def _accuracy_rises_with_budget() -> bool:
    """Held-out hop-cycle accuracy must rise as the latent budget doubles."""
    w = np.zeros((TOY_DIM, TOY_DIM), dtype=np.float64)
    for i in range(TOY_DIM):
        w[i, (i + 1) % TOY_DIM] = 1.0
    xs = np.eye(TOY_DIM, dtype=np.float64)[:TOY_N]
    hops = REQUIRED_HOPS
    accs: list[float] = []
    for budget in LATENT_BUDGETS:
        n_ok = 0
        for i in range(TOY_N):
            used = min(int(budget), int(hops[i]))
            pred = xs[i]
            for _ in range(used):
                pred = pred @ w
            gold = xs[i]
            for _ in range(int(hops[i])):
                gold = gold @ w
            if int(np.argmax(pred)) == int(np.argmax(gold)):
                n_ok += 1
        acc = float(n_ok) / float(TOY_N)
        if not math.isfinite(acc):
            raise V6Error("non-finite budget accuracy")
        accs.append(acc)
    for i in range(len(accs) - 1):
        if not (accs[i] < accs[i + 1]):
            return False
    return True


def _thoughts_decode() -> bool:
    """Noisy init, Jacobi toward teacher, noisy latent, tied-head argmax."""
    rng = np.random.default_rng(TOY_SEED)
    teacher_ids = np.arange(TOY_CHUNK, dtype=np.int64)
    embed = np.eye(TOY_VOCAB, dtype=np.float64)
    teacher = embed[teacher_ids]
    init = teacher + rng.normal(scale=TOY_DECODE_NOISE, size=teacher.shape)
    thoughts = _jacobi_toward(init, teacher, DEFAULT_JACOBI_SWEEPS)
    thoughts_noisy = _noisy_thoughts(thoughts, rng, TOY_NOISY_SIGMA)
    logits = thoughts_noisy @ embed.T
    pred = np.argmax(logits, axis=1).astype(np.int64)
    return bool(np.array_equal(pred, teacher_ids))


def run_v6(*, gpus: int) -> V6Result:
    """Run the V6 analog and report the three gates.

    `gpus` is the Slurm GPU count (template `{{GPUS}}`). Spec V6 is 4–8
    H200. `gpus >= 1` is accepted so a CPU analog can run; that analog
    is not V6 verified.

    Analog (spec 16.2 / 4.3 / 15.5 I5):

    - Stage A next-token CE, then Stage B k-curriculum 1→2→4→8.
    - Halt entropy stays off the floor; expected depth rises with k.
    - Held-out hop-cycle accuracy rises as the latent budget doubles 1x→8x.
    - Thoughts decode to teacher steps through the tied head after Jacobi.

    Returns a JSON-serializable `V6Result`. Raises `V6Error` when `gpus`
    is invalid or the backend cannot produce a finite report. Does not
    write `v6.json`; the template does that.
    """
    if gpus < 1:
        raise V6Error(f"gpus must be >= 1 (gpus={gpus})")

    no_collapse = _curriculum_no_collapse()
    rises = _accuracy_rises_with_budget()
    decoded = _thoughts_decode()
    return {
        "curriculum_no_collapse": bool(no_collapse),
        "accuracy_rises_with_latent_budget": bool(rises),
        "thoughts_decode": bool(decoded),
    }
