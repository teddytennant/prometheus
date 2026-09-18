"""Independent NumPy V1 parity protocol (spec 16.2 / 16-h200-verification).

Slow and obvious. Does **not** import ``model/``, ``train/``,
``prometheus.verify.v1_parity``, or JAX. Production ``run_v1`` must not import
this module; tests import both.

V1 meaning
----------
``logits_max_diff``
    ``max |a - b|`` over FP32 logits from production JAX ``model.forward`` vs an
    independent (non-JAX, not ``tests/``) forward on ``model.tiny_config``.
    Gate: finite and ``<= LOGITS_MAX_ABS`` (1e-5).
``grad_ok``
    Reverse-mode grads match central finite differences on a tiny parameter
    slice. Gate: True.
``overfit_ok``
    One-batch train: loss after the step(s) is **strictly** below the loss
    before. Gate: True.

The toy LM below is the protocol's reference math (embedding + linear unembed +
next-token CE). It is not the flagship / tiny_config net; production V1 runs
that net. Tests use this module to check shapes, dtypes, golden max-abs scale,
FD vs reverse-mode, and that a deliberately wrong forward exceeds 1e-5.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

Array = np.ndarray

LOGITS_MAX_ABS = 1e-5

# Toy LM (protocol only). Small so FD is cheap and obvious.
TOY_VOCAB = 8
TOY_DIM = 4
TOY_BATCH = 2
TOY_SEQ = 5
TOY_SEED = 0
TOY_FD_EPS = 1e-3
TOY_GRAD_RTOL = 5e-2
TOY_GRAD_ATOL = 5e-3
TOY_OVERFIT_LR = 0.5
TOY_OVERFIT_STEPS = 1

# Uniform logit shift that must fail the 1e-5 gate (fault injection).
WRONG_FORWARD_SHIFT = 1e-3

# Golden: identical logits → 0; a 2e-5 uniform shift sits just above the gate.
GOLDEN_MATCH_DIFF = 0.0
GOLDEN_SHIFT_ABOVE_GATE = 2e-5


def logits_max_abs_diff(a: Array, b: Array) -> float:
    """Max absolute difference of two logit tensors, compared as float32.

    Both must have the same shape. Empty arrays yield 0.0. Non-finite inputs
    yield a non-finite result (the V1 gate requires a finite value).
    """
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    if x.shape != y.shape:
        raise ValueError(f"logit shape mismatch: {x.shape} vs {y.shape}")
    if x.size == 0:
        return 0.0
    diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
    return float(np.max(diff))


def meets_logits_gate(diff: float, *, limit: float = LOGITS_MAX_ABS) -> bool:
    """True iff ``diff`` is a finite real and ``diff <= limit``."""
    d = float(diff)
    return bool(np.isfinite(d) and d <= float(limit))


def grad_match_ok(
    analytic: Array,
    finite_diff: Array,
    *,
    rtol: float = TOY_GRAD_RTOL,
    atol: float = TOY_GRAD_ATOL,
) -> bool:
    """Elementwise reverse-mode vs central differences (relative + absolute)."""
    g = np.asarray(analytic, dtype=np.float64)
    h = np.asarray(finite_diff, dtype=np.float64)
    if g.shape != h.shape:
        raise ValueError(f"grad shape mismatch: {g.shape} vs {h.shape}")
    if g.size == 0:
        return True
    if not (np.isfinite(g).all() and np.isfinite(h).all()):
        return False
    scale = np.maximum(np.abs(g), np.abs(h))
    abs_err = np.abs(g - h)
    ok = (abs_err <= float(atol) + float(rtol) * scale).all()
    return bool(ok)


def overfit_loss_dropped(loss_before: float, loss_after: float) -> bool:
    """One-batch overfit gate: loss must strictly decrease and stay finite."""
    a = float(loss_before)
    b = float(loss_after)
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return b < a


def shift_logits(logits: Array, delta: float) -> Array:
    """Deliberately wrong forward: add a uniform FP32 shift."""
    x = np.asarray(logits, dtype=np.float32)
    return (x.astype(np.float32) + np.float32(delta)).astype(np.float32)


def mean_cross_entropy(logits: Array, targets: Array) -> float:
    """Mean token CE. ``logits`` (..., V), ``targets`` (...) int ids in ``[0, V)``."""
    z = np.asarray(logits, dtype=np.float64)
    t = np.asarray(targets)
    if z.ndim < 1:
        raise ValueError("logits must have a vocab axis")
    if z.shape[:-1] != t.shape:
        raise ValueError(f"CE shape mismatch: logits {z.shape} vs targets {t.shape}")
    if z.size == 0:
        return 0.0
    v = int(z.shape[-1])
    flat = z.reshape(-1, v)
    idx = t.reshape(-1).astype(np.int64)
    if int(idx.min()) < 0 or int(idx.max()) >= v:
        raise ValueError("target id out of vocab")
    m = np.max(flat, axis=-1, keepdims=True)
    shifted = flat - m
    log_z = np.log(np.exp(shifted).sum(axis=-1)) + m.reshape(-1)
    nll = log_z - flat[np.arange(flat.shape[0]), idx]
    return float(np.mean(nll))


def toy_init(seed: int = TOY_SEED) -> dict[str, Array]:
    """Tiny embedding + unembed. N(0, 0.02) FP32, deterministic seed."""
    rng = np.random.default_rng(int(seed))
    scale = 0.02
    embed = rng.normal(0.0, scale, size=(TOY_VOCAB, TOY_DIM)).astype(np.float32)
    unembed = rng.normal(0.0, scale, size=(TOY_VOCAB, TOY_DIM)).astype(np.float32)
    return {"embed": embed, "unembed": unembed}


def toy_tokens(seed: int = TOY_SEED) -> Array:
    """``(TOY_BATCH, TOY_SEQ)`` int32 ids in ``[0, TOY_VOCAB)``."""
    rng = np.random.default_rng(int(seed) + 1)
    return rng.integers(0, TOY_VOCAB, size=(TOY_BATCH, TOY_SEQ), dtype=np.int32)


def toy_forward(tokens: Array, params: Mapping[str, Array]) -> Array:
    """``logits[b,s,:] = embed[tokens[b,s]] @ unembed.T``. FP32 ``(B, S, V)``."""
    tok = np.asarray(tokens)
    if tok.ndim != 2:
        raise ValueError("tokens must be (batch, seq)")
    embed = np.asarray(params["embed"], dtype=np.float32)
    unembed = np.asarray(params["unembed"], dtype=np.float32)
    if embed.ndim != 2 or unembed.ndim != 2:
        raise ValueError("embed and unembed must be 2D")
    if embed.shape[1] != unembed.shape[1]:
        raise ValueError("embed/unembed dim mismatch")
    if tok.size and (int(tok.min()) < 0 or int(tok.max()) >= embed.shape[0]):
        raise ValueError("token id out of vocab")
    hidden = embed[tok]  # (B, S, D)
    logits = hidden @ unembed.T
    return np.asarray(logits, dtype=np.float32)


def toy_next_token_loss(tokens: Array, params: Mapping[str, Array]) -> float:
    """Mean CE of position t predicting token t+1. Last position dropped."""
    tok = np.asarray(tokens)
    if tok.shape[1] < 2:
        raise ValueError("need seq >= 2 for next-token loss")
    logits = toy_forward(tok, params)
    return mean_cross_entropy(logits[:, :-1, :], tok[:, 1:])


def toy_reverse_mode_grads(
    tokens: Array, params: Mapping[str, Array]
) -> dict[str, Array]:
    """Hand reverse-mode of ``toy_next_token_loss`` w.r.t. embed and unembed.

    ``dL/dlogits = (softmax - onehot) / n_tokens`` on next-token positions,
    zeros on the last position. Then chain through the linear unembed and
    gather into the embedding table. Loops stay obvious; no autodiff.
    """
    tok = np.asarray(tokens)
    embed = np.asarray(params["embed"], dtype=np.float64)
    unembed = np.asarray(params["unembed"], dtype=np.float64)
    batch, seq = int(tok.shape[0]), int(tok.shape[1])
    hidden = embed[tok]  # (B, S, D)
    logits = hidden @ unembed.T  # (B, S, V)
    n_tok = float(batch * (seq - 1))

    dlogits = np.zeros_like(logits, dtype=np.float64)
    for b in range(batch):
        for s in range(seq - 1):
            row = logits[b, s]
            row = row - np.max(row)
            ex = np.exp(row)
            prob = ex / ex.sum()
            target = int(tok[b, s + 1])
            dlogits[b, s] = prob / n_tok
            dlogits[b, s, target] -= 1.0 / n_tok

    # logits = hidden @ unembed.T  →  d_unembed = dlogits^T-style @ hidden
    d_unembed = np.zeros_like(unembed, dtype=np.float64)
    d_hidden = np.zeros_like(hidden, dtype=np.float64)
    for b in range(batch):
        for s in range(seq):
            d_unembed += np.outer(dlogits[b, s], hidden[b, s])
            d_hidden[b, s] = dlogits[b, s] @ unembed

    d_embed = np.zeros_like(embed, dtype=np.float64)
    for b in range(batch):
        for s in range(seq):
            d_embed[int(tok[b, s])] += d_hidden[b, s]

    return {
        "embed": d_embed.astype(np.float32),
        "unembed": d_unembed.astype(np.float32),
    }


def finite_difference_scalar(
    loss_fn: Callable[[], float],
    getter: Callable[[], float],
    setter: Callable[[float], None],
    *,
    eps: float = TOY_FD_EPS,
) -> float:
    """Central difference ``(L(θ+ε) - L(θ-ε)) / (2ε)`` on one scalar weight."""
    theta = float(getter())
    setter(theta + float(eps))
    plus = float(loss_fn())
    setter(theta - float(eps))
    minus = float(loss_fn())
    setter(theta)
    return float((plus - minus) / (2.0 * float(eps)))


def toy_finite_diff_slice(
    tokens: Array,
    params: Mapping[str, Array],
    *,
    name: str = "unembed",
    coords: tuple[tuple[int, int], ...] = ((0, 0), (1, 2), (3, 1)),
    eps: float = TOY_FD_EPS,
) -> tuple[Array, Array]:
    """Analytic vs FD on a few ``(row, col)`` entries of one toy matrix."""
    work = {
        "embed": np.array(params["embed"], dtype=np.float32, copy=True),
        "unembed": np.array(params["unembed"], dtype=np.float32, copy=True),
    }
    analytic_full = toy_reverse_mode_grads(tokens, work)
    matrix = work[name]
    analytic = []
    numeric = []
    for i, j in coords:
        analytic.append(float(analytic_full[name][i, j]))

        def _loss() -> float:
            return toy_next_token_loss(tokens, work)

        def _get(i: int = i, j: int = j) -> float:
            return float(matrix[i, j])

        def _set(val: float, i: int = i, j: int = j) -> None:
            matrix[i, j] = np.float32(val)

        numeric.append(finite_difference_scalar(_loss, _get, _set, eps=eps))
    return (
        np.asarray(analytic, dtype=np.float32),
        np.asarray(numeric, dtype=np.float32),
    )


def toy_grad_ok(
    tokens: Array | None = None,
    params: Mapping[str, Array] | None = None,
    *,
    seed: int = TOY_SEED,
) -> bool:
    """Reverse-mode matches finite differences on a tiny unembed slice."""
    if params is None:
        params = toy_init(seed)
    if tokens is None:
        tokens = toy_tokens(seed)
    analytic, numeric = toy_finite_diff_slice(tokens, params)
    return grad_match_ok(analytic, numeric)


def toy_sgd_step(
    tokens: Array,
    params: Mapping[str, Array],
    *,
    lr: float = TOY_OVERFIT_LR,
) -> dict[str, Array]:
    """One full-batch SGD step on embed and unembed."""
    grads = toy_reverse_mode_grads(tokens, params)
    out: dict[str, Array] = {}
    for key in ("embed", "unembed"):
        out[key] = (
            np.asarray(params[key], dtype=np.float32)
            - np.float32(lr) * np.asarray(grads[key], dtype=np.float32)
        ).astype(np.float32)
    return out


def toy_overfit_ok(
    tokens: Array | None = None,
    params: Mapping[str, Array] | None = None,
    *,
    seed: int = TOY_SEED,
    lr: float = TOY_OVERFIT_LR,
    steps: int = TOY_OVERFIT_STEPS,
) -> bool:
    """One-batch train: loss after ``steps`` SGD updates is strictly lower."""
    if params is None:
        params = toy_init(seed)
    if tokens is None:
        tokens = toy_tokens(seed)
    work: dict[str, Array] = {
        "embed": np.array(params["embed"], dtype=np.float32, copy=True),
        "unembed": np.array(params["unembed"], dtype=np.float32, copy=True),
    }
    before = toy_next_token_loss(tokens, work)
    for _ in range(int(steps)):
        work = toy_sgd_step(tokens, work, lr=lr)
    after = toy_next_token_loss(tokens, work)
    return overfit_loss_dropped(before, after)


def evaluate_toy_protocol(seed: int = TOY_SEED) -> dict[str, Any]:
    """Run the three V1 checks on the toy LM (self-parity of the numpy path).

    ``logits_max_diff`` here is toy-forward vs itself (exactly 0) — the golden
    matching-forward scale. ``grad_ok`` / ``overfit_ok`` use the protocol
    definitions above.
    """
    params = toy_init(seed)
    tokens = toy_tokens(seed)
    logits = toy_forward(tokens, params)
    self_diff = logits_max_abs_diff(logits, logits)
    return {
        "logits_max_diff": self_diff,
        "grad_ok": toy_grad_ok(tokens, params),
        "overfit_ok": toy_overfit_ok(tokens, params),
        "logits": logits,
        "tokens": tokens,
        "params": params,
    }


def wrong_forward_exceeds_gate(
    logits: Array,
    *,
    shift: float = WRONG_FORWARD_SHIFT,
    limit: float = LOGITS_MAX_ABS,
) -> bool:
    """True iff a uniform shift of ``shift`` makes max-abs-diff fail the gate."""
    wrong = shift_logits(logits, shift)
    diff = logits_max_abs_diff(logits, wrong)
    return (not meets_logits_gate(diff, limit=limit)) and diff > float(limit)
