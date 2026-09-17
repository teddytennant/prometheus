"""Slow, obvious NumPy reference for A2 train math (spec 5.3, 5.4, 3.1).

This is the oracle, not the production JAX implementation. Production
``train.*`` functions must match these shapes, dtypes, and values
(atol/rtol 1e-5 in FP32) on the same inputs.

Do not import ``train``, production ``model`` math, or jax from this file.
Config objects are duck-typed (``peak_lr``, ``warmup_steps``, ``stable_steps``,
``decay_steps``, ``z_loss_weight``).

Formulas (spec 5.3 MuonClip / precision, 5.4 WSD, 3.1 soft-cap / MTP / z-loss)
--------------------------------------------------------------------------
Muon + QK-clip on 2D hidden weights; AdamW on embeddings, norms, router
biases, and latent sigma. Master weights stay conceptually FP32.

Newton–Schulz (Muon polar factor, Keller Jordan / Kimi K2)::

    X_0 = G / ||G||_F
    if G is tall (rows > cols), iterate on X^T so the Gramian is the
    smaller square; transpose back at the end.
    for s in 1..steps:                  # steps odd and >= 1
        A = X X^T
        X ← a X + (b A + c A²) X        # odd quintic
    (a, b, c) = (3.4445, -4.7750, 2.0315)

Muon Nesterov then Newton–Schulz::

    m_t = β m_{t-1} + g_t
    g_nesterov = g_t + β m_t
    Δ = lr * √max(1, rows/cols) * NS(g_nesterov)
    # caller does param ← param − Δ

QK-clip (Kimi K2, α = 1/2)::

    μ = max |q k^T|
    if μ > τ: γ = τ/μ; q ← √γ q; k ← √γ k
    else identity. After clip, max |q k^T| ≤ τ.

Decoupled AdamW, 1-based bias correction::

    m_t = β1 m + (1-β1) g
    v_t = β2 v + (1-β2) g²
    m̂ = m_t / (1 − β1^t)          # t is 1-based
    v̂ = v_t / (1 − β2^t)
    p ← p (1 − lr·wd) − lr · m̂ / (√v̂ + ε)

WSD, step 1-based, L = peak_lr, W/S/D = warmup/stable/decay::

    1 ≤ t ≤ W:           L · t / W          (linear warmup to peak)
    W < t ≤ W+S:         L                  (stable = peak)
    W+S < t ≤ W+S+D:     L · (1 − (t−W−S)/D)  (linear decay to 0)
    t = W+S+D:           0
    t > W+S+D:           0

Logit soft-capping (3.1)::

    cap · tanh(logits / cap)

Masked mean CE (logits already soft-capped). MTP head i predicts token
t+i+1 (one extra step per head index). z-loss is mean over tokens of
(logsumexp(router_probs))^2. Total::

    ce + mtp + z_loss_weight · z
"""

from __future__ import annotations

from typing import Any

import numpy as np

Array = np.ndarray

# Odd quintic coefficients for Muon Newton–Schulz (Keller Jordan).
_NS_A = 3.4445
_NS_B = -4.7750
_NS_C = 2.0315
_NS_EPS = 1e-7


def _as_f32(x: Any) -> Array:
    return np.asarray(x, dtype=np.float32)


def _logsumexp(x: Array, axis: int = -1) -> Array:
    x = _as_f32(x)
    m = np.max(x, axis=axis, keepdims=True)
    s = np.log(np.maximum(np.exp(x - m).sum(axis=axis, keepdims=True), 1e-12))
    return np.squeeze(m + s, axis=axis)


def newton_schulz(matrix: Array, steps: int) -> Array:
    """Newton–Schulz orthogonalization of a 2D gradient. Odd ``steps`` >= 1.

    Polar-factor approximation via Muon's odd quintic (see module docstring).
    Tall matrices (rows > cols) are transposed so ``X X^T`` is the compact
    Gramian; the output has the same shape as ``G``. Zero / tiny inputs map
    to zeros. Returns float32.
    """
    if int(steps) < 1 or int(steps) % 2 == 0:
        raise ValueError("newton_schulz steps must be odd and >= 1")
    G = _as_f32(matrix)
    if G.ndim != 2:
        raise ValueError(f"newton_schulz expects a 2D matrix, got shape {G.shape}")
    transposed = G.shape[0] > G.shape[1]
    X = G.T if transposed else G
    fro = float(np.linalg.norm(X))
    if fro <= _NS_EPS:
        return np.zeros_like(G)
    X = X / np.float32(fro)
    a = np.float32(_NS_A)
    b = np.float32(_NS_B)
    c = np.float32(_NS_C)
    for _ in range(int(steps)):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposed:
        X = X.T
    return np.asarray(X, dtype=np.float32)


def muon_update(
    grad: Array,
    momentum: Array,
    *,
    lr: float,
    momentum_coeff: float,
    ns_steps: int,
) -> tuple[Array, Array]:
    """Nesterov momentum then Newton–Schulz. Returns ``(delta, new_momentum)``.

    m_t = β m_{t-1} + g_t
    g_nesterov = g_t + β m_t
    Δ = lr * sqrt(max(1, rows/cols)) * NS(g_nesterov)

    ``delta`` is subtracted from the parameter (param ← param − delta).
    """
    g = _as_f32(grad)
    m = _as_f32(momentum)
    if g.shape != m.shape:
        raise ValueError(f"grad shape {g.shape} != momentum shape {m.shape}")
    if g.ndim != 2:
        raise ValueError(f"muon_update expects a 2D grad, got shape {g.shape}")
    beta = np.float32(momentum_coeff)
    new_m = beta * m + g
    nesterov = g + beta * new_m
    orth = newton_schulz(nesterov, int(ns_steps))
    rows, cols = int(orth.shape[0]), int(orth.shape[1])
    rms = np.float32(np.sqrt(max(1.0, rows / max(cols, 1))))
    delta = np.float32(lr) * rms * orth
    return delta.astype(np.float32), new_m.astype(np.float32)


def qk_clip(q: Array, k: Array, max_logit: float) -> tuple[Array, Array]:
    """Scale Q and K so max |q k^T| does not exceed ``max_logit`` (Kimi K2).

    q, k are ``(..., n, d)``; scores = einsum('...id,...jd->...ij').
    μ = max |scores|. If μ > τ, both tensors are multiplied by √(τ/μ)
    (α = 1/2 split). If μ ≤ τ, values are unchanged (float32 copies).
    """
    if max_logit <= 0:
        raise ValueError("qk_clip max_logit must be > 0")
    q32 = _as_f32(q)
    k32 = _as_f32(k)
    if q32.shape[-1] != k32.shape[-1]:
        raise ValueError("q and k last dim (head dim) must match")
    scores = np.einsum("...id,...jd->...ij", q32, k32)
    mu = float(np.max(np.abs(scores)))
    tau = float(max_logit)
    if mu <= tau or mu == 0.0:
        return q32.copy(), k32.copy()
    scale = np.float32(np.sqrt(tau / mu))
    return (q32 * scale).astype(np.float32), (k32 * scale).astype(np.float32)


def adamw_update(
    param: Array,
    grad: Array,
    m: Array,
    v: Array,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    step: int,
) -> tuple[Array, Array, Array]:
    """Decoupled AdamW. ``step`` is 1-based for bias correction.

    Returns ``(p, m, v)``. Weight decay is not inside the adaptive denom:
    p ← p (1 − lr·wd) − lr · m̂ / (√v̂ + ε).
    """
    if int(step) < 1:
        raise ValueError("adamw_update step must be 1-based (>= 1)")
    p = _as_f32(param)
    g = _as_f32(grad)
    m32 = _as_f32(m)
    v32 = _as_f32(v)
    b1 = np.float32(beta1)
    b2 = np.float32(beta2)
    m_t = b1 * m32 + (np.float32(1.0) - b1) * g
    v_t = b2 * v32 + (np.float32(1.0) - b2) * (g * g)
    t = int(step)
    m_hat = m_t / (np.float32(1.0) - b1**t)
    v_hat = v_t / (np.float32(1.0) - b2**t)
    adam_step = m_hat / (np.sqrt(v_hat) + np.float32(eps))
    p_t = p * (np.float32(1.0) - np.float32(lr) * np.float32(wd)) - np.float32(lr) * adam_step
    return p_t.astype(np.float32), m_t.astype(np.float32), v_t.astype(np.float32)


def wsd_lr(step: int, config: Any) -> float:
    """Warmup (linear), stable (peak), decay (linear to 0). ``step`` is 1-based.

    See module docstring. ``config`` needs peak_lr, warmup_steps, stable_steps,
    decay_steps. At step = warmup+stable+decay the rate is exactly 0.
    """
    if int(step) < 1:
        raise ValueError("wsd_lr step must be 1-based (>= 1)")
    peak = float(config.peak_lr)
    w = int(config.warmup_steps)
    s = int(config.stable_steps)
    d = int(config.decay_steps)
    t = int(step)
    if w > 0 and t <= w:
        return peak * (t / w)
    if t <= w + s:
        return peak
    if d <= 0 or t >= w + s + d:
        return 0.0
    decayed = t - w - s
    return peak * (1.0 - decayed / d)


def soft_cap(logits: Array, cap: float) -> Array:
    """Logit soft-capping: cap * tanh(logits / cap)."""
    if cap <= 0:
        raise ValueError("soft_cap cap must be > 0")
    x = _as_f32(logits)
    c = np.float32(cap)
    return (c * np.tanh(x / c)).astype(np.float32)


def cross_entropy(logits: Array, targets: Array, loss_mask: Array) -> Array:
    """Masked mean CE. ``logits`` are already soft-capped.

    logits (..., V), targets (...,) int, loss_mask (...,) in {0, 1}.
    Returns mean_{mask=1} [ −log softmax(logits)[target] ]. All-zero mask → 0.
    """
    logits_f = _as_f32(logits)
    targets_i = np.asarray(targets)
    mask = _as_f32(loss_mask)
    if logits_f.shape[:-1] != tuple(targets_i.shape) or logits_f.shape[:-1] != tuple(mask.shape):
        raise ValueError(
            f"shape mismatch: logits {logits_f.shape}, targets {targets_i.shape}, mask {mask.shape}"
        )
    v = int(logits_f.shape[-1])
    flat = logits_f.reshape(-1, v)
    t = targets_i.reshape(-1).astype(np.int64)
    m = mask.reshape(-1)
    shifted = flat - np.max(flat, axis=-1, keepdims=True)
    log_z = np.log(np.maximum(np.exp(shifted).sum(axis=-1), 1e-12))
    nll = -(shifted[np.arange(flat.shape[0]), t] - log_z)
    denom = float(m.sum())
    if denom <= 0.0:
        return np.float32(0.0)
    return np.float32(float((nll * m).sum() / denom))


def mtp_loss(mtp_logits: tuple[Array, ...], tokens: Array, loss_mask: Array) -> Array:
    """Mean CE of each MTP head on the token ``head_index + 1`` steps ahead.

    Head i at position t predicts tokens[..., t + i + 1]. Source and target
    positions must both be unmasked. Heads with no valid positions are skipped.
    Mean over heads of the per-head masked CE. Empty → 0.
    """
    tokens_i = np.asarray(tokens)
    mask = _as_f32(loss_mask)
    seq = int(tokens_i.shape[-1])
    losses: list[Array] = []
    for i, logits in enumerate(mtp_logits):
        offset = i + 1
        if offset >= seq:
            continue
        logits_f = _as_f32(logits)
        src = logits_f[..., : seq - offset, :]
        tgt = tokens_i[..., offset:]
        m = mask[..., offset:] * mask[..., : seq - offset]
        losses.append(cross_entropy(src, tgt, m))
    if not losses:
        return np.float32(0.0)
    stacked = np.stack([np.asarray(x, dtype=np.float32) for x in losses])
    return np.float32(float(np.mean(stacked)))


def z_loss(router_probs: Array) -> Array:
    """Mean squared log-sum-exp of router probabilities, per token then mean.

    router_probs (..., n_experts) → mean( logsumexp(probs, axis=-1)^2 ).
    """
    p = _as_f32(router_probs)
    lse = _logsumexp(p, axis=-1)
    return np.float32(float(np.mean(lse**2)))


def total_loss(ce: Array, mtp: Array, z: Array, config: Any) -> Array:
    """ce + mtp + z_loss_weight * z."""
    w = np.float32(float(config.z_loss_weight))
    return (_as_f32(ce) + _as_f32(mtp) + w * _as_f32(z)).astype(np.float32)
