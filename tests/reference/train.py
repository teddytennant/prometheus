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

μP-for-Muon (Moonlight arXiv 2502.16982 Lemma 1, equations 4 and 7).
This does not replace the Jordan scale inside ``muon_update``.

Lemma 1: a full-rank semi-orthogonal matrix of shape (A, B) has
RMS √(1 / max(A, B)), where RMS is √mean(square). Equation 4 multiplies
that factor by 0.2 so every shape has update RMS 0.2::

    scale = 0.2 * √max(A, B)
    m_t = β m + g                         # same Nesterov as muon_update
    O = NS(g + β m_t)
    new = param − lr * (scale * O + λ param)

Weight decay is decoupled: it hits ``param``, never the matrix that
Newton–Schulz sees. Equation 7 is the same per-matrix factor, described
as an adjusted learning rate. It is not a width ratio. Under this RMS
match the rate tuned at ``base_width`` is reused unchanged at ``width``.

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

import math
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


# Moonlight update RMS (arXiv 2502.16982 eq. 4). Duplicated so this file
# does not import production ``train``.
MOONLIGHT_UPDATE_RMS = 0.2


def _require_pos_int(name: str, value: object) -> int:
    """Integer >= 1. Bool is rejected (it is a subclass of int, not a width)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    return value


def _require_finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def muon_rms_scale(rows: int, cols: int) -> float:
    """``0.2 * sqrt(max(rows, cols))``. Lemma 1 × equation 4.

    ``rows`` and ``cols`` must be integers >= 1.
    """
    rows_i = _require_pos_int("rows", rows)
    cols_i = _require_pos_int("cols", cols)
    return float(MOONLIGHT_UPDATE_RMS * math.sqrt(max(rows_i, cols_i)))


def transferred_muon_lr(base_lr: float, *, width: int, base_width: int) -> float:
    """μP-for-Muon: the transferred rate is ``base_lr`` at every width.

    The shape factor lives in ``muon_rms_scale``, not here. ``width`` and
    ``base_width`` must be integers >= 1. ``base_lr`` must be finite and >= 0.
    """
    _require_pos_int("width", width)
    _require_pos_int("base_width", base_width)
    lr = _require_finite_real("base_lr", base_lr)
    if lr < 0.0:
        raise ValueError(f"base_lr must be >= 0, got {base_lr!r}")
    return float(base_lr)


def _as_matrix(name: str, value: Array) -> np.ndarray:
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a 2D array") from exc
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got ndim {arr.ndim}")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{name} dimensions must be >= 1, got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError(f"{name} must be a real array, got dtype {arr.dtype}")
    return arr.astype(np.float32, copy=False)


def muon_transfer_step(
    param: Array,
    grad: Array,
    momentum: Array,
    *,
    lr: float,
    momentum_coeff: float,
    ns_steps: int,
    weight_decay: float,
) -> tuple[Array, Array]:
    """One Moonlight Muon step. Slow NumPy, same Nesterov and NS as ``muon_update``.

    Equation 4::

        new = param − lr * (muon_rms_scale(A, B) * O + weight_decay * param)

    ``O`` is Newton–Schulz of the Nesterov momentum. Weight decay is applied
    to ``param`` after orthogonalization, never inside Newton–Schulz.
    ``weight_decay`` must be >= 0. ``lr`` must be finite. The three matrices
    must be 2D and the same shape. Returns float32 ``(new_param, new_momentum)``.
    """
    lr_f = _require_finite_real("lr", lr)
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)):
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay!r}")
    # NaN fails ``>= 0``. +inf is finite-checked so the reference stays defined;
    # the spec only requires ``weight_decay >= 0``.
    if not (weight_decay >= 0) or not math.isfinite(float(weight_decay)):
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay!r}")
    wd_f = float(weight_decay)
    p = _as_matrix("param", param)
    g = _as_matrix("grad", grad)
    m = _as_matrix("momentum", momentum)
    if p.shape != g.shape or p.shape != m.shape:
        raise ValueError(
            f"param, grad, and momentum must share a shape, got {p.shape}, {g.shape}, {m.shape}"
        )
    rows, cols = int(p.shape[0]), int(p.shape[1])
    beta = np.float32(momentum_coeff)
    # Same Nesterov buffer as reference ``muon_update``.
    new_m = beta * m + g
    nesterov = g + beta * new_m
    orth = newton_schulz(nesterov, ns_steps)
    scale = np.float32(muon_rms_scale(rows, cols))
    lr32 = np.float32(lr_f)
    wd32 = np.float32(wd_f)
    new_p = p - lr32 * (scale * orth + wd32 * p)
    return new_p.astype(np.float32), new_m.astype(np.float32)


def check_mup_muon_reference() -> None:
    """Identity self-check for the Moonlight oracle. Not a pytest.

    Raises AssertionError if Lemma 1, equation 4, or the width-independent
    learning rate fails on this reference.
    """
    shapes = (
        (1, 1),
        (1, 2),
        (2, 1),
        (3, 5),
        (8, 3),
        (3, 8),
        (4, 4),
        (7, 13),
        (16, 9),
        (32, 1),
        (1, 32),
        (64, 64),
        (2, 100),
        (100, 2),
        (9, 2),
        (6, 6),
    )
    for rows, cols in shapes:
        scale = muon_rms_scale(rows, cols)
        root = math.sqrt(max(rows, cols))
        assert scale == MOONLIGHT_UPDATE_RMS * root
        # 1 ulp: x * (1/x) is not always exact.
        assert abs(scale * (1.0 / root) - MOONLIGHT_UPDATE_RMS) < 1e-12
        assert muon_rms_scale(rows, cols) == muon_rms_scale(cols, rows)
        # Lemma 1 on an exact semi-orthogonal matrix, not on Newton–Schulz.
        rng = np.random.default_rng(1000 + rows * 100 + cols)
        draw = rng.standard_normal((rows, cols))
        if rows >= cols:
            q, _ = np.linalg.qr(draw)
            orth = q[:, :cols]
        else:
            q, _ = np.linalg.qr(draw.T)
            orth = q[:, :rows].T
        rms_orth = float(np.sqrt(np.mean(np.square(orth))))
        assert abs(rms_orth - 1.0 / root) < 1e-6
        rms_scaled = float(np.sqrt(np.mean(np.square(scale * orth))))
        assert abs(rms_scaled - MOONLIGHT_UPDATE_RMS) < 1e-6

    for base in (0.0, 0.02, 1.0, 1e-12, 100.0):
        for width, base_width in ((1, 1), (128, 2048), (2048, 128), (7, 13), (10**6, 3)):
            got = transferred_muon_lr(base, width=width, base_width=base_width)
            assert got == base
            assert got == transferred_muon_lr(base, width=base_width, base_width=width)

    rng = np.random.default_rng(15)
    for shape in ((1, 1), (3, 2), (2, 5), (4, 4), (8, 1), (1, 8)):
        param = rng.standard_normal(shape).astype(np.float32)
        grad = rng.standard_normal(shape).astype(np.float32)
        momentum = rng.standard_normal(shape).astype(np.float32)
        lr = 0.02
        beta = 0.95
        steps = 5
        wd = 0.1
        p_copy = param.copy()
        g_copy = grad.copy()
        m_copy = momentum.copy()
        new_p, new_m = muon_transfer_step(
            param, grad, momentum, lr=lr, momentum_coeff=beta, ns_steps=steps, weight_decay=wd
        )
        assert np.array_equal(param, p_copy)
        assert np.array_equal(grad, g_copy)
        assert np.array_equal(momentum, m_copy)
        assert new_p.shape == shape and new_m.shape == shape
        assert new_p.dtype == np.float32 and new_m.dtype == np.float32
        beta32 = np.float32(beta)
        expect_m = beta32 * momentum + grad
        assert np.allclose(new_m, expect_m, rtol=0.0, atol=0.0)
        new_m_ref = beta32 * momentum + grad
        nesterov = grad + beta32 * new_m_ref
        orth = newton_schulz(nesterov, steps)
        scale = np.float32(muon_rms_scale(*shape))
        expect_p = param - np.float32(lr) * (scale * orth + np.float32(wd) * param)
        assert np.allclose(new_p, expect_p, rtol=1e-6, atol=1e-6)
        # Weight decay is outside Newton–Schulz: the two calls share O.
        bare_p, bare_m = muon_transfer_step(
            param, grad, momentum, lr=lr, momentum_coeff=beta, ns_steps=steps, weight_decay=0.0
        )
        assert np.array_equal(bare_m, new_m)
        decay = bare_p - new_p
        assert np.allclose(decay, np.float32(lr) * np.float32(wd) * param, rtol=1e-5, atol=1e-5)
        jordan = np.float32(math.sqrt(max(1.0, shape[0] / max(shape[1], 1))))
        if abs(float(scale) - float(jordan)) > 1e-3:
            wrong = param - np.float32(lr) * jordan * orth
            assert not np.allclose(bare_p, wrong, rtol=1e-4, atol=1e-4)
        # Zero gradient and zero momentum: NS input is 0, update is pure decay.
        zeros = np.zeros(shape, dtype=np.float32)
        decay_p, decay_m = muon_transfer_step(
            param, zeros, zeros, lr=lr, momentum_coeff=beta, ns_steps=steps, weight_decay=wd
        )
        assert np.allclose(decay_m, 0.0, atol=0.0)
        assert np.allclose(decay_p, param * np.float32(1.0 - lr * wd), rtol=1e-5, atol=1e-5)
        # lr = 0 leaves the parameter alone and still writes momentum.
        held_p, held_m = muon_transfer_step(
            param, grad, momentum, lr=0.0, momentum_coeff=beta, ns_steps=steps, weight_decay=wd
        )
        assert np.array_equal(held_p, param)
        assert np.array_equal(held_m, expect_m)

    # Frozen scalars shared with tests/test_mup_muon.py. Computed here, not
    # from production.
    frozen_p = np.array([[0.5, -0.25], [0.1, 0.8], [-0.3, 0.4]], dtype=np.float32)
    frozen_g = np.array([[0.2, -0.1], [0.0, 0.3], [-0.4, 0.05]], dtype=np.float32)
    frozen_m = np.array([[0.1, 0.2], [-0.3, 0.0], [0.4, -0.2]], dtype=np.float32)
    frozen_new_p, frozen_new_m = muon_transfer_step(
        frozen_p,
        frozen_g,
        frozen_m,
        lr=0.02,
        momentum_coeff=0.95,
        ns_steps=5,
        weight_decay=0.1,
    )
    assert abs(float(frozen_new_p[0, 0]) - 0.4941366910934448) < 1e-6
    assert abs(float(frozen_new_p.sum()) - 1.2451510429382324) < 1e-6
    assert abs(float(frozen_new_m[0, 0]) - 0.29500001668930054) < 1e-6
    assert abs(float(frozen_new_m.sum()) - 0.24) < 1e-6
    bare_p, _bare_m = muon_transfer_step(
        frozen_p,
        frozen_g,
        frozen_m,
        lr=0.02,
        momentum_coeff=0.95,
        ns_steps=5,
        weight_decay=0.0,
    )
    assert abs(float(bare_p[0, 0]) - 0.49513667821884155) < 1e-6
    assert abs(float(bare_p.sum()) - 1.2476511001586914) < 1e-6

    for bad in (0, -1, 1.0, 1.5, None, "4"):
        try:
            muon_rms_scale(bad, 2)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(f"rows {bad!r} should raise")
        try:
            muon_rms_scale(2, bad)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(f"cols {bad!r} should raise")
    for bad_lr in (-0.1, float("nan"), float("inf"), float("-inf")):
        try:
            transferred_muon_lr(bad_lr, width=4, base_width=4)
        except ValueError:
            pass
        else:
            raise AssertionError(f"base_lr {bad_lr!r} should raise")
    try:
        transferred_muon_lr(0.02, width=0, base_width=4)
    except ValueError:
        pass
    else:
        raise AssertionError("width 0 should raise")
    p = np.ones((2, 3), dtype=np.float32)
    try:
        muon_transfer_step(
            p, p, p, lr=float("nan"), momentum_coeff=0.9, ns_steps=1, weight_decay=0.0
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite lr should raise")
    try:
        muon_transfer_step(p, p, p, lr=0.02, momentum_coeff=0.9, ns_steps=1, weight_decay=-1e-4)
    except ValueError:
        pass
    else:
        raise AssertionError("negative weight decay should raise")
    try:
        muon_transfer_step(
            p, p.reshape(-1), p, lr=0.02, momentum_coeff=0.9, ns_steps=1, weight_decay=0.0
        )
    except ValueError:
        pass
    else:
        raise AssertionError("1D grad should raise")
    print("mup-muon reference identities: ok")
