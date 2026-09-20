"""Slow, obvious NumPy reference for A3 kernels (spec 5.1, 3.1 routing).

This is the oracle math, not the production JAX / Triton implementation.
Production ``kernels.*`` must match these shapes, dtypes, and values on the
same inputs. Chunk size does not appear in the linear-attention math.

Do not import ``kernels`` or ``jax``. Duck-type production dataclasses.

Gated delta-rule
----------------
q, k, v : (B, S, H, D) used as-is (no L2-norm, no 1/sqrt(D) scale).
beta    : (B, S, H) in (0, 1].
state   : (B, H, D, D) left-multiplied by q; zeros if omitted.

Per timestep, with I the D×D identity::

    S_t = (I - β_t k_t k_t^T) S_{t-1} + β_t k_t v_t^T
    o_t = q_t S_t

This is the Gated DeltaNet / KDA recurrence with decay gate 1. The fused
reverse-state VJP is ``gated_delta_rule_vjp``; production
``jax.grad(chunked_delta_rule)`` must match it (spec 5.1). The A1
``model.linear_attention`` wrapper may L2-normalise and pass β=1 into this
kernel; the kernel itself does not.

FP8
---
E4M3FN (Hopper training default): 1 sign, 4 exp, 3 mantissa, bias 7, no inf,
max finite 448, exp=15 mantissa=7 is NaN. Per-block abs-max along the last
(contracting) axis; scale = amax/448 (1.0 if the block is all zeros). The
quantized payload ``q`` is the E4M3 bit pattern stored as int8.

``fp8_linear`` is y = x_hat @ w_hat^T with both sides quantised then dequantised.
Backward (STE) reuses those forward dequantised tensors; scales are not
differentiated. ``fp8_linear_vjp(x, weight, g)`` is the jax.grad-facing
form of that VJP (freeze x_hat, w_hat then FP32 linear backward).
``fp8_linear_scale_path_grad`` is the *wrong* VJP that differentiates
dequant through live per-block scales; production jax.grad must not match it.

EP dispatch
-----------
Each token is copied to each of its top-k experts, padded to max_per_expert
along the slot axis. Token order within an expert is increasing token index,
then k. A token whose chosen experts span more than ``max_racks`` distinct
racks is an error. ``ep_combine`` is the inverse: weighted sum with ``probs``.

The fused permutation VJP is ``ep_dispatch_vjp`` (scatter cotangents onto
tokens via residual) and ``ep_combine_vjp`` (weighted-scatter cotangents
onto expert slots using ``probs`` and residual). Production
``jax.grad(ep_dispatch)`` / ``jax.grad(ep_combine)`` must match these.
``meta`` routing ids and residual are not differentiated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

Array = np.ndarray

DEFAULT_CHUNK = 64
DEFAULT_FP8_BLOCK = 128
MAX_RACKS = 4
FP8_E4M3_MAX = 448.0
# E4M3FN min positive subnormal: 2^{-9}.
_FP8_E4M3_MIN_SUB = 2.0 ** -9


def _as_f32(x: Any) -> Array:
    return np.asarray(x, dtype=np.float32)


def _e4m3_positive_table() -> tuple[Array, Array]:
    """Finite non-negative E4M3FN values and their 8-bit codes (sign=0)."""
    vals: list[float] = []
    codes: list[int] = []
    for m in range(8):
        vals.append(m * _FP8_E4M3_MIN_SUB)
        codes.append(m)
    for e in range(1, 15):
        for m in range(8):
            vals.append((2.0 ** (e - 7)) * (1.0 + m / 8.0))
            codes.append((e << 3) | m)
    for m in range(7):
        vals.append((2.0 ** (15 - 7)) * (1.0 + m / 8.0))
        codes.append((15 << 3) | m)
    return np.asarray(vals, dtype=np.float32), np.asarray(codes, dtype=np.uint8)


_E4M3_POS, _E4M3_POS_CODES = _e4m3_positive_table()


def _to_e4m3_bits(x: Array) -> Array:
    """Round to nearest finite E4M3FN; ties to the smaller-magnitude code."""
    x = np.asarray(x, dtype=np.float32)
    sign = np.signbit(x).astype(np.uint8)
    ax = np.abs(x)
    ax = np.nan_to_num(ax, nan=FP8_E4M3_MAX, posinf=FP8_E4M3_MAX, neginf=FP8_E4M3_MAX)
    ax = np.minimum(ax, np.float32(FP8_E4M3_MAX))
    diffs = np.abs(ax[..., None] - _E4M3_POS)
    idx = np.argmin(diffs, axis=-1)
    codes = _E4M3_POS_CODES[idx]
    return np.asarray(codes | (sign << 7), dtype=np.uint8)


def _from_e4m3_bits(bits: Array) -> Array:
    bits = np.asarray(bits, dtype=np.uint8)
    sign = np.where((bits & np.uint8(0x80)) != 0, np.float32(-1.0), np.float32(1.0))
    exp = ((bits >> np.uint8(3)) & np.uint8(0x0F)).astype(np.int32)
    man = (bits & np.uint8(0x07)).astype(np.int32)
    sub = exp == 0
    nan = (exp == 15) & (man == 7)
    val_sub = man.astype(np.float32) * np.float32(_FP8_E4M3_MIN_SUB)
    val_norm = (np.float32(2.0) ** (exp.astype(np.float32) - np.float32(7.0))) * (
        np.float32(1.0) + man.astype(np.float32) / np.float32(8.0)
    )
    val = np.where(sub, val_sub, val_norm)
    val = np.where(nan, np.float32(np.nan), val)
    return sign * val


@dataclass
class Fp8Meta:
    """Quantized payload + per-block scales. Mirrors production ``kernels.Fp8Meta``."""

    q: Array
    scale: Array
    block: int = DEFAULT_FP8_BLOCK
    dtype: str = "fp8"


def gated_delta_rule(
    q: Array,
    k: Array,
    v: Array,
    beta: Array,
    state: Array | None = None,
    config: Any | None = None,
) -> tuple[Array, Array]:
    """Explicit per-timestep gated delta-rule. ``config.chunk`` is ignored."""
    del config
    q = _as_f32(q)
    k = _as_f32(k)
    v = _as_f32(v)
    beta = _as_f32(beta)
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, v must all have shape (B, S, H, D)")
    batch, seq, heads, dim = q.shape
    if beta.shape != (batch, seq, heads):
        raise ValueError(f"beta shape {beta.shape} != {(batch, seq, heads)}")
    if state is None:
        s = np.zeros((batch, heads, dim, dim), dtype=np.float32)
    else:
        s = _as_f32(state).copy()
        if s.shape != (batch, heads, dim, dim):
            raise ValueError(f"state shape {s.shape} != {(batch, heads, dim, dim)}")
    eye = np.eye(dim, dtype=np.float32)
    out = np.zeros((batch, seq, heads, dim), dtype=np.float32)
    for t in range(seq):
        kt = k[:, t, :, :]
        vt = v[:, t, :, :]
        qt = q[:, t, :, :]
        bt = beta[:, t, :, None, None]
        kk = np.einsum("bhd,bhe->bhde", kt, kt)
        decay = eye[None, None, :, :] - bt * kk
        s = np.einsum("bhij,bhjk->bhik", decay, s)
        s = s + bt * np.einsum("bhd,bhe->bhde", kt, vt)
        out[:, t, :, :] = np.einsum("bhd,bhde->bhe", qt, s)
    return out.astype(np.float32), s.astype(np.float32)


def gated_delta_rule_vjp(
    q: Array,
    k: Array,
    v: Array,
    beta: Array,
    go: Array,
    gs: Array,
    state: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array]:
    """Fused reverse-state VJP of ``gated_delta_rule``.

    ``go`` is dL/d out, ``gs`` is dL/d next_state.
    Returns ``(gq, gk, gv, gbeta, gstate0)``.

    Production ``jax.grad(chunked_delta_rule)`` / ``jax.vjp`` must match this
    kernel and ``chunked_delta_rule_bwd`` (spec 5.1).
    """
    q = _as_f32(q)
    k = _as_f32(k)
    v = _as_f32(v)
    beta = _as_f32(beta)
    go = _as_f32(go)
    gs = _as_f32(gs)
    batch, seq, heads, dim = q.shape
    if state is None:
        s0 = np.zeros((batch, heads, dim, dim), dtype=np.float32)
    else:
        s0 = _as_f32(state).copy()
    eye = np.eye(dim, dtype=np.float32)
    s = s0
    s_hist = [s.copy()]
    a_hist: list[Array] = []
    for t in range(seq):
        kt = k[:, t, :, :]
        vt = v[:, t, :, :]
        bt = beta[:, t, :, None, None]
        kk = np.einsum("bhd,bhe->bhde", kt, kt)
        decay = eye[None, None, :, :] - bt * kk
        s = np.einsum("bhij,bhjk->bhik", decay, s)
        s = s + bt * np.einsum("bhd,bhe->bhde", kt, vt)
        a_hist.append(decay)
        s_hist.append(s.copy())

    gq = np.zeros_like(q)
    gk = np.zeros_like(k)
    gv = np.zeros_like(v)
    gbeta = np.zeros_like(beta)
    g_s = gs.copy()
    for t in range(seq - 1, -1, -1):
        qt = q[:, t, :, :]
        kt = k[:, t, :, :]
        vt = v[:, t, :, :]
        bt = beta[:, t, :]
        s_t = s_hist[t + 1]
        s_prev = s_hist[t]
        a_t = a_hist[t]
        g_o = go[:, t, :, :]
        gq[:, t, :, :] = np.einsum("bhe,bhde->bhd", g_o, s_t)
        g_s = g_s + np.einsum("bhd,bhe->bhde", qt, g_o)
        g_s_prev = np.einsum("bhij,bhik->bhjk", a_t, g_s)
        g_a = np.einsum("bhik,bhjk->bhij", g_s, s_prev)
        gv[:, t, :, :] = np.einsum("bhde,bhd->bhe", g_s, kt) * bt[:, :, None]
        gk_v = np.einsum("bhde,bhe->bhd", g_s, vt) * bt[:, :, None]
        gbeta_v = np.einsum("bhde,bhd,bhe->bh", g_s, kt, vt)
        gbeta_a = -np.einsum("bhde,bhd,bhe->bh", g_a, kt, kt)
        gk_a = -bt[:, :, None] * (
            np.einsum("bhde,bhe->bhd", g_a, kt) + np.einsum("bhed,bhe->bhd", g_a, kt)
        )
        gk[:, t, :, :] = gk_v + gk_a
        gbeta[:, t, :] = gbeta_v + gbeta_a
        g_s = g_s_prev
    return (
        gq.astype(np.float32),
        gk.astype(np.float32),
        gv.astype(np.float32),
        gbeta.astype(np.float32),
        g_s.astype(np.float32),
    )


def _pad_blocks(x: Array, block: int) -> tuple[Array, int, int]:
    x = np.asarray(x)
    n = int(x.shape[-1])
    if block < 1:
        raise ValueError("fp8 block must be >= 1")
    n_blocks = (n + block - 1) // block
    pad = n_blocks * block - n
    if pad:
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, pad)]
        x = np.pad(x, pad_width)
    return x.reshape(*x.shape[:-1], n_blocks, block), n, n_blocks


def fp8_quantize(x: Array, *, block: int = DEFAULT_FP8_BLOCK) -> Fp8Meta:
    """Per-block abs-max scale, round to E4M3FN. ``x`` is FP32 (or cast to it)."""
    x = _as_f32(x)
    if x.ndim < 1:
        raise ValueError("fp8_quantize expects at least a 1-D tensor")
    blocked, n, n_blocks = _pad_blocks(x, block)
    amax = np.max(np.abs(blocked), axis=-1)
    scale = np.empty_like(amax, dtype=np.float32)
    zero = amax == 0
    scale[zero] = np.float32(1.0)
    scale[~zero] = amax[~zero] / np.float32(FP8_E4M3_MAX)
    scaled = blocked / scale[..., None]
    bits = _to_e4m3_bits(scaled)
    bits = bits.reshape(*x.shape[:-1], n_blocks * block)[..., :n]
    q = bits.astype(np.int8)
    return Fp8Meta(q=q, scale=scale.astype(np.float32), block=int(block), dtype="fp8")


def fp8_dequantize(meta: Any) -> Array:
    """Unpack int8-view E4M3 + per-block scales to FP32."""
    q = np.asarray(meta.q)
    scale = _as_f32(meta.scale)
    block = int(meta.block)
    bits = q.astype(np.uint8)
    blocked, n, n_blocks = _pad_blocks(bits, block)
    decoded = _from_e4m3_bits(blocked)
    if scale.shape != tuple(decoded.shape[:-1]):
        raise ValueError(
            f"scale shape {scale.shape} does not match blocked payload {decoded.shape[:-1]}"
        )
    restored = decoded * scale[..., None]
    restored = restored.reshape(*restored.shape[:-2], n_blocks * block)[..., :n]
    return restored.astype(np.float32)


def fp8_linear(x: Array, weight: Array, *, block: int = DEFAULT_FP8_BLOCK) -> Array:
    """y = x_hat @ w_hat^T with both sides fake-quantised per-block to E4M3."""
    x = _as_f32(x)
    weight = _as_f32(weight)
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D (out, in), got {weight.shape}")
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"contracting dim mismatch: x[..., {x.shape[-1]}] vs weight[..., {weight.shape[-1]}]"
        )
    x_hat = fp8_dequantize(fp8_quantize(x, block=block))
    w_hat = fp8_dequantize(fp8_quantize(weight, block=block))
    return np.matmul(x_hat, np.swapaxes(w_hat, -1, -2)).astype(np.float32)


def fp8_linear_fwd(
    x: Array, weight: Array, block: int
) -> tuple[Array, tuple[Fp8Meta, Fp8Meta]]:
    """Forward of ``fp8_linear``; residual holds the forward quantised tensors."""
    x_meta = fp8_quantize(x, block=block)
    w_meta = fp8_quantize(weight, block=block)
    y = np.matmul(fp8_dequantize(x_meta), np.swapaxes(fp8_dequantize(w_meta), -1, -2))
    return y.astype(np.float32), (x_meta, w_meta)


def fp8_linear_bwd(residual: Any, g: Array) -> tuple[Array, Array]:
    """STE backward using forward scales / dequantised tensors, not new quantisation.

    residual is ``(x_meta, w_meta)`` from ``fp8_linear_fwd``.
    dL/dx = g @ w_hat, dL/dw = g^T @ x_hat (batch axes flattened).
    """
    x_meta, w_meta = residual
    x_hat = fp8_dequantize(x_meta)
    w_hat = fp8_dequantize(w_meta)
    g = _as_f32(g)
    in_f = int(x_hat.shape[-1])
    out_f = int(w_hat.shape[0])
    g_f = g.reshape(-1, out_f)
    x_f = x_hat.reshape(-1, in_f)
    grad_x = np.matmul(g_f, w_hat).reshape(x_hat.shape)
    grad_w = np.matmul(g_f.T, x_f)
    return grad_x.astype(np.float32), grad_w.astype(np.float32)


def fp8_linear_vjp(
    x: Array, weight: Array, g: Array, *, block: int = DEFAULT_FP8_BLOCK
) -> tuple[Array, Array]:
    """STE VJP of ``fp8_linear`` for comparison with ``jax.vjp`` / ``jax.grad``.

    Independent of production internals: fake-quant both operands (per-block
    E4M3FN), freeze those dequantised tensors, then the FP32 linear VJP.
    Scales are not differentiated.
    """
    x = _as_f32(x)
    weight = _as_f32(weight)
    g = _as_f32(g)
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D (out, in), got {weight.shape}")
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"contracting dim mismatch: x[..., {x.shape[-1]}] vs weight[..., {weight.shape[-1]}]"
        )
    x_hat = fp8_dequantize(fp8_quantize(x, block=block))
    w_hat = fp8_dequantize(fp8_quantize(weight, block=block))
    in_f = int(x_hat.shape[-1])
    out_f = int(w_hat.shape[0])
    g_f = g.reshape(-1, out_f)
    x_f = x_hat.reshape(-1, in_f)
    grad_x = np.matmul(g_f, w_hat).reshape(x_hat.shape)
    grad_w = np.matmul(g_f.T, x_f)
    return grad_x.astype(np.float32), grad_w.astype(np.float32)


def _grad_through_live_block_scale(operand: Array, g_hat: Array, block: int) -> Array:
    """dL/doperand with hat = decode(q)*scale, q frozen, scale = amax/448."""
    operand = _as_f32(operand)
    g_hat = _as_f32(g_hat)
    blocked, n, n_blocks = _pad_blocks(operand, block)
    g_blocked, _, _ = _pad_blocks(g_hat, block)
    amax = np.max(np.abs(blocked), axis=-1)
    scale = np.where(
        amax == 0, np.float32(1.0), amax / np.float32(FP8_E4M3_MAX)
    ).astype(np.float32)
    hat = fp8_dequantize(fp8_quantize(operand, block=block))
    hat_blocked, _, _ = _pad_blocks(hat, block)
    raw = hat_blocked / scale[..., None]
    g_scale = np.sum(g_blocked * raw, axis=-1)
    g_amax = np.where(amax == 0, np.float32(0.0), g_scale / np.float32(FP8_E4M3_MAX))
    is_amax = np.abs(blocked) == amax[..., None]
    g_blocked_out = np.where(is_amax, g_amax[..., None] * np.sign(blocked), np.float32(0.0))
    g_out = g_blocked_out.reshape(*operand.shape[:-1], n_blocks * block)[..., :n]
    return g_out.astype(np.float32)


def fp8_linear_scale_path_grad(
    x: Array, weight: Array, g: Array, *, block: int = DEFAULT_FP8_BLOCK
) -> tuple[Array, Array]:
    """Wrong VJP: freeze E4M3 codes, differentiate dequant through live scales.

    Used only as a counterexample. STE / ``jax.grad(fp8_linear)`` must not
    match this: scales are not differentiated.
    """
    x = _as_f32(x)
    weight = _as_f32(weight)
    g = _as_f32(g)
    x_hat = fp8_dequantize(fp8_quantize(x, block=block))
    w_hat = fp8_dequantize(fp8_quantize(weight, block=block))
    in_f = int(x_hat.shape[-1])
    out_f = int(w_hat.shape[0])
    g_f = g.reshape(-1, out_f)
    g_xhat = np.matmul(g_f, w_hat).reshape(x_hat.shape)
    g_what = np.matmul(g_f.T, x_hat.reshape(-1, in_f))
    grad_x = _grad_through_live_block_scale(x, g_xhat, block)
    grad_w = _grad_through_live_block_scale(weight, g_what, block)
    return grad_x.astype(np.float32), grad_w.astype(np.float32)


def _meta_fields(meta: Any) -> tuple[Array, Array, Array, int, int]:
    expert_ids = np.asarray(meta.expert_ids)
    probs = _as_f32(meta.probs)
    racks = np.asarray(meta.racks)
    n_experts = int(meta.n_experts)
    max_racks = int(getattr(meta, "max_racks", MAX_RACKS))
    return expert_ids, probs, racks, n_experts, max_racks


@dataclass
class _DispatchResidual:
    token_index: Array
    k_index: Array
    max_per_expert: int


def ep_dispatch(tokens: Array, meta: Any) -> tuple[Array, _DispatchResidual]:
    """Copy each token to its top-k experts; pad slots to max_per_expert.

    Raises ValueError if a token's chosen experts span more than max_racks
    distinct racks, or if an expert id is out of range.
    """
    tokens = _as_f32(tokens)
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be (n_tokens, d_model), got {tokens.shape}")
    expert_ids, probs, racks, n_experts, max_racks = _meta_fields(meta)
    del probs
    n_tokens, d_model = tokens.shape
    if expert_ids.ndim != 2 or expert_ids.shape[0] != n_tokens:
        raise ValueError("expert_ids must be (n_tokens, top_k)")
    if racks.shape != (n_experts,):
        raise ValueError(f"racks length {racks.shape} != n_experts={n_experts}")
    top_k = int(expert_ids.shape[1])
    for t in range(n_tokens):
        chosen = [int(expert_ids[t, k]) for k in range(top_k)]
        for e in chosen:
            if e < 0 or e >= n_experts:
                raise ValueError(f"expert id {e} out of range [0, {n_experts})")
        rack_ids = {int(racks[e]) for e in chosen}
        if len(rack_ids) > max_racks:
            raise ValueError(
                f"token {t} experts span {len(rack_ids)} racks > max_racks={max_racks}"
            )
    counts = np.zeros((n_experts,), dtype=np.int32)
    for t in range(n_tokens):
        for k in range(top_k):
            counts[int(expert_ids[t, k])] += 1
    max_per = int(counts.max()) if n_experts > 0 else 0
    dispatched = np.zeros((n_experts, max_per, d_model), dtype=np.float32)
    token_index = np.full((n_experts, max_per), -1, dtype=np.int32)
    k_index = np.full((n_experts, max_per), -1, dtype=np.int32)
    fill = np.zeros((n_experts,), dtype=np.int32)
    for t in range(n_tokens):
        for k in range(top_k):
            e = int(expert_ids[t, k])
            slot = int(fill[e])
            dispatched[e, slot, :] = tokens[t]
            token_index[e, slot] = t
            k_index[e, slot] = k
            fill[e] += 1
    residual = _DispatchResidual(
        token_index=token_index, k_index=k_index, max_per_expert=max_per
    )
    return dispatched, residual


def ep_combine(expert_out: Array, meta: Any, residual: Any) -> Array:
    """Weighted sum of expert outputs back to tokens. Inverts ``ep_dispatch``."""
    expert_out = _as_f32(expert_out)
    expert_ids, probs, racks, n_experts, max_racks = _meta_fields(meta)
    del racks, max_racks
    n_tokens = int(expert_ids.shape[0])
    d_model = int(expert_out.shape[-1])
    if expert_out.ndim != 3 or expert_out.shape[0] != n_experts:
        raise ValueError("expert_out must be (n_experts, max_per_expert, d_model)")
    out = np.zeros((n_tokens, d_model), dtype=np.float32)
    token_index = np.asarray(residual.token_index)
    k_index = np.asarray(residual.k_index)
    max_per = int(expert_out.shape[1])
    for e in range(n_experts):
        for slot in range(max_per):
            t = int(token_index[e, slot])
            if t < 0:
                continue
            k = int(k_index[e, slot])
            out[t] += probs[t, k] * expert_out[e, slot]
    return out.astype(np.float32)


def ep_dispatch_vjp(tokens: Array, meta: Any, g_dispatched: Array, residual: Any) -> Array:
    """Scatter dispatch cotangents onto tokens via residual (inverse permutation).

    ``g_dispatched`` is (n_experts, max_per_expert, d_model). Each occupied
    slot ``(e, s)`` with ``token_index[e, s] == t`` adds ``g_dispatched[e, s]``
    onto ``grad_tokens[t]``. Padded slots (token_index < 0) contribute nothing.
    A token routed to several experts (top_k > 1) receives the sum of those
    slot cotangents.

    ``meta`` is not differentiated. ``residual`` is not differentiated.
    Independent of production ``kernels`` (duck-types residual / meta).
    """
    tokens = _as_f32(tokens)
    g_dispatched = _as_f32(g_dispatched)
    _expert_ids, _probs, _racks, n_experts, _max_racks = _meta_fields(meta)
    del _expert_ids, _probs, _racks, _max_racks
    if g_dispatched.ndim != 3 or int(g_dispatched.shape[0]) != n_experts:
        raise ValueError("g_dispatched must be (n_experts, max_per_expert, d_model)")
    n_tokens = int(tokens.shape[0])
    d_model = int(tokens.shape[1])
    token_index = np.asarray(residual.token_index)
    max_per = int(g_dispatched.shape[1])
    grad = np.zeros((n_tokens, d_model), dtype=np.float32)
    for e in range(n_experts):
        for slot in range(max_per):
            t = int(token_index[e, slot])
            if t < 0:
                continue
            grad[t] += g_dispatched[e, slot]
    return grad.astype(np.float32)


def ep_combine_vjp(expert_out: Array, meta: Any, residual: Any, g_combined: Array) -> Array:
    """Weighted-scatter combine cotangents onto expert slots.

    ``grad_expert_out[e, s] = probs[t, k] * g_combined[t]`` for occupied
    slots with ``token_index[e, s] == t`` and ``k_index[e, s] == k``.
    Padded / unused slots get zero. Routing ids (expert_ids, racks) and
    residual are not differentiated.

    Independent of production ``kernels`` (duck-types residual / meta).
    """
    expert_out = _as_f32(expert_out)
    g_combined = _as_f32(g_combined)
    _expert_ids, probs, _racks, n_experts, _max_racks = _meta_fields(meta)
    del _expert_ids, _racks, _max_racks
    if expert_out.ndim != 3 or int(expert_out.shape[0]) != n_experts:
        raise ValueError("expert_out must be (n_experts, max_per_expert, d_model)")
    token_index = np.asarray(residual.token_index)
    k_index = np.asarray(residual.k_index)
    max_per = int(expert_out.shape[1])
    d_model = int(expert_out.shape[2])
    grad = np.zeros((n_experts, max_per, d_model), dtype=np.float32)
    for e in range(n_experts):
        for slot in range(max_per):
            t = int(token_index[e, slot])
            if t < 0:
                continue
            k = int(k_index[e, slot])
            grad[e, slot] = probs[t, k] * g_combined[t]
    return grad.astype(np.float32)


# ---------------------------------------------------------------------------
# Explicit EP VJP pair (A3-ep-fwd-jit). Slow Python loops, not production _ep_*.
# ---------------------------------------------------------------------------


class _CombineResidual:
    """Bwd residual for ``ep_combine``: weighted gather needs probs and slot indices."""

    def __init__(self, probs: Array, token_index: Array, k_index: Array) -> None:
        self.probs = np.asarray(probs, dtype=np.float32)
        self.token_index = np.asarray(token_index)
        self.k_index = np.asarray(k_index)


def ep_dispatch_fwd(tokens: Array, meta: Any) -> tuple[Array, _DispatchResidual]:
    """Explicit VJP forward. Residual carries ``token_index`` / ``k_index`` / ``max_per_expert``."""
    return ep_dispatch(tokens, meta)


def ep_dispatch_bwd(residual: Any, g: Array) -> Array:
    """Scatter-add ``g`` (dispatched cotangent) into ``grad_tokens`` via ``residual.token_index``.

    Padded slots (``token_index < 0``) contribute nothing. ``n_tokens`` is
    inferred as ``max(token_index) + 1`` because every token is routed.
    """
    g = _as_f32(g)
    token_index = np.asarray(residual.token_index)
    if g.ndim != 3:
        raise ValueError(f"g must be (n_experts, max_per_expert, d_model), got {g.shape}")
    n_experts, max_per = int(token_index.shape[0]), int(token_index.shape[1])
    d_model = int(g.shape[-1])
    occupied = token_index[token_index >= 0]
    n_tokens = int(occupied.max()) + 1 if occupied.size else 0
    grad = np.zeros((n_tokens, d_model), dtype=np.float32)
    for e in range(n_experts):
        for slot in range(max_per):
            t = int(token_index[e, slot])
            if t < 0:
                continue
            grad[t] += g[e, slot]
    return grad.astype(np.float32)


def ep_combine_fwd(
    expert_out: Array, meta: Any, residual: Any
) -> tuple[Array, _CombineResidual]:
    """Explicit VJP forward. Residual is probs + slot indices for the weighted gather."""
    combined = ep_combine(expert_out, meta, residual)
    _, probs, _, _, _ = _meta_fields(meta)
    bwd = _CombineResidual(
        probs=probs,
        token_index=np.asarray(residual.token_index),
        k_index=np.asarray(residual.k_index),
    )
    return combined, bwd


def _unpack_combine_residual(residual: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(residual, (tuple, list)) and len(residual) >= 3:
        probs, token_index, k_index = residual[0], residual[1], residual[2]
        return _as_f32(probs), np.asarray(token_index), np.asarray(k_index)
    if hasattr(residual, "probs") and hasattr(residual, "token_index"):
        return (
            _as_f32(residual.probs),
            np.asarray(residual.token_index),
            np.asarray(residual.k_index),
        )
    raise TypeError(
        "combine residual must be a pytree with probs/token_index/k_index "
        f"(attributes or a 3-tuple); got {type(residual)!r}"
    )


def ep_combine_bwd(residual: Any, g: Array) -> Array:
    """Weighted gather: ``grad_expert_out[e,s] = probs[t,k] * g[t]`` for occupied slots.

    Padded slots stay zero. Duck-types a 3-tuple or an object with
    ``probs`` / ``token_index`` / ``k_index``.
    """
    g = _as_f32(g)
    if g.ndim != 2:
        raise ValueError(f"g must be (n_tokens, d_model), got {g.shape}")
    probs, token_index, k_index = _unpack_combine_residual(residual)
    n_experts, max_per = int(token_index.shape[0]), int(token_index.shape[1])
    d_model = int(g.shape[-1])
    grad = np.zeros((n_experts, max_per, d_model), dtype=np.float32)
    for e in range(n_experts):
        for slot in range(max_per):
            t = int(token_index[e, slot])
            if t < 0:
                continue
            k = int(k_index[e, slot])
            grad[e, slot] = probs[t, k] * g[t]
    return grad.astype(np.float32)
