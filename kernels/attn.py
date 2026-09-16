"""Gated linear attention (chunked delta-rule), MLA with RoPE + QK-norm."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

ROPE_BASE = 10000.0


def _as_f32(x: Any) -> Array:
    return jnp.asarray(x, dtype=jnp.float32)


def _softmax(x: Array, axis: int = -1) -> Array:
    x = _as_f32(x)
    x = x - jnp.max(x, axis=axis, keepdims=True)
    e = jnp.exp(x)
    return e / jnp.maximum(e.sum(axis=axis, keepdims=True), jnp.float32(1e-12))


def _l2_normalize(x: Array, eps: float = 1e-6) -> Array:
    n = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(n, jnp.float32(eps))


def _ensure_heads(x: Array) -> tuple[Array, bool]:
    x = _as_f32(x)
    if x.ndim == 3:
        return x[:, :, None, :], True
    if x.ndim == 4:
        return x, False
    raise ValueError(f"expected 3-D or 4-D tensor, got shape {x.shape}")


def rms_norm(x: Array, weight: Array, eps: float = 1e-6) -> Array:
    x = _as_f32(x)
    weight = _as_f32(weight)
    ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return (x * (1.0 / jnp.sqrt(ms + jnp.float32(eps))) * weight).astype(jnp.float32)


def apply_rope(x: Array, positions: Array, *, partial: bool = True) -> Array:
    x = _as_f32(x)
    d = int(x.shape[-1])
    n_pairs = d // 2
    if n_pairs == 0:
        return x
    n_rot = n_pairs if not partial else max(1, n_pairs // 2)
    rot_dim = n_rot * 2
    inv_freq = jnp.float32(ROPE_BASE) ** (
        -jnp.float32(2.0) * jnp.arange(n_rot, dtype=jnp.float32) / jnp.float32(rot_dim)
    )
    pos = _as_f32(positions)
    if pos.ndim == 1:
        angle = pos[:, None] * inv_freq
        if x.ndim == 3:
            angle = angle[None, :, :]
        elif x.ndim == 4:
            angle = angle[None, :, None, :]
    elif pos.ndim == 2:
        angle = pos[:, :, None] * inv_freq
        if x.ndim == 4:
            angle = angle[:, :, None, :]
    else:
        raise ValueError("positions must be (seq,) or (batch, seq)")
    rotated = x[..., :rot_dim]
    passthrough = x[..., rot_dim:]
    even, odd = rotated[..., 0::2], rotated[..., 1::2]
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    stacked = jnp.stack([even * cos - odd * sin, even * sin + odd * cos], axis=-1)
    out_rot = stacked.reshape(*rotated.shape)
    if passthrough.shape[-1] == 0:
        return out_rot.astype(jnp.float32)
    return jnp.concatenate([out_rot, passthrough], axis=-1).astype(jnp.float32)


def _delta_step(s: Array, inputs: tuple[Array, Array, Array]) -> tuple[Array, Array]:
    qt, kt, vt = inputs
    dim = kt.shape[-1]
    eye = jnp.eye(dim, dtype=jnp.float32)
    decay = eye[None, None, :, :] - jnp.einsum("bhd,bhe->bhde", kt, kt)
    s = jnp.einsum("bhij,bhjk->bhik", decay, s)
    s = s + jnp.einsum("bhd,bhe->bhde", kt, vt)
    out_t = jnp.einsum("bhd,bhde->bhe", qt, s)
    return s, out_t


def linear_attention(
    q: Array,
    k: Array,
    v: Array,
    state: Array | None = None,
    *,
    chunk_size: int | None = None,
) -> tuple[Array, Array]:
    """Chunked gated delta-rule. Default path matches model.linear_attention."""
    q, squeeze = _ensure_heads(q)
    k, _ = _ensure_heads(k)
    v, _ = _ensure_heads(v)
    if q.shape != k.shape or q.shape[:3] != v.shape[:3] or q.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, v must share batch/seq/heads and key/value dim")
    batch, seq, heads, dim = q.shape
    qn, kn = _l2_normalize(q), _l2_normalize(k)
    if state is None:
        s0 = jnp.zeros((batch, heads, dim, dim), dtype=jnp.float32)
    else:
        s0 = _as_f32(state)
        if s0.ndim == 3:
            s0 = s0[:, None, :, :]
        if s0.shape != (batch, heads, dim, dim):
            raise ValueError(f"state shape {s0.shape} != {(batch, heads, dim, dim)}")
    qn_t = jnp.transpose(qn, (1, 0, 2, 3))
    kn_t = jnp.transpose(kn, (1, 0, 2, 3))
    v_t = jnp.transpose(v, (1, 0, 2, 3))
    cs = seq if chunk_size is None or chunk_size <= 0 else min(int(chunk_size), seq)
    if cs >= seq:
        s, out_t = jax.lax.scan(_delta_step, s0, (qn_t, kn_t, v_t), length=seq)
    else:
        s = s0
        chunks: list[Array] = []
        for start in range(0, seq, cs):
            sl = slice(start, min(start + cs, seq))
            s, out_c = jax.lax.scan(_delta_step, s, (qn_t[sl], kn_t[sl], v_t[sl]))
            chunks.append(out_c)
        out_t = jnp.concatenate(chunks, axis=0)
    out = jnp.transpose(out_t, (1, 0, 2, 3)).astype(jnp.float32)
    s = s.astype(jnp.float32)
    if squeeze:
        return out[:, :, 0, :], s[:, 0, :, :]
    return out, s


def mla_attention(
    q: Array,
    k_nope: Array,
    k_rope: Array,
    *,
    qk_norm: bool = True,
    positions: Array | None = None,
) -> Array:
    """MLA: concat compressed KV + RoPE, QK-RMSNorm, causal softmax."""
    q = _as_f32(q)
    ckv = _as_f32(k_nope)
    rk = _as_f32(k_rope)
    if q.ndim != 4:
        raise ValueError(f"q must be 4-D, got {q.shape}")
    if ckv.ndim == 3:
        ckv = ckv[:, :, None, :]
    if rk.ndim == 3:
        rk = rk[:, :, None, :]
    batch, seq_q, n_heads, d_q = q.shape
    seq_k = ckv.shape[1]
    d_nope = ckv.shape[-1]
    d_rope = rk.shape[-1]
    if d_q != d_nope + d_rope:
        raise ValueError(f"q last dim {d_q} must equal d_nope ({d_nope}) + d_rope ({d_rope})")
    if positions is not None:
        q_nope, q_rope = q[..., :d_nope], q[..., d_nope:]
        q_rope = apply_rope(q_rope, positions, partial=False)
        rk = apply_rope(rk, positions, partial=False)
        q = jnp.concatenate([q_nope, q_rope], axis=-1)

    def _bcast(x: Array) -> Array:
        if x.shape[2] == n_heads:
            return x
        if x.shape[2] == 1:
            return jnp.broadcast_to(x, (x.shape[0], x.shape[1], n_heads, x.shape[3]))
        raise ValueError(f"cannot broadcast heads {x.shape[2]} to {n_heads}")

    ckv_h, rk_h = _bcast(ckv), _bcast(rk)
    k = jnp.concatenate([ckv_h, rk_h], axis=-1)
    v = ckv_h
    if qk_norm:
        q = rms_norm(q, jnp.ones((d_q,), dtype=jnp.float32))
        k = rms_norm(k, jnp.ones((d_q,), dtype=jnp.float32))
    scale = 1.0 / jnp.sqrt(jnp.float32(d_q))
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
    q_idx = jnp.arange(seq_q)[:, None]
    k_idx = jnp.arange(seq_k)[None, :]
    causal = k_idx > (q_idx + (seq_k - seq_q))
    scores = jnp.where(causal[None, None, :, :], jnp.float32(-1e9), scores)
    attn = _softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", attn, v).astype(jnp.float32)


def flash_attention_probe(q: Array, k: Array, v: Array) -> Array:
    q, k, v = _as_f32(q), _as_f32(k), _as_f32(v)
    scale = 1.0 / jnp.sqrt(jnp.float32(q.shape[-1]))
    logits = jnp.einsum("bthd,bshd->bhts", q, k) * scale
    seq_q, seq_k = q.shape[1], k.shape[1]
    causal = jnp.arange(seq_k)[None, :] > jnp.arange(seq_q)[:, None]
    logits = jnp.where(causal[None, None, :, :], jnp.float32(-1e9), logits)
    w = _softmax(logits, axis=-1)
    return jnp.einsum("bhts,bshd->bthd", w, v)
