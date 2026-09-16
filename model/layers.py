"""Attention, MoE, RoPE, RMSNorm, latent adapter (spec 3)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from model.config import (
    DTYPE,
    FLAGSHIP_MAX_RACKS,
    LINEAR_TO_MLA,
    ROPE_BASE,
    AttentionKind,
    FfnKind,
    ModelConfig,
    _attn_geometry,
    attention_kind,
    ffn_kind,
)

def _as_f32(x: Any) -> Array:
    return jnp.asarray(x, dtype=jnp.float32)


def _silu(x: Array) -> Array:
    x = _as_f32(x)
    return x * (1.0 / (1.0 + jnp.exp(-jnp.clip(x, -80.0, 80.0))))


def _sigmoid(x: Array) -> Array:
    x = _as_f32(x)
    return 1.0 / (1.0 + jnp.exp(-jnp.clip(x, -80.0, 80.0)))


def _softmax(x: Array, axis: int = -1) -> Array:
    x = _as_f32(x)
    x = x - jnp.max(x, axis=axis, keepdims=True)
    e = jnp.exp(x)
    return e / jnp.maximum(e.sum(axis=axis, keepdims=True), jnp.float32(1e-12))


def _l2_normalize(x: Array, eps: float = 1e-6) -> Array:
    n = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(n, jnp.float32(eps))


def rms_norm(x: Array, weight: Array, eps: float = 1e-6) -> Array:
    x = _as_f32(x)
    weight = _as_f32(weight)
    ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    out = x * (1.0 / jnp.sqrt(ms + jnp.float32(eps))) * weight
    return out.astype(jnp.float32)


def _rope_angles(positions: Array, n_rot: int, rot_dim: int, x: Array) -> Array:
    inv_freq = jnp.float32(ROPE_BASE) ** (
        -jnp.float32(2.0) * jnp.arange(n_rot, dtype=jnp.float32) / jnp.float32(rot_dim)
    )
    pos = _as_f32(positions)
    if pos.ndim == 1:
        angle = pos[:, None] * inv_freq
        if x.ndim == 2:
            return angle
        if x.ndim == 3:
            return angle[None, :, :]
        if x.ndim == 4:
            return angle[None, :, None, :]
        raise ValueError(f"unsupported rank {x.ndim} for rope")
    if pos.ndim == 2:
        angle = pos[:, :, None] * inv_freq
        if x.ndim == 3:
            return angle
        if x.ndim == 4:
            return angle[:, :, None, :]
        raise ValueError(f"unsupported rank {x.ndim} for 2-D positions")
    raise ValueError("positions must be (seq,) or (batch, seq)")


def _apply_rope(x: Array, positions: Array, *, partial: bool) -> Array:
    x = _as_f32(x)
    d = int(x.shape[-1])
    n_pairs = d // 2
    if n_pairs == 0:
        return x
    n_rot = n_pairs if not partial else max(1, n_pairs // 2)
    rot_dim = n_rot * 2
    rotated = x[..., :rot_dim]
    passthrough = x[..., rot_dim:]
    even = rotated[..., 0::2]
    odd = rotated[..., 1::2]
    angle = _rope_angles(positions, n_rot, rot_dim, x)
    cos = jnp.cos(angle)
    sin = jnp.sin(angle)
    out_even = even * cos - odd * sin
    out_odd = even * sin + odd * cos
    stacked = jnp.stack([out_even, out_odd], axis=-1)
    out_rot = stacked.reshape(*rotated.shape)
    if passthrough.shape[-1] == 0:
        return out_rot.astype(jnp.float32)
    return jnp.concatenate([out_rot, passthrough], axis=-1).astype(jnp.float32)


def rope(q: Array, k: Array, positions: Array, *, partial: bool = True) -> tuple[Array, Array]:
    """RoPE on MLA. `partial` is the 3.1 partial-dim variant. 2D RoPE is separate."""
    return _apply_rope(q, positions, partial=partial), _apply_rope(
        k, positions, partial=partial
    )


def rope_2d(q: Array, k: Array, row: Array, col: Array) -> tuple[Array, Array]:
    """2D RoPE on ARC grid spans."""
    q = _as_f32(q)
    k = _as_f32(k)
    d = int(q.shape[-1])
    if d % 2 != 0:
        raise ValueError("rope_2d requires an even last dimension")
    half = d // 2
    q1, k1 = rope(q[..., :half], k[..., :half], row, partial=False)
    q2, k2 = rope(q[..., half:], k[..., half:], col, partial=False)
    return (
        jnp.concatenate([q1, q2], axis=-1).astype(jnp.float32),
        jnp.concatenate([k1, k2], axis=-1).astype(jnp.float32),
    )


def _ensure_heads(x: Array) -> tuple[Array, bool]:
    x = _as_f32(x)
    if x.ndim == 3:
        return x[:, :, None, :], True
    if x.ndim == 4:
        return x, False
    raise ValueError(f"expected 3-D or 4-D tensor, got shape {x.shape}")


def linear_attention(
    q: Array, k: Array, v: Array, state: Array | None = None
) -> tuple[Array, Array]:
    """Gated DeltaNet / KDA family. Returns (output, next_state)."""
    q, squeeze = _ensure_heads(q)
    k, _ = _ensure_heads(k)
    v, _ = _ensure_heads(v)
    if q.shape != k.shape or q.shape[:3] != v.shape[:3] or q.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, v must share batch/seq/heads and key/value dim")
    batch, seq, heads, dim = q.shape
    qn = _l2_normalize(q)
    kn = _l2_normalize(k)
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
    eye = jnp.eye(dim, dtype=jnp.float32)

    def step(s: Array, inputs: tuple[Array, Array, Array]) -> tuple[Array, Array]:
        qt, kt, vt = inputs
        kk = jnp.einsum("bhd,bhe->bhde", kt, kt)
        decay = eye[None, None, :, :] - kk
        s = jnp.einsum("bhij,bhjk->bhik", decay, s)
        s = s + jnp.einsum("bhd,bhe->bhde", kt, vt)
        out_t = jnp.einsum("bhd,bhde->bhe", qt, s)
        return s, out_t

    s, out_t = jax.lax.scan(step, s0, (qn_t, kn_t, v_t), length=seq)
    out = jnp.transpose(out_t, (1, 0, 2, 3)).astype(jnp.float32)
    s = s.astype(jnp.float32)
    if squeeze:
        return out[:, :, 0, :], s[:, 0, :, :]
    return out, s


def mla_attention(
    q: Array, compressed_kv: Array, rope_k: Array, *, qk_norm: bool = True
) -> Array:
    """DeepSeek-V3-style MLA with QK-norm on by default."""
    q = _as_f32(q)
    ckv = _as_f32(compressed_kv)
    rk = _as_f32(rope_k)
    if q.ndim != 4:
        raise ValueError(f"q must be 4-D, got {q.shape}")
    if ckv.ndim == 3:
        ckv = ckv[:, :, None, :]
    if rk.ndim == 3:
        rk = rk[:, :, None, :]
    if ckv.ndim != 4 or rk.ndim != 4:
        raise ValueError("compressed_kv and rope_k must be 3-D or 4-D")
    batch, seq_q, n_heads, d_q = q.shape
    seq_k = ckv.shape[1]
    d_nope = ckv.shape[-1]
    d_rope = rk.shape[-1]
    if d_q != d_nope + d_rope:
        raise ValueError(
            f"q last dim {d_q} must equal d_nope ({d_nope}) + d_rope ({d_rope})"
        )

    def _bcast(x: Array) -> Array:
        if x.shape[2] == n_heads:
            return x
        if x.shape[2] == 1:
            return jnp.broadcast_to(x, (x.shape[0], x.shape[1], n_heads, x.shape[3]))
        raise ValueError(f"cannot broadcast heads {x.shape[2]} to {n_heads}")

    ckv_h = _bcast(ckv)
    rk_h = _bcast(rk)
    k = jnp.concatenate([ckv_h, rk_h], axis=-1)
    v = ckv_h
    if qk_norm:
        ones_q = jnp.ones((d_q,), dtype=jnp.float32)
        ones_k = jnp.ones((d_q,), dtype=jnp.float32)
        q = rms_norm(q, ones_q)
        k = rms_norm(k, ones_k)
    scale = 1.0 / jnp.sqrt(jnp.float32(d_q))
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
    q_idx = jnp.arange(seq_q)[:, None]
    k_idx = jnp.arange(seq_k)[None, :]
    causal = k_idx > (q_idx + (seq_k - seq_q))
    scores = jnp.where(causal[None, None, :, :], jnp.float32(-1e9), scores)
    attn = _softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", attn, v).astype(jnp.float32)


def _expert_to_rack(n_experts: int, max_racks: int) -> Array:
    n_racks = min(int(max_racks), int(n_experts))
    n_racks = max(n_racks, 1)
    e = jnp.arange(n_experts, dtype=jnp.int32)
    return jnp.minimum(
        jnp.int32(n_racks - 1), e * jnp.int32(n_racks) // jnp.int32(n_experts)
    )


def _topk_one(
    scores: Array, top_k: int, max_racks: int, racks: Array, n_racks: int
) -> Array:
    n_exp = scores.shape[0]
    order = jnp.argsort(-scores, stable=True).astype(jnp.int32)
    chosen = jnp.full((top_k,), -1, dtype=jnp.int32)
    used = jnp.zeros((n_racks,), dtype=bool)
    n_chosen = jnp.int32(0)

    def first_step(i: int, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        chosen_i, used_i, n_i = carry
        e = order[i]
        rack = racks[e]
        n_used = used_i.astype(jnp.int32).sum()
        can = jnp.logical_and(
            jnp.logical_or(used_i[rack], n_used < jnp.int32(max_racks)),
            n_i < jnp.int32(top_k),
        )
        idx = jnp.minimum(n_i, jnp.int32(top_k - 1))
        chosen_i = chosen_i.at[idx].set(jnp.where(can, e, chosen_i[idx]))
        used_i = used_i.at[rack].set(jnp.where(can, True, used_i[rack]))
        n_i = n_i + can.astype(jnp.int32)
        return chosen_i, used_i, n_i

    chosen, used, n_chosen = jax.lax.fori_loop(
        0, n_exp, first_step, (chosen, used, n_chosen)
    )

    def second_step(i: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        chosen_i, n_i = carry
        e = order[i]
        already = jnp.any(chosen_i == e)
        can = jnp.logical_and(~already, n_i < jnp.int32(top_k))
        idx = jnp.minimum(n_i, jnp.int32(top_k - 1))
        chosen_i = chosen_i.at[idx].set(jnp.where(can, e, chosen_i[idx]))
        n_i = n_i + can.astype(jnp.int32)
        return chosen_i, n_i

    chosen, n_chosen = jax.lax.fori_loop(0, n_exp, second_step, (chosen, n_chosen))
    return chosen


def _node_limited_topk(scores: Array, top_k: int, max_racks: int) -> Array:
    n_exp = int(scores.shape[-1])
    n_racks = max(min(int(max_racks), n_exp), 1)
    racks = _expert_to_rack(n_exp, max_racks)

    def one(sc: Array) -> Array:
        return _topk_one(sc, top_k, max_racks, racks, n_racks)

    return jax.vmap(one)(scores)


def _swiglu(x: Array, w_gate: Array, w_up: Array, w_down: Array) -> Array:
    h = _silu(x @ w_gate) * (x @ w_up)
    return h @ w_down


def moe(
    x: Array,
    *,
    router_weight: Array,
    routed_weights: Array,
    shared_weights: Array,
    top_k: int,
    max_racks: int = FLAGSHIP_MAX_RACKS,
) -> tuple[Array, Array, Array]:
    """Sigmoid gates, top-k routed + shared experts, node-limited routing.

    Returns (output, router_probs, expert_ids). Aux-loss-free bias balancing is
    a training concern (A2); the reference forward still returns router_probs
    so z-loss can be computed.
    """
    x = _as_f32(x)
    orig = x.shape
    d_model = orig[-1]
    flat = x.reshape(-1, d_model)
    logits = flat @ _as_f32(router_weight)
    probs = _sigmoid(logits)
    expert_ids = _node_limited_topk(probs, int(top_k), int(max_racks))
    tok = jnp.arange(flat.shape[0])[:, None]
    top_scores = probs[tok, expert_ids]
    gates = top_scores / jnp.maximum(top_scores.sum(axis=-1, keepdims=True), jnp.float32(1e-9))
    w_gate, w_up, w_down = (_as_f32(t) for t in routed_weights)
    wg = w_gate[expert_ids]
    wu = w_up[expert_ids]
    wd = w_down[expert_ids]
    hidden = _silu(jnp.einsum("nd,nkdh->nkh", flat, wg)) * jnp.einsum(
        "nd,nkdh->nkh", flat, wu
    )
    routed = jnp.einsum("nkh,nkhd->nkd", hidden, wd)
    out = jnp.einsum("nk,nkd->nd", gates, routed)
    s_gate, s_up, s_down = (_as_f32(t) for t in shared_weights)
    n_shared = int(s_gate.shape[0])
    if n_shared > 0:

        def one_shared(wg_s: Array, wu_s: Array, wd_s: Array) -> Array:
            return _swiglu(flat, wg_s, wu_s, wd_s)

        shared = jax.vmap(one_shared)(s_gate, s_up, s_down)
        out = out + shared.sum(axis=0)
    n_routed = probs.shape[-1]
    router_probs = probs.reshape(*orig[:-1], n_routed).astype(jnp.float32)
    ids = expert_ids.reshape(*orig[:-1], int(top_k)).astype(jnp.int32)
    return out.reshape(orig).astype(jnp.float32), router_probs, ids


def _apply_latent_adapter(x: Array, w1: Array, w2: Array, norm_weight: Array) -> Array:
    x = _as_f32(x)
    h = _silu(x @ _as_f32(w1)) @ _as_f32(w2)
    return rms_norm(h, norm_weight)


def latent_adapter(x: Array, *, hidden: int) -> Array:
    """2-layer MLP + norm. Maps a continuous thought vector into residual stream."""
    x = _as_f32(x)
    d = int(x.shape[-1])
    inv = jax.lax.rsqrt(jnp.float32(max(int(hidden), 1)))
    w1 = jnp.eye(d, int(hidden), dtype=jnp.float32) * inv
    w2 = jnp.eye(int(hidden), d, dtype=jnp.float32) * inv
    return _apply_latent_adapter(x, w1, w2, jnp.ones((d,), dtype=jnp.float32))

