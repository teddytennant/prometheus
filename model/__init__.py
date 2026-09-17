"""FP32 reference model (spec 3, 4.1, 15.5 A1).

Architecture is the flagship shape in 3.1: hybrid 3:1 linear vs MLA, DeepSeek-V3
MLA, Gated DeltaNet / KDA linear attention, sigmoid-gated MoE, 8-layer core
iterated r times, MTP (2 extra heads), pre-norm RMSNorm, QK-norm on MLA, logit
soft-capping, router z-loss, and the 2-layer MLP + norm latent adapter.

This package is the CPU/FP32 JAX reference. Kernels (A3) and Stage-B latent
training (I5: halt, thought-decode loss, Jacobi, noisy latents) come later.
The adapter module is part of the V1 shape and lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Array = Any  # numpy.ndarray or jax.Array; FP32 reference dtype

ROPE_BASE = 10000.0
RMS_NORM_EPS = 1e-6
INIT_SCALE = 0.02


class AttentionKind(StrEnum):
    LINEAR = "linear"
    MLA = "mla"


class FfnKind(StrEnum):
    DENSE = "dense"
    MOE = "moe"


class AttentionKindError(ValueError):
    """Layer index is outside the hybrid 3:1 pattern."""


class ConfigError(ValueError):
    """Config violates a 3.1 invariant (counts, ratios, required fields)."""


@dataclass(frozen=True)
class ModelConfig:
    """Discrete architecture. Flagship numbers are 3.1; tiny() is the V1 stand-in."""

    d_model: int
    n_layers: int
    n_dense: int
    n_moe: int
    n_linear_attn: int
    n_mla: int
    n_routed_experts: int
    n_shared_experts: int
    top_k: int
    expert_hidden: int
    mtp_heads: int
    core_block_layers: int
    recurrence_train_mean: float
    recurrence_max: int
    vocab_size: int
    max_context: int
    max_racks: int = 4
    prelude_layers: int | None = None
    coda_layers: int | None = None
    adapter_hidden: int | None = None


# spec 3.1 flagship. Vocab here is 256k; the F1 tokenizer schema is 128k BPE.
# The model takes vocab_size from config, not from the tokenizer artifact.
FLAGSHIP_D_MODEL = 12288
FLAGSHIP_N_LAYERS = 96
FLAGSHIP_N_DENSE = 3
FLAGSHIP_N_MOE = 93
FLAGSHIP_N_LINEAR_ATTN = 72
FLAGSHIP_N_MLA = 24
FLAGSHIP_N_ROUTED_EXPERTS = 512
FLAGSHIP_N_SHARED_EXPERTS = 2
FLAGSHIP_TOP_K = 20
FLAGSHIP_EXPERT_HIDDEN = 4096
FLAGSHIP_MTP_HEADS = 2
FLAGSHIP_CORE_BLOCK_LAYERS = 8
FLAGSHIP_RECURRENCE_TRAIN_MEAN = 3.0
FLAGSHIP_RECURRENCE_MAX = 16
FLAGSHIP_VOCAB_SIZE = 256_000
FLAGSHIP_MAX_CONTEXT = 16_384
FLAGSHIP_MAX_RACKS = 4
FLAGSHIP_TOTAL_PARAMS = 7_400_000_000_000
FLAGSHIP_UNIQUE_ACTIVE_PARAMS = 400_000_000_000
FLAGSHIP_COMPUTE_ACTIVE_PARAMS = 450_000_000_000

# Hybrid pattern is 3 linear : 1 MLA, repeating. Index 0 is linear.
LINEAR_TO_MLA = 3


def flagship_config() -> ModelConfig:
    return ModelConfig(
        d_model=FLAGSHIP_D_MODEL,
        n_layers=FLAGSHIP_N_LAYERS,
        n_dense=FLAGSHIP_N_DENSE,
        n_moe=FLAGSHIP_N_MOE,
        n_linear_attn=FLAGSHIP_N_LINEAR_ATTN,
        n_mla=FLAGSHIP_N_MLA,
        n_routed_experts=FLAGSHIP_N_ROUTED_EXPERTS,
        n_shared_experts=FLAGSHIP_N_SHARED_EXPERTS,
        top_k=FLAGSHIP_TOP_K,
        expert_hidden=FLAGSHIP_EXPERT_HIDDEN,
        mtp_heads=FLAGSHIP_MTP_HEADS,
        core_block_layers=FLAGSHIP_CORE_BLOCK_LAYERS,
        recurrence_train_mean=FLAGSHIP_RECURRENCE_TRAIN_MEAN,
        recurrence_max=FLAGSHIP_RECURRENCE_MAX,
        vocab_size=FLAGSHIP_VOCAB_SIZE,
        max_context=FLAGSHIP_MAX_CONTEXT,
        max_racks=FLAGSHIP_MAX_RACKS,
    )


def tiny_config() -> ModelConfig:
    """V1 CPU stand-in (~10M). Same discrete choices as flagship, smaller sizes.

    8 unique layers, 3:1 linear/MLA, 1 dense + 7 MoE, 8 routed + 2 shared,
    top-2, MTP 2, core block 2 iterated r times, vocab 256, context 128.
    """
    return ModelConfig(
        d_model=128,
        n_layers=8,
        n_dense=1,
        n_moe=7,
        n_linear_attn=6,
        n_mla=2,
        n_routed_experts=8,
        n_shared_experts=2,
        top_k=2,
        expert_hidden=256,
        mtp_heads=2,
        core_block_layers=2,
        recurrence_train_mean=FLAGSHIP_RECURRENCE_TRAIN_MEAN,
        recurrence_max=FLAGSHIP_RECURRENCE_MAX,
        vocab_size=256,
        max_context=128,
        max_racks=FLAGSHIP_MAX_RACKS,
        prelude_layers=3,
        coda_layers=3,
        adapter_hidden=128,
    )


@dataclass(frozen=True)
class ForwardOutput:
    """Next-token logits plus MTP heads and the router aux used in the loss."""

    logits: Array
    mtp_logits: tuple[Array, ...]
    z_loss: Array
    router_probs: Array
    expert_ids: Array
    hidden: Array
    r_used: int


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


def _attn_geometry(d_model: int) -> tuple[int, int, int, int]:
    """(n_heads, d_head, d_nope, d_rope) derived from d_model."""
    d_head = 1
    for cand in (64, 32, 16, 8, 4, 2, 1):
        if d_model % cand == 0:
            d_head = cand
            break
    n_heads = d_model // d_head
    d_nope = d_head
    d_rope = (d_head // 2) // 2 * 2
    if d_rope < 2:
        d_rope = 2 if d_head >= 2 else d_head
    return n_heads, d_head, d_nope, d_rope


def _recurrent_split(config: ModelConfig) -> tuple[int, int, int]:
    """Prelude / core / coda unique-layer counts. Sum equals n_layers."""
    core = int(config.core_block_layers)
    if core < 1 or core > config.n_layers:
        raise ConfigError("core_block_layers out of range")
    if config.prelude_layers is not None and config.coda_layers is not None:
        prelude = int(config.prelude_layers)
        coda = int(config.coda_layers)
        if prelude + core + coda != config.n_layers:
            raise ConfigError("prelude + core + coda must equal n_layers")
        if prelude < 0 or coda < 0:
            raise ConfigError("prelude_layers and coda_layers must be >= 0")
        return prelude, core, coda
    rest = config.n_layers - core
    prelude = rest // 2
    coda = rest - prelude
    return prelude, core, coda


def validate_config(config: ModelConfig) -> None:
    """Raise ConfigError if counts or the 3:1 hybrid ratio do not hold."""
    if config.n_layers < 1:
        raise ConfigError("n_layers must be >= 1")
    if config.n_dense + config.n_moe != config.n_layers:
        raise ConfigError("n_dense + n_moe must equal n_layers")
    if config.n_linear_attn + config.n_mla != config.n_layers:
        raise ConfigError("n_linear_attn + n_mla must equal n_layers")
    period = LINEAR_TO_MLA + 1
    if config.n_layers % period != 0:
        raise ConfigError("n_layers must be a multiple of the 3:1 hybrid period")
    if config.n_mla * LINEAR_TO_MLA != config.n_linear_attn:
        raise ConfigError("hybrid attention must be 3 linear : 1 MLA")
    if config.n_dense < 0 or config.n_moe < 0:
        raise ConfigError("n_dense and n_moe must be >= 0")
    for name in (
        "d_model",
        "vocab_size",
        "max_context",
        "n_routed_experts",
        "top_k",
        "expert_hidden",
        "core_block_layers",
        "max_racks",
    ):
        if getattr(config, name) < 1:
            raise ConfigError(f"{name} must be >= 1")
    if config.n_shared_experts < 0:
        raise ConfigError("n_shared_experts must be >= 0")
    if config.top_k > config.n_routed_experts:
        raise ConfigError("top_k cannot exceed n_routed_experts")
    if config.mtp_heads < 0:
        raise ConfigError("mtp_heads must be >= 0")
    if config.recurrence_max < 1:
        raise ConfigError("recurrence_max must be >= 1")
    if config.recurrence_train_mean <= 0:
        raise ConfigError("recurrence_train_mean must be > 0")
    if config.adapter_hidden is not None and config.adapter_hidden < 1:
        raise ConfigError("adapter_hidden must be >= 1")
    _recurrent_split(config)


def attention_kind(layer_index: int, config: ModelConfig) -> AttentionKind:
    """3:1 linear vs MLA, starting at linear. Raises AttentionKindError if OOB."""
    if layer_index < 0 or layer_index >= config.n_layers:
        raise AttentionKindError(
            f"layer_index {layer_index} outside [0, {config.n_layers})"
        )
    period = LINEAR_TO_MLA + 1
    if layer_index % period == LINEAR_TO_MLA:
        return AttentionKind.MLA
    return AttentionKind.LINEAR


def ffn_kind(layer_index: int, config: ModelConfig) -> FfnKind:
    """First n_dense layers are dense; the rest are MoE."""
    if layer_index < 0 or layer_index >= config.n_layers:
        raise ConfigError(f"layer_index {layer_index} outside [0, {config.n_layers})")
    if layer_index < config.n_dense:
        return FfnKind.DENSE
    return FfnKind.MOE


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


def _rng_key(rng: Any) -> Array:
    if rng is None:
        return jax.random.PRNGKey(0)
    if isinstance(rng, (int, np.integer)):
        return jax.random.PRNGKey(int(rng))
    if isinstance(rng, np.random.Generator):
        return jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
    return rng


def init_params(config: ModelConfig, rng: Any) -> dict[str, Any]:
    """FP32 parameter tree. Embedding, per-layer attn/ffn, MTP heads, adapter."""
    validate_config(config)
    key = _rng_key(rng)
    d = config.d_model
    n_heads, d_head, d_nope, d_rope = _attn_geometry(d)
    kv_rank = d_head
    hidden = config.expert_hidden
    adapter_h = config.adapter_hidden if config.adapter_hidden is not None else d

    def take() -> Array:
        nonlocal key
        key, sub = jax.random.split(key)
        return sub

    def w(shape: tuple[int, ...]) -> Array:
        return jax.random.normal(take(), shape, dtype=jnp.float32) * jnp.float32(INIT_SCALE)

    layers: list[dict[str, Array]] = []
    for i in range(config.n_layers):
        layer: dict[str, Array] = {
            "pre_attn_norm": jnp.ones((d,), dtype=jnp.float32),
            "pre_ffn_norm": jnp.ones((d,), dtype=jnp.float32),
        }
        kind = attention_kind(i, config)
        if kind is AttentionKind.LINEAR:
            layer["W_q"] = w((d, n_heads, d_head))
            layer["W_k"] = w((d, n_heads, d_head))
            layer["W_v"] = w((d, n_heads, d_head))
            layer["W_o"] = w((n_heads, d_head, d))
        else:
            layer["W_q"] = w((d, n_heads, d_nope + d_rope))
            layer["W_kv_compress"] = w((d, kv_rank))
            layer["W_kv_up"] = w((kv_rank, n_heads, d_nope))
            layer["W_rope_k"] = w((d, d_rope))
            layer["W_o"] = w((n_heads, d_nope, d))
        if ffn_kind(i, config) is FfnKind.DENSE:
            layer["ffn_gate"] = w((d, hidden))
            layer["ffn_up"] = w((d, hidden))
            layer["ffn_down"] = w((hidden, d))
        else:
            n_r = config.n_routed_experts
            n_s = config.n_shared_experts
            layer["router"] = w((d, n_r))
            layer["routed_gate"] = w((n_r, d, hidden))
            layer["routed_up"] = w((n_r, d, hidden))
            layer["routed_down"] = w((n_r, hidden, d))
            layer["shared_gate"] = w((n_s, d, hidden))
            layer["shared_up"] = w((n_s, d, hidden))
            layer["shared_down"] = w((n_s, hidden, d))
        layers.append(layer)
    mtp = []
    for _ in range(config.mtp_heads):
        mtp.append(
            {
                "norm": jnp.ones((d,), dtype=jnp.float32),
                "proj": w((d, d)),
                "unembed": w((config.vocab_size, d)),
            }
        )
    return {
        "embed": w((config.vocab_size, d)),
        "unembed": w((config.vocab_size, d)),
        "final_norm": jnp.ones((d,), dtype=jnp.float32),
        "layers": layers,
        "mtp": mtp,
        "adapter_w1": w((d, adapter_h)),
        "adapter_w2": w((adapter_h, d)),
        "adapter_norm": jnp.ones((d,), dtype=jnp.float32),
    }


def param_count(params: dict[str, Any]) -> int:
    n = 0

    def walk(obj: Any) -> None:
        nonlocal n
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)
        elif hasattr(obj, "size"):
            n += int(obj.size)
        elif hasattr(obj, "shape"):
            prod = 1
            for s in obj.shape:
                prod *= int(s)
            n += prod

    walk(params)
    return n


def _dense_ffn(x: Array, layer: dict[str, Array]) -> Array:
    return _swiglu(x, layer["ffn_gate"], layer["ffn_up"], layer["ffn_down"])


def _linear_attn(
    x: Array, layer: dict[str, Array], state: Array | None
) -> tuple[Array, Array]:
    q = jnp.einsum("bsd,dhe->bshe", x, _as_f32(layer["W_q"]))
    k = jnp.einsum("bsd,dhe->bshe", x, _as_f32(layer["W_k"]))
    v = jnp.einsum("bsd,dhe->bshe", x, _as_f32(layer["W_v"]))
    y, state = linear_attention(q, k, v, state)
    y = jnp.einsum("bshe,hed->bsd", y, _as_f32(layer["W_o"]))
    return y, state


def _mla_attn(x: Array, layer: dict[str, Array], positions: Array) -> Array:
    q = jnp.einsum("bsd,dhe->bshe", x, _as_f32(layer["W_q"]))
    ckv = jnp.einsum("bsd,dc->bsc", x, _as_f32(layer["W_kv_compress"]))
    k_nope = jnp.einsum("bsc,che->bshe", ckv, _as_f32(layer["W_kv_up"]))
    rope_k = jnp.einsum("bsd,dr->bsr", x, _as_f32(layer["W_rope_k"]))
    d_nope = k_nope.shape[-1]
    q_nope = q[..., :d_nope]
    q_rope = q[..., d_nope:]
    rk = rope_k[:, :, None, :]
    rk = jnp.broadcast_to(rk, q_rope.shape)
    q_rope, rk = rope(q_rope, rk, positions, partial=False)
    q = jnp.concatenate([q_nope, q_rope], axis=-1)
    y = mla_attention(q, k_nope, rk, qk_norm=True)
    return jnp.einsum("bshe,hed->bsd", y, _as_f32(layer["W_o"]))


def _block(
    h: Array,
    layer: dict[str, Array],
    config: ModelConfig,
    layer_index: int,
    positions: Array,
    lin_state: Array | None,
) -> tuple[Array, Array | None, Array | None, Array | None, Array | None]:
    """One pre-norm block. Returns h, lin_state, router_logits, probs, ids."""
    n = rms_norm(h, layer["pre_attn_norm"])
    if attention_kind(layer_index, config) is AttentionKind.LINEAR:
        attn, lin_state = _linear_attn(n, layer, lin_state)
    else:
        attn = _mla_attn(n, layer, positions)
        lin_state = None
    h = h + attn
    n = rms_norm(h, layer["pre_ffn_norm"])
    router_logits = router_probs = expert_ids = None
    if ffn_kind(layer_index, config) is FfnKind.DENSE:
        h = h + _dense_ffn(n, layer)
    else:
        routed = (layer["routed_gate"], layer["routed_up"], layer["routed_down"])
        shared = (layer["shared_gate"], layer["shared_up"], layer["shared_down"])
        y, router_probs, expert_ids = moe(
            n,
            router_weight=layer["router"],
            routed_weights=routed,
            shared_weights=shared,
            top_k=config.top_k,
            max_racks=config.max_racks,
        )
        router_logits = n.reshape(-1, n.shape[-1]) @ _as_f32(layer["router"])
        h = h + y
    return h, lin_state, router_logits, router_probs, expert_ids


def _sample_r(config: ModelConfig, tokens: Array) -> int:
    seed = int(np.asarray(tokens).sum()) % (2**31)
    rng = np.random.default_rng(seed)
    mean = float(config.recurrence_train_mean)
    max_r = int(config.recurrence_max)
    p = min(max(1.0 / mean, 1e-6), 0.999)
    for _ in range(64):
        u = float(rng.random())
        k = int(np.floor(np.log(max(1.0 - u, 1e-12)) / np.log(1.0 - p))) + 1
        if 1 <= k <= max_r:
            return k
    return min(max_r, max(1, int(round(mean))))


def forward(
    tokens: Array,
    params: dict[str, Any],
    config: ModelConfig,
    *,
    r: int | None = None,
    thoughts: Array | None = None,
) -> ForwardOutput:
    """FP32 forward.

    `r` is the core-block iteration count. None means sample from the training
    heavy-tailed distribution with mean `config.recurrence_train_mean` and cap
    `config.recurrence_max`. KV is shared across iterations (Huginn-style).

    `thoughts` is an optional (batch, n_thoughts, d_model) tensor inserted via
    the latent adapter before the discrete tokens. Empty/None is discrete-only.
    """
    validate_config(config)
    tokens = jnp.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError("tokens must be (batch, seq)")
    batch, seq = int(tokens.shape[0]), int(tokens.shape[1])
    if seq > config.max_context:
        raise ConfigError("sequence longer than max_context")
    if tokens.size and (
        int(jnp.min(tokens)) < 0 or int(jnp.max(tokens)) >= config.vocab_size
    ):
        raise ConfigError("token id out of vocab")
    embed = _as_f32(params["embed"])
    h = embed[tokens]
    n_thoughts = 0
    if thoughts is not None:
        thoughts_a = _as_f32(thoughts)
        if thoughts_a.ndim != 3 or thoughts_a.shape[0] != batch:
            raise ValueError("thoughts must be (batch, n_thoughts, d_model)")
        if thoughts_a.shape[-1] != config.d_model:
            raise ValueError("thoughts last dim must equal d_model")
        n_thoughts = int(thoughts_a.shape[1])
        if n_thoughts:
            adapted = _apply_latent_adapter(
                thoughts_a,
                params["adapter_w1"],
                params["adapter_w2"],
                params["adapter_norm"],
            )
            h = jnp.concatenate([adapted, h], axis=1)
    full_seq = int(h.shape[1])
    positions = jnp.arange(full_seq, dtype=jnp.float32)
    r_used = int(r) if r is not None else _sample_r(config, tokens)
    if r_used < 1:
        raise ConfigError("r must be >= 1")
    prelude, core, coda = _recurrent_split(config)
    z_sum = jnp.asarray(0.0, dtype=jnp.float32)
    z_n = jnp.asarray(0, dtype=jnp.int32)
    last_probs = jnp.zeros(
        (batch, full_seq, config.n_routed_experts), dtype=jnp.float32
    )
    last_ids = jnp.zeros((batch, full_seq, config.top_k), dtype=jnp.int32)

    def apply_layer(
        h_in: Array,
        idx: int,
        z_sum_in: Array,
        z_n_in: Array,
        probs_in: Array,
        ids_in: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        h_out, _, logits, probs, ids = _block(
            h_in, params["layers"][idx], config, idx, positions, None
        )
        if logits is not None:
            lse = jax.nn.logsumexp(logits, axis=-1)
            z_sum_in = z_sum_in + jnp.mean(jnp.square(lse)).astype(jnp.float32)
            z_n_in = z_n_in + jnp.int32(1)
            probs_in, ids_in = probs, ids
        return h_out, z_sum_in, z_n_in, probs_in, ids_in

    def apply_range(
        h_in: Array,
        start: int,
        n: int,
        z_sum_in: Array,
        z_n_in: Array,
        probs_in: Array,
        ids_in: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        for j in range(n):
            h_in, z_sum_in, z_n_in, probs_in, ids_in = apply_layer(
                h_in, start + j, z_sum_in, z_n_in, probs_in, ids_in
            )
        return h_in, z_sum_in, z_n_in, probs_in, ids_in

    h, z_sum, z_n, last_probs, last_ids = apply_range(
        h, 0, prelude, z_sum, z_n, last_probs, last_ids
    )
    injected = h
    h, z_sum, z_n, last_probs, last_ids = apply_range(
        h, prelude, core, z_sum, z_n, last_probs, last_ids
    )

    def core_body(
        carry: tuple[Array, Array, Array, Array, Array], _: Array
    ) -> tuple[tuple[Array, Array, Array, Array, Array], Array]:
        h_in, z_sum_in, z_n_in, probs_in, ids_in = carry
        h_out, z_sum_out, z_n_out, probs_out, ids_out = apply_range(
            h_in + injected, prelude, core, z_sum_in, z_n_in, probs_in, ids_in
        )
        return (h_out, z_sum_out, z_n_out, probs_out, ids_out), jnp.array(
            0, dtype=jnp.int32
        )

    if r_used > 1:
        carry, _ = jax.lax.scan(
            core_body,
            (h, z_sum, z_n, last_probs, last_ids),
            xs=None,
            length=r_used - 1,
        )
        h, z_sum, z_n, last_probs, last_ids = carry
    h, z_sum, z_n, last_probs, last_ids = apply_range(
        h, prelude + core, coda, z_sum, z_n, last_probs, last_ids
    )

    h = rms_norm(h, params["final_norm"])
    hidden = h[:, n_thoughts:, :]
    logits = hidden @ _as_f32(params["unembed"]).T
    mtp_logits = []
    for head in params["mtp"]:
        mh = rms_norm(hidden, head["norm"])
        mh = mh @ _as_f32(head["proj"]) + hidden
        mtp_logits.append(mh @ _as_f32(head["unembed"]).T)
    router_probs = last_probs[:, n_thoughts:, :]
    expert_ids = last_ids[:, n_thoughts:, :]
    z_n_f = z_n.astype(jnp.float32)
    z_loss = jnp.where(z_n > 0, z_sum / jnp.maximum(z_n_f, jnp.float32(1.0)), jnp.float32(0.0))
    return ForwardOutput(
        logits=logits.astype(jnp.float32),
        hidden=hidden.astype(jnp.float32),
        mtp_logits=tuple(m.astype(jnp.float32) for m in mtp_logits),
        router_probs=router_probs,
        expert_ids=expert_ids,
        z_loss=z_loss,
        r_used=r_used,
    )
