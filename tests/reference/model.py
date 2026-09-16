"""Slow, obvious NumPy reference for the A1 FP32 model (spec 3.1 + 4.1 adapter).

This is the oracle math, not the production JAX implementation. Production
``model.*`` functions must match these shapes, dtypes, and values (atol/rtol
1e-5 in FP32) on the same inputs.

Layout conventions
------------------
Tensors are ``(batch, seq, ...)`` with sequence on axis 1. Attention tensors
are ``(batch, seq, n_heads, dim)``. 3-D ``(batch, seq, dim)`` is treated as
one head.

RoPE base is 10000 (the usual transformer default; the spec does not name a
base). RMSNorm eps is 1e-6. Parameter init is N(0, 0.02) with ones for
RMSNorm scales. None of these are flagship architecture numbers.

Gated DeltaNet here is the delta rule with β=1 after L2-normalising q and k
(no extra gate tensors on the public signature). MLA is the attention kernel
after projections: keys = concat(compressed_kv, rope_k), values =
compressed_kv, causal softmax, optional QK-RMSNorm with unit weight.

Do not call ``init_params`` / ``forward`` on ``flagship_config()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from model import (
    LINEAR_TO_MLA,
    AttentionKind,
    AttentionKindError,
    ConfigError,
    FfnKind,
    ForwardOutput,
    ModelConfig,
)

Array = np.ndarray

ROPE_BASE = 10000.0
RMS_NORM_EPS = 1e-6
INIT_SCALE = 0.02


def attn_geometry(d_model: int) -> tuple[int, int, int, int]:
    """(n_heads, d_head, d_nope, d_rope) derived from d_model.

    d_head is the largest of {64, 32, 16, 8, 4, 2, 1} that divides d_model.
    MLA content dim equals d_head; RoPE dim is the even part of d_head // 2.
    """
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


def recurrent_split(config: ModelConfig) -> tuple[int, int, int]:
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


def _as_f32(x: Any) -> Array:
    return np.asarray(x, dtype=np.float32)


def _silu(x: Array) -> Array:
    x = _as_f32(x)
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0))))


def _sigmoid(x: Array) -> Array:
    x = _as_f32(x)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def _softmax(x: Array, axis: int = -1) -> Array:
    x = _as_f32(x)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-12)


def _l2_normalize(x: Array, eps: float = 1e-6) -> Array:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def validate_config(config: ModelConfig) -> None:
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
    recurrent_split(config)


def attention_kind(layer_index: int, config: ModelConfig) -> AttentionKind:
    if layer_index < 0 or layer_index >= config.n_layers:
        raise AttentionKindError(
            f"layer_index {layer_index} outside [0, {config.n_layers})"
        )
    period = LINEAR_TO_MLA + 1
    if layer_index % period == LINEAR_TO_MLA:
        return AttentionKind.MLA
    return AttentionKind.LINEAR


def ffn_kind(layer_index: int, config: ModelConfig) -> FfnKind:
    if layer_index < 0 or layer_index >= config.n_layers:
        raise ConfigError(f"layer_index {layer_index} outside [0, {config.n_layers})")
    if layer_index < config.n_dense:
        return FfnKind.DENSE
    return FfnKind.MOE


def rms_norm(x: Array, weight: Array, eps: float = RMS_NORM_EPS) -> Array:
    x = _as_f32(x)
    weight = _as_f32(weight)
    ms = np.mean(np.square(x), axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(ms + eps)) * weight


def _rope_angles(positions: Array, n_rot: int, rot_dim: int, x: Array) -> Array:
    inv_freq = ROPE_BASE ** (-2.0 * np.arange(n_rot, dtype=np.float32) / float(rot_dim))
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
    cos = np.cos(angle)
    sin = np.sin(angle)
    out_even = even * cos - odd * sin
    out_odd = even * sin + odd * cos
    stacked = np.stack([out_even, out_odd], axis=-1)
    out_rot = stacked.reshape(*rotated.shape)
    if passthrough.shape[-1] == 0:
        return out_rot
    return np.concatenate([out_rot, passthrough], axis=-1)


def rope(
    q: Array, k: Array, positions: Array, *, partial: bool = True
) -> tuple[Array, Array]:
    return _apply_rope(q, positions, partial=partial), _apply_rope(
        k, positions, partial=partial
    )


def rope_2d(q: Array, k: Array, row: Array, col: Array) -> tuple[Array, Array]:
    q = _as_f32(q)
    k = _as_f32(k)
    d = int(q.shape[-1])
    if d % 2 != 0:
        raise ValueError("rope_2d requires an even last dimension")
    half = d // 2
    q1, k1 = rope(q[..., :half], k[..., :half], row, partial=False)
    q2, k2 = rope(q[..., half:], k[..., half:], col, partial=False)
    return np.concatenate([q1, q2], axis=-1), np.concatenate([k1, k2], axis=-1)


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
    """Causal Gated-DeltaNet-family recurrence (delta rule, β=1).

    q, k, v: (batch, seq, dim) or (batch, seq, heads, dim)
    state: (batch, dim, dim) or (batch, heads, dim, dim)
    """
    q, squeeze = _ensure_heads(q)
    k, _ = _ensure_heads(k)
    v, _ = _ensure_heads(v)
    if q.shape != k.shape or q.shape[:3] != v.shape[:3] or q.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, v must share batch/seq/heads and key/value dim")
    batch, seq, heads, dim = q.shape
    qn = _l2_normalize(q)
    kn = _l2_normalize(k)
    if state is None:
        s = np.zeros((batch, heads, dim, dim), dtype=np.float32)
    else:
        s = _as_f32(state)
        if s.ndim == 3:
            s = s[:, None, :, :]
        s = s.copy()
        if s.shape != (batch, heads, dim, dim):
            raise ValueError(f"state shape {s.shape} != {(batch, heads, dim, dim)}")
    out = np.zeros_like(q, dtype=np.float32)
    eye = np.eye(dim, dtype=np.float32)
    for t in range(seq):
        kt = kn[:, t, :, :]
        vt = v[:, t, :, :]
        qt = qn[:, t, :, :]
        kk = np.einsum("bhd,bhe->bhde", kt, kt)
        decay = eye[None, None, :, :] - kk
        s = np.einsum("bhij,bhjk->bhik", decay, s)
        s = s + np.einsum("bhd,bhe->bhde", kt, vt)
        out[:, t, :, :] = np.einsum("bhd,bhde->bhe", qt, s)
    if squeeze:
        return out[:, :, 0, :], s[:, 0, :, :]
    return out, s


def mla_attention(
    q: Array,
    compressed_kv: Array,
    rope_k: Array,
    *,
    qk_norm: bool = True,
) -> Array:
    """Causal MLA kernel after projections.

    q: (batch, seq_q, heads, d_nope + d_rope)
    compressed_kv: (batch, seq_k, kv_heads, d_nope)  — kv_heads is 1 or heads
    rope_k: (batch, seq_k, kv_heads, d_rope) or (batch, seq_k, d_rope)

    Keys are concat(compressed_kv, rope_k); values are compressed_kv.
    Output: (batch, seq_q, heads, d_nope).
    """
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
            return np.broadcast_to(x, (x.shape[0], x.shape[1], n_heads, x.shape[3]))
        raise ValueError(f"cannot broadcast heads {x.shape[2]} to {n_heads}")

    ckv_h = _bcast(ckv)
    rk_h = _bcast(rk)
    k = np.concatenate([ckv_h, rk_h], axis=-1)
    v = ckv_h
    if qk_norm:
        ones_q = np.ones((d_q,), dtype=np.float32)
        ones_k = np.ones((d_q,), dtype=np.float32)
        q = rms_norm(q, ones_q)
        k = rms_norm(k, ones_k)
    scale = 1.0 / np.sqrt(np.float32(d_q))
    scores = np.einsum("bqhd,bkhd->bhqk", q, k) * scale
    q_idx = np.arange(seq_q)[:, None]
    k_idx = np.arange(seq_k)[None, :]
    causal = k_idx > (q_idx + (seq_k - seq_q))
    scores = np.where(causal[None, None, :, :], np.float32(-1e9), scores)
    attn = _softmax(scores, axis=-1)
    return np.einsum("bhqk,bkhd->bqhd", attn, v)


def _expert_to_rack(n_experts: int, max_racks: int) -> Array:
    n_racks = min(int(max_racks), int(n_experts))
    n_racks = max(n_racks, 1)
    return np.array(
        [min(n_racks - 1, e * n_racks // n_experts) for e in range(n_experts)],
        dtype=np.int32,
    )


def node_limited_topk(scores: Array, top_k: int, max_racks: int) -> Array:
    """Greedy top-k with at most ``max_racks`` racks per token.

    scores: (tokens, n_routed)
    returns expert ids (tokens, top_k)
    """
    scores = _as_f32(scores)
    n_tok, n_exp = scores.shape
    racks = _expert_to_rack(n_exp, max_racks)
    ids = np.zeros((n_tok, top_k), dtype=np.int32)
    for t in range(n_tok):
        order = np.argsort(-scores[t], kind="stable")
        chosen: list[int] = []
        used: set[int] = set()
        for e in order:
            e_i = int(e)
            rack = int(racks[e_i])
            if rack in used or len(used) < max_racks:
                chosen.append(e_i)
                used.add(rack)
            if len(chosen) == top_k:
                break
        if len(chosen) < top_k:
            for e in order:
                e_i = int(e)
                if e_i not in chosen:
                    chosen.append(e_i)
                if len(chosen) == top_k:
                    break
        ids[t] = np.asarray(chosen[:top_k], dtype=np.int32)
    return ids


def _swiglu(x: Array, w_gate: Array, w_up: Array, w_down: Array) -> Array:
    h = _silu(x @ w_gate) * (x @ w_up)
    return h @ w_down


def moe(
    x: Array,
    *,
    router_weight: Array,
    routed_weights: tuple[Array, Array, Array],
    shared_weights: tuple[Array, Array, Array],
    top_k: int,
    max_racks: int = 4,
) -> tuple[Array, Array, Array]:
    """Sigmoid-gated top-k routed experts + always-on shared experts (SwiGLU).

    routed_weights / shared_weights are (W_gate, W_up, W_down) with
    W_gate/W_up (n, d_model, hidden) and W_down (n, hidden, d_model).
    Returns (output, sigmoid router_probs, expert_ids).
    """
    x = _as_f32(x)
    orig = x.shape
    d_model = orig[-1]
    flat = x.reshape(-1, d_model)
    logits = flat @ _as_f32(router_weight)
    probs = _sigmoid(logits)
    expert_ids = node_limited_topk(probs, int(top_k), int(max_racks))
    tok = np.arange(flat.shape[0])[:, None]
    top_scores = probs[tok, expert_ids]
    gates = top_scores / np.maximum(top_scores.sum(axis=-1, keepdims=True), 1e-9)
    w_gate, w_up, w_down = (_as_f32(t) for t in routed_weights)
    out = np.zeros_like(flat)
    for t in range(flat.shape[0]):
        xt = flat[t]
        acc = np.zeros((d_model,), dtype=np.float32)
        for j in range(int(top_k)):
            e = int(expert_ids[t, j])
            acc = acc + gates[t, j] * _swiglu(xt, w_gate[e], w_up[e], w_down[e])
        out[t] = acc
    s_gate, s_up, s_down = (_as_f32(t) for t in shared_weights)
    for s in range(s_gate.shape[0]):
        out = out + _swiglu(flat, s_gate[s], s_up[s], s_down[s])
    n_routed = probs.shape[-1]
    router_probs = probs.reshape(*orig[:-1], n_routed)
    ids = expert_ids.reshape(*orig[:-1], int(top_k))
    return out.reshape(orig), router_probs, ids


def apply_latent_adapter(
    x: Array, w1: Array, w2: Array, norm_weight: Array
) -> Array:
    """2-layer MLP (SiLU) + RMSNorm. Maps a thought vector to residual width."""
    x = _as_f32(x)
    h = _silu(x @ _as_f32(w1)) @ _as_f32(w2)
    return rms_norm(h, norm_weight)


def latent_adapter(
    x: Array,
    *,
    hidden: int,
    w1: Array | None = None,
    w2: Array | None = None,
    norm_weight: Array | None = None,
) -> Array:
    """Reference adapter. Production has no weights; tests go through forward."""
    if w1 is None or w2 is None or norm_weight is None:
        raise TypeError("reference latent_adapter requires w1, w2, and norm_weight")
    if w1.shape[-1] != hidden:
        raise ValueError(f"w1 inner dim {w1.shape[-1]} != hidden {hidden}")
    return apply_latent_adapter(x, w1, w2, norm_weight)


def _rng(rng: Any) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng(0)
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    raise TypeError(f"unsupported rng type {type(rng)!r}")


def _w(rng: np.random.Generator, shape: tuple[int, ...]) -> Array:
    return (rng.standard_normal(shape) * INIT_SCALE).astype(np.float32)


def init_params(config: ModelConfig, rng: Any) -> dict[str, Any]:
    """FP32 parameter tree. See module docstring for layout."""
    validate_config(config)
    rng = _rng(rng)
    d = config.d_model
    n_heads, d_head, d_nope, d_rope = attn_geometry(d)
    kv_rank = d_head
    hidden = config.expert_hidden
    adapter_h = config.adapter_hidden if config.adapter_hidden is not None else d
    layers: list[dict[str, Array]] = []
    for i in range(config.n_layers):
        layer: dict[str, Array] = {
            "pre_attn_norm": np.ones((d,), dtype=np.float32),
            "pre_ffn_norm": np.ones((d,), dtype=np.float32),
        }
        kind = attention_kind(i, config)
        if kind is AttentionKind.LINEAR:
            layer["W_q"] = _w(rng, (d, n_heads, d_head))
            layer["W_k"] = _w(rng, (d, n_heads, d_head))
            layer["W_v"] = _w(rng, (d, n_heads, d_head))
            layer["W_o"] = _w(rng, (n_heads, d_head, d))
        else:
            layer["W_q"] = _w(rng, (d, n_heads, d_nope + d_rope))
            layer["W_kv_compress"] = _w(rng, (d, kv_rank))
            layer["W_kv_up"] = _w(rng, (kv_rank, n_heads, d_nope))
            layer["W_rope_k"] = _w(rng, (d, d_rope))
            layer["W_o"] = _w(rng, (n_heads, d_nope, d))
        if ffn_kind(i, config) is FfnKind.DENSE:
            layer["ffn_gate"] = _w(rng, (d, hidden))
            layer["ffn_up"] = _w(rng, (d, hidden))
            layer["ffn_down"] = _w(rng, (hidden, d))
        else:
            n_r = config.n_routed_experts
            n_s = config.n_shared_experts
            layer["router"] = _w(rng, (d, n_r))
            layer["routed_gate"] = _w(rng, (n_r, d, hidden))
            layer["routed_up"] = _w(rng, (n_r, d, hidden))
            layer["routed_down"] = _w(rng, (n_r, hidden, d))
            layer["shared_gate"] = _w(rng, (n_s, d, hidden))
            layer["shared_up"] = _w(rng, (n_s, d, hidden))
            layer["shared_down"] = _w(rng, (n_s, hidden, d))
        layers.append(layer)
    mtp = []
    for _ in range(config.mtp_heads):
        mtp.append(
            {
                "norm": np.ones((d,), dtype=np.float32),
                "proj": _w(rng, (d, d)),
                "unembed": _w(rng, (config.vocab_size, d)),
            }
        )
    return {
        "embed": _w(rng, (config.vocab_size, d)),
        "unembed": _w(rng, (config.vocab_size, d)),
        "final_norm": np.ones((d,), dtype=np.float32),
        "layers": layers,
        "mtp": mtp,
        "adapter_w1": _w(rng, (d, adapter_h)),
        "adapter_w2": _w(rng, (adapter_h, d)),
        "adapter_norm": np.ones((d,), dtype=np.float32),
    }


def param_count(params: Any) -> int:
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
    q = np.einsum("bsd,dhe->bshe", x, layer["W_q"])
    k = np.einsum("bsd,dhe->bshe", x, layer["W_k"])
    v = np.einsum("bsd,dhe->bshe", x, layer["W_v"])
    y, state = linear_attention(q, k, v, state)
    y = np.einsum("bshe,hed->bsd", y, layer["W_o"])
    return y, state


def _mla_attn(x: Array, layer: dict[str, Array], positions: Array) -> Array:
    q = np.einsum("bsd,dhe->bshe", x, layer["W_q"])
    ckv = np.einsum("bsd,dc->bsc", x, layer["W_kv_compress"])
    k_nope = np.einsum("bsc,che->bshe", ckv, layer["W_kv_up"])
    rope_k = np.einsum("bsd,dr->bsr", x, layer["W_rope_k"])
    d_nope = k_nope.shape[-1]
    q_nope = q[..., :d_nope]
    q_rope = q[..., d_nope:]
    rk = rope_k[:, :, None, :]
    rk = np.broadcast_to(rk, q_rope.shape)
    q_rope, rk = rope(q_rope, rk, positions, partial=False)
    q = np.concatenate([q_nope, q_rope], axis=-1)
    y = mla_attention(q, k_nope, rk, qk_norm=True)
    return np.einsum("bshe,hed->bsd", y, layer["W_o"])


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
        router_logits = n.reshape(-1, n.shape[-1]) @ layer["router"]
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
    validate_config(config)
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError("tokens must be (batch, seq)")
    batch, seq = tokens.shape
    if seq > config.max_context:
        raise ConfigError("sequence longer than max_context")
    if tokens.size and (
        int(tokens.min()) < 0 or int(tokens.max()) >= config.vocab_size
    ):
        raise ConfigError("token id out of vocab")
    h = params["embed"][tokens]
    n_thoughts = 0
    if thoughts is not None:
        thoughts = _as_f32(thoughts)
        if thoughts.ndim != 3 or thoughts.shape[0] != batch:
            raise ValueError("thoughts must be (batch, n_thoughts, d_model)")
        if thoughts.shape[-1] != config.d_model:
            raise ValueError("thoughts last dim must equal d_model")
        n_thoughts = int(thoughts.shape[1])
        if n_thoughts:
            hidden_ad = (
                config.adapter_hidden
                if config.adapter_hidden is not None
                else config.d_model
            )
            adapted = latent_adapter(
                thoughts,
                hidden=hidden_ad,
                w1=params["adapter_w1"],
                w2=params["adapter_w2"],
                norm_weight=params["adapter_norm"],
            )
            h = np.concatenate([adapted, h], axis=1)
    full_seq = h.shape[1]
    positions = np.arange(full_seq, dtype=np.float32)
    r_used = int(r) if r is not None else _sample_r(config, tokens)
    if r_used < 1:
        raise ConfigError("r must be >= 1")
    prelude, core, coda = recurrent_split(config)
    z_acc = []
    last_probs = None
    last_ids = None

    def run_layer(
        h: Array, idx: int, lin_state: Array | None
    ) -> tuple[Array, Array | None]:
        nonlocal last_probs, last_ids
        h, lin_state, logits, probs, ids = _block(
            h, params["layers"][idx], config, idx, positions, lin_state
        )
        if logits is not None:
            lse = np.logaddexp.reduce(logits, axis=-1)
            z_acc.append(np.mean(np.square(lse)))
            last_probs, last_ids = probs, ids
        return h, lin_state

    for i in range(prelude):
        h, _ = run_layer(h, i, None)
    injected = h.copy()
    core0 = prelude
    for it in range(r_used):
        step = h + injected if it > 0 else h
        for j in range(core):
            step, _ = run_layer(step, core0 + j, None)
        h = step
    for k in range(coda):
        h, _ = run_layer(h, prelude + core + k, None)

    h = rms_norm(h, params["final_norm"])
    hidden = h[:, n_thoughts:, :]
    logits = hidden @ params["unembed"].T
    mtp_logits = []
    for head in params["mtp"]:
        mh = rms_norm(hidden, head["norm"])
        mh = mh @ head["proj"] + hidden
        mtp_logits.append(mh @ head["unembed"].T)
    if last_probs is None:
        router_probs = np.zeros(
            (batch, seq, config.n_routed_experts), dtype=np.float32
        )
        expert_ids = np.zeros((batch, seq, config.top_k), dtype=np.int32)
    else:
        router_probs = last_probs[:, n_thoughts:, :]
        expert_ids = last_ids[:, n_thoughts:, :]
    if z_acc:
        z_loss = np.asarray(float(np.mean(z_acc)), dtype=np.float32)
    else:
        z_loss = np.asarray(0.0, dtype=np.float32)
    return ForwardOutput(
        logits=logits,
        hidden=hidden,
        mtp_logits=mtp_logits,
        router_probs=router_probs,
        expert_ids=expert_ids,
        z_loss=z_loss,
        r_used=r_used,
    )
