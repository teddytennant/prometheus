"""Independent NumPy forward for V1 logit parity (not JAX ``model.forward``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from model import (
    AttentionKind,
    ConfigError,
    FfnKind,
    ModelConfig,
    attention_kind,
    ffn_kind,
    validate_config,
)

Array = np.ndarray

RMS_NORM_EPS = 1e-6
ROPE_BASE = 10_000.0


def _as_f32(x: Any) -> Array:
    return np.asarray(x, dtype=np.float32)


def _as_tree(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _as_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_as_tree(v) for v in obj]
        return list(seq) if isinstance(obj, list) else tuple(seq)
    if hasattr(obj, "ndim"):
        return np.asarray(obj)
    return obj


def _silu(x: Array) -> Array:
    x = _as_f32(x)
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0))))


def _sigmoid(x: Array) -> Array:
    x = _as_f32(x)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def _softmax(x: Array, axis: int = -1) -> Array:
    x = _as_f32(x)
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(z, -80.0, 80.0))
    return e / np.sum(e, axis=axis, keepdims=True)


def _l2_normalize(x: Array, axis: int = -1) -> Array:
    x = _as_f32(x)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def rms_norm(x: Array, weight: Array) -> Array:
    x = _as_f32(x)
    w = _as_f32(weight)
    ms = np.mean(np.square(x), axis=-1, keepdims=True)
    return x * np.reciprocal(np.sqrt(ms + np.float32(RMS_NORM_EPS))) * w


def _rope_angles(positions: Array, n_rot: int, rot_dim: int, like: Array) -> Array:
    pos = _as_f32(positions)
    if pos.ndim == 0:
        pos = pos.reshape(1)
    idx = np.arange(n_rot, dtype=np.float32)
    freq = np.power(np.float32(ROPE_BASE), -(2.0 * idx) / np.float32(rot_dim))
    if pos.ndim == 1:
        angle = pos[:, None] * freq[None, :]
        b = int(like.shape[0])
        s = int(like.shape[1])
        h = int(like.shape[2]) if like.ndim == 4 else 1
        angle = np.broadcast_to(angle[None, :, None, :], (b, s, h, n_rot))
        if like.ndim == 3:
            angle = angle[:, :, 0, :]
        return angle
    raise ValueError("positions must be 0-D or 1-D")


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


def _recurrent_split(config: ModelConfig) -> tuple[int, int, int]:
    n = int(config.n_layers)
    prelude = min(int(config.prelude_layers), n)
    coda = min(int(config.coda_layers), n - prelude)
    core = n - prelude - coda
    if core < 1:
        raise ConfigError("need at least one core layer for recurrence")
    return prelude, core, coda


def _block(
    h: Array,
    layer: dict[str, Array],
    config: ModelConfig,
    layer_index: int,
    positions: Array,
    lin_state: Array | None,
) -> tuple[Array, Array | None]:
    n = rms_norm(h, layer["pre_attn_norm"])
    if attention_kind(layer_index, config) is AttentionKind.LINEAR:
        attn, lin_state = _linear_attn(n, layer, lin_state)
    else:
        attn = _mla_attn(n, layer, positions)
        lin_state = None
    h = h + attn
    n = rms_norm(h, layer["pre_ffn_norm"])
    if ffn_kind(layer_index, config) is FfnKind.DENSE:
        h = h + _dense_ffn(n, layer)
    else:
        routed = (layer["routed_gate"], layer["routed_up"], layer["routed_down"])
        shared = (layer["shared_gate"], layer["shared_up"], layer["shared_down"])
        y, _, _ = moe(
            n,
            router_weight=layer["router"],
            routed_weights=routed,
            shared_weights=shared,
            top_k=config.top_k,
            max_racks=config.max_racks,
        )
        h = h + y
    return h, lin_state


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
    tokens: Any,
    params: dict[str, Any],
    config: ModelConfig,
    *,
    r: int | None = None,
) -> Array:
    """Next-token logits, same architecture/params as production ``model.forward``."""
    validate_config(config)
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError("tokens must be (batch, seq)")
    _batch, seq = tokens.shape
    if seq > config.max_context:
        raise ConfigError("sequence longer than max_context")
    if tokens.size and (
        int(tokens.min()) < 0 or int(tokens.max()) >= config.vocab_size
    ):
        raise ConfigError("token id out of vocab")
    params = _as_tree(params)
    embed = np.asarray(params["embed"])
    h = embed[tokens]
    positions = np.arange(h.shape[1], dtype=np.float32)
    r_used = int(r) if r is not None else _sample_r(config, tokens)
    if r_used < 1:
        raise ConfigError("r must be >= 1")
    prelude, core, coda = _recurrent_split(config)
    layers = params["layers"]

    def run_layer(h: Array, idx: int, lin_state: Array | None) -> tuple[Array, Array | None]:
        return _block(h, layers[idx], config, idx, positions, lin_state)

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
    return h @ np.asarray(params["unembed"]).T
