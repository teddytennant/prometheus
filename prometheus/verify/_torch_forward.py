"""Independent PyTorch flagship-shape forward for V1 logit parity (spec 16.2).

Not JAX ``model.forward``. Not NumPy. Not imported from ``tests/``.
Production ``prometheus.verify.v1_parity.run_v1`` compares JAX logits to
this module. Gate: finite ``max |a - b|`` <= 1e-5 in FP32 on
``model.tiny_config``. CPU analog stays callable without a GPU; that does
not count as V1 verified.

Torch is the independent stack. Do not re-export NumPy math wrapped in
``torch.tensor``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from model import (
    AttentionKind,
    ConfigError,
    FfnKind,
    ModelConfig,
    attention_kind,
    ffn_kind,
)

ROPE_BASE = 10000.0
RMS_NORM_EPS = 1e-6


def _as_f32(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32)
    return torch.tensor(np.asarray(x), dtype=torch.float32)


def _as_long(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.long)
    return torch.tensor(np.asarray(x), dtype=torch.long)


def _tree_to_torch(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _tree_to_torch(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_tree_to_torch(v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_to_torch(v) for v in tree)
    if isinstance(tree, torch.Tensor):
        return tree
    arr = np.array(np.asarray(tree), copy=True)
    return torch.tensor(arr)


def _silu(x: torch.Tensor) -> torch.Tensor:
    x = _as_f32(x)
    return x * (1.0 / (1.0 + torch.exp(-torch.clamp(x, -80.0, 80.0))))


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = _as_f32(x)
    return 1.0 / (1.0 + torch.exp(-torch.clamp(x, -80.0, 80.0)))


def _softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = _as_f32(x)
    x = x - torch.amax(x, dim=dim, keepdim=True)
    e = torch.exp(x)
    return e / torch.clamp(e.sum(dim=dim, keepdim=True), min=1e-12)


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    n = torch.linalg.norm(x, dim=-1, keepdim=True)
    return x / torch.clamp(n, min=eps)


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_NORM_EPS
) -> torch.Tensor:
    x = _as_f32(x)
    weight = _as_f32(weight)
    ms = torch.mean(torch.square(x), dim=-1, keepdim=True)
    return x * (1.0 / torch.sqrt(ms + float(eps))) * weight


def _recurrent_split(config: ModelConfig) -> tuple[int, int, int]:
    core = int(config.core_block_layers)
    if config.prelude_layers is not None and config.coda_layers is not None:
        prelude = int(config.prelude_layers)
        coda = int(config.coda_layers)
        return prelude, core, coda
    rest = config.n_layers - core
    prelude = rest // 2
    coda = rest - prelude
    return prelude, core, coda


def _rope_angles(
    positions: torch.Tensor, n_rot: int, rot_dim: int, x: torch.Tensor
) -> torch.Tensor:
    inv_freq = torch.tensor(ROPE_BASE, dtype=torch.float32) ** (
        -torch.tensor(2.0, dtype=torch.float32)
        * torch.arange(n_rot, dtype=torch.float32)
        / torch.tensor(rot_dim, dtype=torch.float32)
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


def _apply_rope(
    x: torch.Tensor, positions: torch.Tensor, *, partial: bool
) -> torch.Tensor:
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
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    out_even = even * cos - odd * sin
    out_odd = even * sin + odd * cos
    stacked = torch.stack([out_even, out_odd], dim=-1)
    out_rot = stacked.reshape(*rotated.shape)
    if passthrough.shape[-1] == 0:
        return out_rot
    return torch.cat([out_rot, passthrough], dim=-1)


def _ensure_heads(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    x = _as_f32(x)
    if x.ndim == 3:
        return x[:, :, None, :], True
    if x.ndim == 4:
        return x, False
    raise ValueError(f"expected 3-D or 4-D tensor, got shape {x.shape}")


def linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, squeeze = _ensure_heads(q)
    k, _ = _ensure_heads(k)
    v, _ = _ensure_heads(v)
    if q.shape != k.shape or q.shape[:3] != v.shape[:3] or q.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, v must share batch/seq/heads and key/value dim")
    batch, seq, heads, dim = q.shape
    qn = _l2_normalize(q)
    kn = _l2_normalize(k)
    if state is None:
        s = torch.zeros((batch, heads, dim, dim), dtype=torch.float32)
    else:
        s = _as_f32(state)
        if s.ndim == 3:
            s = s[:, None, :, :]
        if tuple(s.shape) != (batch, heads, dim, dim):
            raise ValueError(f"state shape {tuple(s.shape)} != {(batch, heads, dim, dim)}")
    out = torch.zeros_like(q, dtype=torch.float32)
    eye = torch.eye(dim, dtype=torch.float32)
    for t in range(seq):
        kt = kn[:, t, :, :]
        vt = v[:, t, :, :]
        qt = qn[:, t, :, :]
        kk = torch.einsum("bhd,bhe->bhde", kt, kt)
        decay = eye[None, None, :, :] - kk
        s = torch.einsum("bhij,bhjk->bhik", decay, s)
        s = s + torch.einsum("bhd,bhe->bhde", kt, vt)
        out[:, t, :, :] = torch.einsum("bhd,bhde->bhe", qt, s)
    if squeeze:
        return out[:, :, 0, :], s[:, 0, :, :]
    return out, s


def mla_attention(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    rope_k: torch.Tensor,
    *,
    qk_norm: bool = True,
) -> torch.Tensor:
    q = _as_f32(q)
    ckv = _as_f32(compressed_kv)
    rk = _as_f32(rope_k)
    if q.ndim != 4:
        raise ValueError(f"q must be 4-D, got {tuple(q.shape)}")
    if ckv.ndim == 3:
        ckv = ckv[:, :, None, :]
    if rk.ndim == 3:
        rk = rk[:, :, None, :]
    batch, seq_q, n_heads, d_q = q.shape
    seq_k = int(ckv.shape[1])
    d_nope = int(ckv.shape[-1])
    d_rope = int(rk.shape[-1])
    if d_q != d_nope + d_rope:
        raise ValueError(
            f"q last dim {d_q} must equal d_nope ({d_nope}) + d_rope ({d_rope})"
        )

    def _bcast(x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] == n_heads:
            return x
        if x.shape[2] == 1:
            return x.expand(x.shape[0], x.shape[1], n_heads, x.shape[3])
        raise ValueError(f"cannot broadcast heads {x.shape[2]} to {n_heads}")

    ckv_h = _bcast(ckv)
    rk_h = _bcast(rk)
    k = torch.cat([ckv_h, rk_h], dim=-1)
    v = ckv_h
    if qk_norm:
        ones_q = torch.ones((d_q,), dtype=torch.float32)
        ones_k = torch.ones((d_q,), dtype=torch.float32)
        q = rms_norm(q, ones_q)
        k = rms_norm(k, ones_k)
    scale = 1.0 / torch.sqrt(torch.tensor(d_q, dtype=torch.float32))
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    q_idx = torch.arange(seq_q)[:, None]
    k_idx = torch.arange(seq_k)[None, :]
    causal = k_idx > (q_idx + (seq_k - seq_q))
    scores = torch.where(
        causal[None, None, :, :],
        torch.tensor(-1e9, dtype=torch.float32),
        scores,
    )
    attn = _softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", attn, v)


def _expert_to_rack(n_experts: int, max_racks: int) -> list[int]:
    n_racks = min(int(max_racks), int(n_experts))
    n_racks = max(n_racks, 1)
    return [min(n_racks - 1, e * n_racks // n_experts) for e in range(n_experts)]


def node_limited_topk(
    scores: torch.Tensor, top_k: int, max_racks: int
) -> torch.Tensor:
    scores = _as_f32(scores)
    n_tok, n_exp = scores.shape
    racks = _expert_to_rack(n_exp, max_racks)
    ids = torch.zeros((n_tok, top_k), dtype=torch.int32)
    for t in range(n_tok):
        order = torch.argsort(-scores[t], stable=True)
        chosen: list[int] = []
        used: set[int] = set()
        for e in order.tolist():
            e_i = int(e)
            rack = int(racks[e_i])
            if rack in used or len(used) < max_racks:
                chosen.append(e_i)
                used.add(rack)
            if len(chosen) == top_k:
                break
        if len(chosen) < top_k:
            for e in order.tolist():
                e_i = int(e)
                if e_i not in chosen:
                    chosen.append(e_i)
                if len(chosen) == top_k:
                    break
        ids[t] = torch.tensor(chosen[:top_k], dtype=torch.int32)
    return ids


def _swiglu(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    h = _silu(x @ w_gate) * (x @ w_up)
    return h @ w_down


def moe(
    x: torch.Tensor,
    *,
    router_weight: torch.Tensor,
    routed_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    shared_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    top_k: int,
    max_racks: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = _as_f32(x)
    orig = tuple(x.shape)
    d_model = orig[-1]
    flat = x.reshape(-1, d_model)
    logits = flat @ _as_f32(router_weight)
    probs = _sigmoid(logits)
    expert_ids = node_limited_topk(probs, int(top_k), int(max_racks))
    tok = torch.arange(flat.shape[0])[:, None]
    ids_l = expert_ids.to(dtype=torch.long)
    top_scores = probs[tok, ids_l]
    gates = top_scores / torch.clamp(top_scores.sum(dim=-1, keepdim=True), min=1e-9)
    w_gate, w_up, w_down = (_as_f32(t) for t in routed_weights)
    wg = w_gate[ids_l]
    wu = w_up[ids_l]
    wd = w_down[ids_l]
    hidden = _silu(torch.einsum("nd,nkdh->nkh", flat, wg)) * torch.einsum(
        "nd,nkdh->nkh", flat, wu
    )
    routed = torch.einsum("nkh,nkhd->nkd", hidden, wd)
    out = torch.einsum("nk,nkd->nd", gates, routed)
    s_gate, s_up, s_down = (_as_f32(t) for t in shared_weights)
    for s in range(int(s_gate.shape[0])):
        out = out + _swiglu(flat, s_gate[s], s_up[s], s_down[s])
    n_routed = int(probs.shape[-1])
    router_probs = probs.reshape(*orig[:-1], n_routed)
    ids = expert_ids.reshape(*orig[:-1], int(top_k))
    return out.reshape(orig), router_probs, ids


def _linear_attn(n: torch.Tensor, layer: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.einsum("bsd,dhe->bshe", n, _as_f32(layer["W_q"]))
    k = torch.einsum("bsd,dhe->bshe", n, _as_f32(layer["W_k"]))
    v = torch.einsum("bsd,dhe->bshe", n, _as_f32(layer["W_v"]))
    y, state = linear_attention(q, k, v, None)
    y = torch.einsum("bshe,hed->bsd", y, _as_f32(layer["W_o"]))
    return y, state


def _mla_attn(
    n: torch.Tensor, layer: dict[str, Any], positions: torch.Tensor
) -> torch.Tensor:
    q = torch.einsum("bsd,dhe->bshe", n, _as_f32(layer["W_q"]))
    ckv = torch.einsum("bsd,dc->bsc", n, _as_f32(layer["W_kv_compress"]))
    k_nope = torch.einsum("bsc,che->bshe", ckv, _as_f32(layer["W_kv_up"]))
    rope_k = torch.einsum("bsd,dr->bsr", n, _as_f32(layer["W_rope_k"]))
    d_nope = int(k_nope.shape[-1])
    q_nope = q[..., :d_nope]
    q_rope = q[..., d_nope:]
    rk = rope_k[:, :, None, :].expand_as(q_rope)
    q_rope, rk = _apply_rope(q_rope, positions, partial=False), _apply_rope(
        rk, positions, partial=False
    )
    q = torch.cat([q_nope, q_rope], dim=-1)
    y = mla_attention(q, k_nope, rk, qk_norm=True)
    return torch.einsum("bshe,hed->bsd", y, _as_f32(layer["W_o"]))


def _dense_ffn(n: torch.Tensor, layer: dict[str, Any]) -> torch.Tensor:
    return _swiglu(n, layer["ffn_gate"], layer["ffn_up"], layer["ffn_down"])


def _sample_r(config: ModelConfig, tokens: torch.Tensor) -> int:
    seed = int(tokens.detach().cpu().sum().item()) % (2**31)
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


def _apply_layer(
    h: torch.Tensor,
    layer: dict[str, Any],
    layer_index: int,
    config: ModelConfig,
    positions: torch.Tensor,
) -> torch.Tensor:
    n = rms_norm(h, layer["pre_attn_norm"])
    if attention_kind(layer_index, config) is AttentionKind.LINEAR:
        attn, _ = _linear_attn(n, layer)
    else:
        attn = _mla_attn(n, layer, positions)
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
    return h


def _run_unique(
    h: torch.Tensor,
    layers: list[dict[str, Any]],
    indices: range,
    config: ModelConfig,
    positions: torch.Tensor,
) -> torch.Tensor:
    for i in indices:
        h = _apply_layer(h, layers[i], i, config, positions)
    return h


def forward(
    tokens: Any,
    params: dict[str, Any],
    config: ModelConfig,
    *,
    r: int | None = None,
) -> torch.Tensor:
    """Next-token logits, same architecture/params as production ``model.forward``.

    Implementation is PyTorch (CPU is enough). Returns logits of shape
    ``(batch, seq, vocab_size)`` in float32.
    """
    params_t = _tree_to_torch(params)
    tok = _as_long(tokens)
    if tok.ndim != 2:
        raise ValueError("tokens must be (batch, seq)")
    seq = int(tok.shape[1])
    if seq > config.max_context:
        raise ConfigError("sequence longer than max_context")
    if tok.numel() and (int(tok.min()) < 0 or int(tok.max()) >= config.vocab_size):
        raise ConfigError("token id out of vocab")
    embed = _as_f32(params_t["embed"])
    h = embed[tok]
    positions = torch.arange(seq, dtype=torch.float32)
    layers: list[dict[str, Any]] = params_t["layers"]
    prelude, core, coda = _recurrent_split(config)
    r_used = int(r) if r is not None else _sample_r(config, tok)
    if r_used < 1:
        raise ConfigError("r must be >= 1")
    h = _run_unique(h, layers, range(0, prelude), config, positions)
    injected = h
    h = _run_unique(h, layers, range(prelude, prelude + core), config, positions)
    for _ in range(r_used - 1):
        h = _run_unique(
            h + injected,
            layers,
            range(prelude, prelude + core),
            config,
            positions,
        )
    h = _run_unique(
        h,
        layers,
        range(prelude + core, prelude + core + coda),
        config,
        positions,
    )
    h = rms_norm(h, params_t["final_norm"])
    logits = h @ _as_f32(params_t["unembed"]).T
    return logits.to(dtype=torch.float32)
