"""Param init and forward (spec 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

INIT_SCALE = 0.02

from model.config import (
    AttentionKind,
    FfnKind,
    ModelConfig,
    _attn_geometry,
    _recurrent_split,
    attention_kind,
    ffn_kind,
    validate_config,
)
from model.layers import (
    _apply_latent_adapter,
    _as_f32,
    _l2_normalize,
    _sigmoid,
    _silu,
    _softmax,
    _swiglu,
    linear_attention,
    mla_attention,
    moe,
    rms_norm,
    rope,
)

@dataclass
class ForwardOutput:
    """Next-token logits plus MTP heads and the router aux used in the loss."""

    logits: Array
    mtp_logits: tuple[Array, ...]
    z_loss: Array
    router_probs: Array
    expert_ids: Array
    hidden: Array
    r_used: int


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
