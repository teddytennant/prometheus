"""Attention kernels. Linear attn is the model reference; flash is a probe."""

from __future__ import annotations

from typing import Any

from jax import Array

import model as M


def linear_attention(
    q: Array, k: Array, v: Array, state: Array | None = None
) -> tuple[Array, Array]:
    return M.linear_attention(q, k, v, state=state)


def mla_attention(q, k_nope, k_rope, *, qk_norm: bool = True) -> Array:
    return M.mla_attention(q, k_nope, k_rope, qk_norm=qk_norm)


def flash_attention_probe(q: Array, k: Array, v: Array) -> Array:
    import jax.numpy as jnp

    scale = 1.0 / jnp.sqrt(q.shape[-1])
    logits = jnp.einsum("bthd,bshd->bhts", q, k) * scale
    w = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
    w = w / (jnp.sum(w, axis=-1, keepdims=True) + 1e-9)
    return jnp.einsum("bhts,bshd->bthd", w, v)
