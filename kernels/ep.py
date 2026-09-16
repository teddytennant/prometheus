"""Expert-parallel dispatch/combine (spec 4). CPU stand-in for all-to-all."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def ep_dispatch(tokens: Array, expert_ids: Array, n_experts: int) -> list[Array]:
    tokens = jnp.asarray(tokens)
    expert_ids = jnp.asarray(expert_ids)
    buckets = []
    flat = tokens.reshape(-1, tokens.shape[-1])
    ids = expert_ids.reshape(-1)
    for e in range(n_experts):
        buckets.append(flat[ids == e])
    return buckets


def dispatch(tokens: Array, expert_ids: Array, n_experts: int) -> list[Array]:
    return ep_dispatch(tokens, expert_ids, n_experts)


def combine(expert_out: Array, weights: Array, expert_ids: Array) -> Array:
    gathered = expert_out[
        expert_ids,
        jnp.arange(expert_ids.shape[0])[:, None, None],
        jnp.arange(expert_ids.shape[1])[None, :, None],
    ]
    return (gathered * weights[..., None]).sum(axis=2)
