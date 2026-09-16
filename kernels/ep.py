"""Expert-parallel dispatch/combine and all-to-all (spec 5.1)."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array


def expert_home(expert_id: int, n_ep: int) -> int:
    return int(expert_id) % int(n_ep)


def ep_dispatch(tokens: Array, expert_ids: Array, n_experts: int) -> list[Array]:
    tokens = jnp.asarray(tokens)
    expert_ids = jnp.asarray(expert_ids)
    flat = tokens.reshape(-1, tokens.shape[-1])
    ids = expert_ids.reshape(-1)
    return [flat[ids == e] for e in range(int(n_experts))]


def dispatch(tokens: Array, expert_ids: Array, n_experts: int) -> list[Array]:
    return ep_dispatch(tokens, expert_ids, n_experts)


def combine(expert_out: Array, weights: Array, expert_ids: Array) -> Array:
    expert_out = jnp.asarray(expert_out)
    weights = jnp.asarray(weights)
    expert_ids = jnp.asarray(expert_ids)
    gathered = expert_out[expert_ids]
    while weights.ndim < gathered.ndim:
        weights = weights[..., None]
    return gathered * weights


def all_to_all(send_bufs: list[list[Any]]) -> list[list[Any]]:
    """EP all-to-all: recv[dst][src] = send[src][dst]. Applying twice is identity."""
    n = len(send_bufs)
    if any(len(row) != n for row in send_bufs):
        raise ValueError("send_bufs must be an n_ep x n_ep grid")
    return [[send_bufs[src][dst] for src in range(n)] for dst in range(n)]


def partition_by_rank(tokens: Array, expert_ids: Array, n_ep: int) -> list[Array]:
    """Send each token to the EP rank that owns its expert (expert_id % n_ep)."""
    tokens = np.asarray(tokens, dtype=np.float32)
    ids = np.asarray(expert_ids, dtype=np.int32).reshape(-1)
    if tokens.ndim == 1:
        tokens = tokens[:, None]
    homes = np.array([expert_home(int(e), n_ep) for e in ids], dtype=np.int32)
    out: list[Array] = []
    for r in range(n_ep):
        sel = homes == r
        out.append(tokens[sel] if np.any(sel) else np.zeros((0, tokens.shape[1]), dtype=np.float32))
    return out


def reconstruct_from_dispatch(
    buckets: list[Array], expert_ids: Array, n_tokens: int, dim: int
) -> np.ndarray:
    ids = np.asarray(expert_ids, dtype=np.int32).reshape(-1)
    out = np.zeros((n_tokens, dim), dtype=np.float32)
    for e, bucket in enumerate(buckets):
        b = np.asarray(bucket)
        if b.size:
            out[ids == e] = b
    return out


def _swiglu_np(
    x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray
) -> np.ndarray:
    h = x @ w_gate
    h = h * (1.0 / (1.0 + np.exp(-np.clip(h, -20, 20))))
    return (h * (x @ w_up)) @ w_down


def ep_moe_match(
    n_ep: int = 2,
    n_experts: int = 8,
    top_k: int = 2,
    d: int = 16,
    hidden: int = 32,
    n_tok: int = 12,
    n_shared: int = 1,
) -> dict[str, Any]:
    """Full MoE vs expert-parallel split. Routing is computed once, then sharded."""
    from model.layers import moe

    rng = np.random.default_rng(2)
    x = rng.standard_normal((n_tok, d)).astype(np.float32)
    router = rng.standard_normal((d, n_experts)).astype(np.float32) * 0.2
    w_gate = rng.standard_normal((n_experts, d, hidden)).astype(np.float32) * 0.05
    w_up = rng.standard_normal((n_experts, d, hidden)).astype(np.float32) * 0.05
    w_down = rng.standard_normal((n_experts, hidden, d)).astype(np.float32) * 0.05
    s_gate = rng.standard_normal((n_shared, d, hidden)).astype(np.float32) * 0.05
    s_up = rng.standard_normal((n_shared, d, hidden)).astype(np.float32) * 0.05
    s_down = rng.standard_normal((n_shared, hidden, d)).astype(np.float32) * 0.05

    full, probs, ids = moe(
        jnp.asarray(x),
        router_weight=jnp.asarray(router),
        routed_weights=(jnp.asarray(w_gate), jnp.asarray(w_up), jnp.asarray(w_down)),
        shared_weights=(jnp.asarray(s_gate), jnp.asarray(s_up), jnp.asarray(s_down)),
        top_k=top_k,
    )
    full_np = np.asarray(full)
    ids_np = np.asarray(ids)
    probs_np = np.asarray(probs)
    top_scores = np.take_along_axis(probs_np, ids_np, axis=-1)
    gates = top_scores / np.maximum(top_scores.sum(axis=-1, keepdims=True), 1e-9)

    routed = np.zeros_like(x)
    for rank in range(int(n_ep)):
        for n in range(n_tok):
            for k in range(top_k):
                e = int(ids_np[n, k])
                if expert_home(e, n_ep) != rank:
                    continue
                y = _swiglu_np(x[n : n + 1], w_gate[e], w_up[e], w_down[e])
                routed[n] += float(gates[n, k]) * y[0]

    shared = np.zeros_like(x)
    for s in range(n_shared):
        shared += _swiglu_np(x, s_gate[s], s_up[s], s_down[s])
    ep_out = routed + shared
    denom = float(np.linalg.norm(full_np) + 1e-8)
    rel = float(np.linalg.norm(ep_out - full_np) / denom)
    send = [[np.array([src * n_ep + dst]) for dst in range(n_ep)] for src in range(n_ep)]
    a2a_ok = all(
        np.array_equal(np.asarray(all_to_all(all_to_all(send))[i][j]), np.asarray(send[i][j]))
        for i in range(n_ep)
        for j in range(n_ep)
    )
    return {
        "relative_err": rel,
        "routing_identical": bool(a2a_ok),
        "n_ep": int(n_ep),
        "n_experts": int(n_experts),
        "identity_mesh": False,
    }
