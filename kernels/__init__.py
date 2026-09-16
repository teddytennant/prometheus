"""Linear attention, FP8 linears, EP dispatch (spec 5.3, 15.5 A3).

CUDA/Triton kernels are the cluster path (S3). This module is the JAX/XLA
reference the CPU tests and V1/V3 checkers run.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import model as M

Array = jax.Array

FP8_MAX = 448.0  # e4m3


def linear_attention(
    q: Array, k: Array, v: Array, state: Array | None = None
) -> tuple[Array, Array]:
    return M.linear_attention(q, k, v, state=state)


def mla_attention(q, k_nope, k_rope, *, qk_norm: bool = True) -> Array:
    return M.mla_attention(q, k_nope, k_rope, qk_norm=qk_norm)


def fake_quant_fp8(x: Array) -> Array:
    x = jnp.asarray(x, dtype=jnp.float32)
    scale = jnp.maximum(jnp.max(jnp.abs(x)) / FP8_MAX, 1e-8)
    q = jnp.round(jnp.clip(x / scale, -FP8_MAX, FP8_MAX))
    return q * scale


def fake_quant_nvfp4(x: Array) -> Array:
    """Numerics-only NVFP4 emulation (spec 16: never a speed claim)."""
    x = jnp.asarray(x, dtype=jnp.float32)
    maxv = 6.0  # e2m1-ish
    scale = jnp.maximum(jnp.max(jnp.abs(x)) / maxv, 1e-8)
    q = jnp.round(jnp.clip(x / scale, -maxv, maxv))
    return q * scale


def fp8_linear(x: Array, weight: Array, bias: Array | None = None) -> Array:
    y = jnp.asarray(x) @ fake_quant_fp8(weight).T
    if bias is not None:
        y = y + bias
    return y


def ep_dispatch(tokens: Array, expert_ids: Array, n_experts: int) -> list[Array]:
    """Split tokens onto expert buckets. CPU stand-in for all-to-all."""
    tokens = jnp.asarray(tokens)
    expert_ids = jnp.asarray(expert_ids)
    buckets = []
    flat = tokens.reshape(-1, tokens.shape[-1])
    ids = expert_ids.reshape(-1)
    for e in range(n_experts):
        buckets.append(flat[ids == e])
    return buckets


def precision_probe() -> dict:
    rng = np.random.default_rng(4)
    w = rng.standard_normal((32, 32)).astype(np.float32)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    bf = x @ w
    fp8 = np.asarray(fp8_linear(jnp.asarray(x), jnp.asarray(w.T)))
    rel = float(np.linalg.norm(fp8 - bf) / (np.linalg.norm(bf) + 1e-8))
    nv = np.asarray(fake_quant_nvfp4(jnp.asarray(w)))
    return {
        "fp8_vs_bf16_rel": rel,
        "nvfp4_numerics_ok": bool(np.all(np.isfinite(nv))),
    }


__all__ = [
    "linear_attention",
    "mla_attention",
    "fake_quant_fp8",
    "fake_quant_nvfp4",
    "fp8_linear",
    "ep_dispatch",
    "precision_probe",
]
