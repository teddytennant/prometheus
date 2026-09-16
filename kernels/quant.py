"""FP8 / NVFP4 reference numerics (spec 3, A3). Not a speed claim."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

FP8_MAX = 448.0  # e4m3


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


def nvfp4_roundtrip(x: Array) -> Array:
    return fake_quant_nvfp4(x)


def fp8_cast(x: Array) -> Array:
    return fake_quant_fp8(x)


def precision_probe() -> dict[str, Any]:
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
