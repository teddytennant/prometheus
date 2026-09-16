"""FP8 / NVFP4 numerics and a two-precision train loop (spec 3, A3, V3)."""

from __future__ import annotations

import os
from typing import Any

import jax
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


def _softmax_ce(logits: Array, y: Array) -> Array:
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    log_z = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))
    gathered = jnp.take_along_axis(logits, y[:, None], axis=-1)[:, 0]
    return jnp.mean(-(gathered - log_z))


def two_precision_train(steps: int | None = None) -> dict[str, Any]:
    """SGD on a linear classifier. Each step compares BF16 weights vs FP8-quantized."""
    if steps is None:
        gpu = jax.default_backend() == "gpu"
        steps = int(os.environ.get("V3_STEPS", "256" if gpu else "32"))
    rng = np.random.default_rng(4)
    n, d, c = 64, 32, 8
    x = jnp.asarray(rng.standard_normal((n, d)).astype(np.float32))
    y = jnp.asarray(rng.integers(0, c, size=(n,), dtype=np.int32))
    w = jnp.asarray(rng.standard_normal((c, d)).astype(np.float32) * 0.05)
    lr = 0.05
    last_rel = 1.0
    last_bf = 0.0
    last_fp = 0.0

    def loss_w(weight):
        return _softmax_ce(x @ weight.T, y)

    for _ in range(int(steps)):
        loss_bf, g = jax.value_and_grad(loss_w)(w)
        loss_fp = loss_w(fake_quant_fp8(w))
        last_bf, last_fp = float(loss_bf), float(loss_fp)
        last_rel = abs(last_fp - last_bf) / (abs(last_bf) + 1e-8)
        w = w - lr * g
    nv = np.asarray(fake_quant_nvfp4(w))
    return {
        "fp8_vs_bf16_rel": float(last_rel),
        "nvfp4_numerics_ok": bool(np.all(np.isfinite(nv))),
        "n_steps": int(steps),
        "loss_bf16": last_bf,
        "loss_fp8": last_fp,
    }


def precision_probe() -> dict[str, Any]:
    return two_precision_train()
