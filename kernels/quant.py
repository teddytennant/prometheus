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


def _quant_tree(params):
    def q(x):
        a = jnp.asarray(x)
        if a.ndim >= 2:
            return fake_quant_fp8(a)
        return a

    return jax.tree.map(q, params)


def two_precision_train(steps: int | None = None) -> dict[str, Any]:
    """Train the flagship-shaped model; each report compares BF16 vs FP8-quantized weights."""
    import model as M
    from train.muon import init_opt_state
    from train.schedule import TrainConfig
    from train.step import loss_fn, train_step

    gpu = jax.default_backend() == "gpu"
    if steps is None:
        steps = int(os.environ.get("V3_STEPS", "2000" if gpu else "4"))
    if gpu:
        from train.rung0 import rung0_gpu_config

        cfg = rung0_gpu_config()
    else:
        cfg = M.cpu_config()
    rng = np.random.default_rng(4)
    tokens = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32))
    params = M.init_params(cfg, jax.random.PRNGKey(4))
    opt = init_opt_state(params)
    tcfg = TrainConfig(
        warmup=0, stable=max(int(steps), 8), decay=0, peak_lr=0.02,
        ns_steps=3, qk_clip=100.0, weight_decay=0.0,
    )
    last_bf = 0.0
    last_fp = 0.0
    last_rel = 1.0
    for _ in range(int(steps)):
        params, opt, loss_bf = train_step(params, opt, tokens, cfg, tcfg)
        last_bf = float(loss_bf)
    last_fp = float(loss_fn(_quant_tree(params), tokens, cfg))
    last_rel = abs(last_fp - last_bf) / (abs(last_bf) + 1e-8)
    nv = np.asarray(fake_quant_nvfp4(params["unembed"]))
    return {
        "fp8_vs_bf16_rel": float(last_rel),
        "nvfp4_numerics_ok": bool(np.all(np.isfinite(nv))),
        "n_steps": int(steps),
        "n_params": int(M.param_count(params)),
        "loss_bf16": last_bf,
        "loss_fp8": last_fp,
    }


def precision_probe() -> dict[str, Any]:
    return two_precision_train()
