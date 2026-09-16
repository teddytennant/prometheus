"""Per-block FP8, NVFP4 nibble codecs, two-precision train (spec 5.1)."""

from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

FP8_E4M3_MAX = 448.0
FP8_MAX = FP8_E4M3_MAX
NVFP4_MAX = 6.0
NVFP4_BLOCK = 16
FP8_BLOCK = 32
NVFP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _pad_last(x: Array, block: int) -> tuple[Array, int]:
    last = int(x.shape[-1])
    pad = (block - last % block) % block
    if pad:
        pads = [(0, 0)] * (x.ndim - 1) + [(0, pad)]
        x = jnp.pad(x, pads)
    return x, last


def fp8_block_scales(x: Any, block_size: int = FP8_BLOCK) -> Array:
    x = jnp.asarray(x, dtype=jnp.float32)
    x, _last = _pad_last(x, block_size)
    blocked = x.reshape(*x.shape[:-1], -1, block_size)
    amax = jnp.max(jnp.abs(blocked), axis=-1)
    return jnp.maximum(amax / jnp.float32(FP8_E4M3_MAX), jnp.float32(1e-8))


def fake_quant_fp8(x: Any, block_size: int = FP8_BLOCK) -> Array:
    """E4M3-range fake quant with independent scale per last-axis block."""
    x = jnp.asarray(x, dtype=jnp.float32)
    x, last = _pad_last(x, block_size)
    blocked = x.reshape(*x.shape[:-1], -1, block_size)
    scale = jnp.maximum(
        jnp.max(jnp.abs(blocked), axis=-1, keepdims=True) / jnp.float32(FP8_E4M3_MAX),
        jnp.float32(1e-8),
    )
    q = jnp.round(jnp.clip(blocked / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX))
    y = (q * scale).reshape(*x.shape[:-1], x.shape[-1])
    return y[..., :last]


def fake_quant_nvfp4(x: Any, block_size: int = NVFP4_BLOCK) -> Array:
    dequant, _, _ = _nvfp4_block_dequant(np.asarray(x), block_size=block_size)
    return jnp.asarray(dequant, dtype=jnp.float32)


def _nvfp4_block_dequant(
    x: np.ndarray, block_size: int = NVFP4_BLOCK
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    last = x.shape[-1]
    pad = (block_size - last % block_size) % block_size
    if pad:
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
    blocked = x.reshape(*x.shape[:-1], -1, block_size)
    amax = np.max(np.abs(blocked), axis=-1, keepdims=True)
    scale = np.maximum(amax / NVFP4_MAX, 1e-8)
    y = blocked / scale
    levels = np.array(NVFP4_LEVELS, dtype=np.float32)
    idx = np.abs(np.abs(y)[..., None] - levels).argmin(axis=-1)
    sign = np.where(np.sign(y) == 0, 1.0, np.sign(y))
    signed = levels[idx] * sign
    codes = idx.astype(np.uint8)
    codes = np.where(signed < 0, codes + 8, codes).astype(np.uint8)
    dequant = (signed * scale).reshape(*x.shape[:-1], x.shape[-1])[..., :last]
    codes = codes.reshape(*x.shape[:-1], x.shape[-1])[..., :last]
    return dequant.astype(np.float32), codes, scale.squeeze(-1)


def nvfp4_pack(codes: Any) -> np.ndarray:
    c = np.asarray(codes, dtype=np.uint8).reshape(-1)
    if c.size % 2:
        c = np.concatenate([c, np.zeros(1, dtype=np.uint8)])
    return ((c[0::2] & 0xF) << 4) | (c[1::2] & 0xF)


def nvfp4_unpack(packed: Any, n: int) -> np.ndarray:
    p = np.asarray(packed, dtype=np.uint8).reshape(-1)
    hi = (p >> 4) & 0xF
    lo = p & 0xF
    out = np.empty(p.size * 2, dtype=np.uint8)
    out[0::2] = hi
    out[1::2] = lo
    return out[:n]


def nvfp4_roundtrip(x: Any, block_size: int = NVFP4_BLOCK) -> Array:
    x_np = np.asarray(x, dtype=np.float32)
    dequant, codes, _ = _nvfp4_block_dequant(x_np, block_size=block_size)
    packed = nvfp4_pack(codes)
    restored = nvfp4_unpack(packed, codes.size).reshape(codes.shape)
    if not np.array_equal(restored, codes):
        raise RuntimeError("NVFP4 nibble pack/unpack mismatch")
    return jnp.asarray(dequant, dtype=jnp.float32)


def fp8_linear(x: Array, weight: Array, bias: Array | None = None) -> Array:
    y = jnp.asarray(x) @ fake_quant_fp8(weight).T
    if bias is not None:
        y = y + bias
    return y


def fp8_cast(x: Array) -> Array:
    return fake_quant_fp8(x)


def _quant_tree(params):
    """FP8 on linear weights; leave router / norm / 1-D scales in BF16."""

    def q(path: tuple, x):
        name = ".".join(str(p) for p in path)
        a = jnp.asarray(x)
        if a.ndim < 2 or "router" in name or "norm" in name:
            return a
        return fake_quant_fp8(a)

    return jax.tree.map_with_path(q, params)


def two_precision_train(
    steps: int | None = None, *, n_steps: int | None = None, seed: int = 4
) -> dict[str, Any]:
    """Train the flagship-shaped model; report BF16 vs FP8-linear (BF16 router/norm)."""
    import model as M
    from train.muon import init_opt_state
    from train.schedule import TrainConfig
    from train.step import loss_fn, train_step

    if n_steps is not None:
        steps = n_steps
    gpu = jax.default_backend() == "gpu"
    if steps is None:
        steps = int(os.environ.get("V3_STEPS", "2000" if gpu else "4"))
    if gpu:
        from train.rung0 import rung0_gpu_config

        cfg = rung0_gpu_config()
    else:
        cfg = M.cpu_config()
    rng = np.random.default_rng(seed)
    tokens = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32))
    params = M.init_params(cfg, jax.random.PRNGKey(seed))
    opt = init_opt_state(params)
    tcfg = TrainConfig(
        warmup=0,
        stable=max(int(steps), 8),
        decay=0,
        peak_lr=0.02,
        ns_steps=3,
        qk_clip=100.0,
        weight_decay=0.0,
    )
    last_bf = 0.0
    for _ in range(int(steps)):
        params, opt, loss_bf = train_step(params, opt, tokens, cfg, tcfg)
        last_bf = float(loss_bf)
    last_fp = float(loss_fn(_quant_tree(params), tokens, cfg))
    last_rel = abs(last_fp - last_bf) / (abs(last_bf) + 1e-8)
    nv = np.asarray(nvfp4_roundtrip(params["unembed"]))
    return {
        "fp8_vs_bf16_rel": float(last_rel),
        "nvfp4_numerics_ok": bool(np.all(np.isfinite(nv))),
        "n_steps": int(steps),
        "n_params": int(M.param_count(params)),
        "loss_bf16": last_bf,
        "loss_fp8": last_fp,
        "fp8_linears": True,
        "bf16_router_norm": True,
    }


def precision_probe(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((8, 8)).astype(np.float32)
    q8 = np.asarray(fake_quant_fp8(x))
    q4 = np.asarray(nvfp4_roundtrip(x))
    rel8 = float(np.linalg.norm(q8 - x) / (np.linalg.norm(x) + 1e-8))
    rel4 = float(np.linalg.norm(q4 - x) / (np.linalg.norm(x) + 1e-8))
    return {
        "fp8_vs_bf16_rel": rel8,
        "nvfp4_vs_bf16_rel": rel4,
        "nvfp4_numerics_ok": bool(np.isfinite(rel4) and rel4 < 0.5),
    }
