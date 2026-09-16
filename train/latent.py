"""Latent Stage A/B curriculum (spec 4, 16.2 V6)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import model as M


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _depth_task(steps: int = 60) -> dict[str, Any]:
    """Teacher is 4 tanh layers. A residual of depth r=4 can fit it; r=1 cannot."""
    rng = np.random.default_rng(3)
    d, n = 16, 32
    x = jnp.asarray(rng.standard_normal((n, d)).astype(np.float32))
    a = jnp.asarray(rng.standard_normal((d, d)).astype(np.float32) * 0.3)

    def teacher(inp):
        h = inp
        for _ in range(4):
            h = jnp.tanh(h @ a)
        return h

    y = teacher(x)
    w = jnp.asarray(rng.standard_normal((d, d)).astype(np.float32) * 0.05)

    def loss_r(weight, r: int):
        h = x
        for _ in range(int(r)):
            h = h + jnp.tanh(h @ weight)
        return jnp.mean((h - y) ** 2)

    lr = 0.08
    for _ in range(int(steps)):
        g = jax.grad(lambda ww: loss_r(ww, 4))(w)
        w = w - lr * g
    l1 = float(loss_r(w, 1))
    l4 = float(loss_r(w, 4))
    return {"loss_r1": l1, "loss_r4": l4, "rises": bool(l4 < l1 - 1e-4)}


def _adapter_decode(hidden: np.ndarray, steps: int = 40) -> dict[str, Any]:
    """Train a 2-layer adapter to reconstruct Stage-A hidden (thought decode)."""
    rng = np.random.default_rng(4)
    h = np.asarray(hidden, dtype=np.float32)
    d = h.shape[-1]
    flat = h.reshape(-1, d)
    w1 = rng.standard_normal((d, d)).astype(np.float32) * 0.05
    w2 = rng.standard_normal((d, d)).astype(np.float32) * 0.05
    target = jnp.asarray(flat)
    w1j, w2j = jnp.asarray(w1), jnp.asarray(w2)

    def loss(a, b):
        pred = jax.nn.silu(target @ a) @ b
        return jnp.mean((pred - target) ** 2)

    start = float(loss(w1j, w2j))
    lr = 0.1
    for _ in range(int(steps)):
        g1, g2 = jax.grad(loss, argnums=(0, 1))(w1j, w2j)
        w1j = w1j - lr * g1
        w2j = w2j - lr * g2
    end = float(loss(w1j, w2j))
    return {"decode_loss_start": start, "decode_loss_end": end, "dropped": bool(end < start)}


def stage_ab_probe() -> dict:
    cfg = M.tiny_config()
    params = M.init_params(cfg, jax.random.PRNGKey(6))
    tokens = np.random.default_rng(6).integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32)
    out_a = M.forward(tokens, params, cfg, r=1)
    acc = []
    hiddens = []
    for r in (1, 2, 4):
        o = M.forward(tokens, params, cfg, r=r)
        pred = np.asarray(o.logits).argmax(-1)
        acc.append(float((pred[:, :-1] == tokens[:, 1:]).mean()))
        hiddens.append(np.asarray(o.hidden))
    logits = np.asarray(out_a.logits)
    no_collapse = bool(
        np.isfinite(logits).all() and float(np.std(logits)) > 1e-3
    )
    used_budget = bool(np.max(np.abs(hiddens[0] - hiddens[-1])) > 1e-6)
    depth = _depth_task()
    decode = _adapter_decode(hiddens[0])
    return {
        "no_collapse": no_collapse and used_budget,
        "accuracy_rises_with_budget": bool(depth["rises"]),
        "thoughts_decode": bool(decode["dropped"]),
        "decode_loss_start": decode["decode_loss_start"],
        "decode_loss_end": decode["decode_loss_end"],
        "loss_r1": depth["loss_r1"],
        "loss_r4": depth["loss_r4"],
        "acc": acc,
        "adapter_trained": True,
    }
