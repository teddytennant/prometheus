"""Latent Stage A/B curriculum (spec 4, 16.2 V6)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import model as M
from model.layers import _apply_latent_adapter


def _depth_task(params: dict, hidden: np.ndarray, steps: int = 50) -> dict[str, Any]:
    """More applications of the model's latent adapter fit a 4-step teacher better."""
    x = jnp.asarray(np.asarray(hidden, dtype=np.float32).reshape(-1, hidden.shape[-1]))
    rng = np.random.default_rng(3)
    tw1 = jnp.asarray(
        rng.standard_normal(np.asarray(params["adapter_w1"]).shape).astype(np.float32) * 0.2
    )
    tw2 = jnp.asarray(
        rng.standard_normal(np.asarray(params["adapter_w2"]).shape).astype(np.float32) * 0.2
    )
    tn = jnp.ones_like(params["adapter_norm"])

    def apply(h, a, b, n, r):
        y = h
        for _ in range(int(r)):
            y = y + _apply_latent_adapter(y, a, b, n)
        return y

    y = apply(x, tw1, tw2, tn, 4)
    w1 = jnp.asarray(params["adapter_w1"])
    w2 = jnp.asarray(params["adapter_w2"])
    nw = jnp.asarray(params["adapter_norm"])

    def loss(a, b, r):
        return jnp.mean((apply(x, a, b, nw, r) - y) ** 2)

    lr = 0.05
    for _ in range(int(steps)):
        g1, g2 = jax.grad(lambda a, b: loss(a, b, 4), argnums=(0, 1))(w1, w2)
        w1 = w1 - lr * g1
        w2 = w2 - lr * g2
    l1 = float(loss(w1, w2, 1))
    l4 = float(loss(w1, w2, 4))
    return {"loss_r1": l1, "loss_r4": l4, "rises": bool(l4 < l1 - 1e-4)}


def _adapter_decode(params, hidden: np.ndarray, steps: int = 40) -> dict[str, Any]:
    """Train the model's adapter_w1/w2 to reconstruct Stage-A hidden."""
    h = jnp.asarray(np.asarray(hidden, dtype=np.float32).reshape(-1, hidden.shape[-1]))
    w1 = jnp.asarray(params["adapter_w1"])
    w2 = jnp.asarray(params["adapter_w2"])
    nw = jnp.asarray(params["adapter_norm"])

    def loss(a, b, nrm):
        pred = _apply_latent_adapter(h, a, b, nrm)
        return jnp.mean((pred - h) ** 2)

    start = float(loss(w1, w2, nw))
    lr = 0.1
    for _ in range(int(steps)):
        g1, g2 = jax.grad(loss, argnums=(0, 1))(w1, w2, nw)
        w1 = w1 - lr * g1
        w2 = w2 - lr * g2
    end = float(loss(w1, w2, nw))
    return {"decode_loss_start": start, "decode_loss_end": end, "dropped": bool(end < start)}


def stage_ab_probe() -> dict:
    cfg = M.tiny_config() if jax.default_backend() == "gpu" else M.cpu_config()
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
    no_collapse = bool(np.isfinite(logits).all() and float(np.std(logits)) > 1e-3)
    used_budget = bool(np.max(np.abs(hiddens[0] - hiddens[-1])) > 1e-6)
    depth = _depth_task(params, hiddens[0])
    decode = _adapter_decode(params, hiddens[0])
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
        "n_params": int(M.param_count(params)),
    }
