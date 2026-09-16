"""V1 loops: flagship-shape parity, real grad check, one-batch overfit."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

import model as M
from train.muon import init_opt_state
from train.schedule import TrainConfig
from train.step import loss_fn, train_step


def _flagship_shape(cfg: M.ModelConfig) -> bool:
    return (
        cfg.n_layers >= 4
        and cfg.n_routed_experts >= 2
        and cfg.mtp_heads >= 1
        and cfg.recurrence_max >= 2
        and cfg.adapter_hidden is not None
        and any(M.attention_kind(i, cfg).name == "MLA" for i in range(cfg.n_layers))
        and any(M.attention_kind(i, cfg).name == "LINEAR" for i in range(cfg.n_layers))
        and any(M.ffn_kind(i, cfg).name == "MOE" for i in range(cfg.n_layers))
    )


def _finite_diff_grad_check(cfg=None, params=None, tokens=None) -> bool:
    """jax.grad vs central difference of CE wrt unembed on the flagship-shaped model."""
    if cfg is None:
        cfg = M.tiny_config()
        params = M.init_params(cfg, jax.random.PRNGKey(0))
        rng = np.random.default_rng(0)
        tokens = rng.integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32)
    tokens = jnp.asarray(tokens)
    u0 = params["unembed"]

    def ce(unembed):
        p = dict(params)
        p["unembed"] = unembed
        return loss_fn(p, tokens, cfg)

    g = jax.grad(ce)(u0)
    if not bool(jnp.isfinite(g).all()):
        return False
    eps = jnp.float32(1e-3)
    e = jnp.zeros_like(u0).at[0, 0].set(eps)
    fd = (ce(u0 + e) - ce(u0 - e)) / (jnp.float32(2.0) * eps)
    return bool(jnp.isfinite(fd)) and abs(float(g[0, 0]) - float(fd)) < 5e-2


def overfit_one_batch(steps: int = 40) -> dict:
    """Muon train_step on the flagship-shaped cpu_config until CE drops."""
    cfg = M.cpu_config()
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(4, 8), dtype=np.int32))
    params = M.init_params(cfg, jax.random.PRNGKey(0))
    tcfg = TrainConfig(
        warmup=0,
        stable=max(int(steps), 8),
        decay=0,
        peak_lr=0.05,
        ns_steps=3,
        qk_clip=100.0,
        weight_decay=0.0,
    )
    opt = init_opt_state(params)
    start = None
    end = None
    for _ in range(int(steps)):
        params, opt, loss = train_step(params, opt, tokens, cfg, tcfg)
        if start is None:
            start = float(loss)
        end = float(loss)
    return {
        "ok": end < start,
        "loss_start": float(start),
        "loss_end": float(end),
        "n_params": int(M.param_count(params)),
        "toy_embed": False,
    }


def v1_parity() -> dict:
    """JAX tiny_config vs the independent numpy reference, plus a real grad check."""
    from tests.reference import model as ref

    cfg = M.tiny_config()
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32)
    params = M.init_params(cfg, jax.random.PRNGKey(1))
    out = M.forward(tokens, params, cfg, r=1)
    np_params = jax.tree.map(lambda x: np.asarray(x), params)
    ref_out = ref.forward(tokens, np_params, cfg, r=1)
    err = float(np.max(np.abs(np.asarray(out.logits) - np.asarray(ref_out.logits))))
    return {
        "max_abs_err": err,
        "grad_check": _finite_diff_grad_check(cfg, params, tokens),
        "n_params": int(M.param_count(params)),
        "flagship_shape": _flagship_shape(cfg),
    }


def v1_gpu() -> dict:
    """Spec 16.2 V1: ~10M flagship-shape model, JAX vs numpy, grad check, overfit."""
    parity = v1_parity()
    cfg = M.tiny_config()
    params = M.init_params(cfg, jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32))
    opt = init_opt_state(params)
    tcfg = TrainConfig(
        warmup=0, stable=100, decay=0, peak_lr=0.02, ns_steps=3, qk_clip=100.0, weight_decay=0.0
    )
    steps = int(os.environ.get("V1_STEPS", "12"))
    start = None
    end = None
    for _ in range(steps):
        params, opt, loss = train_step(params, opt, tokens, cfg, tcfg)
        if start is None:
            start = float(loss)
        end = float(loss)
    return {
        "logit_max_abs_err": float(parity["max_abs_err"]),
        "grad_check": bool(parity["grad_check"]),
        "overfit_one_batch": bool(end < start),
        "loss_start": float(start),
        "loss_end": float(end),
        "n_params": int(parity["n_params"]),
        "flagship_shape": bool(parity["flagship_shape"]),
        "toy_embed": False,
        "v1_steps": steps,
    }
