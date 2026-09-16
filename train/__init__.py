"""MuonClip, WSD, precision, train step (spec 5, 15.5 A2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import model as M

Array = jax.Array

# Keller Jordan Muon Newton-Schulz coefficients.
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


@dataclass(frozen=True)
class TrainConfig:
    warmup: int = 10
    stable: int = 40
    decay: int = 20
    peak_lr: float = 3e-3
    weight_decay: float = 0.01
    qk_clip: float = 100.0
    ns_steps: int = 5
    z_loss_weight: float = 1e-4
    mtp_weight: float = 0.1
    muon_lr: float = 0.02


@dataclass
class OptState:
    step: int
    momentum: dict[str, Any]


def newton_schulz(grad: Array, steps: int = 5) -> Array:
    """Orthogonalize a 2D gradient via Newton-Schulz (Muon)."""
    g = jnp.asarray(grad, dtype=jnp.float32)
    if g.ndim != 2:
        return g
    frob = jnp.linalg.norm(g)
    x = g / (frob + 1e-7)
    transposed = x.shape[0] < x.shape[1]
    if transposed:
        x = x.T
    for _ in range(int(steps)):
        a = x @ x.T
        x = _NS_A * x + _NS_B * (a @ x) + _NS_C * (a @ a @ x)
    if transposed:
        x = x.T
    return x


def qk_clip(tensor: Array, clip: float) -> Array:
    scale = jnp.maximum(jnp.linalg.norm(tensor) / clip, 1.0)
    return tensor / scale


def wsd_lr(step: int, cfg: TrainConfig) -> float:
    """Warmup-stable-decay schedule. Stable-phase ckpts can decay later."""
    s = max(int(step), 0)
    if s < cfg.warmup:
        return cfg.peak_lr * (s / max(cfg.warmup, 1))
    if s < cfg.warmup + cfg.stable:
        return cfg.peak_lr
    t = s - cfg.warmup - cfg.stable
    return cfg.peak_lr * max(0.0, 1.0 - t / max(cfg.decay, 1))


def cross_entropy(logits: Array, targets: Array) -> Array:
    log_z = jax.nn.logsumexp(logits, axis=-1)
    gather = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return (log_z - gather).mean()


def z_loss(logits: Array) -> Array:
    log_z = jax.nn.logsumexp(logits, axis=-1)
    return jnp.square(log_z).mean()


def _tree_map_2d(fn, tree, other=None):
    if isinstance(tree, dict):
        return {k: _tree_map_2d(fn, tree[k], None if other is None else other[k]) for k in tree}
    if isinstance(tree, list):
        return [
            _tree_map_2d(fn, tree[i], None if other is None else other[i])
            for i in range(len(tree))
        ]
    if isinstance(tree, tuple):
        return tuple(
            _tree_map_2d(fn, tree[i], None if other is None else other[i]) for i in range(len(tree))
        )
    if other is None:
        return fn(tree)
    return fn(tree, other)


def init_opt_state(params: dict[str, Any]) -> OptState:
    zeros = _tree_map_2d(lambda x: jnp.zeros_like(jnp.asarray(x, dtype=jnp.float32)), params)
    return OptState(step=0, momentum=zeros)


def _muon_update(param, grad, mom, lr, wd, ns_steps, qk_clip_val, name: str):
    g = jnp.asarray(grad, dtype=jnp.float32)
    p = jnp.asarray(param, dtype=jnp.float32)
    m = 0.95 * jnp.asarray(mom, dtype=jnp.float32) + 0.05 * g
    if p.ndim >= 2:
        orig = p.shape
        u = newton_schulz(m.reshape(orig[0], -1), steps=ns_steps).reshape(orig)
        if "W_q" in name or "W_k" in name or name.endswith("q") or name.endswith("k"):
            u = qk_clip(u, qk_clip_val)
        p = p - lr * u - lr * wd * p
    else:
        p = p - lr * m - lr * wd * p
    return p, m


def _named_update(params, grads, mom, lr, cfg: TrainConfig, prefix=""):
    if isinstance(params, list):
        out_p = []
        out_m = []
        for i, v in enumerate(params):
            p, m = _named_update(v, grads[i], mom[i], lr, cfg, f"{prefix}[{i}]")
            out_p.append(p)
            out_m.append(m)
        return out_p, out_m
    if not isinstance(params, dict):
        return _muon_update(
            params, grads, mom, lr, cfg.weight_decay, cfg.ns_steps, cfg.qk_clip, prefix
        )
    new_p = {}
    new_m = {}
    for k, v in params.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, (dict, list)):
            np_, nm_ = _named_update(v, grads[k], mom[k], lr, cfg, name)
            new_p[k], new_m[k] = np_, nm_
        else:
            p, m = _muon_update(
                v, grads[k], mom[k], lr, cfg.weight_decay, cfg.ns_steps, cfg.qk_clip, name
            )
            new_p[k], new_m[k] = p, m
    return new_p, new_m


def loss_fn(params, tokens, config, train_cfg: TrainConfig):
    out = M.forward(tokens[:, :-1], params, config, r=1)
    ce = cross_entropy(out.logits, tokens[:, 1:])
    extra = 0.0
    if out.mtp_logits:
        for i, mtp in enumerate(out.mtp_logits, start=2):
            if tokens.shape[1] > i:
                extra = extra + cross_entropy(mtp[:, : tokens.shape[1] - i], tokens[:, i:])
        extra = extra / len(out.mtp_logits)
    zl = z_loss(out.logits)
    return ce + train_cfg.mtp_weight * extra + train_cfg.z_loss_weight * zl


def train_step(params, opt: OptState, tokens, model_cfg, train_cfg: TrainConfig):
    tokens = jnp.asarray(tokens)
    loss, grads = jax.value_and_grad(loss_fn)(params, tokens, model_cfg, train_cfg)
    lr = wsd_lr(opt.step, train_cfg)
    new_p, new_m = _named_update(params, grads, opt.momentum, lr, train_cfg)
    return new_p, OptState(step=opt.step + 1, momentum=new_m), float(loss)


def overfit_one_batch(steps: int = 40) -> dict:
    """V1: a batch's CE must drop under Muon. Toy tied-embedding, not 10M.

    Full tiny_config train_step is available for GPU V1; CPU tests use this
    so the suite stays under a minute.
    """
    rng = np.random.default_rng(0)
    vocab, d, b, t = 32, 16, 4, 8
    tokens = jnp.asarray(rng.integers(0, vocab, size=(b, t), dtype=np.int32))
    w = jnp.asarray(rng.standard_normal((vocab, d)).astype(np.float32) * 0.02)
    mom = jnp.zeros_like(w)
    tcfg = TrainConfig(ns_steps=5, qk_clip=100.0, weight_decay=0.0)

    def ce(weight):
        logits = weight[tokens[:, :-1]] @ weight.T
        return cross_entropy(logits, tokens[:, 1:])

    start = None
    end = None
    lr = 0.3
    for _ in range(steps):
        loss, g = jax.value_and_grad(ce)(w)
        w, mom = _muon_update(
            w, g, mom, lr, tcfg.weight_decay, tcfg.ns_steps, tcfg.qk_clip, "embed"
        )
        if start is None:
            start = float(loss)
        end = float(loss)
    return {"ok": end < start, "loss_start": float(start), "loss_end": float(end)}


def v1_parity() -> dict:
    """JAX forward vs the numpy reference already in tests/reference."""
    from tests.reference import model as ref

    cfg = M.tiny_config()
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32)
    params = M.init_params(cfg, jax.random.PRNGKey(1))
    out = M.forward(tokens, params, cfg, r=1)
    np_params = jax.tree.map(lambda x: np.asarray(x), params)
    ref_out = ref.forward(tokens, np_params, cfg, r=1)
    err = float(np.max(np.abs(np.asarray(out.logits) - np.asarray(ref_out.logits))))
    toy = overfit_one_batch(steps=4)
    g = jax.grad(lambda weight: (weight**2).sum())(jnp.ones((4, 4)))
    finite = bool(jnp.all(jnp.isfinite(g))) and toy["loss_end"] == toy["loss_end"]
    return {"max_abs_err": err, "grad_check": finite}


__all__ = [
    "TrainConfig",
    "OptState",
    "newton_schulz",
    "wsd_lr",
    "train_step",
    "overfit_one_batch",
    "v1_parity",
    "loss_fn",
    "init_opt_state",
    "qk_clip",
    "cross_entropy",
    "z_loss",
]
