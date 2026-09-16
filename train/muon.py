"""MuonClip: Newton-Schulz orthogonalization and QK-clip (spec 5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class

from train.schedule import TrainConfig

# Keller Jordan Muon Newton-Schulz coefficients.
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


@register_pytree_node_class
@dataclass
class OptState:
    step: int
    momentum: dict[str, Any]

    def tree_flatten(self):
        return (self.step, self.momentum), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(step=children[0], momentum=children[1])


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


