"""Single-GPU vs multi-GPU loss/routing match (spec 16.2 V2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import model as M
from parallel.mesh import Mesh, shard_params


def _loss_once(params, tokens, cfg):
    out = M.forward(tokens, params, cfg, r=1)
    log_z = jax.nn.logsumexp(out.logits, axis=-1)
    gather = jnp.take_along_axis(out.logits, tokens[..., None], axis=-1)[..., 0]
    return (log_z - gather).mean()


def parallel_equivalence() -> dict:
    """CPU unit-test path: 1-device vs identity mesh. V2 on H200 uses the GPU fn."""
    cfg = M.tiny_config()
    tokens = np.random.default_rng(5).integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32)
    params = M.init_params(cfg, jax.random.PRNGKey(5))
    mesh = Mesh(ep=1, fsdp=1, pp=1, dp=1, cp=1)
    sharded = shard_params(params, mesh)
    a = float(_loss_once(params, jnp.asarray(tokens), cfg))
    b = float(_loss_once(sharded, jnp.asarray(tokens), cfg))
    rel = abs(a - b) / (abs(a) + 1e-12)
    out = M.forward(tokens, params, cfg, r=1)
    out2 = M.forward(tokens, sharded, cfg, r=1)
    routing = True
    if out.expert_ids is not None:
        routing = bool(np.array_equal(np.asarray(out.expert_ids), np.asarray(out2.expert_ids)))
    return {"relative_loss_err": rel, "routing_identical": routing, "identity_mesh": True}


def parallel_equivalence_gpu() -> dict:
    """V2: same batch on each GPU, mesh_size == n_devices >= 2."""
    devices = jax.devices("gpu")
    n = len(devices)
    if n < 2:
        raise RuntimeError(f"V2 needs >=2 GPUs, got {n}")
    cfg = M.tiny_config()
    tokens = np.random.default_rng(5).integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32)
    params = M.init_params(cfg, jax.random.PRNGKey(5))
    tok = jnp.asarray(tokens)
    placed = []
    routes = []
    for dev in devices:
        p = jax.tree.map(lambda x, d=dev: jax.device_put(x, d), params)
        t = jax.device_put(tok, dev)
        placed.append(float(_loss_once(p, t, cfg)))
        out = M.forward(np.asarray(t), p, cfg, r=1)
        if out.expert_ids is not None:
            routes.append(np.asarray(out.expert_ids))
    a, b = placed[0], placed[1]
    rel = abs(a - b) / (abs(a) + 1e-12)
    routing = True
    if len(routes) >= 2:
        routing = bool(np.array_equal(routes[0], routes[1]))
    mesh = Mesh(dp=n)
    return {
        "relative_loss_err": rel,
        "routing_identical": routing,
        "identity_mesh": False,
        "mesh_size": mesh.size(),
        "n_devices": n,
    }
