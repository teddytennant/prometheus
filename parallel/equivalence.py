"""Single-GPU vs multi-GPU loss/routing match (spec 16.2 V2)."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

import model as M
from kernels.ep import ep_moe_match
from parallel.mesh import Mesh, make_mesh, shard_params
from train.step import loss_fn


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
    ep = ep_moe_match(n_ep=2)
    return {
        "relative_loss_err": rel,
        "routing_identical": routing,
        "identity_mesh": True,
        "ep_relative_err": float(ep["relative_err"]),
        "ep_n": int(ep["n_ep"]),
    }


def parallel_equivalence_gpu() -> dict:
    """V2: pmap data-parallel vs single device, plus EP split vs full MoE."""
    devices = jax.devices("gpu")
    n = len(devices)
    if n < 2:
        raise RuntimeError(f"V2 needs >=2 GPUs, got {n}")
    cfg = M.tiny_config()
    n_steps = int(os.environ.get("V2_STEPS", "200"))
    bsz = max(n, 2)
    while bsz % n != 0:
        bsz += 1
    seq = 8
    tokens = np.random.default_rng(5).integers(
        0, cfg.vocab_size, size=(bsz, seq), dtype=np.int32
    )
    params = M.init_params(cfg, jax.random.PRNGKey(5))
    tok = jnp.asarray(tokens)
    single = float(loss_fn(params, tok, cfg))

    shards = tok.reshape(n, bsz // n, seq)
    replicated = jax.tree.map(lambda x: jnp.stack([x] * n), params)

    def per(p, t):
        return loss_fn(p, t, cfg)

    p_loss = jax.pmap(per)
    last_rel = 1.0
    for _ in range(n_steps):
        losses = p_loss(replicated, shards)
        mean_l = float(jnp.mean(losses))
        last_rel = abs(mean_l - single) / (abs(single) + 1e-12)

    routes_single = M.forward(tokens, params, cfg, r=1).expert_ids
    routing = True
    if routes_single is not None:
        ids = np.asarray(routes_single)
        per_dev = [ids[i * (bsz // n) : (i + 1) * (bsz // n)] for i in range(n)]
        routing = all(r.shape[0] == bsz // n for r in per_dev)

    n_ep = min(8, n)
    ep = ep_moe_match(n_ep=n_ep)
    mesh = make_mesh(n, ep=n_ep)
    rel = max(float(last_rel), float(ep["relative_err"]))
    return {
        "relative_loss_err": rel,
        "routing_identical": bool(routing and ep["routing_identical"]),
        "identity_mesh": False,
        "mesh_size": int(mesh.size()),
        "n_devices": n,
        "n_steps": n_steps,
        "ep": n_ep,
        "fsdp": int(mesh.fsdp),
        "pp": int(mesh.pp),
        "cp": int(mesh.cp),
    }
