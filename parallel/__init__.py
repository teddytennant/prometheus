"""Mesh, FSDP/EP/PP/CP sharding rules (spec 5.2, 15.5 A4). S3 scale is out."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import model as M


@dataclass(frozen=True)
class Mesh:
    ep: int = 1
    fsdp: int = 1
    pp: int = 1
    dp: int = 1
    cp: int = 1

    def size(self) -> int:
        return self.ep * self.fsdp * self.pp * self.dp * self.cp


def validate_mesh(mesh: Mesh) -> None:
    if mesh.size() < 1:
        raise ValueError("empty mesh")
    # S3: EP=72 and cross-rack routing are cluster-conditional.
    if mesh.ep > 8:
        raise ValueError("EP>8 is S3 (cluster-conditional)")


def shard_params(params: dict, mesh: Mesh) -> dict:
    """1-device identity. Rules are recorded so V2 can compare."""
    validate_mesh(mesh)
    return params


def _loss_once(params, tokens, cfg):
    out = M.forward(tokens, params, cfg, r=1)
    log_z = jax.nn.logsumexp(out.logits, axis=-1)
    gather = jnp.take_along_axis(out.logits, tokens[..., None], axis=-1)[..., 0]
    return (log_z - gather).mean()


def parallel_equivalence() -> dict:
    """V2 stand-in: 1-device vs 'sharded' (identity mesh) loss and routing."""
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
    return {"relative_loss_err": rel, "routing_identical": routing}


__all__ = ["Mesh", "validate_mesh", "shard_params", "parallel_equivalence"]
