"""Device mesh and sharding rules (spec 4 / 5.2). S3 scale is out."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Mesh:
    ep: int = 1
    fsdp: int = 1
    pp: int = 1
    dp: int = 1
    cp: int = 1

    def size(self) -> int:
        return self.ep * self.fsdp * self.pp * self.dp * self.cp


MeshSpec = Mesh


def validate_mesh(mesh: Mesh) -> None:
    if mesh.size() < 1:
        raise ValueError("empty mesh")
    # S3: EP=72 and cross-rack routing are cluster-conditional.
    if mesh.ep > 8:
        raise ValueError("EP>8 is S3 (cluster-conditional)")


def make_mesh(n_devices: int, ep: int = 1, fsdp: int = 1, cp: int = 1, pp: int = 1) -> Mesh:
    if n_devices < 1:
        raise ValueError("n_devices must be >= 1")
    product = ep * fsdp * cp * pp
    if n_devices % product != 0:
        raise ValueError(f"{n_devices} devices not divisible by ep*fsdp*cp*pp={product}")
    dp = n_devices // product
    mesh = Mesh(dp=dp, ep=ep, fsdp=fsdp, cp=cp, pp=pp)
    validate_mesh(mesh)
    return mesh


def _shard_array(x: Any, n: int, rank: int) -> Any:
    arr = np.asarray(x)
    if arr.ndim == 0 or n <= 1:
        return arr
    size = arr.shape[0]
    chunk = (size + n - 1) // n
    start = rank * chunk
    end = min(start + chunk, size)
    return arr[start:end]


def shard_params(params: dict, mesh: Mesh, rank: int = 0) -> dict:
    """Slice axis-0 of every leaf across FSDP ranks. Identity when fsdp==1."""
    validate_mesh(mesh)

    def walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(walk(v) for v in obj)
        return _shard_array(obj, mesh.fsdp, rank)

    return walk(params)


def unshard_params(shards: list[dict]) -> dict:
    """Concatenate FSDP shards along axis 0."""

    def walk(objs: list[Any]) -> Any:
        if isinstance(objs[0], dict):
            return {k: walk([o[k] for o in objs]) for k in objs[0]}
        if isinstance(objs[0], (list, tuple)):
            return type(objs[0])(walk([o[i] for o in objs]) for i in range(len(objs[0])))
        return np.concatenate([np.asarray(o) for o in objs], axis=0)

    return walk(shards)


def shard_for(kind: str, mesh: Mesh) -> dict[str, int]:
    if kind == "moe_expert":
        return {"ep": mesh.ep, "fsdp": mesh.fsdp}
    if kind == "dense":
        return {"fsdp": mesh.fsdp, "dp": mesh.dp}
    if kind == "embed":
        return {"fsdp": mesh.fsdp}
    if kind == "seq":
        return {"cp": mesh.cp}
    raise ValueError(f"unknown shard kind {kind}")
