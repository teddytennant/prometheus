"""Device mesh, sharding, FLOP-balanced PP, context-parallel ring (spec 5.2)."""

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

    def __post_init__(self) -> None:
        if self.ep > 8:
            raise ValueError("EP>8 is S3 (cluster-conditional)")


MeshSpec = Mesh


def validate_mesh(mesh: Mesh) -> None:
    if mesh.size() < 1:
        raise ValueError("empty mesh")
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


def shard_params(params: dict, mesh: Mesh, rank: int | None = None) -> dict:
    """Slice axis-0 of every leaf across FSDP ranks.

    rank=None stacks every rank on axis 0 so unshard_params can concat.
    """
    validate_mesh(mesh)
    if rank is None:
        shards = [shard_params(params, mesh, r) for r in range(max(mesh.fsdp, 1))]

        def stack(objs: list[Any]) -> Any:
            if isinstance(objs[0], dict):
                return {k: stack([o[k] for o in objs]) for k in objs[0]}
            if isinstance(objs[0], (list, tuple)):
                return type(objs[0])(stack([o[i] for o in objs]) for i in range(len(objs[0])))
            return np.stack([np.asarray(o) for o in objs], axis=0)

        return stack(shards)

    def walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(walk(v) for v in obj)
        return _shard_array(obj, mesh.fsdp, rank)

    return walk(params)


def unshard_params(
    shards: list[dict] | dict, mesh: Mesh | None = None, orig_len: int | None = None
) -> dict:
    """Concatenate FSDP shards along axis 0."""

    def concat_leaves(objs: list[Any]) -> Any:
        if isinstance(objs[0], dict):
            return {k: concat_leaves([o[k] for o in objs]) for k in objs[0]}
        if isinstance(objs[0], (list, tuple)):
            return type(objs[0])(concat_leaves([o[i] for o in objs]) for i in range(len(objs[0])))
        arrs = [np.asarray(o) for o in objs]
        if arrs[0].ndim == 0:
            return arrs[0]
        out = np.concatenate(arrs, axis=0)
        if orig_len is not None:
            out = out[:orig_len]
        return out

    if isinstance(shards, dict):

        def unstack(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: unstack(v) for k, v in obj.items()}
            arr = np.asarray(obj)
            if arr.ndim == 0:
                return arr
            parts = [arr[i] for i in range(arr.shape[0])]
            return concat_leaves(parts)

        return unstack(shards)
    return concat_leaves(shards)


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


def pipeline_stages(
    n_layers: int,
    pp: int,
    *,
    core_start: int = 0,
    core_len: int = 8,
    r: float = 3.0,
) -> list[list[int]]:
    """Contiguous PP split. Recurrent core layers cost `r` so they get extra stages."""
    if pp < 1:
        raise ValueError("pp must be >= 1")
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    pp = min(pp, n_layers)
    core_end = min(n_layers, core_start + max(0, core_len))
    weights = [float(r) if core_start <= i < core_end else 1.0 for i in range(n_layers)]
    total = sum(weights)
    targets = [(k + 1) * total / pp for k in range(pp)]
    stages: list[list[int]] = [[] for _ in range(pp)]
    acc = 0.0
    stage = 0
    for i, w in enumerate(weights):
        if stage < pp - 1 and acc + w / 2.0 >= targets[stage] and stages[stage]:
            stage += 1
        stages[stage].append(i)
        acc += w
    return stages


def context_parallel_ring(x: Any, cp: int) -> dict[str, Any]:
    """Split sequence across CP ranks; ring all-gather reconstructs the full context."""
    if cp < 1:
        raise ValueError("cp must be >= 1")
    arr = np.asarray(x)
    axis = 1 if arr.ndim >= 2 else 0
    seq = int(arr.shape[axis])
    pad = (cp - seq % cp) % cp
    if pad:
        pads = [(0, 0)] * arr.ndim
        pads[axis] = (0, pad)
        arr = np.pad(arr, pads)
    chunks = np.split(arr, cp, axis=axis)
    assembled: list[np.ndarray] = []
    for rank in range(cp):
        buf: list[np.ndarray | None] = [None] * cp
        buf[rank] = chunks[rank]
        src = rank
        for _ in range(cp - 1):
            src = (src - 1) % cp
            buf[src] = chunks[src]
        assembled.append(np.concatenate(buf, axis=axis))
    full = assembled[0]
    ranks_agree = all(np.allclose(a, full) for a in assembled)
    if pad:
        sl = [slice(None)] * full.ndim
        sl[axis] = slice(0, seq)
        full = full[tuple(sl)]
    return {
        "full": full,
        "ranks_agree": bool(ranks_agree),
        "cp": cp,
        "n_ring_steps": max(0, cp - 1),
    }
