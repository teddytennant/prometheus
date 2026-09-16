"""Device mesh and sharding rules (spec 4 / 5.2). S3 scale is out."""

from __future__ import annotations

from dataclasses import dataclass


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


def shard_params(params: dict, mesh: Mesh) -> dict:
    """1-device identity. Rules are recorded so V2 can compare."""
    validate_mesh(mesh)
    return params


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
