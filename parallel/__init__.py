"""Parallelism package (spec 4): mesh, sharding, equivalence."""

from __future__ import annotations

from parallel.equivalence import parallel_equivalence, parallel_equivalence_gpu
from parallel.mesh import Mesh, MeshSpec, make_mesh, shard_for, shard_params, validate_mesh

__all__ = [
    "Mesh",
    "MeshSpec",
    "make_mesh",
    "shard_for",
    "shard_params",
    "validate_mesh",
    "parallel_equivalence",
    "parallel_equivalence_gpu",
]
