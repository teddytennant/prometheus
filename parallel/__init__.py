"""Parallelism package (spec 5.2): mesh, sharding, equivalence."""

from __future__ import annotations

from parallel.equivalence import parallel_equivalence, parallel_equivalence_gpu
from parallel.mesh import (
    Mesh,
    MeshSpec,
    context_parallel_ring,
    make_mesh,
    pipeline_stages,
    shard_for,
    shard_params,
    unshard_params,
    validate_mesh,
)

__all__ = [
    "Mesh",
    "MeshSpec",
    "make_mesh",
    "shard_for",
    "shard_params",
    "unshard_params",
    "validate_mesh",
    "pipeline_stages",
    "context_parallel_ring",
    "parallel_equivalence",
    "parallel_equivalence_gpu",
]
