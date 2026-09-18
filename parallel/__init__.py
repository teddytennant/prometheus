"""SPMD mesh, FSDP, EP, PP, CP (spec 5.1, 5.2, 15.5 A4).

One replica is 12 racks of 72 H200s (EP=72, PP=12). FSDP shards attention,
dense, and shared-expert params over the rack. Routed experts stay unsharded
on the GPU that owns them (~7 of 512 per GPU). Context parallel is off in
pretrain (CP=1) and on in mid-training.

This package is the CPU-facing mesh contract. It does not launch multi-host
jobs. Device IDs are integers in ``range(n_devices)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

Array = Any


class Axis(StrEnum):
    """Named mesh axes. FSDP shares the EP axis (the rack)."""

    DP = "dp"
    FSDP = "fsdp"
    EP = "ep"
    PP = "pp"
    CP = "cp"


class ParamKind(StrEnum):
    """Which FSDP vs EP rule applies (spec 5.2)."""

    ATTENTION = "attention"
    DENSE = "dense"
    SHARED_EXPERT = "shared_expert"
    ROUTED_EXPERT = "routed_expert"
    OTHER = "other"


class MeshError(ValueError):
    """Mesh, partition, or pipeline layout violates spec 5.2."""


# Flagship replica (spec 5.2). 72*12 = 864 GPUs per replica.
FLAGSHIP_EP = 72
FLAGSHIP_FSDP = 72
FLAGSHIP_PP = 12
FLAGSHIP_DP = 115
FLAGSHIP_CP_PRETRAIN = 1
FLAGSHIP_GPUS_PER_REPLICA = FLAGSHIP_EP * FLAGSHIP_PP


@dataclass(frozen=True)
class MeshSpec:
    """Integer device mesh. ``n_devices`` must equal dp * pp * ep * cp.

    ``fsdp`` is the shard count for attention/dense/shared and must equal
    ``ep`` (FSDP over the rack). Routed experts ignore FSDP.
    """

    dp: int
    fsdp: int
    ep: int
    pp: int
    cp: int

    @property
    def n_devices(self) -> int:
        return self.dp * self.pp * self.ep * self.cp


@dataclass(frozen=True)
class PartitionSpec:
    """Axis names per tensor dimension. None means replicated on that dim."""

    axes: tuple[str | None, ...]


@dataclass(frozen=True)
class StageRange:
    """Half-open unique-layer range assigned to one PP stage."""

    start: int
    end: int


def flagship_mesh() -> MeshSpec:
    """One flagship replica: EP=72, FSDP=72, PP=12, DP=115, CP=1."""
    return MeshSpec(
        dp=FLAGSHIP_DP,
        fsdp=FLAGSHIP_FSDP,
        ep=FLAGSHIP_EP,
        pp=FLAGSHIP_PP,
        cp=FLAGSHIP_CP_PRETRAIN,
    )


def tiny_mesh() -> MeshSpec:
    """CPU stand-in: 8 devices, EP=2, FSDP=2, PP=2, DP=2, CP=1."""
    return MeshSpec(dp=2, fsdp=2, ep=2, pp=2, cp=1)


def validate_mesh(spec: MeshSpec) -> None:
    """Raise MeshError if counts are non-positive or fsdp != ep."""
    raise NotImplementedError("A4 validate_mesh")


def device_ids(spec: MeshSpec) -> tuple[int, ...]:
    """``range(spec.n_devices)`` after validate_mesh."""
    raise NotImplementedError("A4 device_ids")


def coord_of(device: int, spec: MeshSpec) -> tuple[int, int, int, int]:
    """(dp, pp, ep, cp) coordinate of ``device``. FSDP index equals EP."""
    raise NotImplementedError("A4 coord_of")


def fsdp_partition(kind: ParamKind) -> PartitionSpec:
    """Attention/dense/shared shard on FSDP. Routed experts replicate (None,)."""
    raise NotImplementedError("A4 fsdp_partition")


def pipeline_stage_layers(
    n_layers: int,
    n_pp: int,
    core_block_layers: int,
    recurrence: int,
) -> tuple[StageRange, ...]:
    """FLOP-balanced unique-layer ranges per PP stage (spec 5.2).

    Recurrence multiplies FLOP of the last ``core_block_layers`` unique
    layers. ``n_pp`` stages cover ``[0, n_layers)`` without overlap.
    """
    raise NotImplementedError("A4 pipeline_stage_layers")


def context_parallel_on(seq_len: int, max_context_pretrain: int) -> bool:
    """True when seq_len exceeds the pretrain context (spec 5.2 CP)."""
    raise NotImplementedError("A4 context_parallel_on")


def ep_dispatch(tokens: Array, meta: Any) -> tuple[Array, Any]:
    """All-to-all tokens to experts. Wraps A3 ``kernels.ep_dispatch``."""
    raise NotImplementedError("A4 ep_dispatch")


def ep_combine(expert_out: Array, meta: Any, residual: Any) -> Array:
    """Weighted sum back to token order. Wraps A3 ``kernels.ep_combine``."""
    raise NotImplementedError("A4 ep_combine")
