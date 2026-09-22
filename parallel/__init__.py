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

import kernels
import parallel.fsdp as _fsdp

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
    for name in ("dp", "fsdp", "ep", "pp", "cp"):
        if getattr(spec, name) < 1:
            raise MeshError(f"{name} must be positive")
    if spec.fsdp != spec.ep:
        raise MeshError("fsdp must equal ep")


def device_ids(spec: MeshSpec) -> tuple[int, ...]:
    """``range(spec.n_devices)`` after validate_mesh."""
    validate_mesh(spec)
    return tuple(range(spec.n_devices))


def coord_of(device: int, spec: MeshSpec) -> tuple[int, int, int, int]:
    """(dp, pp, ep, cp) coordinate of ``device``. FSDP index equals EP."""
    validate_mesh(spec)
    n = spec.n_devices
    if device < 0 or device >= n:
        raise MeshError("device id out of range")
    cp_i = device % spec.cp
    rest = device // spec.cp
    ep_i = rest % spec.ep
    rest = rest // spec.ep
    pp_i = rest % spec.pp
    dp_i = rest // spec.pp
    return (dp_i, pp_i, ep_i, cp_i)


def fsdp_partition(kind: ParamKind) -> PartitionSpec:
    """Attention/dense/shared shard on FSDP. Routed experts replicate (None,)."""
    if kind == ParamKind.ROUTED_EXPERT:
        return PartitionSpec((None,))
    return PartitionSpec(("fsdp",))


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
    if n_layers < 1 or n_pp < 1:
        raise MeshError("n_layers and n_pp must be positive")
    if n_pp > n_layers:
        raise MeshError("n_pp cannot exceed n_layers")
    if core_block_layers < 0 or core_block_layers > n_layers:
        raise MeshError("core_block_layers out of range")
    if recurrence < 1:
        raise MeshError("recurrence must be positive")

    prefix = [0]
    for i in range(n_layers):
        cost = recurrence if i >= n_layers - core_block_layers else 1
        prefix.append(prefix[-1] + cost)
    total = prefix[n_layers]
    ranges: list[StageRange] = []
    start = 0
    for s in range(n_pp - 1):
        target_num = total * (s + 1)
        max_end = n_layers - (n_pp - 1 - s)
        best_end = start + 1
        best_err = abs(n_pp * prefix[best_end] - target_num)
        for end in range(start + 2, max_end + 1):
            err = abs(n_pp * prefix[end] - target_num)
            if err < best_err:
                best_err = err
                best_end = end
        ranges.append(StageRange(start, best_end))
        start = best_end
    ranges.append(StageRange(start, n_layers))
    return tuple(ranges)


def context_parallel_on(seq_len: int, max_context_pretrain: int) -> bool:
    """True when seq_len exceeds the pretrain context (spec 5.2 CP)."""
    return seq_len > max_context_pretrain


def ep_dispatch(tokens: Array, meta: Any) -> tuple[Array, Any]:
    """All-to-all tokens to experts. Wraps A3 ``kernels.ep_dispatch``."""
    return kernels.ep_dispatch(tokens, meta)


def ep_combine(expert_out: Array, meta: Any, residual: Any) -> Array:
    """Weighted sum back to token order. Wraps A3 ``kernels.ep_combine``."""
    return kernels.ep_combine(expert_out, meta, residual)


fsdp_shard = _fsdp.fsdp_shard
fsdp_all_gather = _fsdp.fsdp_all_gather
fsdp_reduce_scatter = _fsdp.fsdp_reduce_scatter
zero3_views = _fsdp.zero3_views
