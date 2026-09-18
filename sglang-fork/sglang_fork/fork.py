"""SGLang fork deltas (spec 13.2-13.8, 4.1-4.2, 9.2, 10, 15.5 I4).

Discrete serving contract for latent decode, recurrence buckets, KV
tiering, routing capture, sub-agent prefix groups, and the TTT LoRA
hook. This module does not talk to a live SGLang engine and does not
import ``model``. Weight conversion stays C1; architecture counts stay
C2. JAX TTT training stays I8.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ForkError(ValueError):
    """A fork delta violates a 13.x invariant."""


class DecodeMode(StrEnum):
    """13.2 / 4.1: verbal samples a token; latent feeds the adapter."""

    VERBAL = "verbal"
    LATENT = "latent"


class KvTier(StrEnum):
    """13.4: HBM then Grace LPDDR5X then local NVMe then distributed."""

    HBM = "hbm"
    GRACE = "grace"
    NVME = "nvme"
    DISTRIBUTED = "distributed"


# 13.3: r is quantized so batches stay regular.
RECURRENCE_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16)
RECURRENCE_MAX = 16

# 4.2: latent chunks sit between discrete anchors.
LATENT_CHUNK_MIN = 4
LATENT_CHUNK_MAX = 64

# 14.4: sub-agents may spawn to depth 3.
SUBAGENT_DEPTH_MAX = 3

KV_TIER_ORDER: tuple[KvTier, ...] = (
    KvTier.HBM,
    KvTier.GRACE,
    KvTier.NVME,
    KvTier.DISTRIBUTED,
)


@dataclass(frozen=True)
class RecurrencePlan:
    """Per-request depth. ``r`` is the bucketed iteration count."""

    r: int
    budget: int
    kv_shared: bool = True


@dataclass(frozen=True)
class LatentChunk:
    """One continuous-thought span inside ``<think>``. No tokens emitted."""

    n_thoughts: int
    mode: DecodeMode = DecodeMode.LATENT


@dataclass(frozen=True)
class RoutingRecord:
    """Expert ids chosen for one token at one MoE layer (9.2 routing replay)."""

    token_index: int
    layer_index: int
    expert_ids: tuple[int, ...]


@dataclass(frozen=True)
class PrefixGroup:
    """13.7: spawned sub-agents share the parent's prefix KV."""

    parent_session: str
    child_sessions: tuple[str, ...]
    depth: int


@dataclass(frozen=True)
class TttHook:
    """13.8 / 10: SGLang hot-loads a LoRA the JAX sidecar produced."""

    task_id: str
    lora_id: str


def bucket_recurrence(r: int, budget: int) -> RecurrencePlan:
    """Ceil ``r`` to the next value in ``RECURRENCE_BUCKETS``, cap by
    ``min(budget, RECURRENCE_MAX)``. ``kv_shared`` is always True.
    Raises ForkError on non-positive ``r`` or ``budget``.
    """
    raise NotImplementedError("I4 bucket_recurrence")


def validate_latent_chunk(n_thoughts: int) -> LatentChunk:
    """Accept ``n_thoughts`` in ``[LATENT_CHUNK_MIN, LATENT_CHUNK_MAX]``.
    Raises ForkError outside that range.
    """
    raise NotImplementedError("I4 validate_latent_chunk")


def decode_mode(in_think: bool, halt: bool) -> DecodeMode:
    """LATENT while inside ``<think>`` and not halted; VERBAL otherwise.
    Latent steps must not emit a token (13.2).
    """
    raise NotImplementedError("I4 decode_mode")


def next_kv_tier(tier: KvTier) -> KvTier | None:
    """The next colder tier, or None at DISTRIBUTED."""
    raise NotImplementedError("I4 next_kv_tier")


def swap_waiting_session(decoding: bool) -> KvTier:
    """Decoding stays on HBM. Waiting sessions move to GRACE (13.4, 13.7)."""
    raise NotImplementedError("I4 swap_waiting_session")


def capture_routing(
    token_index: int,
    layer_index: int,
    expert_ids: tuple[int, ...],
    top_k: int,
) -> RoutingRecord:
    """Record expert ids for the trainer's routing replay (9.2).
    Length of ``expert_ids`` must equal ``top_k``; ids unique and
    non-negative. Raises ForkError otherwise.
    """
    raise NotImplementedError("I4 capture_routing")


def schedule_prefix_group(
    parent_session: str,
    child_sessions: tuple[str, ...],
    depth: int,
) -> PrefixGroup:
    """Schedule children as one group sharing the parent's prefix KV.
    ``parent_session`` non-empty, ``child_sessions`` non-empty, no
    overlap with the parent, depth in ``[1, SUBAGENT_DEPTH_MAX]``.
    Raises ForkError otherwise.
    """
    raise NotImplementedError("I4 schedule_prefix_group")


def register_ttt(task_id: str, lora_id: str) -> TttHook:
    """Register a sidecar LoRA for hot-load. Both ids non-empty.
    Raises ForkError otherwise. Does not train the LoRA (I8).
    """
    raise NotImplementedError("I4 register_ttt")
