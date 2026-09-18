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
    if r <= 0 or budget <= 0:
        raise ForkError("r and budget must be positive")
    cap = min(budget, RECURRENCE_MAX)
    bucketed = RECURRENCE_MAX
    for bucket in RECURRENCE_BUCKETS:
        if bucket >= r:
            bucketed = bucket
            break
    return RecurrencePlan(r=min(bucketed, cap), budget=budget, kv_shared=True)


def validate_latent_chunk(n_thoughts: int) -> LatentChunk:
    """Accept ``n_thoughts`` in ``[LATENT_CHUNK_MIN, LATENT_CHUNK_MAX]``.
    Raises ForkError outside that range.
    """
    if n_thoughts < LATENT_CHUNK_MIN or n_thoughts > LATENT_CHUNK_MAX:
        raise ForkError(
            f"n_thoughts {n_thoughts} outside [{LATENT_CHUNK_MIN}, {LATENT_CHUNK_MAX}]"
        )
    return LatentChunk(n_thoughts=n_thoughts, mode=DecodeMode.LATENT)


def decode_mode(in_think: bool, halt: bool) -> DecodeMode:
    """LATENT while inside ``<think>`` and not halted; VERBAL otherwise.
    Latent steps must not emit a token (13.2).
    """
    if in_think and not halt:
        return DecodeMode.LATENT
    return DecodeMode.VERBAL


def next_kv_tier(tier: KvTier) -> KvTier | None:
    """The next colder tier, or None at DISTRIBUTED."""
    for index, current in enumerate(KV_TIER_ORDER):
        if current == tier:
            if index + 1 < len(KV_TIER_ORDER):
                return KV_TIER_ORDER[index + 1]
            return None
    raise ForkError(f"unknown kv tier {tier!r}")


def swap_waiting_session(decoding: bool) -> KvTier:
    """Decoding stays on HBM. Waiting sessions move to GRACE (13.4, 13.7)."""
    if decoding:
        return KvTier.HBM
    return KvTier.GRACE


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
    ids = tuple(expert_ids)
    if top_k <= 0:
        raise ForkError("top_k must be positive")
    if len(ids) != top_k:
        raise ForkError(f"expert_ids length {len(ids)} != top_k {top_k}")
    seen: set[int] = set()
    for expert_id in ids:
        if expert_id < 0:
            raise ForkError(f"negative expert id {expert_id}")
        if expert_id in seen:
            raise ForkError(f"duplicate expert id {expert_id}")
        seen.add(expert_id)
    return RoutingRecord(
        token_index=token_index,
        layer_index=layer_index,
        expert_ids=ids,
    )


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
    if not parent_session:
        raise ForkError("parent_session must be non-empty")
    children = tuple(child_sessions)
    if not children:
        raise ForkError("child_sessions must be non-empty")
    if parent_session in children:
        raise ForkError("parent_session must not appear in child_sessions")
    if depth < 1 or depth > SUBAGENT_DEPTH_MAX:
        raise ForkError(f"depth {depth} outside [1, {SUBAGENT_DEPTH_MAX}]")
    return PrefixGroup(
        parent_session=parent_session,
        child_sessions=children,
        depth=depth,
    )


def register_ttt(task_id: str, lora_id: str) -> TttHook:
    """Register a sidecar LoRA for hot-load. Both ids non-empty.
    Raises ForkError otherwise. Does not train the LoRA (I8).
    """
    if not task_id or not lora_id:
        raise ForkError("task_id and lora_id must be non-empty")
    return TttHook(task_id=task_id, lora_id=lora_id)
