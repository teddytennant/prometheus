"""Independent I4 reference: SGLang fork deltas (spec 13.2-13.8, 4.1-4.2, 9.2, 10).

Plain Python, slow and obvious. Does not import production ``sglang_fork``
or ``sglang_fork.fork``. Enums, dataclasses, and constants are a local
mirror so tests can compare field-by-field without sharing code.

Rules
-----
bucket_recurrence: ceil r to the next value in RECURRENCE_BUCKETS, then
    cap by min(budget, RECURRENCE_MAX). kv_shared is always True.
    Non-positive r or budget -> ForkError.
validate_latent_chunk: n_thoughts in [LATENT_CHUNK_MIN, LATENT_CHUNK_MAX].
decode_mode: LATENT iff in_think and not halt; VERBAL otherwise.
next_kv_tier: HBM -> GRACE -> NVME -> DISTRIBUTED -> None.
swap_waiting_session: decoding True -> HBM; False -> GRACE.
capture_routing: len(expert_ids) == top_k, unique non-negative ids.
schedule_prefix_group: non-empty parent and children, parent not in
    children, depth in [1, SUBAGENT_DEPTH_MAX].
register_ttt: both ids non-empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ForkError(ValueError):
    """A fork delta violates a 13.x invariant."""


class DecodeMode(StrEnum):
    VERBAL = "verbal"
    LATENT = "latent"


class KvTier(StrEnum):
    HBM = "hbm"
    GRACE = "grace"
    NVME = "nvme"
    DISTRIBUTED = "distributed"


RECURRENCE_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16)
RECURRENCE_MAX = 16

LATENT_CHUNK_MIN = 4
LATENT_CHUNK_MAX = 64

SUBAGENT_DEPTH_MAX = 3

KV_TIER_ORDER: tuple[KvTier, ...] = (
    KvTier.HBM,
    KvTier.GRACE,
    KvTier.NVME,
    KvTier.DISTRIBUTED,
)


@dataclass(frozen=True)
class RecurrencePlan:
    r: int
    budget: int
    kv_shared: bool = True


@dataclass(frozen=True)
class LatentChunk:
    n_thoughts: int
    mode: DecodeMode = DecodeMode.LATENT


@dataclass(frozen=True)
class RoutingRecord:
    token_index: int
    layer_index: int
    expert_ids: tuple[int, ...]


@dataclass(frozen=True)
class PrefixGroup:
    parent_session: str
    child_sessions: tuple[str, ...]
    depth: int


@dataclass(frozen=True)
class TttHook:
    task_id: str
    lora_id: str


def _ceil_to_bucket(r: int) -> int:
    """Smallest bucket >= r, or RECURRENCE_MAX if r exceeds every bucket."""
    for bucket in RECURRENCE_BUCKETS:
        if bucket >= r:
            return bucket
    return RECURRENCE_MAX


def bucket_recurrence(r: int, budget: int) -> RecurrencePlan:
    """Ceil r to the next bucket, then cap by min(budget, RECURRENCE_MAX)."""
    if r <= 0 or budget <= 0:
        raise ForkError("r and budget must be positive")
    cap = min(budget, RECURRENCE_MAX)
    bucketed = _ceil_to_bucket(r)
    return RecurrencePlan(r=min(bucketed, cap), budget=budget, kv_shared=True)


def validate_latent_chunk(n_thoughts: int) -> LatentChunk:
    """Accept n_thoughts in [LATENT_CHUNK_MIN, LATENT_CHUNK_MAX]."""
    if n_thoughts < LATENT_CHUNK_MIN or n_thoughts > LATENT_CHUNK_MAX:
        raise ForkError(
            f"n_thoughts {n_thoughts} outside [{LATENT_CHUNK_MIN}, {LATENT_CHUNK_MAX}]"
        )
    return LatentChunk(n_thoughts=n_thoughts, mode=DecodeMode.LATENT)


def decode_mode(in_think: bool, halt: bool) -> DecodeMode:
    """LATENT while inside <think> and not halted; VERBAL otherwise."""
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
    """Decoding stays on HBM. Waiting sessions move to GRACE."""
    if decoding:
        return KvTier.HBM
    return KvTier.GRACE


def capture_routing(
    token_index: int,
    layer_index: int,
    expert_ids: tuple[int, ...],
    top_k: int,
) -> RoutingRecord:
    """Record expert ids. Length == top_k; ids unique and non-negative."""
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
    """Children share the parent's prefix KV. Depth in [1, SUBAGENT_DEPTH_MAX]."""
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
    """Register a sidecar LoRA for hot-load. Both ids non-empty."""
    if not task_id or not lora_id:
        raise ForkError("task_id and lora_id must be non-empty")
    return TttHook(task_id=task_id, lora_id=lora_id)
