"""Slow, obvious NumPy / plain-Python reference for A4 parallel (spec 5.1-5.2).

This is the oracle, not the production JAX mesh. Production ``parallel.*``
must match these counts, coordinates, PartitionSpec axes, stage ranges, and
EP buffers (atol/rtol 1e-5 in FP32) on the same inputs.

Do not import production ``parallel``, ``kernels``, or jax from this file.
MeshSpec / DispatchMeta objects are duck-typed (``dp``, ``fsdp``, ``ep``,
``pp``, ``cp``, ``n_devices``; ``expert_ids``, ``probs``, ``racks``,
``n_experts``, ``max_racks``).

Mesh (spec 5.1)
---------------
A replica is PP racks times EP GPUs in a rack. FSDP is the same axis as EP
(``fsdp == ep``). CP is 1 at pretrain. Device ids are C-ordered in the shape
``(dp, pp, ep, cp)``: CP changes fastest, then EP (the FSDP index), then PP,
then DP::

    device = ((dp * pp + pp_i) * ep + ep_i) * cp + cp_i

Pipeline stages (spec 5.2)
--------------------------
Layer ``i`` costs ``recurrence`` FLOP if it is among the last
``core_block_layers`` unique layers, else 1. ``n_pp`` contiguous half-open
ranges cover ``[0, n_layers)`` without overlap. Each stage gets at least one
unique layer.

Split points are chosen left to right so the cumulative cost at the end of
stage ``s`` (s = 0 .. n_pp-2) is as close as possible to
``(s+1)/n_pp * total_cost``, leaving at least one layer for each later stage.
Closeness is the integer ``|n_pp * prefix[end] - total * (s+1)|``. Ties keep
the smaller end (do not take the extra layer). Later stages that hold the
recurrent core therefore get fewer unique layers.

FSDP PartitionSpec
------------------
Attention, dense, shared-expert, and other params shard the leading axis on
``fsdp``. Routed experts replicate: axes ``(None,)``.

EP dispatch / combine
---------------------
Independent nested-loop copy of the A3 contract: tokens -> (n_experts,
max_per_expert, d_model) with padding zeros; combine scatters ``prob *``
expert outputs back. A token whose chosen experts span more than
``max_racks`` distinct racks is an error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FSDP_AXIS = "fsdp"
ROUTED_REPLICATE = (None,)
SHARD_FSDP = (FSDP_AXIS,)

_ROUTED = "routed_expert"


def _as_int(value: object, name: str) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an int") from exc
    if iv != value:
        raise ValueError(f"{name} must be an int")
    return iv


def _as_f32(x: object) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def validate_mesh(spec: object) -> None:
    """Reject non-positive axis counts and fsdp != ep."""
    for name in ("dp", "fsdp", "ep", "pp", "cp"):
        val = _as_int(getattr(spec, name), name)
        if val < 1:
            raise ValueError(f"{name} must be positive")
    if int(spec.fsdp) != int(spec.ep):
        raise ValueError("fsdp must equal ep")


def device_ids(spec: object) -> tuple[int, ...]:
    validate_mesh(spec)
    return tuple(range(int(spec.n_devices)))


def coord_of(device: int, spec: object) -> tuple[int, int, int, int]:
    """(dp, pp, ep, cp). FSDP index is the EP coordinate."""
    validate_mesh(spec)
    n = int(spec.n_devices)
    d = _as_int(device, "device")
    if d < 0 or d >= n:
        raise ValueError("device id out of range")
    cp = int(spec.cp)
    ep = int(spec.ep)
    pp = int(spec.pp)
    cp_i = d % cp
    rest = d // cp
    ep_i = rest % ep
    rest = rest // ep
    pp_i = rest % pp
    dp_i = rest // pp
    return (dp_i, pp_i, ep_i, cp_i)


def device_from_coord(
    dp: int, pp: int, ep: int, cp: int, spec: object
) -> int:
    """Inverse of coord_of for the C-order (dp, pp, ep, cp) layout."""
    validate_mesh(spec)
    if not (0 <= dp < int(spec.dp)):
        raise ValueError("dp coordinate out of range")
    if not (0 <= pp < int(spec.pp)):
        raise ValueError("pp coordinate out of range")
    if not (0 <= ep < int(spec.ep)):
        raise ValueError("ep coordinate out of range")
    if not (0 <= cp < int(spec.cp)):
        raise ValueError("cp coordinate out of range")
    return ((dp * int(spec.pp) + pp) * int(spec.ep) + ep) * int(spec.cp) + cp


def fsdp_partition(kind: object) -> tuple[str | None, ...]:
    """Leading-axis FSDP shard, except routed experts which replicate."""
    value = kind.value if hasattr(kind, "value") else kind
    if value == _ROUTED:
        return ROUTED_REPLICATE
    return SHARD_FSDP


def layer_flop(
    layer_index: int,
    n_layers: int,
    core_block_layers: int,
    recurrence: int,
) -> int:
    """FLOP weight of unique layer ``layer_index`` (spec 5.2 recurrence)."""
    if layer_index >= n_layers - core_block_layers:
        return int(recurrence)
    return 1


def pipeline_stage_layers(
    n_layers: int,
    n_pp: int,
    core_block_layers: int,
    recurrence: int,
) -> tuple[tuple[int, int], ...]:
    """FLOP-balanced unique-layer ranges per PP stage.

    Returns half-open ``(start, end)`` pairs covering ``[0, n_layers)``.
    """
    n_layers = _as_int(n_layers, "n_layers")
    n_pp = _as_int(n_pp, "n_pp")
    core_block_layers = _as_int(core_block_layers, "core_block_layers")
    recurrence = _as_int(recurrence, "recurrence")
    if n_layers < 1 or n_pp < 1:
        raise ValueError("n_layers and n_pp must be positive")
    if n_pp > n_layers:
        raise ValueError("n_pp cannot exceed n_layers")
    if core_block_layers < 0 or core_block_layers > n_layers:
        raise ValueError("core_block_layers out of range")
    if recurrence < 1:
        raise ValueError("recurrence must be positive")

    prefix = [0]
    for i in range(n_layers):
        prefix.append(
            prefix[-1] + layer_flop(i, n_layers, core_block_layers, recurrence)
        )
    total = prefix[n_layers]
    ranges: list[tuple[int, int]] = []
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
        ranges.append((start, best_end))
        start = best_end
    ranges.append((start, n_layers))
    return tuple(ranges)


def context_parallel_on(seq_len: int, max_context_pretrain: int) -> bool:
    """True when seq_len exceeds the pretrain context."""
    return int(seq_len) > int(max_context_pretrain)


@dataclass(frozen=True)
class _EpResidual:
    slots: np.ndarray
    expert_ids: np.ndarray
    probs: np.ndarray
    n_tokens: int
    d_model: int


def ep_dispatch(tokens: object, meta: object) -> tuple[np.ndarray, _EpResidual]:
    """All-to-all tokens -> (n_experts, max_per_expert, d_model)."""
    x = _as_f32(tokens)
    if x.ndim != 2:
        raise ValueError("tokens must be (n_tokens, d_model)")
    n_tokens, d_model = x.shape
    expert_ids = np.asarray(meta.expert_ids, dtype=np.int64)
    probs = _as_f32(meta.probs)
    racks = np.asarray(meta.racks, dtype=np.int64)
    n_experts = int(meta.n_experts)
    max_racks = int(meta.max_racks)
    if expert_ids.ndim != 2 or expert_ids.shape[0] != n_tokens:
        raise ValueError("expert_ids must be (n_tokens, top_k)")
    if probs.shape != expert_ids.shape:
        raise ValueError("probs must match expert_ids")
    if racks.shape != (n_experts,):
        raise ValueError("racks must be (n_experts,)")
    if np.any(expert_ids < 0) or np.any(expert_ids >= n_experts):
        raise ValueError("expert id out of range")

    top_k = int(expert_ids.shape[1])
    for t in range(n_tokens):
        chosen: set[int] = set()
        for k in range(top_k):
            chosen.add(int(racks[int(expert_ids[t, k])]))
        if len(chosen) > max_racks:
            raise ValueError("token experts span more than max_racks")

    counts = np.zeros((n_experts,), dtype=np.int64)
    for t in range(n_tokens):
        for k in range(top_k):
            counts[int(expert_ids[t, k])] += 1
    max_per_expert = int(counts.max()) if n_experts > 0 else 0

    dispatched = np.zeros((n_experts, max_per_expert, d_model), dtype=np.float32)
    slot = np.zeros((n_experts,), dtype=np.int64)
    slots = np.full((n_tokens, top_k), -1, dtype=np.int64)
    for t in range(n_tokens):
        for k in range(top_k):
            e = int(expert_ids[t, k])
            s = int(slot[e])
            dispatched[e, s] = x[t]
            slots[t, k] = s
            slot[e] += 1

    residual = _EpResidual(
        slots=slots,
        expert_ids=np.array(expert_ids, copy=True),
        probs=np.array(probs, copy=True),
        n_tokens=n_tokens,
        d_model=d_model,
    )
    return dispatched, residual


def ep_combine(expert_out: object, meta: object, residual: object) -> np.ndarray:
    """Inverse of ep_dispatch: weighted scatter-add back to tokens."""
    y = _as_f32(expert_out)
    slots = np.asarray(residual.slots, dtype=np.int64)
    expert_ids = np.asarray(residual.expert_ids, dtype=np.int64)
    probs = _as_f32(residual.probs)
    n_tokens = int(residual.n_tokens)
    d_model = int(residual.d_model)
    out = np.zeros((n_tokens, d_model), dtype=np.float32)
    top_k = int(expert_ids.shape[1]) if expert_ids.ndim == 2 else 0
    for t in range(n_tokens):
        for k in range(top_k):
            e = int(expert_ids[t, k])
            s = int(slots[t, k])
            if s < 0:
                continue
            out[t] += probs[t, k] * y[e, s]
    return out
