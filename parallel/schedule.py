"""Circular pipeline schedule (spec 5.1). CPU analog of Praxis SPMD PP.

Spec 5.1: scan over stages, permute activations to the next stage, stages
balanced by FLOPs (see ``pipeline_stage_layers``). This module is the
forward schedule and the execution that follows it. It does not launch
multi-host jobs. The ``shard_map`` + ``ppermute`` device placement is a
GPU stage (V2 / V5). On CPU the contract is the tick schedule and
numerical match to single-device composition.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

Array = Any
StageFn = Callable[[Array], Array]


def pipeline_forward_schedule(
    n_microbatches: int,
    n_stages: int,
) -> tuple[tuple[int | None, ...], ...]:
    """Circular forward schedule (spec 5.1).

    Returns one entry per tick. Each tick is a tuple of length
    ``n_stages``. ``tick[s]`` is the microbatch index stage ``s`` runs, or
    ``None`` if that stage is idle.

    At tick ``t`` (0-based), stage ``s`` runs microbatch ``t - s`` when
    ``0 <= t - s < n_microbatches``, else ``None``. The number of ticks is
    ``n_microbatches + n_stages - 1``.

    Raises ``parallel.MeshError`` if ``n_microbatches < 1`` or
    ``n_stages < 1``. Both arguments are Python ints.
    """
    # Local import: parallel/__init__.py imports this module before MeshError exists.
    from parallel import MeshError

    if n_microbatches < 1 or n_stages < 1:
        raise MeshError(
            "n_microbatches and n_stages must be >= 1, "
            f"got n_microbatches={n_microbatches}, n_stages={n_stages}"
        )
    n_ticks = n_microbatches + n_stages - 1
    ticks: list[tuple[int | None, ...]] = []
    for t in range(n_ticks):
        row: list[int | None] = []
        for s in range(n_stages):
            mb = t - s
            row.append(mb if 0 <= mb < n_microbatches else None)
        ticks.append(tuple(row))
    return tuple(ticks)


def circular_pipeline(
    microbatches: Sequence[Array],
    stage_fns: Sequence[StageFn],
) -> tuple[Array, ...]:
    """Run ``stage_fns`` on each microbatch in pipeline order.

    ``stage_fns[s]`` is the pure forward of PP stage ``s``. The activation
    leaving stage ``s`` on microbatch ``m`` is the input to stage ``s + 1``
    on the same microbatch (the ``ppermute``). Stage 0's input is the
    microbatch itself.

    Execution follows :func:`pipeline_forward_schedule`. Within a tick,
    live stages run in increasing stage index (CPU is sequential; a real
    pipeline runs them together). A stage is not called on a tick where
    the schedule says ``None``. Each stage is called once per microbatch,
    and the call order recorded as ``(stage, microbatch)`` pairs must
    equal the non-``None`` schedule entries in tick-major order.

    ``result[m]`` equals
    ``stage_fns[-1](...stage_fns[0](microbatches[m]))``. Stage functions
    may change rank and shape. They must be pure. An exception from a
    stage function propagates.

    Raises ``parallel.MeshError`` if ``microbatches`` is empty or
    ``stage_fns`` is empty.

    Returns a tuple in microbatch order, not tick order. Elements are
    whatever the last stage returned (not re-wrapped).
    """
    from parallel import MeshError

    activations = list(microbatches)
    fns = tuple(stage_fns)
    n_mb = len(activations)
    n_st = len(fns)
    if n_mb < 1 or n_st < 1:
        raise MeshError(
            "microbatches and stage_fns must be non-empty, "
            f"got {n_mb} microbatches and {n_st} stages"
        )
    # Schedule order guarantees stage s sees the output of stage s-1 for that
    # microbatch (or the microbatch itself when s == 0). One slot per microbatch.
    for tick in pipeline_forward_schedule(n_mb, n_st):
        for s, m in enumerate(tick):
            if m is None:
                continue
            activations[m] = fns[s](activations[m])
    return tuple(activations)
