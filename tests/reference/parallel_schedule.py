"""Independent reference for the circular pipeline schedule (spec 5.1, A4).

Slow and obvious. Does **not** call ``parallel.pipeline_forward_schedule`` or
``parallel.circular_pipeline``. The only production import allowed here is
``parallel.MeshError``.

Schedule
--------
``n_ticks = n_microbatches + n_stages - 1``. Tick ``t`` has one slot per stage.
Stage ``s`` on that tick runs microbatch ``t - s`` when
``0 <= t - s < n_microbatches``, otherwise it is idle (``None``).

Golden (3 microbatches, 2 stages)::

    ((0, None),
     (1, 0),
     (2, 1),
     (None, 2))

Call order is the non-``None`` slots in tick-major order, and within a tick in
increasing stage index (ppermute order). That is not "all stages on microbatch
0, then all stages on microbatch 1" when both counts are greater than 1.

Composition
-----------
Each microbatch is its own chain: stage 0 first, then stage 1, and so on.
Schedule order changes only when stages run, not the per-microbatch result.
``sequential_compose`` is that chain. ``circular_pipeline`` runs the same
functions in schedule order and must return the same tuple.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from parallel import MeshError

Array = Any


def pipeline_forward_schedule(
    n_microbatches: int, n_stages: int
) -> tuple[tuple[int | None, ...], ...]:
    """Tick table. Stage ``s`` at tick ``t`` is microbatch ``t - s`` or ``None``."""
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
            if 0 <= mb < n_microbatches:
                row.append(mb)
            else:
                row.append(None)
        ticks.append(tuple(row))
    return tuple(ticks)


def schedule_call_order(
    n_microbatches: int, n_stages: int
) -> tuple[tuple[int, int], ...]:
    """Non-``None`` schedule slots as ``(stage_index, microbatch_index)``.

    Tick-major. Within a tick, increasing stage index. Idle slots are omitted:
    a stage is not called on a tick where its slot is ``None``.
    """
    calls: list[tuple[int, int]] = []
    for tick in pipeline_forward_schedule(n_microbatches, n_stages):
        for stage, mb in enumerate(tick):
            if mb is not None:
                calls.append((stage, mb))
    return tuple(calls)


def sequential_compose(
    microbatches: Sequence[Array],
    stage_fns: Sequence[Callable[[Array], Array]],
) -> tuple[Array, ...]:
    """Apply ``stage_fns`` in order (stage 0 first) to each microbatch.

    Independent of the tick schedule. Empty inputs raise ``MeshError``.
    """
    mbs = tuple(microbatches)
    fns = tuple(stage_fns)
    if len(mbs) < 1 or len(fns) < 1:
        raise MeshError(
            "microbatches and stage_fns must be non-empty, "
            f"got {len(mbs)} microbatches and {len(fns)} stages"
        )
    results: list[Array] = []
    for mb in mbs:
        x: Array = mb
        for fn in fns:
            x = fn(x)
        results.append(x)
    return tuple(results)


def circular_pipeline(
    microbatches: Sequence[Array],
    stage_fns: Sequence[Callable[[Array], Array]],
) -> tuple[Array, ...]:
    """Run ``stage_fns`` in :func:`pipeline_forward_schedule` order.

    Does not call production. Idle (``None``) slots are not invoked. The
    returned tuple is one value per microbatch, in input order — the same
    values as :func:`sequential_compose` when each stage is a pure function of
    its activation.
    """
    mbs = tuple(microbatches)
    fns = tuple(stage_fns)
    n_mb = len(mbs)
    n_st = len(fns)
    if n_mb < 1 or n_st < 1:
        raise MeshError(
            "microbatches and stage_fns must be non-empty, "
            f"got {n_mb} microbatches and {n_st} stages"
        )
    # Activation that stage ``s`` should see for microbatch ``m``.
    entering: dict[tuple[int, int], Array] = {(0, m): mbs[m] for m in range(n_mb)}
    outputs: list[Array] = [None] * n_mb
    for tick in pipeline_forward_schedule(n_mb, n_st):
        for stage, mb in enumerate(tick):
            if mb is None:
                continue
            y = fns[stage](entering[(stage, mb)])
            if stage + 1 == n_st:
                outputs[mb] = y
            else:
                entering[(stage + 1, mb)] = y
    return tuple(outputs)
