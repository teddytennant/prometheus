"""Independent reference for SPMD circular pipeline placement (spec 5.1).

Does not call ``parallel.spmd_circular_pipeline``. The only production import
is ``parallel.MeshError``.

The CPU reference is sequential composition with Python integer stage ids.
Production must match it, but must get there through ``jax.shard_map`` and
``jax.lax.ppermute``, which these helpers never call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from parallel import MeshError

Array = Any
StageFn = Callable[[Array, Any], Array]


def sequential_compose(microbatches: Array, n_stages: int, stage_fn: StageFn) -> np.ndarray:
    """Apply ``stage_fn(x, stage)`` for ``stage`` in ``0 .. n_stages-1`` per row.

    ``microbatches`` must be an array with a non-empty leading axis. ``n_stages``
    must be an int ``>= 1``. ``stage_fn`` must be callable and must not change
    the activation shape. Bool is not an int.
    """
    if isinstance(n_stages, bool) or not isinstance(n_stages, int) or n_stages < 1:
        raise MeshError(f"n_stages must be an int >= 1, got {n_stages!r}")
    if not callable(stage_fn):
        raise MeshError("stage_fn must be callable")
    if not isinstance(microbatches, np.ndarray):
        raise MeshError(f"microbatches must be an array, got {type(microbatches).__name__}")
    if microbatches.ndim < 1 or microbatches.shape[0] < 1:
        raise MeshError("microbatches leading axis must be non-empty")
    out = []
    for i in range(microbatches.shape[0]):
        x = microbatches[i]
        for stage in range(n_stages):
            y = np.asarray(stage_fn(x, stage))
            if y.shape != np.asarray(x).shape:
                raise MeshError(
                    f"stage_fn changed shape from {np.asarray(x).shape} to {y.shape}"
                )
            x = y
        out.append(x)
    return np.stack(out, axis=0)
