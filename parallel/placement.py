"""SPMD circular pipeline placement (spec 5.1).

Praxis-style: scan the circular schedule inside ``jax.shard_map``, and move
the activation to the next stage with ``jax.lax.ppermute``. Host devices
stand in for the pipeline axis. No NCCL.

Tests that need ``n_stages`` devices must start a fresh interpreter with
``XLA_FLAGS=--xla_force_host_platform_device_count=N`` set before the first
JAX import. A process that already imported JAX with one device cannot grow
the host mesh.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Array = Any
StageFn = Callable[[Array, Any], Array]


def spmd_circular_pipeline(
    microbatches: Array,
    *,
    n_stages: int,
    stage_fn: StageFn,
) -> Array:
    """Run each microbatch through stages ``0 .. n_stages-1`` on a PP mesh.

    ``microbatches`` is a JAX or NumPy array whose leading axis is the
    microbatch count (at least 1). ``stage_fn(x, stage)`` is the SPMD body:
    the same function on every device. ``stage`` is ``jax.lax.axis_index("pp")``,
    a traced scalar integer, not a Python int. The body must be JAX-traceable.
    Indexing a stacked parameter array with ``stage`` is fine; ``int(stage)``
    and Python lists are not.

    Placement, not a host loop:

    - Mesh axis ``"pp"`` of size ``n_stages``. Device ``i`` of
      ``jax.local_devices()[:n_stages]`` (that order) owns stage ``i``.
    - The forward is a ``jax.lax.scan`` over the circular schedule, inside
      ``jax.shard_map`` (the top-level function, not
      ``jax.experimental.shard_map``).
    - Between stages the activation moves with ``jax.lax.ppermute`` on axis
      ``"pp"``, permutation ``(i, (i + 1) % n_stages)`` for each ``i``.
    - Calling ``circular_pipeline`` and applying ``stage_fn`` on one device
      is not this function. A Python ``for`` over stages is not either.

    ``n_stages == 1`` still builds that mesh and still calls ``ppermute``
    with ``[(0, 0)]``.

    Bubble ticks (the ``n_stages - 1`` warmup and drain slots of
    ``circular_pipeline``) must not change a completed microbatch. Whether
    ``stage_fn`` runs on a bubble is unspecified, so ``stage_fn`` must be
    safe on a dummy activation of the same shape and dtype.

    Returns one array, fully addressable on the host, same shape as
    ``microbatches``. Microbatch ``i`` equals the sequential composition
    ``stage_fn(...stage_fn(microbatches[i], 0)..., n_stages - 1)`` with
    Python integer stage ids, within ``rtol=1e-5`` and ``atol=1e-5`` for
    floating dtypes, and bitwise for integer dtypes when ``stage_fn`` itself
    is exact. Dtype is whatever ``stage_fn`` returns, and it must be the
    same for every microbatch and every stage.

    ``jax.jit`` of this function, with ``n_stages`` and ``stage_fn`` static,
    matches the eager result to the same tolerance. The jaxpr of that jitted
    call contains a ``ppermute`` primitive. Inside ``stage_fn``,
    ``jax.lax.axis_index("pp")`` equals the ``stage`` argument; calling the
    body outside the mesh is not a valid implementation.

    ``stage_fn`` must preserve the activation shape. A shape change cannot
    be ``ppermute``'d uniformly, so it raises ``parallel.MeshError``.

    Raises ``parallel.MeshError`` (not ``NotImplementedError``, not a bare
    ``ValueError``) when ``n_stages`` is not an int ``>= 1``, the leading
    axis is missing or empty, ``microbatches`` is not an array,
    ``stage_fn`` is not callable, or ``jax.local_device_count() < n_stages``.
    Bool is not an int. A second call with a different ``n_stages`` must
    work; there is no process-global mesh left behind.

    Does not mutate ``microbatches``.
    """
    raise NotImplementedError("spmd_circular_pipeline")
