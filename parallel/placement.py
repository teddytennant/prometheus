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
from typing import Any, NoReturn

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec

Array = Any
StageFn = Callable[[Array, Any], Array]


def _raise(message: str) -> NoReturn:
    # parallel/__init__.py imports this module before MeshError is defined.
    from parallel import MeshError

    raise MeshError(message)


def _is_array(value: Array) -> bool:
    return isinstance(value, np.ndarray) or isinstance(value, jax.Array)


def _check_inputs(microbatches: Array, n_stages: int, stage_fn: StageFn) -> int:
    if isinstance(n_stages, bool) or not isinstance(n_stages, int) or n_stages < 1:
        _raise(f"n_stages must be an int >= 1, got {n_stages!r}")
    if not callable(stage_fn):
        _raise("stage_fn must be callable")
    if not _is_array(microbatches):
        _raise("microbatches must be an array")
    # 0-d has no leading axis. shape[0] is only read once ndim is known.
    if microbatches.ndim < 1 or microbatches.shape[0] < 1:
        _raise("microbatches needs a non-empty leading axis")
    if jax.local_device_count() < n_stages:
        _raise(
            f"need {n_stages} local devices, have {jax.local_device_count()}"
        )
    return n_stages


def _reject_host_shape_change(microbatches: Array, stage_fn: StageFn) -> None:
    """Reject a reshape that tracing cannot see.

    ``np.asarray(x).reshape`` dies with a tracer error inside ``shard_map``,
    so it never reaches the in-mesh shape check. Probe a host copy first.
    Skip tracers (jit) and bodies that need the mesh (``axis_index``).
    """
    if isinstance(microbatches, jax.core.Tracer):
        return
    act_shape = tuple(microbatches.shape[1:])
    try:
        spec = jax.eval_shape(lambda x: stage_fn(x, jnp.int32(0)), microbatches[0])
    except Exception:
        spec = None
    if spec is not None:
        if tuple(spec.shape) != act_shape:
            _raise(
                "stage_fn must preserve activation shape, "
                f"got {tuple(spec.shape)}, expected {act_shape}"
            )
        return
    sample = np.array(np.asarray(microbatches[0]), copy=True)
    try:
        probed = stage_fn(sample, 0)
    except Exception:
        return
    got = tuple(np.shape(probed))
    if got != act_shape:
        _raise(f"stage_fn must preserve activation shape, got {got}, expected {act_shape}")


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
    n_stages = _check_inputs(microbatches, n_stages, stage_fn)
    # 64-bit stage_fns (the int64 oracle) must not be truncated to int32.
    # The context manager restores the process flag, including under jit.
    with jax.enable_x64():
        _reject_host_shape_change(microbatches, stage_fn)
        return _place(microbatches, n_stages, stage_fn)


def _place(microbatches: Array, n_stages: int, stage_fn: StageFn) -> Array:
    # Local mesh only. Do not install it as the process mesh.
    devices = np.array(jax.local_devices()[:n_stages])
    mesh = Mesh(devices, ("pp",))
    perm = tuple((i, (i + 1) % n_stages) for i in range(n_stages))

    def _forward(mbs: Array) -> Array:
        stage = jax.lax.axis_index("pp")
        n_mb = mbs.shape[0]
        act_shape = mbs.shape[1:]
        spec = jax.eval_shape(lambda x: stage_fn(x, stage), mbs[0])
        if tuple(spec.shape) != tuple(act_shape):
            _raise(
                "stage_fn must preserve activation shape, "
                f"got {tuple(spec.shape)}, expected {tuple(act_shape)}"
            )
        out_dtype = spec.dtype
        # ppermute makes the carry varying on "pp". scan rejects a carry whose
        # manual axis appears only on the output, so mark the zeros up front.
        outputs = jax.lax.pcast(jnp.zeros(mbs.shape, out_dtype), ("pp",), to="varying")
        activation = jax.lax.pcast(jnp.zeros(act_shape, out_dtype), ("pp",), to="varying")
        n_ticks = n_mb + n_stages - 1
        same_dtype = mbs.dtype == out_dtype

        def body(carry: tuple[Array, Array], tick: Array) -> tuple[tuple[Array, Array], None]:
            act, outs = carry
            # Circular schedule: stage s at tick t runs microbatch t - s.
            mb_index = tick - stage
            valid = (mb_index >= 0) & (mb_index < n_mb)
            safe = jnp.clip(mb_index, 0, n_mb - 1)
            mb_input = mbs[safe]
            if same_dtype:
                y = stage_fn(jnp.where(stage == 0, mb_input, act), stage)
            else:
                # Stage 0 still sees the microbatch dtype; later stages see
                # the previous stage's output. cond keeps those inputs apart.
                y = jax.lax.cond(
                    stage == 0,
                    lambda _: stage_fn(mb_input, stage),
                    lambda _: stage_fn(act, stage),
                    operand=None,
                )
            if tuple(y.shape) != tuple(act_shape):
                _raise(
                    "stage_fn must preserve activation shape, "
                    f"got {tuple(y.shape)}, expected {tuple(act_shape)}"
                )
            y = jnp.asarray(y)
            # Bubbles must not clobber a microbatch the last stage already wrote.
            write = valid & (stage == n_stages - 1)
            outs = outs.at[safe].set(jnp.where(write, y, outs[safe]))
            act = jax.lax.ppermute(y, "pp", perm)
            return (act, outs), None

        (_, outputs), _ = jax.lax.scan(body, (activation, outputs), jnp.arange(n_ticks))
        # Last stage holds every finished microbatch. Sum the zero buffers
        # from the other stages so the result is replicated, with no host sync.
        return jax.lax.psum(outputs, "pp")

    return jax.shard_map(
        _forward,
        mesh=mesh,
        in_specs=PartitionSpec(),
        out_specs=PartitionSpec(),
    )(microbatches)
