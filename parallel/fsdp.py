"""FSDP ZeRO-3 collectives (spec 5.2). CPU analog of in-rack NVLink shard.

Attention, dense, shared-expert, and other params are sharded on the FSDP
axis. Routed experts stay replicated on the GPU that owns them. The GPU
holds a BF16 compute copy. The FP32 master lives on Grace.

This module does not launch NCCL. Ranks are integers. A multi-rank call
passes every rank's tensor in rank order.
"""

from __future__ import annotations

from typing import Any

Array = Any


def fsdp_shard(param: Array, n: int, axis: int, rank: int) -> Array:
    """Return rank's contiguous shard of ``param`` along ``axis``.

    ``n`` is the FSDP world size (spec 5.2: FSDP over the rack, equal to EP).
    Pieces follow the ``numpy.array_split`` rule: the first ``dim % n`` shards
    have size ``dim // n + 1``, the rest ``dim // n``. ``rank`` is in
    ``[0, n)``. ``axis`` may be negative and is normalized against ``ndim``.

    Raises ``parallel.MeshError`` if ``n < 1``, ``rank`` is out of range,
    ``axis`` is out of range, or the dimension is smaller than ``n`` (a
    zero-size shard is not a valid ZeRO-3 piece).

    ``n == 1`` returns ``param`` unchanged in values (identity of the object
    is not required).

    Accepts a NumPy array or a JAX array. Returns a ``jax.numpy`` array of
    the same dtype. ``n``, ``axis``, and ``rank`` are static under
    ``jax.jit``: traced values must not be turned into NumPy or Python
    scalars, and the jitted result must match eager bitwise for float32,
    int32, and bfloat16.
    """
    raise NotImplementedError("fsdp_shard")


def fsdp_all_gather(shards: tuple[Array, ...] | list[Array], axis: int) -> Array:
    """Concatenate rank-ordered shards along ``axis``.

    ``shards[r]`` is rank ``r``'s piece. The result's size along ``axis`` is
    the sum of the shard sizes. Every other dimension must match. An empty
    sequence raises ``parallel.MeshError``. Mixed dtypes or mismatched
    ranks raise ``parallel.MeshError``. ``axis`` may be negative.

    Inverse of :func:`fsdp_shard`: gathering the ``n`` shards of
    ``fsdp_shard(param, n, axis, r)`` for ``r`` in ``0 .. n-1`` recovers
    ``param`` exactly (bitwise for float32, int32, bfloat16, float64).

    Returns a ``jax.numpy`` array. ``axis`` is static under ``jax.jit``.
    ``shards`` is a concrete sequence of traced arrays, not a stacked
    tensor (uneven splits cannot be stacked). Traced values must not be
    turned into NumPy or Python scalars. Jitted result matches eager
    bitwise for float32, int32, and bfloat16.
    """
    raise NotImplementedError("fsdp_all_gather")


def fsdp_reduce_scatter(
    partials: tuple[Array, ...] | list[Array],
    axis: int,
    rank: int,
) -> Array:
    """Sum full-shaped partial gradients, then return this rank's shard.

    Each partial has the same shape (the unsharded gradient). ``n`` is
    ``len(partials)``. ``rank`` is in ``[0, n)``. The shard of the sum uses
    the same split rule as :func:`fsdp_shard`.

    The reduction op is sum, not mean (spec 5.3: gradient reduce; the train
    step owns any DP average). Floating partials with itemsize <= 4
    (float32, float16, bfloat16) accumulate in float32, then the shard is
    cast back to the input dtype. float64 accumulates in float64. Integer
    partials accumulate in int64 and cast back. bool and complex raise
    ``parallel.MeshError``.

    Empty partials, mismatched shapes, mixed dtypes, a bad ``axis``, or a
    bad ``rank`` raise ``parallel.MeshError``. A dimension smaller than
    ``n`` raises ``parallel.MeshError``, same as :func:`fsdp_shard`.

    Returns a ``jax.numpy`` array. ``axis`` and ``rank`` are static under
    ``jax.jit``. Traced values must not be turned into NumPy or Python
    scalars. Jitted result matches eager at 1e-5 relative for float32, and
    bitwise for int32.
    """
    raise NotImplementedError("fsdp_reduce_scatter")


def zero3_views(
    param: Array,
    kind: Any,
    n: int,
    axis: int,
    rank: int,
) -> tuple[Array, Array]:
    """Return ``(gpu_compute, grace_master)`` for this FSDP rank.

    Spec 5.2: the GPU holds BF16 weights; Grace holds the FP32 master and
    the Muon momentum. ``kind`` is a ``parallel.ParamKind``.

    ``ROUTED_EXPERT`` is not sharded: both views are the full tensor.
    Every other kind is sharded with :func:`fsdp_shard` along ``axis``
    first, then cast. Shard-then-cast, not cast-then-shard.

    ``gpu_compute`` is bfloat16. ``grace_master`` is float32. Values match
    the dtype cast of the (sharded or full) param, with no extra rounding.

    ``kind`` must be a ``ParamKind``. A string or other type raises
    ``parallel.MeshError``. A non-floating ``param`` raises
    ``parallel.MeshError``. Shard errors propagate from :func:`fsdp_shard`.

    Returns two ``jax.numpy`` arrays. ``n``, ``axis``, ``rank``, and
    ``kind`` are static under ``jax.jit``. Traced values must not be turned
    into NumPy or Python scalars. Jitted views match eager bitwise.
    """
    raise NotImplementedError("zero3_views")
