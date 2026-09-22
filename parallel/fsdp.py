"""FSDP ZeRO-3 collectives (spec 5.2).

Shard, all-gather, and reduce-scatter for parameter tensors, plus the two
storage views of a parameter (GPU BF16 shard, Grace FP32 shard). Collective
math only: no device mesh, no NCCL. JAX so the same functions trace under
``jax.jit`` (``n``, ``axis``, ``rank``, and ``kind`` are static).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NoReturn

import jax.numpy as jnp

Array = Any


def _raise(message: str) -> NoReturn:
    # parallel/__init__.py imports this module before MeshError is defined.
    from parallel import MeshError

    raise MeshError(message)


def _static_int(value: Any, label: str) -> int:
    # n, axis, and rank are static under jit. Reject tracers and bools here
    # rather than calling int() on them.
    if isinstance(value, bool) or not isinstance(value, int):
        _raise(f"{label} must be an integer, got {type(value).__name__}")
    return value


def _normalize_axis(axis: int, ndim: int) -> int:
    axis_i = _static_int(axis, "axis")
    if ndim <= 0 or axis_i < -ndim or axis_i >= ndim:
        _raise(f"axis {axis} out of range for ndim {ndim}")
    if axis_i < 0:
        axis_i += ndim
    return axis_i


def _check_n_rank(n: int, rank: int) -> tuple[int, int]:
    n_i = _static_int(n, "n")
    rank_i = _static_int(rank, "rank")
    if n_i < 1:
        _raise(f"n must be >= 1, got {n_i}")
    if rank_i < 0 or rank_i >= n_i:
        _raise(f"rank {rank_i} out of range for n={n_i}")
    return n_i, rank_i


def _as_jax(value: Array) -> Array:
    return jnp.asarray(value)


def _sequence(values: Sequence[Array], what: str) -> tuple[Array, ...]:
    try:
        n = len(values)
    except TypeError:
        _raise(f"{what} must be a non-empty sequence")
    if n < 1:
        _raise(f"{what} must be a non-empty sequence")
    return tuple(_as_jax(v) for v in values)


def _same_dtype_and_shape(arrays: Sequence[Array], *, scatter_axis: int | None) -> None:
    """Require one dtype. Shapes match on every axis, or every axis but ``scatter_axis``."""
    ref = arrays[0]
    for arr in arrays[1:]:
        if arr.dtype != ref.dtype:
            _raise(f"mixed dtypes {ref.dtype} vs {arr.dtype}")
        if arr.ndim != ref.ndim:
            _raise(f"mismatched ranks {ref.ndim} vs {arr.ndim}")
        for i in range(ref.ndim):
            if scatter_axis is not None and i == scatter_axis:
                continue
            if arr.shape[i] != ref.shape[i]:
                _raise(f"mismatched shapes {ref.shape} vs {arr.shape}")


def _split_bounds(dim: int, n: int, rank: int) -> tuple[int, int]:
    """``numpy.array_split`` bounds: the first ``dim % n`` ranks get one extra."""
    base, rem = divmod(dim, n)
    if rank < rem:
        start = rank * (base + 1)
        return start, start + base + 1
    start = rem * (base + 1) + (rank - rem) * base
    return start, start + base


def _slice_axis(param: Array, axis: int, start: int, stop: int) -> Array:
    indexer = [slice(None)] * param.ndim
    indexer[axis] = slice(start, stop)
    return param[tuple(indexer)]


def _cast(arr: Array, dtype: Any) -> Array:
    # Identity cast must keep signed zeros and NaN payloads.
    if arr.dtype == dtype:
        return arr
    return arr.astype(dtype)


def _acc_dtype(dtype: Any) -> Any:
    if dtype == jnp.bool_ or jnp.issubdtype(dtype, jnp.complexfloating):
        _raise(f"reduce-scatter does not support dtype {dtype}")
    if jnp.issubdtype(dtype, jnp.floating):
        # float16 / bfloat16 / float32 accumulate in float32; float64 stays float64.
        if jnp.dtype(dtype).itemsize <= 4:
            return jnp.float32
        return jnp.float64
    if jnp.issubdtype(dtype, jnp.integer):
        return jnp.int64
    _raise(f"reduce-scatter does not support dtype {dtype}")


def fsdp_shard(param: Array, n: int, axis: int, rank: int) -> Array:
    """Return rank's shard of ``param`` along ``axis``.

    Split rule is ``numpy.array_split``: the first ``dim % n`` ranks receive
    ``dim // n + 1`` elements, the rest ``dim // n``. ``axis`` may be negative
    and is normalized against ``param.ndim``. ``n == 1`` returns the values
    unchanged (object identity is not required).

    ``param`` may be a NumPy array or a JAX array. The result is a
    ``jax.numpy`` array of the same dtype. A zero-size shard is invalid, so
    ``dim < n`` raises even for a rank that would otherwise be non-empty.

    Raises ``parallel.MeshError`` if ``n < 1``, ``rank`` is not in
    ``[0, n)``, ``axis`` is out of range, or the split dimension is smaller
    than ``n``.
    """
    n_i, rank_i = _check_n_rank(n, rank)
    param = _as_jax(param)
    axis_i = _normalize_axis(axis, param.ndim)
    # Static shape: bounds are concrete Python ints, safe under jit.
    dim = param.shape[axis_i]
    if dim < n_i:
        _raise(f"axis {axis} length {dim} < n={n_i}; zero-size shard is invalid")
    start, stop = _split_bounds(dim, n_i, rank_i)
    return _slice_axis(param, axis_i, start, stop)


def fsdp_all_gather(shards: tuple[Array, ...] | list[Array], axis: int) -> Array:
    """Concatenate rank-ordered shards along ``axis``.

    Inverse of :func:`fsdp_shard` when ``shards[r]`` is rank ``r``'s shard of
    the same tensor: the gather is bitwise identical to the original, including
    for uneven splits. ``axis`` may be negative. An empty sequence, mixed
    dtypes, mismatched ranks, or an out-of-range axis raises ``parallel.MeshError``.
    """
    arrays = _sequence(shards, "shards")
    axis_i = _normalize_axis(axis, arrays[0].ndim)
    _same_dtype_and_shape(arrays, scatter_axis=axis_i)
    if len(arrays) == 1:
        return arrays[0]
    return jnp.concatenate(arrays, axis=axis_i)


def fsdp_reduce_scatter(partials: tuple[Array, ...] | list[Array], axis: int, rank: int) -> Array:
    """Sum partials, then return this rank's shard along ``axis``.

    The reduction is a sum, not a mean. Partials must share dtype and shape.
    Accumulation dtype is float32 for float32 / float16 / bfloat16 inputs,
    float64 for float64, and int64 for integer inputs. bool and complex raise
    ``parallel.MeshError``. The shard is cast back to the input dtype.

    The sum is a left fold in that accumulation dtype, not a pairwise tree
    and not ``jax.numpy.sum`` of a stack. For ``n > 2`` those are not bitwise
    identical::

        acc = cast(partials[0], acc_dtype)
        for p in partials[1:]:
            acc = acc + cast(p, acc_dtype)
        return cast(shard(acc), input_dtype)

    Empty partials, mixed dtypes or shapes, a bad axis or rank, or a split
    dimension smaller than ``len(partials)`` raise ``parallel.MeshError``.
    """
    arrays = _sequence(partials, "partials")
    _same_dtype_and_shape(arrays, scatter_axis=None)
    n_i, rank_i = _check_n_rank(len(arrays), rank)
    out_dtype = arrays[0].dtype
    acc_dtype = _acc_dtype(out_dtype)
    # Left fold. A tree sum (or jnp.sum on a stack) is not bitwise identical for n > 2.
    acc = _cast(arrays[0], acc_dtype)
    for part in arrays[1:]:
        acc = acc + _cast(part, acc_dtype)
    return _cast(fsdp_shard(acc, n_i, axis, rank_i), out_dtype)


def zero3_views(
    param: Array, kind: Any, n: int, axis: int, rank: int
) -> tuple[Array, Array]:
    """Return ``(gpu_view, grace_view)`` for one parameter.

    ``kind`` must be a ``parallel.ParamKind`` (a bare string raises
    ``parallel.MeshError``). Non-floating ``param`` raises ``parallel.MeshError``.

    ``ParamKind.ROUTED_EXPERT`` is not sharded: both views are the full tensor
    (experts are sharded by expert parallel, not FSDP). Every other kind is
    sharded with :func:`fsdp_shard` in the input dtype, then cast. Shard
    errors propagate from ``fsdp_shard``.

    ``gpu_view`` is bfloat16. ``grace_view`` is float32. Casts are bitwise
    identical to an independent cast of the selected values (shard first, then
    cast — not cast, then shard).
    """
    from parallel import ParamKind

    if not isinstance(kind, ParamKind):
        _raise(f"kind must be a parallel.ParamKind, got {type(kind).__name__}")
    param = _as_jax(param)
    if not jnp.issubdtype(param.dtype, jnp.floating):
        _raise(f"non-floating param dtype {param.dtype}")
    if kind is ParamKind.ROUTED_EXPERT:
        selected = param
    else:
        # Shard in the input dtype, then cast. Cast-then-shard is not bitwise identical.
        selected = fsdp_shard(param, n, axis, rank)
    return _cast(selected, jnp.bfloat16), _cast(selected, jnp.float32)
