"""Independent NumPy / JAX reference for FSDP ZeRO-3 collectives.

This oracle does not call ``parallel.fsdp_shard``, ``parallel.fsdp_all_gather``,
``parallel.fsdp_reduce_scatter``, or ``parallel.zero3_views``. The only
production names it imports are ``parallel.ParamKind`` and ``parallel.MeshError``.

Split (``numpy.array_split``)
-----------------------------
Along ``axis`` (negative axes normalized by ``ndim``), rank ``r`` of ``n``
receives the contiguous half-open slice whose length is::

    dim // n + 1    if r < dim % n
    dim // n        otherwise

The first ``dim % n`` ranks are the longer ones. Offsets are the prefix sums
of those lengths. ``n == 1`` is the full tensor (values unchanged; object
identity is not required). A dimension smaller than ``n`` is invalid: a
zero-size piece is not a ZeRO-3 shard.

Reduce-scatter accumulation
---------------------------
``fsdp_reduce_scatter(partials, axis, rank)`` equals
``shard(sum(partials), n, axis, rank)`` with this sum, then a cast of the
shard back to the input dtype. The op is sum, not mean.

* float32, float16, bfloat16 (floating itemsize <= 4): left fold in float32
* float64: left fold in float64
* integers: left fold in int64, then cast the shard back
* bool and complex: ``MeshError``

Left fold starts at the first partial. There is no leading zero, so one
partial keeps signed zeros::

    acc = cast(partials[0], acc_dtype)
    for p in partials[1:]:
        acc = acc + cast(p, acc_dtype)
    return cast(shard(acc, n, axis, rank), input_dtype)

Do not use a tree reduction (``numpy.sum`` / ``jax.numpy.sum`` on a stack).
For ``n > 2`` that is not bitwise identical to this left fold. Addition is
left-to-right in the order of ``partials``, not reversed.

zero3_views
-----------
``kind`` must be a ``parallel.ParamKind`` instance. A string or any other
type is ``MeshError``. A non-floating param is ``MeshError``.

``ROUTED_EXPERT`` is not sharded: both views are an independent cast of the
full tensor. Every other kind is sharded in the input dtype first, then
cast (shard-then-cast, not cast-then-shard). ``gpu`` is bfloat16. ``grace``
is float32.

Casts go through ``jax.numpy.astype`` (IEEE round-to-nearest-even). That is
the independent cast the contract requires; this module never asks production
for the cast.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from parallel import MeshError, ParamKind

Array = np.ndarray


def normalize_axis(axis: int, ndim: int) -> int:
    """Normalize a possibly negative axis. Raise ``MeshError`` if missing."""
    if ndim <= 0:
        raise MeshError(f"axis {axis} out of range for ndim {ndim}")
    try:
        axis_i = int(axis)
    except (TypeError, ValueError) as exc:
        raise MeshError(f"axis {axis} out of range for ndim {ndim}") from exc
    if axis_i < 0:
        axis_i += ndim
    if axis_i < 0 or axis_i >= ndim:
        raise MeshError(f"axis {axis} out of range for ndim {ndim}")
    return axis_i


def split_sizes(dim: int, n: int) -> list[int]:
    """``numpy.array_split`` lengths. First ``dim % n`` pieces are longer."""
    if n < 1:
        raise MeshError(f"n < 1 ({n})")
    base, rem = divmod(int(dim), int(n))
    return [base + 1] * rem + [base] * (n - rem)


def split_bounds(dim: int, n: int) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` for each rank along the split axis."""
    bounds: list[tuple[int, int]] = []
    start = 0
    for size in split_sizes(dim, n):
        bounds.append((start, start + size))
        start += size
    return bounds


def _as_numpy(param: Array) -> np.ndarray:
    return np.asarray(param)


def _check_n_rank(n: int, rank: int) -> tuple[int, int]:
    try:
        n_i = int(n)
        rank_i = int(rank)
    except (TypeError, ValueError) as exc:
        raise MeshError(f"n and rank must be integers, got n={n!r} rank={rank!r}") from exc
    if n_i < 1:
        raise MeshError(f"n < 1 ({n_i})")
    if rank_i < 0 or rank_i >= n_i:
        raise MeshError(f"rank {rank_i} out of range for n={n_i}")
    return n_i, rank_i


def _ensure_x64() -> None:
    import jax

    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)


def cast_dtype(arr: Array, dtype) -> np.ndarray:
    """Independent cast. Same rounding as ``jax.numpy.astype``.

    Identity casts (same dtype name) are a NumPy copy, so signed zeros and
    NaN payloads are preserved. Every real conversion goes through JAX.
    """
    src = np.asarray(arr)
    name = np.dtype(dtype).name
    if src.dtype.name == name:
        return np.array(src, copy=True)
    _ensure_x64()
    import jax.numpy as jnp

    target = jnp.bfloat16 if name == "bfloat16" else getattr(jnp, name)
    return np.asarray(jnp.asarray(src).astype(target))


def _is_floating(dtype) -> bool:
    dt = np.dtype(dtype)
    if dt.name == "bfloat16":
        return True
    return bool(np.issubdtype(dt, np.floating))


def _is_integer(dtype) -> bool:
    dt = np.dtype(dtype)
    if dt.name == "bool" or dt == np.dtype(bool):
        return False
    return bool(np.issubdtype(dt, np.integer))


def _is_bool_or_complex(dtype) -> bool:
    dt = np.dtype(dtype)
    if dt.name == "bool" or dt == np.dtype(bool):
        return True
    return bool(np.issubdtype(dt, np.complexfloating))


def _acc_dtype(dtype) -> np.dtype:
    """Accumulation dtype for reduce-scatter. ``MeshError`` on bool/complex."""
    dt = np.dtype(dtype)
    if _is_bool_or_complex(dt):
        raise MeshError(f"bool and complex partials are not summable, got {dt}")
    if dt.name == "bfloat16" or (np.issubdtype(dt, np.floating) and dt.itemsize <= 4):
        return np.dtype(np.float32)
    if np.issubdtype(dt, np.floating):
        return np.dtype(np.float64)
    if _is_integer(dt):
        return np.dtype(np.int64)
    raise MeshError(f"unsupported reduce-scatter dtype {dt}")


def fsdp_shard(param: Array, n: int, axis: int, rank: int) -> np.ndarray:
    """Rank's contiguous ``numpy.array_split`` piece. NumPy array, same dtype."""
    arr = _as_numpy(param)
    n_i, rank_i = _check_n_rank(n, rank)
    axis_i = normalize_axis(axis, arr.ndim)
    dim = int(arr.shape[axis_i])
    if dim < n_i:
        raise MeshError(f"dimension {dim} < n {n_i} on axis {axis_i}")
    start, stop = split_bounds(dim, n_i)[rank_i]
    sl = [slice(None)] * arr.ndim
    sl[axis_i] = slice(start, stop)
    return np.array(arr[tuple(sl)], copy=True)


def _check_sequence(parts: Sequence[Array], *, what: str) -> list[np.ndarray]:
    if parts is None:
        raise MeshError(f"empty {what}")
    try:
        arrays = [_as_numpy(p) for p in parts]
    except TypeError as exc:
        raise MeshError(f"empty {what}") from exc
    if len(arrays) < 1:
        raise MeshError(f"empty {what}")
    return arrays


def _check_same_dtype(arrays: Sequence[np.ndarray]) -> np.dtype:
    dt = arrays[0].dtype
    for arr in arrays[1:]:
        if arr.dtype != dt and arr.dtype.name != dt.name:
            raise MeshError(f"mixed dtypes {dt} vs {arr.dtype}")
    return dt


def _check_concat_shapes(arrays: Sequence[np.ndarray], axis: int) -> int:
    ndim = arrays[0].ndim
    axis_i = normalize_axis(axis, ndim)
    for arr in arrays[1:]:
        if arr.ndim != ndim:
            raise MeshError(
                f"mismatched ranks (ndim) {arrays[0].shape} vs {arr.shape}"
            )
        for i, (d0, d1) in enumerate(zip(arrays[0].shape, arr.shape, strict=True)):
            if i == axis_i:
                continue
            if d0 != d1:
                raise MeshError(
                    f"mismatched shapes {arrays[0].shape} vs {arr.shape} on axis {axis_i}"
                )
    return axis_i


def fsdp_all_gather(shards: Sequence[Array], axis: int) -> np.ndarray:
    """Concatenate rank-ordered shards along ``axis``. NumPy array."""
    arrays = _check_sequence(shards, what="shards")
    _check_same_dtype(arrays)
    axis_i = _check_concat_shapes(arrays, axis)
    return np.concatenate(arrays, axis=axis_i)


def _check_same_shapes(arrays: Sequence[np.ndarray]) -> None:
    shape = arrays[0].shape
    ndim = arrays[0].ndim
    for arr in arrays[1:]:
        if arr.ndim != ndim or arr.shape != shape:
            raise MeshError(f"mismatched partial shapes {shape} vs {arr.shape}")


def fsdp_reduce_scatter(partials: Sequence[Array], axis: int, rank: int) -> np.ndarray:
    """Left-fold sum in the accumulation dtype, then shard, then cast back."""
    arrays = _check_sequence(partials, what="partials")
    out_dtype = _check_same_dtype(arrays)
    _check_same_shapes(arrays)
    n_i, rank_i = _check_n_rank(len(arrays), rank)
    acc_dtype = _acc_dtype(out_dtype)
    axis_i = normalize_axis(axis, arrays[0].ndim)
    dim = int(arrays[0].shape[axis_i])
    if dim < n_i:
        raise MeshError(f"dimension {dim} < n {n_i} on axis {axis_i}")
    acc = cast_dtype(arrays[0], acc_dtype)
    for part in arrays[1:]:
        acc = acc + cast_dtype(part, acc_dtype)
    shard = fsdp_shard(acc, n_i, axis_i, rank_i)
    return cast_dtype(shard, out_dtype)


def zero3_views(
    param: Array,
    kind: ParamKind,
    n: int,
    axis: int,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(gpu_bfloat16, grace_float32)``. Shard in the input dtype, then cast.

    ``ROUTED_EXPERT`` skips the shard and casts the full tensor. Every other
    ``ParamKind`` calls this module's ``fsdp_shard`` (not production) and
    casts that piece.
    """
    if not isinstance(kind, ParamKind):
        raise MeshError(f"kind must be ParamKind, got {type(kind).__name__}")
    arr = _as_numpy(param)
    if not _is_floating(arr.dtype):
        raise MeshError(f"non-floating param dtype {arr.dtype}")
    if kind == ParamKind.ROUTED_EXPERT:
        selected = np.array(arr, copy=True)
    else:
        selected = fsdp_shard(arr, n, axis, rank)
    gpu = cast_dtype(selected, "bfloat16")
    grace = cast_dtype(selected, np.float32)
    return gpu, grace
