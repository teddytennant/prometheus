"""Slow NumPy reference for ring context-parallel attention.

Independent of production ``parallel``, ``kernels``, and jax. This is the
oracle for ``cp_split_seq``, ``cp_ring_schedule``, ``ring_attention_rank``,
and ``ring_attention`` (spec 5.1 context parallel; interface on ``parallel.cp``).

Schedule
--------
Length ``n``. Step 0 is ``rank``. Step ``t`` is ``(rank - t) mod n``.
``n=4, rank=1`` is ``(1, 0, 3, 2)``. ``n=1, rank=0`` is ``(0,)``.

Split
-----
Equal contiguous chunk of axis ``axis`` (negative axes allowed). ``n == 1``
returns the same values.

Ring attention
--------------
``q`` / ``k`` blocks are ``(batch, chunk, heads, dim)``. ``v`` may have a
different last dim. Owner ``i`` owns key/value positions
``[i * chunk, (i + 1) * chunk)``. Query positions for ``rank`` start at
``rank * chunk``. Blocks are stored by owner id, not ring order, and consumed
in ``cp_ring_schedule`` order.

Float32 online softmax (the ``1e-12`` floor matches ``model._softmax``; the
causal sentinel ``-1e9`` matches ``mla_attention``)::

    scale = 1/sqrt(dim) if scale is None else scale   # Python float, then f32
    scores = einsum("bqhd,bkhd->bhqk", q, k_block) * scale
    causal: score at global key index > global query index is replaced by -1e9
    state starts at m=-inf, l=0, acc=0
    m_new = max(m, rowmax(scores))
    alpha = exp(m - m_new)
    p = exp(scores - m_new)
    l = alpha * l + sum(p)
    acc = alpha * acc + einsum("bhqk,bkhd->bhqd", p, v_block)
    output = acc / max(l, 1e-12)   # then transpose to (batch, chunk, heads, dim_v)

``n == 1`` of this fold equals ``one_shot_attention`` bitwise in this file.
JAX ``exp`` / ``sum`` are not bit-identical to NumPy; tests apply the
tolerances in ``tests/test_parallel_cp.py``.

Do not import production ``parallel``, ``kernels``, or jax from this file.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# Same sentinel as mla_attention. Same floor as model._softmax.
CAUSAL_MASK_SCORE = np.float32(-1e9)
SOFTMAX_FLOOR = np.float32(1e-12)


def default_scale(dim: int) -> np.float32:
    """``1 / sqrt(dim)`` as a Python float, then float32."""
    return np.float32(1.0 / math.sqrt(int(dim)))


def _as_f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _normalize_axis(axis: int, ndim: int) -> int:
    ax = axis + ndim if axis < 0 else axis
    if ax < 0 or ax >= ndim:
        raise ValueError(f"axis {axis} out of range for ndim {ndim}")
    return ax


def cp_ring_schedule(n: int, rank: int) -> tuple[int, ...]:
    """Ring order for this rank. Step ``t`` is owner ``(rank - t) mod n``."""
    if isinstance(n, bool) or isinstance(rank, bool):
        raise ValueError("n and rank must be integers")
    n = int(n)
    rank = int(rank)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if rank < 0 or rank >= n:
        raise ValueError(f"rank {rank} not in [0, {n})")
    return tuple((rank - t) % n for t in range(n))


def cp_split_seq(x, n: int, rank: int, axis: int = 1) -> np.ndarray:
    """Equal contiguous chunk of ``x`` along ``axis`` for this rank."""
    arr = np.asarray(x)
    if isinstance(n, bool) or isinstance(rank, bool) or isinstance(axis, bool):
        raise ValueError("n, rank, and axis must be integers")
    n = int(n)
    rank = int(rank)
    axis = int(axis)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if rank < 0 or rank >= n:
        raise ValueError(f"rank {rank} not in [0, {n})")
    ax = _normalize_axis(axis, arr.ndim)
    dim = int(arr.shape[ax])
    if dim < n or dim % n != 0:
        raise ValueError(f"axis {axis} size {dim} is not divisible by n={n}")
    chunk = dim // n
    start = rank * chunk
    sl = [slice(None)] * arr.ndim
    sl[ax] = slice(start, start + chunk)
    return arr[tuple(sl)]


def _scale_of(dim: int, scale: float | None) -> np.float32:
    if scale is None:
        return default_scale(dim)
    return np.float32(scale)


def _causal_mask(scores: np.ndarray, q_index: np.ndarray, k_index: np.ndarray) -> np.ndarray:
    """Replace scores whose global key index is greater than the query index."""
    # scores (batch, heads, query, key); indices broadcast over the last two axes.
    mask = k_index.reshape(1, -1) > q_index.reshape(-1, 1)
    return np.where(mask, CAUSAL_MASK_SCORE, scores)


def one_shot_attention(
    q,
    k,
    v,
    *,
    causal: bool = True,
    scale: float | None = None,
) -> np.ndarray:
    """Full-sequence softmax attention in float32. Shape ``(batch, seq, heads, dim_v)``.

    Causal mask uses positions on the sequence axis (axis 1). ``n == 1`` ring
    attention must match this bitwise inside this file (same floor, same sentinel).
    """
    q_a = _as_f32(q)
    k_a = _as_f32(k)
    v_a = _as_f32(v)
    _b, sq, _h, dim = q_a.shape
    sc = _scale_of(dim, scale)
    scores = np.einsum("bqhd,bkhd->bhqk", q_a, k_a) * sc
    if causal:
        scores = _causal_mask(
            scores,
            np.arange(sq, dtype=np.int64),
            np.arange(k_a.shape[1], dtype=np.int64),
        )
    m = np.max(scores, axis=-1, keepdims=True)
    p = np.exp(scores - m)
    ell = np.sum(p, axis=-1, keepdims=True)
    acc = np.einsum("bhqk,bkhd->bhqd", p, v_a)
    out = acc / np.maximum(ell, SOFTMAX_FLOOR)
    return np.transpose(out, (0, 2, 1, 3))


def ring_attention_rank(
    q_local,
    k_by_owner: Sequence,
    v_by_owner: Sequence,
    *,
    rank: int,
    causal: bool = True,
    scale: float | None = None,
) -> np.ndarray:
    """Online-softmax attention for one rank's query chunk.

    ``k_by_owner[i]`` / ``v_by_owner[i]`` are owner ``i``'s blocks. The walk
    follows ``cp_ring_schedule(n, rank)``, not owner order.
    """
    q_a = _as_f32(q_local)
    batch, chunk, heads, dim = q_a.shape
    n = len(k_by_owner)
    if len(v_by_owner) != n:
        raise ValueError("k_by_owner and v_by_owner lengths differ")
    order = cp_ring_schedule(n, rank)
    v0 = _as_f32(v_by_owner[0])
    dim_v = int(v0.shape[-1])
    sc = _scale_of(dim, scale)

    # Running state. m/l/acc stay in (batch, heads, query, ...) so they
    # broadcast against scores (batch, heads, query, key).
    m = np.full((batch, heads, chunk, 1), -np.inf, dtype=np.float32)
    ell = np.zeros((batch, heads, chunk, 1), dtype=np.float32)
    acc = np.zeros((batch, heads, chunk, dim_v), dtype=np.float32)
    q_index = rank * chunk + np.arange(chunk, dtype=np.int64)

    for owner in order:
        k_block = _as_f32(k_by_owner[owner])
        v_block = _as_f32(v_by_owner[owner])
        scores = np.einsum("bqhd,bkhd->bhqk", q_a, k_block) * sc
        if causal:
            k_index = owner * chunk + np.arange(k_block.shape[1], dtype=np.int64)
            scores = _causal_mask(scores, q_index, k_index)
        rowmax = np.max(scores, axis=-1, keepdims=True)
        m_new = np.maximum(m, rowmax)
        alpha = np.exp(m - m_new)
        p = np.exp(scores - m_new)
        ell = alpha * ell + np.sum(p, axis=-1, keepdims=True)
        acc = alpha * acc + np.einsum("bhqk,bkhd->bhqd", p, v_block)
        m = m_new

    out = acc / np.maximum(ell, SOFTMAX_FLOOR)
    return np.transpose(out, (0, 2, 1, 3))


def ring_attention(
    q,
    k,
    v,
    *,
    n: int,
    causal: bool = True,
    scale: float | None = None,
) -> np.ndarray:
    """Split on sequence, run ``ring_attention_rank`` per rank, concat on axis 1."""
    if isinstance(n, bool):
        raise ValueError("n must be an integer")
    n = int(n)
    q_chunks = [cp_split_seq(q, n, r, axis=1) for r in range(n)]
    k_owners = [cp_split_seq(k, n, r, axis=1) for r in range(n)]
    v_owners = [cp_split_seq(v, n, r, axis=1) for r in range(n)]
    parts = [
        ring_attention_rank(
            q_chunks[r],
            k_owners,
            v_owners,
            rank=r,
            causal=causal,
            scale=scale,
        )
        for r in range(n)
    ]
    return np.concatenate(parts, axis=1)
