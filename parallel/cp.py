"""Ring context parallel (spec 5.1, 5.2). CPU analog of the NVLink ring.

Context parallel is off in pretrain (CP=1) and on for the 256k/1M
mid-training phases. This module is the ring attention that those phases
run. It does not launch NCCL. Ranks are integers. A multi-rank call passes
every rank's K/V block in owner order.

The concatenated result must match one-shot causal (or non-causal)
attention. The block order is the ring, not a single softmax over the
full sequence, so an implementation that only slices a dense attention
is not this contract.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

Array = Any

# Same sentinels as model._softmax / mla_attention. Float32 bits, not Python floats,
# so the n==1 fold matches one-shot attention bitwise.
_CAUSAL_MASK = np.float32(-1e9)
_SOFTMAX_FLOOR = np.float32(1e-12)


def _raise_mesh(message: str) -> None:
    # Lazy: parallel/__init__.py imports this module before MeshError exists.
    from parallel import MeshError

    raise MeshError(message)


def _static_int(value: int, name: str) -> int:
    """Reject tracers and bools. n/rank/axis are static under jit."""
    if isinstance(value, bool) or not isinstance(value, int):
        _raise_mesh(f"{name} must be a Python int")
    return value


def cp_split_seq(x: Array, n: int, rank: int, *, axis: int = 1) -> Array:
    """Equal contiguous chunk of ``x`` along ``axis`` for this CP rank.

    ``n`` is the CP world size. ``axis`` defaults to 1, the sequence axis
    of ``(batch, seq, heads, dim)``. ``rank`` is in ``[0, n)``. The split
    size is ``dim // n``; ``dim % n`` must be 0. Unequal chunks are not a
    valid ring piece.

    Raises ``parallel.MeshError`` if ``n < 1``, ``rank`` is out of range,
    ``axis`` is out of range, the dimension is smaller than ``n``, or the
    dimension is not divisible by ``n``. ``axis`` may be negative and is
    normalized against ``ndim``.

    ``n == 1`` returns ``x`` unchanged in values (identity of the object
    is not required).

    Accepts a NumPy array or a JAX array. Returns a ``jax.numpy`` array of
    the same dtype. ``n``, ``rank``, and ``axis`` are static under
    ``jax.jit``: traced values must not be turned into NumPy or Python
    scalars, and the jitted result must match eager bitwise.
    """
    n = _static_int(n, "n")
    rank = _static_int(rank, "rank")
    axis = _static_int(axis, "axis")
    if n < 1:
        _raise_mesh(f"n must be >= 1, got {n}")
    if rank < 0 or rank >= n:
        _raise_mesh(f"rank {rank} out of range for n={n}")
    arr = jnp.asarray(x)
    ax = axis + arr.ndim if axis < 0 else axis
    if ax < 0 or ax >= arr.ndim:
        _raise_mesh(f"axis {axis} out of range for ndim={arr.ndim}")
    # Shape ints are static under jit; do not int() a traced value.
    dim = arr.shape[ax]
    if dim < n:
        _raise_mesh(f"dimension {dim} is smaller than n={n}")
    if dim % n != 0:
        _raise_mesh(f"dimension {dim} is not divisible by n={n}")
    if n == 1:
        return arr
    chunk = dim // n
    start = rank * chunk
    slices = [slice(None)] * arr.ndim
    slices[ax] = slice(start, start + chunk)
    return arr[tuple(slices)]


def cp_ring_schedule(n: int, rank: int) -> tuple[int, ...]:
    """KV-owner index this rank sees at each ring step.

    Length ``n``. Step 0 is this rank's own block. KV travels toward
    increasing rank, so step ``t`` sees owner ``(rank - t) mod n``.

    Raises ``parallel.MeshError`` if ``n < 1`` or ``rank`` is not in
    ``[0, n)``. Both arguments are Python ints. ``n == 1`` returns
    ``(0,)`` for rank 0.
    """
    n = _static_int(n, "n")
    rank = _static_int(rank, "rank")
    if n < 1:
        _raise_mesh(f"n must be >= 1, got {n}")
    if rank < 0 or rank >= n:
        _raise_mesh(f"rank {rank} out of range for n={n}")
    return tuple((rank - t) % n for t in range(n))


def _as_f32(x: Array, name: str) -> Array:
    arr = jnp.asarray(x)
    if not jnp.issubdtype(arr.dtype, jnp.floating):
        _raise_mesh(f"{name} must be a floating dtype, got {arr.dtype}")
    return jnp.asarray(arr, dtype=jnp.float32)


def _scale_factor(dim: int, scale: float | None) -> np.float32:
    if scale is None:
        return np.float32(1.0 / math.sqrt(dim))
    return np.float32(scale)


def _check_block(
    block: Array,
    name: str,
    batch: int,
    chunk: int,
    heads: int,
    feat: int | None,
) -> Array:
    arr = _as_f32(block, name)
    if arr.ndim != 4:
        _raise_mesh(f"{name} must be rank 4, got {arr.ndim}")
    b, seq, h, d = arr.shape
    if b != batch:
        _raise_mesh(f"{name} batch {b} != {batch}")
    if seq != chunk:
        _raise_mesh(f"{name} sequence {seq} != chunk {chunk}")
    if h != heads:
        _raise_mesh(f"{name} heads {h} != {heads}")
    if feat is not None and d != feat:
        _raise_mesh(f"{name} feature dim {d} != {feat}")
    return arr


def _online_fold(
    q: Array,
    ks: tuple[Array, ...],
    vs: tuple[Array, ...],
    schedule: tuple[int, ...],
    *,
    rank: int,
    causal: bool,
    scale_f: np.float32,
) -> Array:
    """Online softmax in ring order. n==1 is one step of this fold."""
    batch, chunk, heads, _dim = q.shape
    dim_v = vs[0].shape[-1]
    # (B, H, Q, ...) matches the score einsum. Init is the spec state:
    # m=-inf, l=0, acc=0. The first step's alpha is exp(-inf)=0, so it is
    # the one-shot softmax bitwise (no extra dense softmax).
    m = jnp.full((batch, heads, chunk, 1), -jnp.inf, dtype=jnp.float32)
    ell = jnp.zeros((batch, heads, chunk, 1), dtype=jnp.float32)
    acc = jnp.zeros((batch, heads, chunk, dim_v), dtype=jnp.float32)
    q_index = rank * chunk + jnp.arange(chunk)
    mask_score = jnp.asarray(_CAUSAL_MASK)
    for owner in schedule:
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, ks[owner]) * scale_f
        if causal:
            k_index = owner * chunk + jnp.arange(chunk)
            scores = jnp.where(k_index[None, :] > q_index[:, None], mask_score, scores)
        rowmax = jnp.max(scores, axis=-1, keepdims=True)
        m_new = jnp.maximum(m, rowmax)
        alpha = jnp.exp(m - m_new)
        p = jnp.exp(scores - m_new)
        ell = alpha * ell + jnp.sum(p, axis=-1, keepdims=True)
        acc = alpha * acc + jnp.einsum("bhqk,bkhd->bhqd", p, vs[owner])
        m = m_new
    out = acc / jnp.maximum(ell, jnp.asarray(_SOFTMAX_FLOOR))
    return jnp.transpose(out, (0, 2, 1, 3))


def ring_attention_rank(
    q_local: Array,
    k_by_owner: Sequence[Array],
    v_by_owner: Sequence[Array],
    *,
    rank: int,
    causal: bool = True,
    scale: float | None = None,
) -> Array:
    """This rank's ring-attention output.

    ``q_local`` is ``(batch, chunk, heads, dim)``, this rank's queries.
    ``k_by_owner[i]`` and ``v_by_owner[i]`` are the blocks owned by rank
    ``i``, not in ring order. Both sequences have length ``n`` (the CP
    world size). Each block is ``(batch, chunk, heads, dim_k)`` for K and
    ``(batch, chunk, heads, dim_v)`` for V. ``dim_v`` may differ from
    ``dim`` (MLA stores V as the nope part). Heads must already match;
    this function does not broadcast a head axis of 1.

    Global query offset is ``rank * chunk``. Global key offset of owner
    ``i`` is ``i * chunk``. The function walks owners in
    ``cp_ring_schedule(n, rank)`` order and folds online softmax:

    - compute in float32
    - ``scale`` defaults to ``1 / sqrt(dim)``; a passed value is a Python
      float and is static under ``jax.jit``
    - scores are ``einsum("bqhd,bkhd->bhqk", q, k_block) * scale``
    - if ``causal``, a score whose global key index is greater than its
      global query index is replaced with ``-1e9`` (same sentinel as
      ``mla_attention``). Non-causal leaves scores unchanged
    - running state starts at ``m = -inf``, ``l = 0``, ``acc = 0``
    - for each block, ``m_new = max(m, rowmax(scores))``,
      ``alpha = exp(m - m_new)``, ``p = exp(scores - m_new)``,
      ``l = alpha * l + sum(p)``, ``acc = alpha * acc + p @ v_block``
    - output is ``acc / max(l, 1e-12)``, float32, shape
      ``(batch, chunk, heads, dim_v)``

    ``n == 1`` is one block, so the result matches one-shot softmax
    attention bitwise in float32 (the ``1e-12`` floor matches
    ``model._softmax``). For ``n > 1`` the block order is the ring, so
    the result must match this online fold, not a single softmax over
    the concatenated keys. It must also stay within ``1e-4`` max abs of
    that one-shot softmax on float32 inputs with entries in ``[-1, 1]``.

    Raises ``parallel.MeshError`` if ``rank`` is out of range, the owner
    sequences differ in length, a block sequence axis is not ``chunk``,
    batch or heads disagree, or an input is not a floating dtype.
    ``rank`` and ``causal`` are static under ``jax.jit``. Traced arrays
    must not be turned into NumPy or Python scalars. The jitted result
    must match eager within ``1e-6`` max abs.
    """
    rank = _static_int(rank, "rank")
    if len(k_by_owner) != len(v_by_owner):
        _raise_mesh("K and V owner sequences differ in length")
    n = len(k_by_owner)
    if n < 1 or rank < 0 or rank >= n:
        _raise_mesh(f"rank {rank} out of range for n={n}")
    q = _as_f32(q_local, "q")
    if q.ndim != 4:
        _raise_mesh(f"q must be rank 4, got {q.ndim}")
    batch, chunk, heads, dim = q.shape
    ks = tuple(
        _check_block(block, f"k[{i}]", batch, chunk, heads, dim)
        for i, block in enumerate(k_by_owner)
    )
    vs = tuple(
        _check_block(block, f"v[{i}]", batch, chunk, heads, None)
        for i, block in enumerate(v_by_owner)
    )
    dim_v = vs[0].shape[-1]
    for i, block in enumerate(vs):
        if block.shape[-1] != dim_v:
            _raise_mesh(f"v[{i}] feature dim {block.shape[-1]} != {dim_v}")
    scale_f = _scale_factor(dim, scale)
    return _online_fold(
        q,
        ks,
        vs,
        cp_ring_schedule(n, rank),
        rank=rank,
        causal=causal,
        scale_f=scale_f,
    )


def ring_attention(
    q: Array,
    k: Array,
    v: Array,
    *,
    n: int,
    causal: bool = True,
    scale: float | None = None,
) -> Array:
    """Full-sequence ring attention, concatenated in rank order.

    ``q``, ``k``, ``v`` are unsharded ``(batch, seq, heads, dim)`` (V's
    last dim may differ). ``seq`` must be divisible by ``n``. Split with
    ``cp_split_seq``, run ``ring_attention_rank`` for each rank, and
    concatenate on the sequence axis.

    The result has shape ``(batch, seq, heads, dim_v)`` and dtype
    float32. ``n == 1`` matches one-shot attention bitwise. ``n > 1``
    matches the online ring fold, and stays within ``1e-4`` max abs of
    one-shot attention on float32 inputs with entries in ``[-1, 1]``.

    Raises ``parallel.MeshError`` on the same conditions as
    ``cp_split_seq`` and ``ring_attention_rank``, and if ``q`` and ``k``
    sequence lengths differ. ``n`` and ``causal`` are static under
    ``jax.jit``.
    """
    n = _static_int(n, "n")
    if n < 1:
        _raise_mesh(f"n must be >= 1, got {n}")
    q_arr = jnp.asarray(q)
    k_arr = jnp.asarray(k)
    v_arr = jnp.asarray(v)
    for name, arr in (("q", q_arr), ("k", k_arr), ("v", v_arr)):
        if arr.ndim != 4:
            _raise_mesh(f"{name} must be rank 4, got {arr.ndim}")
    if q_arr.shape[1] != k_arr.shape[1]:
        _raise_mesh(
            f"q and k sequence lengths differ ({q_arr.shape[1]} vs {k_arr.shape[1]})"
        )
    q_parts = tuple(cp_split_seq(q_arr, n, r) for r in range(n))
    k_parts = tuple(cp_split_seq(k_arr, n, r) for r in range(n))
    v_parts = tuple(cp_split_seq(v_arr, n, r) for r in range(n))
    outs = [
        ring_attention_rank(
            q_parts[r],
            k_parts,
            v_parts,
            rank=r,
            causal=causal,
            scale=scale,
        )
        for r in range(n)
    ]
    if n == 1:
        return outs[0]
    return jnp.concatenate(outs, axis=1)
