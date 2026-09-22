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

from collections.abc import Sequence
from typing import Any

Array = Any


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
    raise NotImplementedError("cp_split_seq")


def cp_ring_schedule(n: int, rank: int) -> tuple[int, ...]:
    """KV-owner index this rank sees at each ring step.

    Length ``n``. Step 0 is this rank's own block. KV travels toward
    increasing rank, so step ``t`` sees owner ``(rank - t) mod n``.

    Raises ``parallel.MeshError`` if ``n < 1`` or ``rank`` is not in
    ``[0, n)``. Both arguments are Python ints. ``n == 1`` returns
    ``(0,)`` for rank 0.
    """
    raise NotImplementedError("cp_ring_schedule")


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
    raise NotImplementedError("ring_attention_rank")


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
    raise NotImplementedError("ring_attention")
