"""CPU tests for ring context-parallel attention.

Public interface under test (imported from ``parallel``, not ``parallel.cp``):
``cp_split_seq``, ``cp_ring_schedule``, ``ring_attention_rank``, ``ring_attention``.

Groups
------
schedule
    Goldens ``n=4, rank=1 -> (1, 0, 3, 2)`` and ``n=1, rank=0 -> (0,)``,
    the ``(rank - t) mod n`` formula, and MeshError on a bad ``n`` / ``rank``.
split
    ``n == 1`` identity, equal contiguous chunks (including a negative axis),
    dtype preserved, JAX array out, jit bitwise with ``n, rank, axis`` static,
    MeshError on a bad split.
ring numeric
    ``n == 1`` matches one-shot softmax bitwise in float32 (JAX transcription of
    ``one_shot_attention``; NumPy ``exp`` is not bit-identical, so the NumPy
    oracle is checked at atol/rtol 1e-5). Causal and non-causal, including a
    V last-dim that differs from Q/K. ``n == 4`` matches the NumPy online fold
    at 1e-5 and is within 1e-4 max abs of one-shot on float32 inputs in [-1, 1].
    Rank 0 and rank 1 are each checked against the reference schedule walk, not
    against each other. A future key block does not move earlier causal queries
    by more than 1e-5 (also compared to the reference).
jit
    Split is bitwise. Ring attention (``n``, ``causal`` static) and
    ``ring_attention_rank`` (``rank``, ``causal`` static) match eager within
    1e-6 max abs.
faults
    MeshError on the conditions in the interface docstrings.
grad
    ``jax.grad`` of ``ring_attention`` matches a finite difference of the
    NumPy oracle (atol/rtol 1e-3).

Not marked ``gpu``. Do not loosen these tolerances.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import parallel
from tests.reference import parallel_cp as ref_cp

# Repo FP32 parity (tests/reference/parallel.py). Not a substitute for the
# tighter spec bounds below.
FP32_RTOL = 1e-5
FP32_ATOL = 1e-5
# Spec: n>1 within 1e-4 max abs of one-shot on float32 inputs in [-1, 1].
ONE_SHOT_MAX_ABS = 1e-4
# Spec: jitted ring attention vs eager within 1e-6 max abs.
JIT_MAX_ABS = 1e-6
# Causal property: a future key block must not move earlier queries by more.
CAUSAL_MAX_ABS = 1e-5
# Finite-difference check of jax.grad against the NumPy oracle (float32, eps=1e-3).
GRAD_ATOL = 1e-3
GRAD_RTOL = 1e-3


def _max_abs(got, exp) -> float:
    return float(np.max(np.abs(np.asarray(got) - np.asarray(exp))))


def _assert_max_abs(got, exp, tol: float) -> None:
    err = _max_abs(got, exp)
    assert err <= tol, err


def _assert_f32_parity(got, exp) -> None:
    np.testing.assert_allclose(np.asarray(got), np.asarray(exp), rtol=FP32_RTOL, atol=FP32_ATOL)


def _assert_jax_f32(out, shape: tuple[int, ...]) -> None:
    assert isinstance(out, jax.Array), type(out)
    assert out.dtype == jnp.float32, out.dtype
    assert tuple(out.shape) == shape, (out.shape, shape)
    assert np.isfinite(np.asarray(out)).all()


def _qkv(
    batch: int,
    seq: int,
    heads: int,
    dim: int,
    dim_v: int,
    seed: int,
    *,
    low: float = -1.0,
    high: float = 1.0,
    dtype=np.float32,
):
    rng = np.random.default_rng(seed)
    q = rng.uniform(low, high, size=(batch, seq, heads, dim)).astype(dtype)
    k = rng.uniform(low, high, size=(batch, seq, heads, dim)).astype(dtype)
    v = rng.uniform(low, high, size=(batch, seq, heads, dim_v)).astype(dtype)
    return q, k, v


def _jax_one_shot(q, k, v, *, causal: bool, scale: float | None = None):
    """JAX transcription of ``ref_cp.one_shot_attention`` (same formula).

    ``n == 1`` ring attention must match this bitwise in float32. NumPy ``exp``
    and ``sum`` are not bit-identical to JAX, so the bitwise check cannot be
    against the NumPy array itself.
    """
    q_a = jnp.asarray(q, dtype=jnp.float32)
    k_a = jnp.asarray(k, dtype=jnp.float32)
    v_a = jnp.asarray(v, dtype=jnp.float32)
    dim = int(q_a.shape[-1])
    sc = ref_cp.default_scale(dim) if scale is None else np.float32(scale)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q_a, k_a) * sc
    if causal:
        q_index = jnp.arange(q_a.shape[1])
        k_index = jnp.arange(k_a.shape[1])
        scores = jnp.where(
            k_index[None, :] > q_index[:, None],
            jnp.asarray(ref_cp.CAUSAL_MASK_SCORE),
            scores,
        )
    m = jnp.max(scores, axis=-1, keepdims=True)
    p = jnp.exp(scores - m)
    ell = jnp.sum(p, axis=-1, keepdims=True)
    acc = jnp.einsum("bhqk,bkhd->bhqd", p, v_a)
    out = acc / jnp.maximum(ell, jnp.asarray(ref_cp.SOFTMAX_FLOOR))
    return jnp.transpose(out, (0, 2, 1, 3))


def _assert_n1_matches_one_shot(got, q, k, v, *, causal: bool, scale: float | None = None) -> None:
    shot = _jax_one_shot(q, k, v, causal=causal, scale=scale)
    ref = ref_cp.one_shot_attention(q, k, v, causal=causal, scale=scale)
    # Spec: n == 1 matches one-shot softmax bitwise in float32.
    np.testing.assert_array_equal(np.asarray(got), np.asarray(shot))
    _assert_f32_parity(got, ref)
    ref_fold = ref_cp.ring_attention(q, k, v, n=1, causal=causal, scale=scale)
    np.testing.assert_array_equal(ref, ref_fold)


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------


def test_cp_ring_schedule_golden_n4_rank1():
    got = parallel.cp_ring_schedule(4, 1)
    assert got == (1, 0, 3, 2)
    assert got == ref_cp.cp_ring_schedule(4, 1)


def test_cp_ring_schedule_golden_n1():
    got = parallel.cp_ring_schedule(1, 0)
    assert got == (0,)
    assert got == ref_cp.cp_ring_schedule(1, 0)


@pytest.mark.parametrize(
    "n,rank",
    [(1, 0), (2, 0), (2, 1), (3, 2), (4, 0), (4, 1), (4, 3), (5, 4), (8, 5)],
)
def test_cp_ring_schedule_matches_reference(n, rank):
    got = parallel.cp_ring_schedule(n, rank)
    assert isinstance(got, tuple)
    assert all(type(step) is int for step in got)
    assert len(got) == n
    assert got[0] == rank
    assert got == tuple((rank - t) % n for t in range(n))
    assert sorted(got) == list(range(n))
    assert got == ref_cp.cp_ring_schedule(n, rank)


@pytest.mark.parametrize(
    "n,rank",
    [(0, 0), (-1, 0), (-4, 0), (1, -1), (1, 1), (4, -1), (4, 4), (4, 5)],
)
def test_cp_ring_schedule_mesh_error(n, rank):
    with pytest.raises(parallel.MeshError):
        parallel.cp_ring_schedule(n, rank)


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_cp_split_seq_n1_identity(axis):
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    got = parallel.cp_split_seq(x, 1, 0, axis=axis)
    assert isinstance(got, jax.Array)
    assert got.dtype == x.dtype
    np.testing.assert_array_equal(np.asarray(got), np.asarray(x))
    np.testing.assert_array_equal(np.asarray(got), ref_cp.cp_split_seq(x, 1, 0, axis=axis))


def test_cp_split_seq_default_axis_is_1():
    x = jnp.arange(2 * 6 * 3, dtype=jnp.float32).reshape(2, 6, 3)
    got = parallel.cp_split_seq(x, 3, 2)
    assert isinstance(got, jax.Array)
    assert got.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(got), np.asarray(x[:, 4:6, :]))
    np.testing.assert_array_equal(np.asarray(got), ref_cp.cp_split_seq(x, 3, 2, axis=1))


@pytest.mark.parametrize(
    "shape,n,axis",
    [
        ((2, 8, 4), 4, 1),
        ((2, 8, 4), 2, 0),
        ((2, 8, 4), 4, 2),
        ((2, 8, 4), 4, -1),
        ((2, 8, 4), 2, -2),
        ((2, 8, 4), 2, -3),
        ((6,), 3, 0),
        ((6,), 3, -1),
        ((3, 4, 2, 8), 4, -1),
    ],
)
def test_cp_split_seq_equal_chunks_match_reference(shape, n, axis):
    x = jnp.arange(int(np.prod(shape)), dtype=jnp.int32).reshape(shape)
    for rank in range(n):
        got = parallel.cp_split_seq(x, n, rank, axis=axis)
        exp = ref_cp.cp_split_seq(x, n, rank, axis=axis)
        assert isinstance(got, jax.Array)
        assert got.dtype == x.dtype
        assert tuple(got.shape) == exp.shape
        np.testing.assert_array_equal(np.asarray(got), exp)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.int32])
def test_cp_split_seq_preserves_dtype(dtype):
    x = jnp.arange(16, dtype=dtype).reshape(2, 8)
    got = parallel.cp_split_seq(x, 4, 3, axis=1)
    assert isinstance(got, jax.Array)
    assert got.dtype == dtype
    np.testing.assert_array_equal(np.asarray(got), np.asarray(x[:, 6:8]))


def test_cp_split_seq_numpy_input_returns_jax_array():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    got = parallel.cp_split_seq(x, 2, 1, axis=1)
    assert isinstance(got, jax.Array)
    assert got.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(got), x[:, 2:4])


def test_cp_split_seq_negative_axis_matches_positive():
    x = jnp.arange(24, dtype=jnp.int32).reshape(2, 3, 4)
    neg = parallel.cp_split_seq(x, 2, 1, axis=-1)
    pos = parallel.cp_split_seq(x, 2, 1, axis=2)
    np.testing.assert_array_equal(np.asarray(neg), np.asarray(pos))
    np.testing.assert_array_equal(np.asarray(neg), ref_cp.cp_split_seq(x, 2, 1, axis=-1))


def test_cp_split_seq_jit_bitwise():
    x = jnp.arange(2 * 8 * 4, dtype=jnp.float32).reshape(2, 8, 4)

    def _split(arr, n, rank, axis):
        # ``axis`` is keyword-only on the public function.
        return parallel.cp_split_seq(arr, n, rank, axis=axis)

    eager = _split(x, 4, 2, -1)
    jit_neg = jax.jit(_split, static_argnums=(1, 2, 3))(x, 4, 2, -1)
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(jit_neg))

    def _default_axis(arr, n, rank):
        return parallel.cp_split_seq(arr, n, rank)

    eager_default = parallel.cp_split_seq(x, 2, 1)
    jit_default = jax.jit(_default_axis, static_argnums=(1, 2))(x, 2, 1)
    np.testing.assert_array_equal(np.asarray(eager_default), np.asarray(jit_default))
    np.testing.assert_array_equal(np.asarray(jit_default), ref_cp.cp_split_seq(x, 2, 1, axis=1))


@pytest.mark.parametrize(
    "n,rank,axis,shape",
    [
        (0, 0, 1, (2, 4, 2)),
        (-1, 0, 1, (2, 4, 2)),
        (2, -1, 1, (2, 4, 2)),
        (2, 2, 1, (2, 4, 2)),
        (2, 5, 1, (2, 4, 2)),
        (2, 0, 3, (2, 4, 2)),
        (2, 0, -4, (2, 4, 2)),
        (4, 0, 1, (2, 3, 2)),  # dim < n
        (4, 0, 1, (2, 6, 2)),  # dim % n != 0
        (4, 0, 0, (3, 8, 2)),  # axis 0 dim < n
        (3, 0, -1, (2, 4, 5)),  # last dim % n != 0
        (2, 0, 1, (8,)),  # axis out of range on a vector
    ],
)
def test_cp_split_seq_mesh_error(n, rank, axis, shape):
    x = jnp.ones(shape, dtype=jnp.float32)
    with pytest.raises(parallel.MeshError):
        parallel.cp_split_seq(x, n, rank, axis=axis)


# ---------------------------------------------------------------------------
# ring numeric
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "batch,seq,heads,dim,dim_v,seed",
    [
        (2, 4, 3, 8, 8, 0),
        (2, 4, 3, 8, 5, 1),  # V last-dim differs from Q/K
        (1, 2, 1, 1, 3, 2),
    ],
)
def test_ring_n1_bitwise_vs_one_shot(causal, batch, seq, heads, dim, dim_v, seed):
    q, k, v = _qkv(batch, seq, heads, dim, dim_v, seed)
    q_j, k_j, v_j = jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)
    got_rank = parallel.ring_attention_rank(q_j, (k_j,), (v_j,), rank=0, causal=causal)
    got = parallel.ring_attention(q_j, k_j, v_j, n=1, causal=causal)
    for out in (got_rank, got):
        _assert_jax_f32(out, (batch, seq, heads, dim_v))
        _assert_n1_matches_one_shot(out, q, k, v, causal=causal)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(got_rank))
    if causal:
        # A single visible key: softmax weight is 1, so query 0 is exactly v[:, 0].
        np.testing.assert_array_equal(np.asarray(got)[:, 0], v[:, 0])


def test_ring_attention_tiny_hand_golden():
    """Hand-checked one-shot. Query 0 causal output is exactly v[0] == 2."""
    q = jnp.array([1.0, 0.5], dtype=jnp.float32).reshape(1, 2, 1, 1)
    k = jnp.array([0.5, -1.0], dtype=jnp.float32).reshape(1, 2, 1, 1)
    v = jnp.array([2.0, 4.0], dtype=jnp.float32).reshape(1, 2, 1, 1)
    golden_c = np.array([2.0, 2.6416426], dtype=np.float32).reshape(1, 2, 1, 1)
    golden_nc = np.array([2.3648512, 2.6416426], dtype=np.float32).reshape(1, 2, 1, 1)

    got_c = parallel.ring_attention(q, k, v, n=1, causal=True)
    got_rank = parallel.ring_attention_rank(q, (k,), (v,), rank=0, causal=True)
    _assert_jax_f32(got_c, (1, 2, 1, 1))
    np.testing.assert_array_equal(np.asarray(got_c)[:, 0], np.asarray(v)[:, 0])
    _assert_f32_parity(got_c, golden_c)
    _assert_f32_parity(got_rank, golden_c)
    _assert_n1_matches_one_shot(got_c, q, k, v, causal=True)

    got_nc = parallel.ring_attention(q, k, v, n=1, causal=False)
    _assert_f32_parity(got_nc, golden_nc)
    _assert_n1_matches_one_shot(got_nc, q, k, v, causal=False)

    got_n2 = parallel.ring_attention(q, k, v, n=2, causal=True)
    _assert_f32_parity(got_n2, golden_c)
    _assert_f32_parity(got_n2, ref_cp.ring_attention(q, k, v, n=2, causal=True))
    _assert_max_abs(got_n2, ref_cp.one_shot_attention(q, k, v, causal=True), ONE_SHOT_MAX_ABS)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "batch,seq,heads,dim,dim_v,seed",
    [
        (2, 16, 3, 8, 5, 7),
        (1, 8, 1, 4, 4, 8),
    ],
)
def test_ring_n4_matches_online_fold_and_one_shot(causal, batch, seq, heads, dim, dim_v, seed):
    n = 4
    q, k, v = _qkv(batch, seq, heads, dim, dim_v, seed)
    got = parallel.ring_attention(
        jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), n=n, causal=causal
    )
    fold = ref_cp.ring_attention(q, k, v, n=n, causal=causal)
    shot = ref_cp.one_shot_attention(q, k, v, causal=causal)
    _assert_jax_f32(got, (batch, seq, heads, dim_v))
    _assert_f32_parity(got, fold)
    _assert_max_abs(got, shot, ONE_SHOT_MAX_ABS)
    if causal:
        np.testing.assert_array_equal(np.asarray(got)[:, 0], v[:, 0])


@pytest.mark.parametrize("causal", [True, False])
def test_rank0_and_rank1_match_schedule_reference(causal):
    """Compare each rank to the reference walk. Do not compare ranks to each other."""
    batch, seq, heads, dim, dim_v, n = 2, 16, 3, 8, 5, 4
    chunk = seq // n
    q, k, v = _qkv(batch, seq, heads, dim, dim_v, seed=9)
    ks = tuple(k[:, i * chunk : (i + 1) * chunk] for i in range(n))
    vs = tuple(v[:, i * chunk : (i + 1) * chunk] for i in range(n))
    q0 = q[:, :chunk]
    q1 = q[:, chunk : 2 * chunk]
    out0 = parallel.ring_attention_rank(jnp.asarray(q0), ks, vs, rank=0, causal=causal)
    out1 = parallel.ring_attention_rank(jnp.asarray(q1), ks, vs, rank=1, causal=causal)
    _assert_jax_f32(out0, (batch, chunk, heads, dim_v))
    _assert_jax_f32(out1, (batch, chunk, heads, dim_v))
    _assert_f32_parity(out0, ref_cp.ring_attention_rank(q0, ks, vs, rank=0, causal=causal))
    _assert_f32_parity(out1, ref_cp.ring_attention_rank(q1, ks, vs, rank=1, causal=causal))
    # The reference walk for rank 1 is the locked golden, not owner order.
    assert ref_cp.cp_ring_schedule(n, 1) == (1, 0, 3, 2)
    full = parallel.ring_attention(
        jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), n=n, causal=causal
    )
    _assert_f32_parity(np.asarray(full)[:, :chunk], out0)
    _assert_f32_parity(np.asarray(full)[:, chunk : 2 * chunk], out1)


def test_causal_future_key_block_does_not_change_earlier_queries():
    batch, seq, heads, dim, dim_v, n = 2, 16, 2, 8, 4, 4
    chunk = seq // n
    q, k, v = _qkv(batch, seq, heads, dim, dim_v, seed=11)
    q_j, k_j, v_j = jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)
    out = parallel.ring_attention(q_j, k_j, v_j, n=n, causal=True)
    ref = ref_cp.ring_attention(q, k, v, n=n, causal=True)
    _assert_f32_parity(np.asarray(out)[:, : 2 * chunk], ref[:, : 2 * chunk])

    # Owners 2 and 3 are strictly in the future for queries in the first half.
    k_pert = k_j.at[:, 2 * chunk :].add(3.0)
    v_pert = v_j.at[:, 2 * chunk :].add(-2.5)
    out_pert = parallel.ring_attention(q_j, k_pert, v_pert, n=n, causal=True)
    _assert_max_abs(
        np.asarray(out)[:, : 2 * chunk],
        np.asarray(out_pert)[:, : 2 * chunk],
        CAUSAL_MAX_ABS,
    )

    # Rank 0 cannot see any key owned by rank >= 1.
    out_r0 = parallel.ring_attention(
        q_j, k_j.at[:, chunk:].add(5.0), v_j.at[:, chunk:].add(-4.0), n=n, causal=True
    )
    _assert_max_abs(np.asarray(out)[:, :chunk], np.asarray(out_r0)[:, :chunk], CAUSAL_MAX_ABS)

    ks = tuple(k[:, i * chunk : (i + 1) * chunk] for i in range(n))
    vs = tuple(v[:, i * chunk : (i + 1) * chunk] for i in range(n))
    base = parallel.ring_attention_rank(q[:, :chunk], ks, vs, rank=0, causal=True)
    ks_p = list(ks)
    vs_p = list(vs)
    ks_p[2] = ks_p[2] + np.float32(1.5)
    vs_p[2] = vs_p[2] - np.float32(1.5)
    pert = parallel.ring_attention_rank(q[:, :chunk], tuple(ks_p), tuple(vs_p), rank=0, causal=True)
    _assert_f32_parity(base, ref_cp.ring_attention_rank(q[:, :chunk], ks, vs, rank=0, causal=True))
    _assert_max_abs(base, pert, CAUSAL_MAX_ABS)

    # Same perturbation must move non-causal outputs, or the mask check is vacuous.
    out_nc = parallel.ring_attention(q_j, k_j, v_j, n=n, causal=False)
    out_nc_pert = parallel.ring_attention(q_j, k_pert, v_pert, n=n, causal=False)
    assert _max_abs(out_nc, out_nc_pert) > 1e-3


@pytest.mark.parametrize("scale", [0.5, 0.0, 1.25])
@pytest.mark.parametrize("causal", [True, False])
def test_ring_attention_explicit_scale(scale, causal):
    q, k, v = _qkv(2, 8, 2, 4, 6, seed=13)
    got = parallel.ring_attention(
        jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), n=4, causal=causal, scale=scale
    )
    _assert_jax_f32(got, (2, 8, 2, 6))
    _assert_f32_parity(got, ref_cp.ring_attention(q, k, v, n=4, causal=causal, scale=scale))
    shot = ref_cp.one_shot_attention(q, k, v, causal=causal, scale=scale)
    _assert_max_abs(got, shot, ONE_SHOT_MAX_ABS)


def test_default_scale_matches_explicit_one_over_sqrt_dim():
    q, k, v = _qkv(1, 4, 2, 8, 3, seed=14)
    dim = q.shape[-1]
    scale = float(ref_cp.default_scale(dim))
    assert scale == float(np.float32(1.0 / math.sqrt(dim)))
    got_default = parallel.ring_attention(
        jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), n=1, causal=True
    )
    got_explicit = parallel.ring_attention(
        jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), n=1, causal=True, scale=scale
    )
    np.testing.assert_array_equal(np.asarray(got_default), np.asarray(got_explicit))
    _assert_n1_matches_one_shot(got_default, q, k, v, causal=True)


def test_ring_attention_accepts_numpy_inputs():
    q, k, v = _qkv(2, 4, 2, 4, 7, seed=15)
    got = parallel.ring_attention(q, k, v, n=1, causal=False)
    _assert_jax_f32(got, (2, 4, 2, 7))
    _assert_n1_matches_one_shot(got, q, k, v, causal=False)


def test_ring_attention_float16_inputs_float32_output():
    q, k, v = _qkv(1, 4, 2, 4, 4, seed=16, dtype=np.float16)
    got = parallel.ring_attention(q, k, v, n=1, causal=True)
    _assert_jax_f32(got, (1, 4, 2, 4))
    _assert_n1_matches_one_shot(got, q, k, v, causal=True)


def test_ring_attention_float64_inputs_float32_output():
    q, k, v = _qkv(1, 4, 1, 4, 2, seed=17, dtype=np.float64)
    got = parallel.ring_attention_rank(q, (k,), (v,), rank=0, causal=False)
    _assert_jax_f32(got, (1, 4, 1, 2))
    _assert_n1_matches_one_shot(got, q, k, v, causal=False)


# ---------------------------------------------------------------------------
# jit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [True, False])
def test_ring_attention_jit_parity(causal):
    q, k, v = _qkv(2, 8, 2, 4, 5, seed=21)
    q_j, k_j, v_j = jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)

    def run(qq, kk, vv, n, causal_flag):
        return parallel.ring_attention(qq, kk, vv, n=n, causal=causal_flag)

    eager = run(q_j, k_j, v_j, 4, causal)
    jit = jax.jit(run, static_argnames=("n", "causal_flag"))(q_j, k_j, v_j, 4, causal)
    _assert_max_abs(eager, jit, JIT_MAX_ABS)
    _assert_f32_parity(eager, ref_cp.ring_attention(q, k, v, n=4, causal=causal))


def test_ring_attention_jit_explicit_scale():
    q, k, v = _qkv(1, 8, 1, 4, 4, seed=22)
    q_j, k_j, v_j = jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)

    def run(qq, kk, vv, n, causal_flag, scale):
        return parallel.ring_attention(qq, kk, vv, n=n, causal=causal_flag, scale=scale)

    eager = run(q_j, k_j, v_j, 2, True, 0.5)
    jit = jax.jit(run, static_argnames=("n", "causal_flag", "scale"))(q_j, k_j, v_j, 2, True, 0.5)
    _assert_max_abs(eager, jit, JIT_MAX_ABS)


@pytest.mark.parametrize("causal", [True, False])
def test_ring_attention_rank_jit_parity(causal):
    batch, seq, heads, dim, dim_v, n = 2, 8, 2, 4, 3, 4
    chunk = seq // n
    q, k, v = _qkv(batch, seq, heads, dim, dim_v, seed=23)
    ks = tuple(jnp.asarray(k[:, i * chunk : (i + 1) * chunk]) for i in range(n))
    vs = tuple(jnp.asarray(v[:, i * chunk : (i + 1) * chunk]) for i in range(n))
    q1 = jnp.asarray(q[:, chunk : 2 * chunk])

    def run(qq, kk, vv, rank, causal_flag):
        return parallel.ring_attention_rank(qq, kk, vv, rank=rank, causal=causal_flag)

    eager = run(q1, ks, vs, 1, causal)
    jit = jax.jit(run, static_argnames=("rank", "causal_flag"))(q1, ks, vs, 1, causal)
    _assert_max_abs(eager, jit, JIT_MAX_ABS)
    _assert_f32_parity(
        eager,
        ref_cp.ring_attention_rank(
            q[:, chunk : 2 * chunk],
            tuple(np.asarray(block) for block in ks),
            tuple(np.asarray(block) for block in vs),
            rank=1,
            causal=causal,
        ),
    )


# ---------------------------------------------------------------------------
# faults
# ---------------------------------------------------------------------------


def _rank_case(case: str):
    batch, chunk, heads, dim, dim_v, n = 2, 4, 2, 4, 3, 4
    rng = np.random.default_rng(0)
    q = rng.normal(size=(batch, chunk, heads, dim)).astype(np.float32)
    ks = [rng.normal(size=(batch, chunk, heads, dim)).astype(np.float32) for _ in range(n)]
    vs = [rng.normal(size=(batch, chunk, heads, dim_v)).astype(np.float32) for _ in range(n)]
    rank = 0
    if case == "rank_neg":
        rank = -1
    elif case == "rank_oob":
        rank = n
    elif case == "unequal_owner_count":
        vs = vs[:-1]
    elif case == "k_block_seq":
        ks[1] = rng.normal(size=(batch, chunk + 1, heads, dim)).astype(np.float32)
    elif case == "v_block_seq":
        vs[0] = rng.normal(size=(batch, chunk - 1, heads, dim_v)).astype(np.float32)
    elif case == "unequal_block_seq":
        ks[2] = rng.normal(size=(batch, chunk * 2, heads, dim)).astype(np.float32)
    elif case == "batch_mismatch":
        ks[0] = rng.normal(size=(batch + 1, chunk, heads, dim)).astype(np.float32)
    elif case == "heads_mismatch":
        vs[1] = rng.normal(size=(batch, chunk, heads + 1, dim_v)).astype(np.float32)
    elif case == "k_heads_mismatch":
        ks[3] = rng.normal(size=(batch, chunk, heads + 2, dim)).astype(np.float32)
    elif case == "non_floating_q":
        q = np.ones((batch, chunk, heads, dim), dtype=np.int32)
    elif case == "non_floating_k":
        ks[0] = np.ones((batch, chunk, heads, dim), dtype=np.int32)
    elif case == "empty_owners":
        ks, vs, rank = [], [], 0
    else:
        raise AssertionError(case)
    return q, ks, vs, rank


@pytest.mark.parametrize(
    "case",
    [
        "rank_neg",
        "rank_oob",
        "unequal_owner_count",
        "k_block_seq",
        "v_block_seq",
        "unequal_block_seq",
        "batch_mismatch",
        "heads_mismatch",
        "k_heads_mismatch",
        "non_floating_q",
        "non_floating_k",
        "empty_owners",
    ],
)
def test_ring_attention_rank_mesh_error(case):
    q, ks, vs, rank = _rank_case(case)
    with pytest.raises(parallel.MeshError):
        parallel.ring_attention_rank(q, ks, vs, rank=rank, causal=True)


def _ring_case(case: str):
    q, k, v = _qkv(2, 8, 2, 4, 4, seed=3)
    n = 2
    causal = True
    if case == "n_lt_1":
        n = 0
    elif case == "n_negative":
        n = -2
    elif case == "seq_not_divisible":
        q, k, v = _qkv(2, 6, 2, 4, 4, seed=4)
        n = 4
    elif case == "n_gt_seq":
        n = 16
    elif case == "q_k_seq_mismatch":
        k = k[:, :4]
    elif case == "q_v_seq_mismatch":
        v = v[:, :6]
    elif case == "batch_mismatch":
        k = k[:1]
    elif case == "heads_mismatch":
        k = k[:, :, :1]
    elif case == "v_heads_mismatch":
        v = v[:, :, :1]
    elif case == "non_floating":
        q = np.ones_like(q, dtype=np.int32)
    else:
        raise AssertionError(case)
    return q, k, v, n, causal


@pytest.mark.parametrize(
    "case",
    [
        "n_lt_1",
        "n_negative",
        "seq_not_divisible",
        "n_gt_seq",
        "q_k_seq_mismatch",
        "q_v_seq_mismatch",
        "batch_mismatch",
        "heads_mismatch",
        "v_heads_mismatch",
        "non_floating",
    ],
)
def test_ring_attention_mesh_error(case):
    q, k, v, n, causal = _ring_case(case)
    with pytest.raises(parallel.MeshError):
        parallel.ring_attention(q, k, v, n=n, causal=causal)


# ---------------------------------------------------------------------------
# grad
# ---------------------------------------------------------------------------


def test_ring_attention_grad_matches_finite_difference():
    q, k, v = _qkv(1, 4, 1, 4, 3, seed=31)
    q_j, k_j, v_j = jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)

    def loss(qq, kk, vv):
        return parallel.ring_attention(qq, kk, vv, n=2, causal=True).sum()

    gq, gk, gv = jax.grad(loss, argnums=(0, 1, 2))(q_j, k_j, v_j)
    grads = {"q": np.asarray(gq), "k": np.asarray(gk), "v": np.asarray(gv)}
    bases = {"q": q, "k": k, "v": v}
    eps = 1e-3
    for which, grad in grads.items():
        for idx in np.ndindex(grad.shape):
            def ev(sign, which=which, idx=idx):
                bumped = {name: arr.copy() for name, arr in bases.items()}
                bumped[which][idx] = np.float32(bumped[which][idx] + sign * eps)
                return float(
                    ref_cp.ring_attention(
                        bumped["q"], bumped["k"], bumped["v"], n=2, causal=True
                    ).sum()
                )

            fd = (ev(1.0) - ev(-1.0)) / (2.0 * eps)
            np.testing.assert_allclose(grad[idx], fd, rtol=GRAD_RTOL, atol=GRAD_ATOL)
