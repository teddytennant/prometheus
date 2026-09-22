"""Oracle tests for FSDP ZeRO-3 collectives (spec 5.2, parallel/fsdp.py).

Production under test (public re-exports):

* ``parallel.fsdp_shard``
* ``parallel.fsdp_all_gather``
* ``parallel.fsdp_reduce_scatter``
* ``parallel.zero3_views``

Expected values come from ``tests.reference.parallel_fsdp``, which does not
call those functions. Every test calls production. The current stubs raise
``NotImplementedError``; a test that passed the stub would be wrong.

No test here is GPU-only. The contract is a CPU analog and does not launch
NCCL, so there is no ``pytest.mark.gpu`` / V-stage skip.

Locked properties
-----------------
1. ``fsdp_shard`` matches ``numpy.array_split`` (negative axis, uneven dim,
   ``n == 1``, several ranks). Result is a ``jax.Array`` of the input dtype.
   NumPy and JAX inputs are both accepted.
2. ``fsdp_all_gather`` of the ``n`` shards recovers the original bitwise for
   float32, int32, bfloat16, and float64. Empty sequence, mixed dtypes, and
   mismatched shapes raise ``parallel.MeshError``.
3. ``fsdp_reduce_scatter`` equals ``shard(sum(partials))`` with the reference
   accumulation (float32/float16/bfloat16 in float32, float64 in float64,
   integers in int64, then cast the shard back). Sum, not mean. float32 and
   bfloat16 match the reference bitwise.
4. ``parallel.MeshError`` (not a bare ``ValueError``) for ``n < 1``, rank out
   of range, axis out of range, ``dim < n``, empty partials, mixed dtypes,
   mismatched partial shapes, bool, and complex.
5. Float32 central differences: the three collectives are linear. Jacobian
   action matches analytic selection/sum at relative tolerance 1e-5.
6. ``zero3_views``: ``ParamKind`` required; non-floating raises; routed
   experts stay full and are cast; every other kind is shard-then-cast.
   gpu is bfloat16, grace is float32, bitwise vs the reference.
7. ``jax.jit`` with ``n``, ``axis``, ``rank``, and ``kind`` static matches
   eager bitwise for float32 and bfloat16. No concrete index of a traced value.
8. Round trip: shard then all_gather is identity; all_gather of those pieces
   then shard returns each piece. Bitwise.
"""

from __future__ import annotations

import os

os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import parallel
from parallel import ParamKind
from tests.reference import parallel_fsdp as ref

jax.config.update("jax_enable_x64", True)

# Central-difference step. Exact in float32, so a linear op's FD is exact
# up to a couple of ulps when the base point is zero.
_EPS = np.float32(2.0**-8)

_RECOVER_DTYPES = (np.float32, np.int32, np.float64, np.float16)


def _bits(arr) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(arr)).view(np.uint8)


def _bitwise_equal(got, expected) -> bool:
    g = np.asarray(got)
    e = np.asarray(expected)
    if g.shape != e.shape or g.dtype.name != e.dtype.name:
        return False
    return bool(np.array_equal(_bits(g), _bits(e)))


def assert_bitwise(got, expected) -> None:
    g = np.asarray(got)
    e = np.asarray(expected)
    assert g.shape == e.shape, (g.shape, e.shape)
    assert g.dtype.name == e.dtype.name, (g.dtype, e.dtype)
    np.testing.assert_array_equal(_bits(g), _bits(e))


def assert_jax_bitwise(got, expected) -> None:
    """Production result is a JAX array and matches ``expected`` bitwise."""
    assert isinstance(got, jax.Array), type(got)
    assert_bitwise(got, expected)


def assert_fd(numeric, analytic) -> None:
    """Relative 1e-5 parity. Absolute 1e-5 covers analytic zeros (no relative)."""
    num = np.asarray(numeric, dtype=np.float64)
    ana = np.asarray(analytic, dtype=np.float64)
    np.testing.assert_allclose(num, ana, rtol=1e-5, atol=1e-5)
    scale = np.maximum(np.abs(ana), 1e-12)
    rel = np.abs(num - ana) / scale
    # Non-tiny analytic entries must satisfy the relative bound on their own.
    mask = np.abs(ana) >= 1e-3
    if np.any(mask):
        assert float(np.max(rel[mask])) <= 1e-5


def _sample(shape, dtype, rng: np.random.Generator) -> np.ndarray:
    if dtype == "bfloat16":
        raw = rng.standard_normal(shape).astype(np.float32)
        return np.asarray(jnp.asarray(raw).astype(jnp.bfloat16))
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.integer):
        return rng.integers(-50, 50, size=shape, dtype=np.int64).astype(dt)
    raw = rng.standard_normal(shape)
    if dt == np.dtype(np.float16):
        raw = raw * 0.25
    return raw.astype(dt)


def _maybe_jax(arr: np.ndarray, jax_input: bool):
    if jax_input:
        return jnp.asarray(arr)
    return arr


def _one_hot_fd(fn, shape, index, eps=_EPS) -> np.ndarray:
    """Central difference of ``fn`` at 0, perturbing a single element."""
    plus = np.zeros(shape, dtype=np.float32)
    minus = np.zeros(shape, dtype=np.float32)
    plus[index] = eps
    minus[index] = -eps
    return (fn(plus) - fn(minus)) / (np.float32(2) * eps)


# ---------------------------------------------------------------------------
# 1. fsdp_shard
# ---------------------------------------------------------------------------


class TestFsdpShard:
    """Shard matches array_split, including uneven, negative axis, n==1."""

    def test_hand_uneven_int32_several_ranks(self):
        param = np.arange(7, dtype=np.int32)
        expected = {
            0: np.array([0, 1, 2], dtype=np.int32),
            1: np.array([3, 4], dtype=np.int32),
            2: np.array([5, 6], dtype=np.int32),
        }
        for rank, exp in expected.items():
            got = parallel.fsdp_shard(param, 3, 0, rank)
            assert_jax_bitwise(got, exp)
            assert got.dtype.name == "int32"
            assert_bitwise(got, ref.fsdp_shard(param, 3, 0, rank))
            assert list(np.array_split(param, 3, axis=0)[rank]) == list(np.asarray(got))

    def test_negative_axis_matches_array_split(self):
        param = np.arange(12, dtype=np.int32).reshape(3, 4)
        # axis -1, dim 4, n 3 -> sizes 2, 1, 1
        for rank in range(3):
            got = parallel.fsdp_shard(param, 3, -1, rank)
            exp = np.array_split(param, 3, axis=-1)[rank]
            assert_jax_bitwise(got, exp)
            assert_bitwise(got, ref.fsdp_shard(param, 3, -1, rank))

    def test_n_eq_1_bitwise_identity_signed_zero_and_nan(self):
        param = np.array([-0.0, 0.0, 1.5, np.nan, np.inf, -np.inf], dtype=np.float32)
        for src in (param, jnp.asarray(param)):
            got = parallel.fsdp_shard(src, 1, 0, 0)
            assert_jax_bitwise(got, param)
            # Object identity is not required (spec: values unchanged only).

    @pytest.mark.parametrize(
        "shape,n,axis",
        [
            ((7,), 3, 0),
            ((7,), 3, -1),
            ((4, 5), 2, -1),
            ((4, 5), 4, 0),
            ((8, 6, 3), 4, 1),
            ((8, 6, 3), 3, -1),
            ((9,), 9, 0),
            ((2, 2, 2), 2, -2),
            ((5, 1, 4), 1, 0),
            ((6, 5), 5, -1),
        ],
    )
    @pytest.mark.parametrize("dtype", [np.float32, np.int32, np.float64, np.float16])
    @pytest.mark.parametrize("jax_input", [False, True])
    def test_matches_reference_all_ranks(self, shape, n, axis, dtype, jax_input):
        rng = np.random.default_rng(shape[0] * 100 + n + (0 if axis >= 0 else 17))
        param = _sample(shape, dtype, rng)
        src = _maybe_jax(param, jax_input)
        for rank in range(n):
            got = parallel.fsdp_shard(src, n, axis, rank)
            exp = ref.fsdp_shard(param, n, axis, rank)
            assert_jax_bitwise(got, exp)
            assert got.dtype.name == np.dtype(dtype).name
            axis_n = axis if axis >= 0 else axis + len(shape)
            assert got.shape[axis_n] == ref.split_sizes(shape[axis_n], n)[rank]
            assert got.shape[:axis_n] + got.shape[axis_n + 1 :] == (
                shape[:axis_n] + shape[axis_n + 1 :]
            )

    def test_bfloat16_numpy_and_jax_inputs(self):
        rng = np.random.default_rng(11)
        param = _sample((5, 7), "bfloat16", rng)
        for src in (param, jnp.asarray(param)):
            for rank in range(3):
                got = parallel.fsdp_shard(src, 3, -1, rank)
                assert_jax_bitwise(got, ref.fsdp_shard(param, 3, -1, rank))
                assert got.dtype.name == "bfloat16"

    def test_middle_axis_3d_uneven(self):
        rng = np.random.default_rng(4)
        param = _sample((3, 5, 4), np.float32, rng)
        got = parallel.fsdp_shard(param, 3, 1, 0)
        # dim 5, n 3 -> sizes 2, 2, 1; rank 0 is the first two slices
        assert_jax_bitwise(got, param[:, 0:2, :])
        assert tuple(got.shape) == (3, 2, 4)

    def test_does_not_mutate_input(self):
        param = np.arange(6, dtype=np.float32).reshape(2, 3)
        before = param.copy()
        parallel.fsdp_shard(param, 2, -1, 1)
        np.testing.assert_array_equal(param.view(np.uint32), before.view(np.uint32))


class TestFsdpShardErrors:
    """MeshError, not a bare ValueError and not a successful empty shard."""

    @pytest.mark.parametrize("n", [0, -1, -4])
    def test_n_lt_1(self, n):
        x = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_shard(x, n, 0, 0)

    @pytest.mark.parametrize("rank", [-1, 2, 3, 99])
    def test_rank_out_of_range(self, rank):
        x = np.ones((6,), dtype=np.float32)
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_shard(x, 2, 0, rank)

    @pytest.mark.parametrize("axis", [2, -3, 8, -4, 5])
    def test_axis_out_of_range(self, axis):
        # ndim 2: valid axes are -2, -1, 0, 1. These are all outside that.
        x = np.ones((4, 5), dtype=np.float32)
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_shard(x, 2, axis, 0)

    def test_dim_lt_n_even_for_rank_that_would_be_nonempty(self):
        x = np.ones((5, 3), dtype=np.float32)
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_shard(x, 4, 1, 0)
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_shard(x, 4, -1, 1)

    def test_zero_length_dimension(self):
        x = np.ones((0, 4), dtype=np.float32)
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_shard(x, 1, 0, 0)

    def test_scalar_axis_out_of_range(self):
        x = np.array(3.0, dtype=np.float32)
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_shard(x, 1, 0, 0)


# ---------------------------------------------------------------------------
# 2. fsdp_all_gather
# ---------------------------------------------------------------------------


class TestFsdpAllGather:
    """Gather is the inverse of shard, bitwise, for the required dtypes."""

    @pytest.mark.parametrize("dtype", [np.float32, np.int32, np.float64])
    @pytest.mark.parametrize("jax_input", [False, True])
    def test_recovers_original_bitwise(self, dtype, jax_input):
        rng = np.random.default_rng(21)
        # Last axis length 4 >= n=3, so the split is a valid uneven gather.
        param = _sample((5, 7, 4), dtype, rng)
        param.reshape(-1)[0] = param.dtype.type(-0.0) if dtype != np.int32 else param.dtype.type(0)
        if dtype == np.float32:
            param.reshape(-1)[1] = np.float32(np.nan)
        n, axis = 3, -1
        shards = [ref.fsdp_shard(param, n, axis, r) for r in range(n)]
        src = [_maybe_jax(s, jax_input) for s in shards]
        got = parallel.fsdp_all_gather(src, axis)
        assert_jax_bitwise(got, param)
        assert_bitwise(got, ref.fsdp_all_gather(shards, axis))

    def test_recovers_bfloat16_bitwise(self):
        rng = np.random.default_rng(22)
        param = _sample((4, 6), "bfloat16", rng)
        shards = [ref.fsdp_shard(param, 4, 0, r) for r in range(4)]
        for src in (shards, [jnp.asarray(s) for s in shards]):
            got = parallel.fsdp_all_gather(src, 0)
            assert_jax_bitwise(got, param)
            assert got.dtype.name == "bfloat16"

    def test_negative_axis_uneven_pieces(self):
        pieces = [
            np.arange(0, 6, dtype=np.float32).reshape(3, 2),
            np.arange(6, 9, dtype=np.float32).reshape(3, 1),
            np.arange(9, 12, dtype=np.float32).reshape(3, 1),
        ]
        got = parallel.fsdp_all_gather(pieces, -1)
        exp = np.concatenate(pieces, axis=-1)
        assert_jax_bitwise(got, exp)
        assert tuple(got.shape) == (3, 4)

    def test_single_shard_identity(self):
        shard = np.array([[-0.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
        got = parallel.fsdp_all_gather((jnp.asarray(shard),), -2)
        assert_jax_bitwise(got, shard)

    def test_accepts_list_and_tuple(self):
        a = np.ones((2, 2), dtype=np.int32)
        b = np.full((2, 2), 3, dtype=np.int32)
        from_list = parallel.fsdp_all_gather([a, b], 0)
        from_tuple = parallel.fsdp_all_gather((a, b), 0)
        assert_jax_bitwise(from_list, np.concatenate([a, b], axis=0))
        assert_bitwise(from_list, from_tuple)

    def test_empty_sequence_raises(self):
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_all_gather([], 0)
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_all_gather((), -1)

    @pytest.mark.parametrize(
        "shards",
        [
            (
                np.ones((2, 2), dtype=np.float32),
                np.ones((2, 2), dtype=np.float64),
            ),
            (
                np.ones((2,), dtype=np.int32),
                np.ones((2,), dtype=np.int64),
            ),
            (
                np.ones((2, 2), dtype=np.float32),
                np.asarray(jnp.ones((2, 2), dtype=jnp.bfloat16)),
            ),
            (
                np.ones((3,), dtype=np.float16),
                np.ones((3,), dtype=np.float32),
            ),
        ],
    )
    def test_mixed_dtypes_raise(self, shards):
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_all_gather(shards, 0)

    def test_mismatched_shapes_raise(self):
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((2, 4), dtype=np.float32)
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_all_gather([a, b], 0)

    def test_mismatched_ranks_raise(self):
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((2, 3, 1), dtype=np.float32)
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_all_gather([a, b], 0)

    @pytest.mark.parametrize("axis", [2, -3, 5])
    def test_axis_out_of_range(self, axis):
        shards = [np.ones((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32)]
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_all_gather(shards, axis)


# ---------------------------------------------------------------------------
# 3. fsdp_reduce_scatter
# ---------------------------------------------------------------------------


class TestFsdpReduceScatter:
    """Sum then shard, with the reference accumulation. Bitwise for f32/bf16."""

    def test_float32_left_fold_not_tree_sum(self):
        rng = np.random.default_rng(0)
        partials = [rng.standard_normal((64, 32)).astype(np.float32) for _ in range(8)]
        # Seed 0 / this shape: left fold and jnp.sum disagree. Rank 0's shard
        # includes some of those bits (checked below after the production call).
        got = parallel.fsdp_reduce_scatter(partials, 0, 0)
        acc = partials[0].copy()
        for part in partials[1:]:
            acc = acc + part
        exp = ref.fsdp_shard(acc, 8, 0, 0)
        assert_jax_bitwise(got, exp)
        assert_bitwise(got, ref.fsdp_reduce_scatter(partials, 0, 0))
        tree = np.asarray(jnp.sum(jnp.stack([jnp.asarray(p) for p in partials]), axis=0))
        tree_shard = ref.fsdp_shard(tree, 8, 0, 0)
        assert not _bitwise_equal(exp, tree_shard)

    def test_float32_left_to_right_not_reversed(self):
        # (a+b)+c != (c+b)+a bitwise on the first component.
        cols = [
            np.float32(-5356.694),
            np.float32(3615.9507),
            np.float32(13040.0),
        ]
        partials = []
        for value in cols:
            row = np.zeros((3, 2), dtype=np.float32)
            row[0, 0] = value
            partials.append(row)
        got = parallel.fsdp_reduce_scatter(partials, 0, 0)
        acc = partials[0].copy()
        for part in partials[1:]:
            acc = acc + part
        exp = ref.fsdp_shard(acc, 3, 0, 0)
        assert_jax_bitwise(got, exp)
        rev = partials[2].copy()
        for part in (partials[1], partials[0]):
            rev = rev + part
        rev_shard = ref.fsdp_shard(rev, 3, 0, 0)
        assert not _bitwise_equal(exp, rev_shard)

    def test_float16_accumulates_in_float32(self):
        # 100 + 0.1 + 0.1 is 100.25 in float16 and 100.2 after an f32 sum.
        partials = []
        for value in (np.float16(100.0), np.float16(0.1), np.float16(0.1)):
            row = np.zeros((3,), dtype=np.float16)
            row[0] = value
            partials.append(row)
        got = parallel.fsdp_reduce_scatter(partials, 0, 0)
        acc = np.float32(0)
        # left fold, no leading zero: start at the first partial
        acc = np.float32(partials[0][0])
        for part in partials[1:]:
            acc = acc + np.float32(part[0])
        expected = np.zeros((1,), dtype=np.float16)
        expected[0] = np.float32(acc).astype(np.float16)
        assert_jax_bitwise(got, expected)
        narrow = partials[0][0]
        for part in partials[1:]:
            narrow = np.float16(narrow + part[0])
        assert not _bitwise_equal(got, np.array([narrow], dtype=np.float16))
        assert got.dtype.name == "float16"

    def test_bfloat16_bitwise_after_f32_sum(self):
        values = [
            np.float32(-40.25),
            np.float32(-51.25),
            np.float32(17.25),
            np.float32(24.75),
            np.float32(-20.25),
        ]
        partials = []
        for value in values:
            row = np.zeros((5,), dtype=np.float32)
            row[0] = value
            partials.append(np.asarray(jnp.asarray(row).astype(jnp.bfloat16)))
        got = parallel.fsdp_reduce_scatter(partials, 0, 0)
        assert_jax_bitwise(got, ref.fsdp_reduce_scatter(partials, 0, 0))
        assert got.dtype.name == "bfloat16"
        # Narrow bf16 sum of the same scalars is a different bit pattern.
        narrow = partials[0][0]
        for part in partials[1:]:
            narrow = np.asarray(
                (jnp.asarray(narrow) + jnp.asarray(part[0])).astype(jnp.bfloat16)
            )
        assert not _bitwise_equal(np.asarray(got).reshape(-1)[:1], np.asarray(narrow).reshape(1))

    def test_float64_accumulates_in_float64(self):
        rng = np.random.default_rng(9)
        partials = [_sample((5, 4), np.float64, rng) for _ in range(3)]
        got = parallel.fsdp_reduce_scatter(tuple(partials), -1, 2)
        assert_jax_bitwise(got, ref.fsdp_reduce_scatter(partials, -1, 2))
        assert got.dtype.name == "float64"

    @pytest.mark.parametrize("dtype", [np.int32, np.int16, np.int8, np.uint8, np.int64])
    def test_integers_accumulate_in_int64_and_cast_back(self, dtype):
        rng = np.random.default_rng(13)
        # axis 0 has length 6 >= n=4 (axis 1 would be dim 3 < n, which is an error).
        partials = [_sample((6, 3), dtype, rng) for _ in range(4)]
        got = parallel.fsdp_reduce_scatter(partials, 0, 1)
        assert_jax_bitwise(got, ref.fsdp_reduce_scatter(partials, 0, 1))
        assert got.dtype.name == np.dtype(dtype).name

    def test_sum_not_mean(self):
        x = np.ones((6, 4), dtype=np.float32)
        partials = [x, x, x]
        got = parallel.fsdp_reduce_scatter(partials, 0, 0)
        # rank 0 of 6/3 is two rows of 3, not of 1
        exp = np.full((2, 4), 3.0, dtype=np.float32)
        assert_jax_bitwise(got, exp)
        assert not _bitwise_equal(got, np.ones((2, 4), dtype=np.float32))

    def test_n1_preserves_signed_zero(self):
        partial = np.array([-0.0, 1.0, 2.0], dtype=np.float32)
        got = parallel.fsdp_reduce_scatter([partial], 0, 0)
        assert_jax_bitwise(got, partial)
        assert_bitwise(got, ref.fsdp_shard(partial, 1, 0, 0))

    @pytest.mark.parametrize("dtype", ["bfloat16", np.float32, np.float16, np.float64])
    @pytest.mark.parametrize("jax_input", [False, True])
    def test_matches_reference_several_ranks(self, dtype, jax_input):
        rng = np.random.default_rng(30)
        n, axis = 3, -1
        partials = [_sample((4, 5), dtype, rng) for _ in range(n)]
        src = [_maybe_jax(p, jax_input) for p in partials]
        for rank in range(n):
            got = parallel.fsdp_reduce_scatter(src, axis, rank)
            assert_jax_bitwise(got, ref.fsdp_reduce_scatter(partials, axis, rank))

    def test_accepts_list_and_tuple(self):
        a = np.arange(4, dtype=np.float32)
        b = np.arange(4, dtype=np.float32) + 1
        from_list = parallel.fsdp_reduce_scatter([a, b], 0, 1)
        from_tuple = parallel.fsdp_reduce_scatter((a, b), 0, 1)
        assert_bitwise(from_list, from_tuple)
        assert_jax_bitwise(from_list, ref.fsdp_reduce_scatter([a, b], 0, 1))


class TestFsdpReduceScatterErrors:
    """MeshError on empty, mixed, mismatched, bad axis/rank, dim < n, bool, complex."""

    def test_empty_partials(self):
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_reduce_scatter([], 0, 0)
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_reduce_scatter((), -1, 0)

    def test_mixed_dtypes(self):
        a = np.ones((4,), dtype=np.float32)
        b = np.ones((4,), dtype=np.float64)
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_reduce_scatter([a, b], 0, 0)
        bf = np.asarray(jnp.ones((4,), dtype=jnp.bfloat16))
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_reduce_scatter([a, bf], 0, 1)

    def test_mismatched_shapes(self):
        a = np.ones((4, 3), dtype=np.float32)
        b = np.ones((4, 2), dtype=np.float32)
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_reduce_scatter([a, b], 0, 0)
        c = np.ones((4, 3, 1), dtype=np.float32)
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_reduce_scatter([a, c], 0, 1)

    @pytest.mark.parametrize("axis", [2, -3, 7])
    def test_axis_out_of_range(self, axis):
        partials = [np.ones((4, 4), dtype=np.float32) for _ in range(2)]
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_reduce_scatter(partials, axis, 0)

    @pytest.mark.parametrize("rank", [-1, 2, 5])
    def test_rank_out_of_range(self, rank):
        partials = [np.ones((4,), dtype=np.float32) for _ in range(2)]
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_reduce_scatter(partials, 0, rank)

    def test_dim_lt_n(self):
        partials = [np.ones((3, 2), dtype=np.float32) for _ in range(3)]
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_reduce_scatter(partials, 1, 0)

    @pytest.mark.parametrize(
        "partials",
        [
            [np.array([True, False, True]), np.array([False, True, False])],
            [jnp.array([True, False, True]), jnp.array([True, True, False])],
            [np.ones((3,), dtype=np.complex64), np.ones((3,), dtype=np.complex64)],
            [np.ones((3,), dtype=np.complex128), np.full((3,), 2.0, dtype=np.complex128)],
        ],
    )
    def test_bool_and_complex(self, partials):
        with pytest.raises(parallel.MeshError):
            parallel.fsdp_reduce_scatter(partials, 0, 0)


# ---------------------------------------------------------------------------
# 5. Finite differences (float32 linearity)
# ---------------------------------------------------------------------------


class TestFiniteDifference:
    """Central differences vs analytic selection/sum. Relative tolerance 1e-5."""

    def test_shard_one_element_and_jvp(self):
        # shape (7,), n=3, rank=1 -> indices [3, 4), local 0 and 1.
        n, axis, rank = 3, 0, 1

        def prod(arr):
            return np.asarray(parallel.fsdp_shard(arr, n, axis, rank), dtype=np.float32)

        inside = _one_hot_fd(prod, (7,), 3)
        analytic_in = np.zeros((2,), dtype=np.float32)
        analytic_in[0] = 1.0
        assert_fd(inside, analytic_in)

        outside = _one_hot_fd(prod, (7,), 0)
        assert_fd(outside, np.zeros((2,), dtype=np.float32))

        rng = np.random.default_rng(5)
        v = rng.standard_normal((7,)).astype(np.float32)
        plus = v * _EPS
        minus = -v * _EPS
        numeric = (prod(plus) - prod(minus)) / (np.float32(2) * _EPS)
        assert_fd(numeric, ref.fsdp_shard(v, n, axis, rank))

    def test_all_gather_one_element_and_jvp(self):
        shards = [
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((4, 3), dtype=np.float32),
        ]

        def prod(pieces):
            return np.asarray(parallel.fsdp_all_gather(pieces, 0), dtype=np.float32)

        def bump(index_shard, index, sign):
            pieces = [s.copy() for s in shards]
            pieces[index_shard][index] = sign * _EPS
            return pieces

        numeric = (prod(bump(1, (1, 2), 1.0)) - prod(bump(1, (1, 2), -1.0))) / (
            np.float32(2) * _EPS
        )
        analytic = np.zeros((6, 3), dtype=np.float32)
        analytic[2 + 1, 2] = 1.0
        assert_fd(numeric, analytic)

        rng = np.random.default_rng(6)
        tangents = [rng.standard_normal(s.shape).astype(np.float32) for s in shards]
        numeric_j = (
            prod([_EPS * t for t in tangents]) - prod([-_EPS * t for t in tangents])
        ) / (np.float32(2) * _EPS)
        assert_fd(numeric_j, ref.fsdp_all_gather(tangents, 0))

    def test_reduce_scatter_one_element_and_jvp(self):
        # shape (6,), n=3, rank=2 -> indices [4, 6), local 0 and 1.
        n, axis, rank = 3, 0, 2
        base = [np.zeros((6,), dtype=np.float32) for _ in range(n)]

        def prod(partials):
            return np.asarray(
                parallel.fsdp_reduce_scatter(partials, axis, rank), dtype=np.float32
            )

        def bump(which, index, sign):
            parts = [p.copy() for p in base]
            parts[which][index] = sign * _EPS
            return parts

        inside = (prod(bump(1, 4, 1.0)) - prod(bump(1, 4, -1.0))) / (
            np.float32(2) * _EPS
        )
        analytic_in = np.zeros((2,), dtype=np.float32)
        analytic_in[0] = 1.0
        assert_fd(inside, analytic_in)

        outside = (prod(bump(0, 0, 1.0)) - prod(bump(0, 0, -1.0))) / (
            np.float32(2) * _EPS
        )
        assert_fd(outside, np.zeros((2,), dtype=np.float32))

        # Sum: each partial contributes derivative 1 at an in-shard index.
        both = (
            prod(bump(0, 5, 1.0))
            - prod(bump(0, 5, -1.0))
            + prod(bump(2, 5, 1.0))
            - prod(bump(2, 5, -1.0))
        ) / (np.float32(2) * _EPS)
        analytic_both = np.zeros((2,), dtype=np.float32)
        analytic_both[1] = 2.0
        assert_fd(both, analytic_both)

        rng = np.random.default_rng(7)
        tangents = [rng.standard_normal((6,)).astype(np.float32) for _ in range(n)]
        numeric_j = (
            prod([_EPS * t for t in tangents]) - prod([-_EPS * t for t in tangents])
        ) / (np.float32(2) * _EPS)
        assert_fd(numeric_j, ref.fsdp_reduce_scatter(tangents, axis, rank))


# ---------------------------------------------------------------------------
# 6. zero3_views
# ---------------------------------------------------------------------------


_SHARDED_KINDS = (
    ParamKind.ATTENTION,
    ParamKind.DENSE,
    ParamKind.SHARED_EXPERT,
    ParamKind.OTHER,
)


class TestZero3Views:
    """Routed stays full. Every other kind is shard-then-cast. Bitwise."""

    def test_string_and_other_types_raise(self):
        param = np.ones((4, 4), dtype=np.float32)
        for kind in ("attention", "routed_expert", ParamKind.ATTENTION.value, "", 0, None):
            with pytest.raises(parallel.MeshError):
                parallel.zero3_views(param, kind, 2, 0, 0)

    @pytest.mark.parametrize(
        "param",
        [
            np.ones((4,), dtype=np.int32),
            np.ones((4,), dtype=np.int64),
            np.array([True, False, True, False]),
            np.ones((4,), dtype=np.complex64),
            jnp.ones((4,), dtype=jnp.int32),
        ],
    )
    def test_non_floating_raises(self, param):
        with pytest.raises(parallel.MeshError):
            parallel.zero3_views(param, ParamKind.DENSE, 2, 0, 0)

    def test_routed_expert_is_full_tensor_cast(self):
        rng = np.random.default_rng(8)
        param = _sample((5, 6), np.float32, rng)
        gpu, grace = parallel.zero3_views(param, ParamKind.ROUTED_EXPERT, 4, -1, 3)
        assert isinstance(gpu, jax.Array)
        assert isinstance(grace, jax.Array)
        assert gpu.dtype.name == "bfloat16"
        assert grace.dtype.name == "float32"
        assert tuple(gpu.shape) == param.shape
        assert tuple(grace.shape) == param.shape
        exp_gpu, exp_grace = ref.zero3_views(param, ParamKind.ROUTED_EXPERT, 4, -1, 3)
        assert_bitwise(gpu, exp_gpu)
        assert_bitwise(grace, exp_grace)
        # Independent cast of the full tensor, not of a shard.
        assert_bitwise(grace, ref.cast_dtype(param, np.float32))
        assert_bitwise(gpu, ref.cast_dtype(param, "bfloat16"))
        gpu_other, grace_other = parallel.zero3_views(
            param, ParamKind.ROUTED_EXPERT, 4, 0, 0
        )
        assert_bitwise(gpu, gpu_other)
        assert_bitwise(grace, grace_other)

    @pytest.mark.parametrize("kind", _SHARDED_KINDS)
    def test_sharded_kinds_shard_then_cast(self, kind):
        param = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        gpu, grace = parallel.zero3_views(param, kind, 2, 0, 1)
        # dim 5, n 2 -> sizes 3, 2; rank 1 is [4, 5], then cast.
        assert isinstance(gpu, jax.Array) and isinstance(grace, jax.Array)
        assert gpu.dtype.name == "bfloat16"
        assert grace.dtype.name == "float32"
        assert tuple(grace.shape) == (2,)
        assert tuple(gpu.shape) == (2,)
        exp_gpu, exp_grace = ref.zero3_views(param, kind, 2, 0, 1)
        assert_bitwise(gpu, exp_gpu)
        assert_bitwise(grace, exp_grace)
        piece = param[3:5]
        assert_bitwise(grace, ref.cast_dtype(piece, np.float32))
        assert_bitwise(gpu, ref.cast_dtype(piece, "bfloat16"))

    @pytest.mark.parametrize("dtype", [np.float64, np.float16, "bfloat16", np.float32])
    @pytest.mark.parametrize("kind", (ParamKind.ATTENTION, ParamKind.ROUTED_EXPERT))
    def test_bitwise_vs_reference_several_dtypes(self, dtype, kind):
        rng = np.random.default_rng(12)
        param = _sample((3, 7), dtype, rng)
        for rank in (0, 2):
            got = parallel.zero3_views(jnp.asarray(param), kind, 3, -1, rank)
            exp = ref.zero3_views(param, kind, 3, -1, rank)
            assert isinstance(got, tuple) and len(got) == 2
            assert_jax_bitwise(got[0], exp[0])
            assert_jax_bitwise(got[1], exp[1])
            assert got[0].dtype.name == "bfloat16"
            assert got[1].dtype.name == "float32"

    def test_negative_axis_uneven_not_full_cast(self):
        rng = np.random.default_rng(14)
        param = _sample((2, 5), np.float64, rng)
        gpu, grace = parallel.zero3_views(param, ParamKind.SHARED_EXPERT, 3, -1, 0)
        exp_gpu, exp_grace = ref.zero3_views(param, ParamKind.SHARED_EXPERT, 3, -1, 0)
        assert_jax_bitwise(gpu, exp_gpu)
        assert_jax_bitwise(grace, exp_grace)
        # Rank 0's piece is shorter than the full tensor.
        assert gpu.shape != param.shape
        full_gpu = ref.cast_dtype(param, "bfloat16")
        assert gpu.shape != full_gpu.shape

    def test_n_eq_1_sharded_is_cast_of_full(self):
        param = np.array([-0.0, 1.25, 3.5], dtype=np.float32)
        gpu, grace = parallel.zero3_views(param, ParamKind.OTHER, 1, 0, 0)
        assert_jax_bitwise(gpu, ref.cast_dtype(param, "bfloat16"))
        assert_jax_bitwise(grace, param)

    def test_shard_errors_propagate(self):
        param = np.ones((4, 3), dtype=np.float32)
        with pytest.raises(parallel.MeshError):
            parallel.zero3_views(param, ParamKind.ATTENTION, 0, 0, 0)
        with pytest.raises(parallel.MeshError):
            parallel.zero3_views(param, ParamKind.DENSE, 2, 0, 5)
        with pytest.raises(parallel.MeshError):
            parallel.zero3_views(param, ParamKind.OTHER, 2, 7, 0)
        with pytest.raises(parallel.MeshError):
            parallel.zero3_views(param, ParamKind.SHARED_EXPERT, 4, -1, 0)


# ---------------------------------------------------------------------------
# 7. jax.jit, static n/axis/rank/kind, no tracer leak
# ---------------------------------------------------------------------------


class TestJit:
    """Jitted collectives match eager bitwise. Static mesh args, traced data."""

    @pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16, np.int32])
    def test_shard_matches_eager_bitwise(self, dtype):
        if dtype == jnp.bfloat16:
            param = jnp.arange(10, dtype=jnp.float32).astype(jnp.bfloat16)
        else:
            param = jnp.arange(10, dtype=dtype)
        eager = parallel.fsdp_shard(param, 3, -1, 1)
        jitted = jax.jit(parallel.fsdp_shard, static_argnums=(1, 2, 3))(param, 3, -1, 1)
        assert isinstance(jitted, jax.Array)
        assert_bitwise(jitted, eager)

    @pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16, np.int32])
    def test_all_gather_matches_eager_bitwise(self, dtype):
        if dtype == jnp.bfloat16:
            pieces = [
                jnp.arange(i * 3, i * 3 + 3, dtype=jnp.float32).astype(jnp.bfloat16)
                for i in range(3)
            ]
        else:
            pieces = [jnp.arange(i * 3, i * 3 + 3, dtype=dtype) for i in range(3)]
        eager = parallel.fsdp_all_gather(pieces, 0)
        jitted = jax.jit(parallel.fsdp_all_gather, static_argnums=(1,))(pieces, 0)
        assert isinstance(jitted, jax.Array)
        assert_bitwise(jitted, eager)

    @pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16])
    def test_reduce_scatter_matches_eager_bitwise(self, dtype):
        if dtype == jnp.bfloat16:
            partials = [
                jnp.asarray(np.linspace(-1, 1, 9).reshape(3, 3)).astype(jnp.bfloat16)
                for _ in range(3)
            ]
        else:
            partials = [
                jnp.asarray(np.linspace(-1.0 + i, 2.0 + i, 9).reshape(3, 3), dtype=jnp.float32)
                for i in range(3)
            ]
        eager = parallel.fsdp_reduce_scatter(partials, -1, 2)
        jitted = jax.jit(parallel.fsdp_reduce_scatter, static_argnums=(1, 2))(
            partials, -1, 2
        )
        assert isinstance(jitted, jax.Array)
        assert_bitwise(jitted, eager)

    @pytest.mark.parametrize("kind", (ParamKind.ATTENTION, ParamKind.ROUTED_EXPERT))
    @pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16])
    def test_zero3_views_matches_eager_bitwise(self, kind, dtype):
        if dtype == jnp.bfloat16:
            param = jnp.arange(8, dtype=jnp.float32).reshape(2, 4).astype(jnp.bfloat16)
        else:
            param = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
        eager = parallel.zero3_views(param, kind, 2, -1, 1)
        jitted = jax.jit(parallel.zero3_views, static_argnums=(1, 2, 3, 4))(
            param, kind, 2, -1, 1
        )
        assert isinstance(jitted, tuple) and len(jitted) == 2
        assert_bitwise(jitted[0], eager[0])
        assert_bitwise(jitted[1], eager[1])

    def test_no_tracer_leak_on_traced_values(self):
        """n/axis/rank/kind stay static. Data is traced; no concrete index of it."""

        def shard_twice(x):
            y = parallel.fsdp_shard(x, 3, -1, 1)
            return y + y

        def gather_twice(pieces):
            y = parallel.fsdp_all_gather(pieces, -1)
            return y * np.float32(2)

        def reduce_twice(partials):
            y = parallel.fsdp_reduce_scatter(partials, 0, 1)
            return y + y

        def views_sum(x):
            gpu, grace = parallel.zero3_views(x, ParamKind.DENSE, 2, 0, 1)
            return grace + grace, gpu

        x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        shard_out = jax.jit(shard_twice)(x)
        eager_shard = parallel.fsdp_shard(x, 3, -1, 1)
        assert_bitwise(shard_out, np.asarray(eager_shard) * 2)

        pieces = [x[:, :2], x[:, 2:3], x[:, 3:4]]
        gather_out = jax.jit(gather_twice)(pieces)
        assert_bitwise(gather_out, np.asarray(parallel.fsdp_all_gather(pieces, -1)) * 2)

        partials = (x, x + 1, x + 2)
        reduce_out = jax.jit(reduce_twice)(partials)
        assert_bitwise(reduce_out, np.asarray(parallel.fsdp_reduce_scatter(partials, 0, 1)) * 2)

        views_out = jax.jit(views_sum)(x)
        eager_views = parallel.zero3_views(x, ParamKind.DENSE, 2, 0, 1)
        assert_bitwise(views_out[0], np.asarray(eager_views[1]) * 2)
        assert_bitwise(views_out[1], eager_views[0])


# ---------------------------------------------------------------------------
# 8. Round trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Shard then all_gather is identity. Gather of those pieces then shard is too."""

    @pytest.mark.parametrize("dtype", [np.float32, np.int32, np.float64, "bfloat16"])
    @pytest.mark.parametrize(
        "shape,n,axis",
        [
            ((7,), 3, 0),
            ((4, 5), 2, -1),
            ((8, 6, 3), 4, 1),
            ((9, 2), 1, -1),
        ],
    )
    def test_shard_then_all_gather_is_identity(self, dtype, shape, n, axis):
        rng = np.random.default_rng(40 + n)
        param = _sample(shape, dtype, rng)
        if dtype == np.float32:
            param.reshape(-1)[0] = np.float32(-0.0)
            param.reshape(-1)[1] = np.float32(np.nan)
        src = jnp.asarray(param)
        pieces = [parallel.fsdp_shard(src, n, axis, r) for r in range(n)]
        gathered = parallel.fsdp_all_gather(pieces, axis)
        assert_jax_bitwise(gathered, param)
        for rank, piece in enumerate(pieces):
            again = parallel.fsdp_shard(gathered, n, axis, rank)
            assert_bitwise(again, piece)
