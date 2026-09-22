"""Oracle tests for the circular pipeline schedule (spec 5.1, A4).

Production ``parallel.pipeline_forward_schedule`` and ``parallel.circular_pipeline``
are stubs (``NotImplementedError``). Every test in this module calls one of
those two and must fail on the stub. Do not skip, xfail, or mark ``gpu``:
device placement is a later V2/V5 stage; this contract is the CPU tick table
and single-device composition. No ``jax.jit``.

Frozen schedule goldens are written literally (not copied from production).
Value goldens are small-integer arrays; comparison is bitwise (``tobytes``),
not ``np.allclose``.

Groups
------
schedule shape / formula: length ``n_mb + n_st - 1``, tick length ``n_st``,
    slot ``t - s`` or ``None``. Literal 3×2 golden. ``n_stages == 1``,
    ``n_microbatches == 1``, and ``n_microbatches < n_stages`` (more bubble).
schedule faults: ``MeshError`` if either count is 0 or negative.
composition: one output per microbatch, stage 0 first, bitwise vs reference
    and vs literal arrays. Same-shape stage, last-dim change, rank change,
    int32 / int64 / float32. ``n_stages == 1`` and a single microbatch.
call order: log of ``(stage_index, microbatch_index)`` equals non-``None``
    schedule slots, tick-major, increasing stage index. 3 microbatches × 2
    stages differs from microbatch-major, so a naive loop fails. Idle slots
    are not called.
empty inputs: empty microbatches or empty ``stage_fns`` raise ``MeshError``.
stage faults: an exception raised inside a stage propagates as that type,
    not swallowed and not wrapped in ``MeshError``.
"""

from __future__ import annotations

import numpy as np
import pytest

import parallel
from tests.reference import parallel_schedule as ref

# Literal spec golden. Not derived from production.
GOLDEN_3_2 = (
    (0, None),
    (1, 0),
    (2, 1),
    (None, 2),
)

# "All stages on microbatch 0, then all on microbatch 1." Not the contract.
NAIVE_MICROBATCH_MAJOR_3_2 = (
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (0, 2),
    (1, 2),
)

# All microbatches on stage 0, then all on stage 1. Also not the contract.
STAGE_MAJOR_3_2 = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
)

# Tick-major, increasing stage index, Nones dropped. Differs from both above.
CALLS_3_2 = (
    (0, 0),
    (0, 1),
    (1, 0),
    (0, 2),
    (1, 1),
    (1, 2),
)

# n_microbatches < n_stages: longer bubble. 2 + 5 - 1 = 6 ticks.
GOLDEN_2_5 = (
    (0, None, None, None, None),
    (1, 0, None, None, None),
    (None, 1, 0, None, None),
    (None, None, 1, 0, None),
    (None, None, None, 1, 0),
    (None, None, None, None, 1),
)

GOLDEN_1_1 = ((0,),)

GOLDEN_4_1 = (
    (0,),
    (1,),
    (2,),
    (3,),
)

GOLDEN_1_4 = (
    (0, None, None, None),
    (None, 0, None, None),
    (None, None, 0, None),
    (None, None, None, 0),
)

# Float32 linear maps. Entries are small integers so float32 is exact.
# y = x @ W, not an affine bias.
W_SAME = np.array(
    [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ],
    dtype=np.float32,
)
W_CHANGE = np.array(
    [
        [1, 0],
        [1, 1],
        [0, 1],
        [2, 1],
    ],
    dtype=np.float32,
)
W_TAIL = np.array(
    [
        [1, 1],
        [0, 2],
    ],
    dtype=np.float32,
)
W_2 = np.array([[1, 2], [0, 1]], dtype=np.float32)

# Hand-evaluated int32 chain: same-shape (2x+1) then pairwise sums (4,) -> (2,).
INT32_MBS = (
    np.array([1, 2, 3, 4], dtype=np.int32),
    np.array([0, 1, 2, 3], dtype=np.int32),
    np.array([5, 1, 1, 2], dtype=np.int32),
)
INT32_EXPECTED = (
    np.array([8, 16], dtype=np.int32),
    np.array([4, 12], dtype=np.int32),
    np.array([14, 8], dtype=np.int32),
)

# Hand-evaluated float32 chain: (4,) -> (4,) -> (2,) -> (2,).
F32_MBS = (
    np.array([1, 2, 3, 4], dtype=np.float32),
    np.array([2, 0, 1, 3], dtype=np.float32),
)
F32_EXPECTED = (
    np.array([22, 54], dtype=np.float32),
    np.array([12, 30], dtype=np.float32),
)

# Leading dimension preserved; last dim 4 -> 2.
F32_2D_MBS = (
    np.array([[1, 2, 3, 4], [0, 1, 0, 1]], dtype=np.float32),
    np.array([[2, 0, 1, 3], [1, 1, 1, 1]], dtype=np.float32),
)
F32_2D_EXPECTED = (
    np.array([[22, 16], [6, 4]], dtype=np.float32),
    np.array([[12, 9], [8, 6]], dtype=np.float32),
)


class StageFault(Exception):
    """Stage failure. Must surface as this type, not as MeshError."""


def _calls_from_schedule(sched: tuple[tuple[int | None, ...], ...]) -> tuple[tuple[int, int], ...]:
    """Non-None slots, tick-major, increasing stage index."""
    return tuple(
        (stage, mb) for tick in sched for stage, mb in enumerate(tick) if mb is not None
    )


def _naive_microbatch_major(n_mb: int, n_st: int) -> tuple[tuple[int, int], ...]:
    """Wrong order: finish every stage on one microbatch before the next."""
    return tuple((s, m) for m in range(n_mb) for s in range(n_st))


def _assert_schedule(got: object, n_mb: int, n_st: int) -> None:
    """Spec formula: length, tick width, slot ``t - s`` or None, Python ints."""
    assert type(got) is tuple
    assert len(got) == n_mb + n_st - 1
    for t, tick in enumerate(got):
        assert type(tick) is tuple
        assert len(tick) == n_st
        for s, slot in enumerate(tick):
            mb = t - s
            if 0 <= mb < n_mb:
                assert type(slot) is int
                assert slot == mb
            else:
                assert slot is None
    for s in range(n_st):
        seen = [tick[s] for tick in got if tick[s] is not None]
        assert seen == list(range(n_mb))


def _assert_bitwise(got: object, expected: np.ndarray, *, label: str) -> None:
    """Exact dtype, shape, and bytes. Not ``np.allclose`` and not a 1e-5 band."""
    assert isinstance(got, np.ndarray), (label, type(got))
    assert not isinstance(got, (tuple, list)), label
    assert got.dtype == expected.dtype, (label, got.dtype, expected.dtype)
    assert got.shape == expected.shape, (label, got.shape, expected.shape)
    assert got.ndim == expected.ndim, (label, got.ndim, expected.ndim)
    assert got.tobytes() == expected.tobytes(), label


def _assert_exact_tuple(got: object, expected: tuple[np.ndarray, ...]) -> None:
    assert type(got) is tuple, type(got)
    assert len(got) == len(expected)
    for i, (g, e) in enumerate(zip(got, expected, strict=True)):
        _assert_bitwise(g, e, label=f"microbatch {i}")


def _i32_same(x: np.ndarray) -> np.ndarray:
    """Same shape, same dtype: y = 2x + 1."""
    x = np.asarray(x, dtype=np.int32)
    return x * np.int32(2) + np.int32(1)


def _i32_change(x: np.ndarray) -> np.ndarray:
    """Last dim 4 -> 2. Pairwise sums. Input dim != output dim."""
    x = np.asarray(x, dtype=np.int32)
    return np.array([x[0] + x[1], x[2] + x[3]], dtype=np.int32)


def _i32_same_reduced(x: np.ndarray) -> np.ndarray:
    """Same shape on the reduced vector: y = 3x - 1."""
    x = np.asarray(x, dtype=np.int32)
    return x * np.int32(3) - np.int32(1)


def _f32_same(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) @ W_SAME


def _f32_change(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) @ W_CHANGE


def _f32_tail(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) @ W_TAIL


def _f32_2(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) @ W_2


def _rank_up(x: np.ndarray) -> np.ndarray:
    """(d,) -> (2, d). Rank and shape both change."""
    x = np.asarray(x, dtype=np.int32)
    return np.stack([x, x + np.int32(1)], axis=0)


def _rank_same(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.int32) + np.int32(1)


def _times4_plus2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int32)
    return x * np.int32(4) + np.int32(2)


def _mirror(x: np.ndarray) -> np.ndarray:
    """(d,) -> (2d,)."""
    x = np.asarray(x, dtype=np.int32)
    return np.concatenate([x, x[::-1]])


def _plus_i32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.int32) + np.int32(1)


def _pack(mb: int, payload: int) -> np.ndarray:
    return np.array([mb, payload], dtype=np.int64)


def _logging_stage(
    stage_index: int,
    log: list[tuple[int, int]],
    payloads: list[int],
):
    """Record ``(stage_index, microbatch_index)`` and add 1 to the payload tag.

    The microbatch index lives in the activation so a copy still identifies
    the microbatch. Payload on entry to stage ``s`` is ``initial + s`` only
    if the previous stage's output was chained in.
    """

    def fn(activation):
        arr = np.asarray(activation, dtype=np.int64).reshape(-1)
        mb = int(arr[0])
        payload = int(arr[1])
        log.append((stage_index, mb))
        payloads.append(payload)
        return np.array([mb, payload + 1], dtype=np.int64)

    return fn


def _assert_ref_matches_expected(
    mbs: tuple[np.ndarray, ...] | list[np.ndarray],
    fns: tuple | list,
    expected: tuple[np.ndarray, ...],
) -> None:
    """Reference schedule-order run and sequential composition agree with the golden."""
    seq = ref.sequential_compose(mbs, fns)
    pip = ref.circular_pipeline(mbs, fns)
    _assert_exact_tuple(seq, expected)
    _assert_exact_tuple(pip, expected)


# ---------------------------------------------------------------------------
# pipeline_forward_schedule
# ---------------------------------------------------------------------------


def test_pipeline_forward_schedule_3_2_golden() -> None:
    """Literal golden required by the contract, plus the slot formula."""
    assert ref.pipeline_forward_schedule(3, 2) == GOLDEN_3_2
    _assert_schedule(ref.pipeline_forward_schedule(3, 2), 3, 2)
    assert _calls_from_schedule(GOLDEN_3_2) == CALLS_3_2
    got = parallel.pipeline_forward_schedule(3, 2)
    assert got == GOLDEN_3_2
    _assert_schedule(got, 3, 2)
    assert got == ref.pipeline_forward_schedule(3, 2)


@pytest.mark.parametrize(
    "n_mb,n_st,golden",
    [
        pytest.param(1, 1, GOLDEN_1_1, id="one-by-one"),
        pytest.param(4, 1, GOLDEN_4_1, id="n_stages_eq_1"),
        pytest.param(1, 4, GOLDEN_1_4, id="n_microbatches_eq_1"),
        pytest.param(2, 5, GOLDEN_2_5, id="more_bubble_mb_lt_stages"),
    ],
)
def test_pipeline_forward_schedule_literal_edges(n_mb: int, n_st: int, golden: tuple) -> None:
    """n_stages == 1, n_microbatches == 1, and a wider bubble, asserted literally."""
    assert ref.pipeline_forward_schedule(n_mb, n_st) == golden
    _assert_schedule(golden, n_mb, n_st)
    # Bubble case has strictly more idle slots than live calls.
    if n_mb < n_st:
        n_ticks = n_mb + n_st - 1
        assert n_ticks * n_st > n_mb * n_st
        assert sum(slot is None for tick in golden for slot in tick) > 0
    got = parallel.pipeline_forward_schedule(n_mb, n_st)
    assert got == golden
    _assert_schedule(got, n_mb, n_st)
    assert type(got) is tuple
    assert all(type(tick) is tuple for tick in got)


def test_pipeline_forward_schedule_formula_grid() -> None:
    """Property: every positive pair matches the t - s formula and the reference."""
    pairs = [(n_mb, n_st) for n_mb in range(1, 9) for n_st in range(1, 9)]
    pairs.extend([(1, 12), (12, 1), (2, 12), (3, 7), (7, 3), (3, 2)])
    for n_mb, n_st in pairs:
        expected = ref.pipeline_forward_schedule(n_mb, n_st)
        _assert_schedule(expected, n_mb, n_st)
        assert len(expected) == n_mb + n_st - 1
        assert all(len(tick) == n_st for tick in expected)
    for n_mb, n_st in pairs:
        got = parallel.pipeline_forward_schedule(n_mb, n_st)
        _assert_schedule(got, n_mb, n_st)
        assert got == ref.pipeline_forward_schedule(n_mb, n_st)
        again = parallel.pipeline_forward_schedule(n_mb, n_st)
        assert again == got


@pytest.mark.parametrize(
    "n_mb,n_st",
    [
        (0, 1),
        (0, 3),
        (-1, 1),
        (-2, 4),
        (1, 0),
        (3, 0),
        (1, -1),
        (4, -3),
        (0, 0),
        (-1, 0),
        (0, -1),
        (-5, -6),
    ],
)
def test_pipeline_forward_schedule_rejects_nonpositive(n_mb: int, n_st: int) -> None:
    """MeshError for 0 and negative counts. Not a bare ValueError, not a swallow."""
    with pytest.raises(parallel.MeshError):
        ref.pipeline_forward_schedule(n_mb, n_st)
    with pytest.raises(parallel.MeshError):
        parallel.pipeline_forward_schedule(n_mb, n_st)


# ---------------------------------------------------------------------------
# circular_pipeline values (bitwise, not a tolerance)
# ---------------------------------------------------------------------------


def test_circular_pipeline_int32_same_shape_and_dim_change() -> None:
    """Stage 0 keeps shape; stage 1 changes last dim. Bitwise int32, 3 microbatches."""
    fns = (_i32_same, _i32_change)
    mid = _i32_same(INT32_MBS[0])
    assert mid.shape == INT32_MBS[0].shape
    assert mid.dtype == np.int32
    changed = _i32_change(mid)
    assert changed.shape[-1] != mid.shape[-1]
    assert changed.shape == (2,)
    _assert_ref_matches_expected(INT32_MBS, fns, INT32_EXPECTED)
    got = parallel.circular_pipeline(list(INT32_MBS), list(fns))
    _assert_exact_tuple(got, INT32_EXPECTED)
    assert len(got) == 3
    # One value per microbatch, not one per tick (3 + 2 - 1 == 4).
    assert len(got) != 3 + 2 - 1
    assert got[0].dtype == np.int32
    assert got[0].shape == (2,)
    assert got[0].shape != INT32_MBS[0].shape
    for i in range(3):
        _assert_bitwise(got[i], INT32_EXPECTED[i], label=f"mb{i}")


def test_circular_pipeline_float32_linear_maps_bitwise() -> None:
    """Same-shape matmul, dim-changing matmul, then same-shape on the new dim.

    Small integers, float32. Exact bytes, not rtol 1e-5.
    """
    fns = (_f32_same, _f32_change, _f32_tail)
    mid = _f32_same(F32_MBS[0])
    assert mid.shape == F32_MBS[0].shape
    assert mid.dtype == np.float32
    narrowed = _f32_change(mid)
    assert narrowed.shape[-1] != mid.shape[-1]
    assert narrowed.dtype == np.float32
    tail = _f32_tail(narrowed)
    assert tail.shape == narrowed.shape
    _assert_ref_matches_expected(F32_MBS, fns, F32_EXPECTED)
    got = parallel.circular_pipeline(F32_MBS, fns)
    _assert_exact_tuple(got, F32_EXPECTED)
    assert len(got) == 2
    assert len(got) != 2 + 3 - 1
    assert got[0].dtype == np.float32
    assert got[0].dtype != np.float64
    assert got[1].tobytes() == F32_EXPECTED[1].tobytes()
    # A 1-ulp twist of the golden must not be what we accept.
    twisted = F32_EXPECTED[0].copy()
    twisted_bits = twisted.view(np.uint32)
    twisted_bits[0] = twisted_bits[0] ^ np.uint32(1)
    assert twisted.tobytes() != F32_EXPECTED[0].tobytes()
    assert got[0].tobytes() != twisted.tobytes()


def test_circular_pipeline_float32_leading_dim_preserved() -> None:
    """2D activation: same-shape then last-dim change. Leading dim stays 2."""
    fns = (_f32_same, _f32_change)
    assert F32_2D_MBS[0].shape == (2, 4)
    assert F32_2D_EXPECTED[0].shape == (2, 2)
    assert F32_2D_MBS[0].shape[-1] != F32_2D_EXPECTED[0].shape[-1]
    _assert_ref_matches_expected(F32_2D_MBS, fns, F32_2D_EXPECTED)
    got = parallel.circular_pipeline(F32_2D_MBS, fns)
    _assert_exact_tuple(got, F32_2D_EXPECTED)
    assert got[0].ndim == 2
    assert got[0].shape[0] == F32_2D_MBS[0].shape[0]
    assert got[0].dtype == np.float32


def test_circular_pipeline_rank_change() -> None:
    """Stage may change rank: (3,) -> (2, 3), then a same-rank add."""
    mbs = (
        np.array([1, 2, 3], dtype=np.int32),
        np.array([4, 0, 7], dtype=np.int32),
    )
    expected = (
        np.array([[2, 3, 4], [3, 4, 5]], dtype=np.int32),
        np.array([[5, 1, 8], [6, 2, 9]], dtype=np.int32),
    )
    fns = (_rank_up, _rank_same)
    assert mbs[0].ndim == 1
    assert _rank_up(mbs[0]).ndim == 2
    assert _rank_up(mbs[0]).shape != mbs[0].shape
    _assert_ref_matches_expected(mbs, fns, expected)
    got = parallel.circular_pipeline(mbs, fns)
    _assert_exact_tuple(got, expected)
    assert got[0].ndim == 2
    assert got[0].shape == (2, 3)
    assert got[0].dtype == np.int32


def test_circular_pipeline_int64_bitwise() -> None:
    """int64 chain, including a negative result, not cast to float or int32."""
    mbs = (
        np.array([1, 2], dtype=np.int64),
        np.array([9, 8], dtype=np.int64),
    )

    def same(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.int64) * np.int64(3)

    def change(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.int64)
        return np.array([x[0] - x[1]], dtype=np.int64)

    expected = (
        np.array([-3], dtype=np.int64),
        np.array([3], dtype=np.int64),
    )
    fns = (same, change)
    assert mbs[0].shape[-1] != expected[0].shape[-1]
    _assert_ref_matches_expected(mbs, fns, expected)
    got = parallel.circular_pipeline(mbs, fns)
    _assert_exact_tuple(got, expected)
    assert got[0].dtype == np.int64
    assert got[0].shape == (1,)


def test_circular_pipeline_n_stages_eq_1_same_shape() -> None:
    """A single stage still matches composition. Shape and dtype unchanged."""
    mbs = (
        np.array([1, 2], dtype=np.int32),
        np.array([3, 4], dtype=np.int32),
    )
    expected = (
        np.array([6, 10], dtype=np.int32),
        np.array([14, 18], dtype=np.int32),
    )
    fns = (_times4_plus2,)
    assert _times4_plus2(mbs[0]).shape == mbs[0].shape
    _assert_ref_matches_expected(mbs, fns, expected)
    got = parallel.circular_pipeline(mbs, fns)
    _assert_exact_tuple(got, expected)
    assert len(got) == 2
    assert got[0].shape == mbs[0].shape
    assert got[0].dtype == np.int32


def test_circular_pipeline_n_stages_eq_1_shape_change() -> None:
    """n_stages == 1 with input dim != output dim."""
    mbs = [
        np.array([1, 2, 3], dtype=np.int32),
        np.array([4, 5, 6], dtype=np.int32),
    ]
    expected = (
        np.array([1, 2, 3, 3, 2, 1], dtype=np.int32),
        np.array([4, 5, 6, 6, 5, 4], dtype=np.int32),
    )
    fns = [_mirror]
    assert _mirror(mbs[0]).shape[-1] != mbs[0].shape[-1]
    _assert_ref_matches_expected(mbs, fns, expected)
    got = parallel.circular_pipeline(mbs, fns)
    _assert_exact_tuple(got, expected)
    assert got[0].shape == (6,)
    assert got[0].dtype == np.int32
    assert got[0].shape != mbs[0].shape


def test_circular_pipeline_single_microbatch() -> None:
    """One microbatch, three stages (same, dim-change, same). Still composition."""
    mbs = (np.array([1, 2, 3, 4], dtype=np.int32),)
    expected = (np.array([23, 47], dtype=np.int32),)
    fns = (_i32_same, _i32_change, _i32_same_reduced)
    assert fns[0](mbs[0]).shape == mbs[0].shape
    assert fns[1](fns[0](mbs[0])).shape[-1] != mbs[0].shape[-1]
    _assert_ref_matches_expected(mbs, fns, expected)
    got = parallel.circular_pipeline(mbs, fns)
    _assert_exact_tuple(got, expected)
    assert len(got) == 1
    assert type(got) is tuple
    assert got[0].shape == (2,)
    assert got[0].dtype == np.int32


def test_circular_pipeline_one_by_one() -> None:
    """n_stages == 1 and a single microbatch. Float32 linear map, exact bytes."""
    mbs = (np.array([8, 9], dtype=np.float32),)
    expected = (np.array([8, 25], dtype=np.float32),)
    fns = (_f32_2,)
    assert _f32_2(mbs[0]).shape == mbs[0].shape
    _assert_ref_matches_expected(mbs, fns, expected)
    got = parallel.circular_pipeline(mbs, fns)
    _assert_exact_tuple(got, expected)
    assert len(got) == 1
    assert got[0].dtype == np.float32
    assert got[0].shape == (2,)
    assert got[0].tobytes() == expected[0].tobytes()


def test_circular_pipeline_more_stages_than_microbatches() -> None:
    """n_microbatches < n_stages still equals sequential +1 composition."""
    mbs = (
        np.array([1, 2], dtype=np.int32),
        np.array([10, 20], dtype=np.int32),
    )
    n_st = 5
    fns = (_plus_i32,) * n_st
    expected = (
        np.array([6, 7], dtype=np.int32),
        np.array([15, 25], dtype=np.int32),
    )
    assert n_st > len(mbs)
    _assert_ref_matches_expected(mbs, fns, expected)
    got = parallel.circular_pipeline(mbs, fns)
    _assert_exact_tuple(got, expected)
    assert len(got) == 2
    assert len(got) != len(mbs) + n_st - 1
    assert got[0].shape == mbs[0].shape
    assert got[0].dtype == np.int32


def test_circular_pipeline_composition_grid() -> None:
    """Property: many (n_mb, n_st) pairs match sequential composition bitwise."""
    pairs = [(n_mb, n_st) for n_mb in (1, 2, 3, 5) for n_st in (1, 2, 4, 6)]
    cases: list[tuple[tuple[np.ndarray, ...], tuple, tuple[np.ndarray, ...]]] = []
    for n_mb, n_st in pairs:
        mbs = tuple(np.array([m, m + 1, 3], dtype=np.int32) for m in range(n_mb))
        fns = (_plus_i32,) * n_st
        expected = tuple(mb + np.int32(n_st) for mb in mbs)
        cases.append((mbs, fns, expected))
    for mbs, fns, expected in cases:
        _assert_ref_matches_expected(mbs, fns, expected)
        assert len(expected) == len(mbs)
        assert expected[0].dtype == np.int32
        assert expected[0].shape == mbs[0].shape
    for mbs, fns, expected in cases:
        got = parallel.circular_pipeline(mbs, fns)
        _assert_exact_tuple(got, expected)
        assert len(got) == len(mbs)
        if len(fns) > 1:
            assert len(got) != len(mbs) + len(fns) - 1


# ---------------------------------------------------------------------------
# call order (ppermute / tick-major, not microbatch-major)
# ---------------------------------------------------------------------------


def _run_logged(n_mb: int, n_st: int, payload0: int):
    log: list[tuple[int, int]] = []
    payloads: list[int] = []
    stages = [_logging_stage(s, log, payloads) for s in range(n_st)]
    mbs = [_pack(m, payload0 * (m + 1)) for m in range(n_mb)]
    return mbs, stages, log, payloads


def _assert_payload_chain(payloads: list[int], log: list[tuple[int, int]], payload0: int) -> None:
    assert len(payloads) == len(log)
    for (stage, mb), payload in zip(log, payloads, strict=True):
        assert payload == payload0 * (mb + 1) + stage


def test_call_order_3_2_is_tick_major_not_microbatch_major() -> None:
    """3×2 schedule order differs from a naive sequential loop. That loop must fail.

    Tick 1 of the golden is (1, 0): stage 0 runs microbatch 1, then stage 1
    runs microbatch 0. Recorded calls are only the non-None slots.
    """
    assert CALLS_3_2 == _calls_from_schedule(GOLDEN_3_2)
    assert CALLS_3_2 != NAIVE_MICROBATCH_MAJOR_3_2
    assert CALLS_3_2 != STAGE_MAJOR_3_2
    # The distinguishing adjacent pair: not (1, 0) before (0, 1).
    assert CALLS_3_2[1:3] == ((0, 1), (1, 0))
    assert ref.schedule_call_order(3, 2) == CALLS_3_2
    assert ref.pipeline_forward_schedule(3, 2) == GOLDEN_3_2
    n_ticks = 3 + 2 - 1
    assert n_ticks * 2 > 3 * 2  # idle slots exist; calling every slot is too many
    assert len(CALLS_3_2) == 3 * 2

    payload0 = 10
    mbs, stages, ref_log, ref_payloads = _run_logged(3, 2, payload0)
    ref_out = ref.circular_pipeline(mbs, stages)
    assert tuple(ref_log) == CALLS_3_2
    _assert_payload_chain(ref_payloads, ref_log, payload0)
    for m, row in enumerate(ref_out):
        _assert_bitwise(
            row,
            np.array([m, payload0 * (m + 1) + 2], dtype=np.int64),
            label=f"ref mb{m}",
        )

    log: list[tuple[int, int]] = []
    payloads: list[int] = []
    stages = [_logging_stage(s, log, payloads) for s in range(2)]
    mbs = [_pack(m, payload0 * (m + 1)) for m in range(3)]
    got = parallel.circular_pipeline(mbs, stages)
    assert tuple(log) == CALLS_3_2
    assert tuple(log) != NAIVE_MICROBATCH_MAJOR_3_2
    assert tuple(log) != STAGE_MAJOR_3_2
    assert len(log) == 3 * 2
    live = {(s, mb) for tick in GOLDEN_3_2 for s, mb in enumerate(tick) if mb is not None}
    assert set(log) == live
    # No call whose (stage, microbatch) is absent from the schedule.
    assert all((s, mb) in live for s, mb in log)
    _assert_payload_chain(payloads, log, payload0)
    _assert_exact_tuple(
        got,
        tuple(np.array([m, payload0 * (m + 1) + 2], dtype=np.int64) for m in range(3)),
    )


def test_call_order_bubble_does_not_call_none_slots() -> None:
    """2 microbatches, 4 stages: idle ticks are not calls. Order != naive."""
    n_mb, n_st = 2, 4
    sched = (
        (0, None, None, None),
        (1, 0, None, None),
        (None, 1, 0, None),
        (None, None, 1, 0),
        (None, None, None, 1),
    )
    assert ref.pipeline_forward_schedule(n_mb, n_st) == sched
    expected = _calls_from_schedule(sched)
    assert expected == (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    )
    naive = _naive_microbatch_major(n_mb, n_st)
    assert expected != naive
    assert len(expected) == n_mb * n_st
    assert sum(slot is None for tick in sched for slot in tick) == (2 + 4 - 1) * 4 - n_mb * n_st
    assert ref.schedule_call_order(n_mb, n_st) == expected

    payload0 = 7
    log: list[tuple[int, int]] = []
    payloads: list[int] = []
    stages = [_logging_stage(s, log, payloads) for s in range(n_st)]
    mbs = [_pack(m, payload0 * (m + 1)) for m in range(n_mb)]
    # Reference itself must not call None slots.
    ref_log: list[tuple[int, int]] = []
    ref_payloads: list[int] = []
    ref_stages = [_logging_stage(s, ref_log, ref_payloads) for s in range(n_st)]
    ref.circular_pipeline(mbs, ref_stages)
    assert tuple(ref_log) == expected

    got = parallel.circular_pipeline(mbs, stages)
    assert tuple(log) == expected
    assert tuple(log) != naive
    assert len(log) == n_mb * n_st
    _assert_payload_chain(payloads, log, payload0)
    _assert_exact_tuple(
        got,
        tuple(np.array([m, payload0 * (m + 1) + n_st], dtype=np.int64) for m in range(n_mb)),
    )


def test_call_order_grid_matches_schedule() -> None:
    """Property: logged calls equal the schedule for every small pair, including bubble."""
    pairs = [(n_mb, n_st) for n_mb in range(1, 6) for n_st in range(1, 6)]
    pairs.append((2, 5))
    for n_mb, n_st in pairs:
        sched = ref.pipeline_forward_schedule(n_mb, n_st)
        calls = _calls_from_schedule(sched)
        assert calls == ref.schedule_call_order(n_mb, n_st)
        assert len(calls) == n_mb * n_st
        if n_mb > 1 and n_st > 1:
            assert calls != _naive_microbatch_major(n_mb, n_st)
    payload0 = 3
    for n_mb, n_st in pairs:
        log: list[tuple[int, int]] = []
        payloads: list[int] = []
        stages = [_logging_stage(s, log, payloads) for s in range(n_st)]
        mbs = [_pack(m, payload0 * (m + 1)) for m in range(n_mb)]
        got = parallel.circular_pipeline(tuple(mbs), tuple(stages))
        expected = ref.schedule_call_order(n_mb, n_st)
        assert tuple(log) == expected
        if n_mb > 1 and n_st > 1:
            assert tuple(log) != _naive_microbatch_major(n_mb, n_st)
        _assert_payload_chain(payloads, log, payload0)
        _assert_exact_tuple(
            got,
            tuple(
                np.array([m, payload0 * (m + 1) + n_st], dtype=np.int64) for m in range(n_mb)
            ),
        )


def test_call_order_n_stages_eq_1() -> None:
    """Single stage: schedule order is one call per microbatch, in index order."""
    expected = ((0, 0), (0, 1), (0, 2))
    assert ref.schedule_call_order(3, 1) == expected
    assert _calls_from_schedule(ref.pipeline_forward_schedule(3, 1)) == expected
    log: list[tuple[int, int]] = []
    payloads: list[int] = []
    stages = [_logging_stage(0, log, payloads)]
    mbs = [_pack(m, 4 * (m + 1)) for m in range(3)]
    got = parallel.circular_pipeline(mbs, stages)
    assert tuple(log) == expected
    assert len(log) == 3
    _assert_payload_chain(payloads, log, 4)
    _assert_exact_tuple(
        got,
        tuple(np.array([m, 4 * (m + 1) + 1], dtype=np.int64) for m in range(3)),
    )


def test_call_order_single_microbatch() -> None:
    """One microbatch, three stages: each stage called once, stage 0 first."""
    expected = ((0, 0), (1, 0), (2, 0))
    assert ref.schedule_call_order(1, 3) == expected
    log: list[tuple[int, int]] = []
    payloads: list[int] = []
    stages = [_logging_stage(s, log, payloads) for s in range(3)]
    mbs = [_pack(0, 5)]
    got = parallel.circular_pipeline(mbs, stages)
    assert tuple(log) == expected
    assert len(log) == 3
    _assert_payload_chain(payloads, log, 5)
    _assert_bitwise(got[0], np.array([0, 8], dtype=np.int64), label="only")


# ---------------------------------------------------------------------------
# empty inputs and stage exceptions
# ---------------------------------------------------------------------------


def _identity(x):
    return x


@pytest.mark.parametrize(
    "microbatches,stage_fns",
    [
        pytest.param((), (_identity,), id="empty-tuple-microbatches"),
        pytest.param([], [_identity], id="empty-list-microbatches"),
        pytest.param((np.array([1, 2, 3], dtype=np.int32),), (), id="empty-tuple-stages"),
        pytest.param([np.array([1, 2, 3], dtype=np.int32)], [], id="empty-list-stages"),
        pytest.param((), (), id="both-empty-tuples"),
        pytest.param([], [], id="both-empty-lists"),
    ],
)
def test_empty_microbatches_or_stage_fns_raise_mesh_error(microbatches, stage_fns) -> None:
    """Empty microbatches or empty stage_fns raise parallel.MeshError."""
    with pytest.raises(parallel.MeshError):
        ref.sequential_compose(microbatches, stage_fns)
    with pytest.raises(parallel.MeshError):
        ref.circular_pipeline(microbatches, stage_fns)
    with pytest.raises(parallel.MeshError):
        parallel.circular_pipeline(microbatches, stage_fns)


def test_stage_exception_propagates_unwrapped() -> None:
    """An exception inside a stage is not swallowed and not wrapped in MeshError."""
    assert not issubclass(StageFault, parallel.MeshError)

    def boom(_x):
        raise StageFault("stage boom")

    mb = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(StageFault, match="stage boom") as ei:
        parallel.circular_pipeline((mb,), (boom,))
    assert type(ei.value) is StageFault
    assert not isinstance(ei.value, parallel.MeshError)


def test_later_stage_exception_propagates_unwrapped() -> None:
    """A fault in a later stage still propagates as that exception, not MeshError."""
    assert not issubclass(StageFault, parallel.MeshError)

    def ok(x):
        return np.asarray(x, dtype=np.int32) + np.int32(1)

    def boom(_x):
        raise StageFault("later stage boom")

    mbs = (
        np.array([1, 2], dtype=np.int32),
        np.array([3, 4], dtype=np.int32),
    )
    with pytest.raises(StageFault, match="later stage boom") as ei:
        parallel.circular_pipeline(mbs, (ok, boom))
    assert type(ei.value) is StageFault
    assert not isinstance(ei.value, parallel.MeshError)


def test_builtin_exception_propagates_not_mesh_error() -> None:
    """A builtin raised by a stage is not replaced by MeshError."""

    def boom(_x):
        raise KeyError("missing-key")

    mb = np.array([8], dtype=np.int32)
    with pytest.raises(KeyError, match="missing-key") as ei:
        parallel.circular_pipeline((mb, mb + 1), (boom, _identity))
    assert type(ei.value) is KeyError
    assert not isinstance(ei.value, parallel.MeshError)
