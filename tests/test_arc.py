"""Implementer-facing tests for I8 ``arc/`` (TTT sidecar, grid DSL, executor).

Groups
------
constants / interface
    N_COLORS, N_DIHEDRAL, size bounds, DIHEDRAL_NAMES, OpKind values,
    ArcError subclassing, frozen dataclasses. These inspect types and
    constants only and MAY pass against the stub.
make_grid
    Freeze as tuple-of-tuples of int; empty / jagged / size / color faults.
apply_dihedral
    Frozen D4 goldens (including rot90 of ``[[1,2,3],[4,5,6]]``), H/W swap,
    1x1, out-of-range index.
inverse_dihedral
    Locked inverse table; inverse(inverse(k)) == k; apply-inverse restores.
apply_color_perm
    Identity, swap including color 0, cycle; bad perm faults.
execute
    Empty program identity; left-to-right; recolor 1->2; src==dst no-op;
    dihedral ops ignore src/dst; missing / out-of-range recolor faults.
consistent
    All pairs must match; empty train faults.
synthesize
    First catalog hit (rot90 vs flip_h); None if none match; empty faults.
vote
    Majority; ties keep earliest; shape mismatch does not tally together;
    empty faults.
ensemble_predict
    Identity program over (0, 1, 4) restores a non-symmetric grid; empty
    and out-of-range dihedrals fault.
sidecar_hook
    ``sidecar_hook("t", "l")``; empty ids fault. Does not train.
properties
    Production matches ``tests.reference.arc_ttt`` on brute-force small
    grids. Discrete CPU only: no gradients, no JAX, no GPU, no live SGLang.

Goldens are frozen from ``tests.reference.arc_ttt``, not from production.
Every call into a stubbed function body hits ``NotImplementedError`` today,
so those tests must be red. After I8 is implemented, production must match
the reference. Production must never import ``tests/``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import arc
from tests.reference import arc_ttt as ref

# ---------------------------------------------------------------------------
# Frozen goldens, computed from tests.reference.arc_ttt (not production).
# ---------------------------------------------------------------------------

SRC_2X3 = ((1, 2, 3), (4, 5, 6))
SRC_1X1 = ((7,),)

# apply_dihedral goldens on SRC_2X3, index order matching E3 / DIHEDRAL_NAMES.
GOLDEN_DIHEDRAL_2X3: tuple[tuple[tuple[int, ...], ...], ...] = (
    ((1, 2, 3), (4, 5, 6)),  # 0 identity
    ((4, 1), (5, 2), (6, 3)),  # 1 rot90 CW
    ((6, 5, 4), (3, 2, 1)),  # 2 rot180
    ((3, 6), (2, 5), (1, 4)),  # 3 rot270 CW
    ((3, 2, 1), (6, 5, 4)),  # 4 flip_h
    ((4, 5, 6), (1, 2, 3)),  # 5 flip_v
    ((1, 4), (2, 5), (3, 6)),  # 6 transpose
    ((6, 3), (5, 2), (4, 1)),  # 7 anti_transpose
)

GOLDEN_ROT90 = GOLDEN_DIHEDRAL_2X3[1]
GOLDEN_IDENTITY_2X3 = GOLDEN_DIHEDRAL_2X3[0]
GOLDEN_ROT180_2X3 = GOLDEN_DIHEDRAL_2X3[2]
GOLDEN_FLIP_H_2X3 = GOLDEN_DIHEDRAL_2X3[4]
GOLDEN_FLIP_V_2X3 = GOLDEN_DIHEDRAL_2X3[5]
GOLDEN_TRANSPOSE_2X3 = GOLDEN_DIHEDRAL_2X3[6]

GOLDEN_1X1 = ((7,),)

GOLDEN_INVERSE: tuple[int, ...] = (0, 3, 2, 1, 4, 5, 6, 7)

GOLDEN_RECOLOR = ((0, 2, 2), (2, 2, 3))

GOLDEN_VOTE_A = ((1, 2), (3, 4))
GOLDEN_VOTE_B = ((5,),)

IDENTITY_PERM: tuple[int, ...] = tuple(range(arc.N_COLORS))
SWAP_0_1: tuple[int, ...] = (1, 0, 2, 3, 4, 5, 6, 7, 8, 9)
CYCLE: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 0)

# Dihedrals that swap height and width.
_SWAP_HW = frozenset({1, 3, 6, 7})


def _cells(grid) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(c) for c in row) for row in grid.cells)


def g_prod(rows: tuple[tuple[int, ...], ...] | list[list[int]]) -> arc.Grid:
    return arc.Grid(cells=tuple(tuple(int(c) for c in row) for row in rows))


def g_ref(rows: tuple[tuple[int, ...], ...] | list[list[int]]) -> ref.Grid:
    return ref.Grid(cells=tuple(tuple(int(c) for c in row) for row in rows))


def _assert_grid(got: arc.Grid, exp: ref.Grid, golden=None) -> None:
    assert _cells(got) == _cells(exp)
    if golden is not None:
        assert _cells(got) == golden
    assert isinstance(got, arc.Grid)
    for row in got.cells:
        for color in row:
            assert type(color) is int
            assert 0 <= color < arc.N_COLORS


def _assert_program(got: arc.Program, exp: ref.Program) -> None:
    assert len(got.ops) == len(exp.ops)
    for g_op, e_op in zip(got.ops, exp.ops, strict=True):
        assert g_op.kind == e_op.kind
        assert g_op.src == e_op.src
        assert g_op.dst == e_op.dst
    assert isinstance(got, arc.Program)


def _op_prod(kind: arc.OpKind, src: int | None = None, dst: int | None = None) -> arc.Op:
    return arc.Op(kind=kind, src=src, dst=dst)


def _op_ref(kind: ref.OpKind, src: int | None = None, dst: int | None = None) -> ref.Op:
    return ref.Op(kind=kind, src=src, dst=dst)


# ---------------------------------------------------------------------------
# constants / interface (MAY pass against the stub)
# ---------------------------------------------------------------------------


def test_constants_match_reference():
    assert arc.N_COLORS == ref.N_COLORS == 10
    assert arc.N_DIHEDRAL == ref.N_DIHEDRAL == 8
    assert arc.MIN_GRID_SIZE == ref.MIN_GRID_SIZE == 1
    assert arc.MAX_GRID_SIZE == ref.MAX_GRID_SIZE == 30
    assert arc.DIHEDRAL_NAMES == ref.DIHEDRAL_NAMES
    assert len(arc.DIHEDRAL_NAMES) == arc.N_DIHEDRAL


def test_dihedral_names_order():
    assert arc.DIHEDRAL_NAMES == (
        "identity",
        "rot90",
        "rot180",
        "rot270",
        "flip_h",
        "flip_v",
        "transpose",
        "anti_transpose",
    )


def test_op_kind_values():
    assert arc.OpKind.IDENTITY == "identity"
    assert arc.OpKind.ROT90 == "rot90"
    assert arc.OpKind.ROT180 == "rot180"
    assert arc.OpKind.ROT270 == "rot270"
    assert arc.OpKind.FLIP_H == "flip_h"
    assert arc.OpKind.FLIP_V == "flip_v"
    assert arc.OpKind.TRANSPOSE == "transpose"
    assert arc.OpKind.ANTI_TRANSPOSE == "anti_transpose"
    assert arc.OpKind.RECOLOR == "recolor"
    assert arc.OpKind.IDENTITY == ref.OpKind.IDENTITY
    assert arc.OpKind.RECOLOR == ref.OpKind.RECOLOR


def test_arc_error_is_value_error():
    assert issubclass(arc.ArcError, ValueError)
    assert issubclass(ref.ArcError, ValueError)


def test_dataclasses_are_frozen():
    grid = arc.Grid(cells=((1, 2), (3, 4)))
    pair = arc.Pair(input=grid, output=grid)
    op = arc.Op(kind=arc.OpKind.ROT90)
    program = arc.Program(ops=(op,))
    hook = arc.SidecarHook(task_id="t", lora_id="l")
    with pytest.raises(FrozenInstanceError):
        grid.cells = ((0,),)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        pair.input = grid  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        op.kind = arc.OpKind.IDENTITY  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        program.ops = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        hook.task_id = "x"  # type: ignore[misc]
    assert op.src is None and op.dst is None


# ---------------------------------------------------------------------------
# make_grid
# ---------------------------------------------------------------------------


def test_make_grid_freezes_tuple_of_tuples():
    raw = [[1, 2], [3, 4]]
    got = arc.make_grid(raw)
    exp = ref.make_grid(raw)
    _assert_grid(got, exp, ((1, 2), (3, 4)))
    assert isinstance(got.cells, tuple)
    assert isinstance(got.cells[0], tuple)
    assert type(got.cells[0][0]) is int


def test_make_grid_copies_input():
    raw = [[1, 2], [3, 4]]
    got = arc.make_grid(raw)
    raw[0][0] = 9
    assert got.cells[0][0] == 1


def test_make_grid_1x1_and_bounds():
    got = arc.make_grid([[0]])
    exp = ref.make_grid([[0]])
    _assert_grid(got, exp, ((0,),))
    wide = [[c % arc.N_COLORS for c in range(arc.MAX_GRID_SIZE)]]
    _assert_grid(arc.make_grid(wide), ref.make_grid(wide))
    tall = [[1] for _ in range(arc.MAX_GRID_SIZE)]
    _assert_grid(arc.make_grid(tall), ref.make_grid(tall))


def test_make_grid_empty_raises():
    with pytest.raises(arc.ArcError):
        arc.make_grid([])
    with pytest.raises(ref.ArcError):
        ref.make_grid([])


def test_make_grid_zero_width_raises():
    with pytest.raises(arc.ArcError):
        arc.make_grid([[]])
    with pytest.raises(ref.ArcError):
        ref.make_grid([[]])


def test_make_grid_jagged_raises():
    with pytest.raises(arc.ArcError):
        arc.make_grid([[1, 2, 3], [4, 5]])
    with pytest.raises(ref.ArcError):
        ref.make_grid([[1, 2, 3], [4, 5]])
    with pytest.raises(arc.ArcError):
        arc.make_grid([[1], [2, 3]])


@pytest.mark.parametrize("height", [0, 31, 32, 100])
def test_make_grid_height_out_of_range_raises(height: int):
    if height == 0:
        cells: list[list[int]] = []
    else:
        cells = [[0] for _ in range(height)]
    with pytest.raises(arc.ArcError):
        arc.make_grid(cells)
    with pytest.raises(ref.ArcError):
        ref.make_grid(cells)


@pytest.mark.parametrize("width", [0, 31, 32, 100])
def test_make_grid_width_out_of_range_raises(width: int):
    cells = [[0] * width] if width else [[]]
    with pytest.raises(arc.ArcError):
        arc.make_grid(cells)
    with pytest.raises(ref.ArcError):
        ref.make_grid(cells)


@pytest.mark.parametrize("color", [-1, 10, 11, 99])
def test_make_grid_color_out_of_range_raises(color: int):
    with pytest.raises(arc.ArcError):
        arc.make_grid([[color]])
    with pytest.raises(ref.ArcError):
        ref.make_grid([[color]])


def test_make_grid_color_zero_and_nine_ok():
    got = arc.make_grid([[0, 9], [9, 0]])
    exp = ref.make_grid([[0, 9], [9, 0]])
    _assert_grid(got, exp, ((0, 9), (9, 0)))


# ---------------------------------------------------------------------------
# apply_dihedral
# ---------------------------------------------------------------------------


def test_rot90_clockwise_golden():
    got = arc.apply_dihedral(g_prod(SRC_2X3), 1)
    exp = ref.apply_dihedral(g_ref(SRC_2X3), 1)
    _assert_grid(got, exp, GOLDEN_ROT90)
    assert _cells(got) == ((4, 1), (5, 2), (6, 3))


@pytest.mark.parametrize("index", range(8))
def test_apply_dihedral_2x3_goldens(index: int):
    got = arc.apply_dihedral(g_prod(SRC_2X3), index)
    exp = ref.apply_dihedral(g_ref(SRC_2X3), index)
    _assert_grid(got, exp, GOLDEN_DIHEDRAL_2X3[index])


def test_apply_dihedral_identity_rot180_flips_transpose_2x3():
    grid = g_prod(SRC_2X3)
    assert _cells(arc.apply_dihedral(grid, 0)) == GOLDEN_IDENTITY_2X3
    assert _cells(arc.apply_dihedral(grid, 2)) == GOLDEN_ROT180_2X3
    assert _cells(arc.apply_dihedral(grid, 4)) == GOLDEN_FLIP_H_2X3
    assert _cells(arc.apply_dihedral(grid, 5)) == GOLDEN_FLIP_V_2X3
    assert _cells(arc.apply_dihedral(grid, 6)) == GOLDEN_TRANSPOSE_2X3


@pytest.mark.parametrize("index", range(8))
def test_apply_dihedral_1x1_is_fixed(index: int):
    got = arc.apply_dihedral(g_prod(SRC_1X1), index)
    exp = ref.apply_dihedral(g_ref(SRC_1X1), index)
    _assert_grid(got, exp, GOLDEN_1X1)


def test_apply_dihedral_swaps_height_width():
    src = g_prod(SRC_2X3)
    for index in range(8):
        got = arc.apply_dihedral(src, index)
        height, width = len(got.cells), len(got.cells[0])
        if index in _SWAP_HW:
            assert (height, width) == (3, 2)
        else:
            assert (height, width) == (2, 3)
        # Still a valid grid.
        assert 1 <= height <= arc.MAX_GRID_SIZE
        assert 1 <= width <= arc.MAX_GRID_SIZE


@pytest.mark.parametrize("index", [-1, 8, 9, 99])
def test_apply_dihedral_out_of_range_raises(index: int):
    with pytest.raises(arc.ArcError):
        arc.apply_dihedral(g_prod(SRC_2X3), index)
    with pytest.raises(ref.ArcError):
        ref.apply_dihedral(g_ref(SRC_2X3), index)


# ---------------------------------------------------------------------------
# inverse_dihedral
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index, want", list(enumerate(GOLDEN_INVERSE)))
def test_inverse_dihedral_table(index: int, want: int):
    got = arc.inverse_dihedral(index)
    exp = ref.inverse_dihedral(index)
    assert got == exp == want


def test_inverse_dihedral_involutive():
    for index in range(arc.N_DIHEDRAL):
        assert arc.inverse_dihedral(arc.inverse_dihedral(index)) == index
        assert ref.inverse_dihedral(ref.inverse_dihedral(index)) == index


@pytest.mark.parametrize("index", [-1, 8, 9, 99])
def test_inverse_dihedral_out_of_range_raises(index: int):
    with pytest.raises(arc.ArcError):
        arc.inverse_dihedral(index)
    with pytest.raises(ref.ArcError):
        ref.inverse_dihedral(index)


def test_property_apply_inverse_restores_grid():
    grids = (
        SRC_1X1,
        SRC_2X3,
        ((0,), (1,), (2,)),
        ((9, 8, 7, 6),),
        ((1, 2), (3, 4)),
    )
    for rows in grids:
        prod = g_prod(rows)
        reference = g_ref(rows)
        for index in range(arc.N_DIHEDRAL):
            inv = arc.inverse_dihedral(index)
            got = arc.apply_dihedral(arc.apply_dihedral(prod, index), inv)
            exp = ref.apply_dihedral(
                ref.apply_dihedral(reference, index),
                ref.inverse_dihedral(index),
            )
            _assert_grid(got, exp, rows)


# ---------------------------------------------------------------------------
# apply_color_perm
# ---------------------------------------------------------------------------


def test_apply_color_perm_identity():
    got = arc.apply_color_perm(g_prod(SRC_2X3), IDENTITY_PERM)
    exp = ref.apply_color_perm(g_ref(SRC_2X3), IDENTITY_PERM)
    _assert_grid(got, exp, SRC_2X3)


def test_apply_color_perm_swaps_zero_not_special():
    src = ((0, 1, 2), (0, 0, 1))
    got = arc.apply_color_perm(g_prod(src), SWAP_0_1)
    exp = ref.apply_color_perm(g_ref(src), SWAP_0_1)
    _assert_grid(got, exp, ((1, 0, 2), (1, 1, 0)))


def test_apply_color_perm_cycle():
    src = ((0, 9), (1, 8))
    got = arc.apply_color_perm(g_prod(src), CYCLE)
    exp = ref.apply_color_perm(g_ref(src), CYCLE)
    _assert_grid(got, exp, ((1, 0), (2, 9)))


def test_apply_color_perm_bad_length_raises():
    with pytest.raises(arc.ArcError):
        arc.apply_color_perm(g_prod(SRC_1X1), (0, 1, 2))
    with pytest.raises(ref.ArcError):
        ref.apply_color_perm(g_ref(SRC_1X1), (0, 1, 2))
    with pytest.raises(arc.ArcError):
        arc.apply_color_perm(g_prod(SRC_1X1), ())


def test_apply_color_perm_duplicate_raises():
    bad = (0, 0, 2, 3, 4, 5, 6, 7, 8, 9)
    with pytest.raises(arc.ArcError):
        arc.apply_color_perm(g_prod(SRC_1X1), bad)
    with pytest.raises(ref.ArcError):
        ref.apply_color_perm(g_ref(SRC_1X1), bad)


def test_apply_color_perm_out_of_range_value_raises():
    bad = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    with pytest.raises(arc.ArcError):
        arc.apply_color_perm(g_prod(SRC_1X1), bad)
    with pytest.raises(ref.ArcError):
        ref.apply_color_perm(g_ref(SRC_1X1), bad)


def test_apply_color_perm_missing_color_raises():
    # Length 10 but 9 never appears (10 is invalid and 0 is duplicated).
    bad = (0, 1, 2, 3, 4, 5, 6, 7, 8, 0)
    with pytest.raises(arc.ArcError):
        arc.apply_color_perm(g_prod(SRC_1X1), bad)


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


def test_execute_empty_program_is_identity():
    program = arc.Program(ops=())
    got = arc.execute(program, g_prod(SRC_2X3))
    exp = ref.execute(ref.Program(ops=()), g_ref(SRC_2X3))
    _assert_grid(got, exp, SRC_2X3)


def test_execute_recolor_1_to_2():
    src = ((0, 1, 2), (1, 1, 3))
    prog = arc.Program(ops=(_op_prod(arc.OpKind.RECOLOR, 1, 2),))
    got = arc.execute(prog, g_prod(src))
    exp = ref.execute(
        ref.Program(ops=(_op_ref(ref.OpKind.RECOLOR, 1, 2),)),
        g_ref(src),
    )
    _assert_grid(got, exp, GOLDEN_RECOLOR)


def test_execute_recolor_src_equals_dst_is_noop():
    prog = arc.Program(ops=(_op_prod(arc.OpKind.RECOLOR, 1, 1),))
    got = arc.execute(prog, g_prod(SRC_2X3))
    exp = ref.execute(
        ref.Program(ops=(_op_ref(ref.OpKind.RECOLOR, 1, 1),)),
        g_ref(SRC_2X3),
    )
    _assert_grid(got, exp, SRC_2X3)


def test_execute_recolor_missing_src_or_dst_raises():
    with pytest.raises(arc.ArcError):
        arc.execute(
            arc.Program(ops=(_op_prod(arc.OpKind.RECOLOR, None, 2),)),
            g_prod(SRC_1X1),
        )
    with pytest.raises(ref.ArcError):
        ref.execute(
            ref.Program(ops=(_op_ref(ref.OpKind.RECOLOR, None, 2),)),
            g_ref(SRC_1X1),
        )
    with pytest.raises(arc.ArcError):
        arc.execute(
            arc.Program(ops=(_op_prod(arc.OpKind.RECOLOR, 1, None),)),
            g_prod(SRC_1X1),
        )
    with pytest.raises(arc.ArcError):
        arc.execute(
            arc.Program(ops=(arc.Op(kind=arc.OpKind.RECOLOR),)),
            g_prod(SRC_1X1),
        )


@pytest.mark.parametrize("src, dst", [(-1, 0), (0, -1), (10, 0), (0, 10), (99, 99)])
def test_execute_recolor_out_of_range_raises(src: int, dst: int):
    with pytest.raises(arc.ArcError):
        arc.execute(
            arc.Program(ops=(_op_prod(arc.OpKind.RECOLOR, src, dst),)),
            g_prod(SRC_1X1),
        )
    with pytest.raises(ref.ArcError):
        ref.execute(
            ref.Program(ops=(_op_ref(ref.OpKind.RECOLOR, src, dst),)),
            g_ref(SRC_1X1),
        )


def test_execute_dihedral_ignores_src_dst():
    prog = arc.Program(ops=(_op_prod(arc.OpKind.ROT90, 99, -1),))
    got = arc.execute(prog, g_prod(SRC_2X3))
    exp = ref.execute(
        ref.Program(ops=(_op_ref(ref.OpKind.ROT90, 99, -1),)),
        g_ref(SRC_2X3),
    )
    _assert_grid(got, exp, GOLDEN_ROT90)


def test_execute_ops_left_to_right():
    src = ((1, 2),)
    first_then_second = arc.Program(
        ops=(
            _op_prod(arc.OpKind.RECOLOR, 1, 2),
            _op_prod(arc.OpKind.RECOLOR, 2, 3),
        )
    )
    second_then_first = arc.Program(
        ops=(
            _op_prod(arc.OpKind.RECOLOR, 2, 3),
            _op_prod(arc.OpKind.RECOLOR, 1, 2),
        )
    )
    got_a = arc.execute(first_then_second, g_prod(src))
    exp_a = ref.execute(
        ref.Program(
            ops=(
                _op_ref(ref.OpKind.RECOLOR, 1, 2),
                _op_ref(ref.OpKind.RECOLOR, 2, 3),
            )
        ),
        g_ref(src),
    )
    got_b = arc.execute(second_then_first, g_prod(src))
    exp_b = ref.execute(
        ref.Program(
            ops=(
                _op_ref(ref.OpKind.RECOLOR, 2, 3),
                _op_ref(ref.OpKind.RECOLOR, 1, 2),
            )
        ),
        g_ref(src),
    )
    _assert_grid(got_a, exp_a, ((3, 3),))
    _assert_grid(got_b, exp_b, ((2, 3),))
    assert _cells(got_a) != _cells(got_b)


def test_execute_rot90_then_rot270_is_identity():
    prog = arc.Program(
        ops=(_op_prod(arc.OpKind.ROT90), _op_prod(arc.OpKind.ROT270))
    )
    got = arc.execute(prog, g_prod(SRC_2X3))
    exp = ref.execute(
        ref.Program(ops=(_op_ref(ref.OpKind.ROT90), _op_ref(ref.OpKind.ROT270))),
        g_ref(SRC_2X3),
    )
    _assert_grid(got, exp, SRC_2X3)


# ---------------------------------------------------------------------------
# consistent
# ---------------------------------------------------------------------------


def test_consistent_true_when_every_pair_matches():
    inp = g_prod(SRC_2X3)
    out = g_prod(GOLDEN_FLIP_H_2X3)
    train = (arc.Pair(input=inp, output=out),)
    prog = arc.Program(ops=(_op_prod(arc.OpKind.FLIP_H),))
    got = arc.consistent(prog, train)
    exp = ref.consistent(
        ref.Program(ops=(_op_ref(ref.OpKind.FLIP_H),)),
        (ref.Pair(input=g_ref(SRC_2X3), output=g_ref(GOLDEN_FLIP_H_2X3)),),
    )
    assert got is True
    assert exp is True


def test_consistent_false_when_a_pair_mismatches():
    train = (
        arc.Pair(input=g_prod(SRC_2X3), output=g_prod(GOLDEN_FLIP_H_2X3)),
        arc.Pair(input=g_prod(SRC_1X1), output=g_prod(((0,),))),
    )
    prog = arc.Program(ops=(_op_prod(arc.OpKind.FLIP_H),))
    got = arc.consistent(prog, train)
    exp = ref.consistent(
        ref.Program(ops=(_op_ref(ref.OpKind.FLIP_H),)),
        (
            ref.Pair(input=g_ref(SRC_2X3), output=g_ref(GOLDEN_FLIP_H_2X3)),
            ref.Pair(input=g_ref(SRC_1X1), output=g_ref(((0,),))),
        ),
    )
    assert got is False
    assert exp is False


def test_consistent_empty_train_raises():
    prog = arc.Program(ops=())
    with pytest.raises(arc.ArcError):
        arc.consistent(prog, ())
    with pytest.raises(ref.ArcError):
        ref.consistent(ref.Program(ops=()), ())


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------


def test_synthesize_first_catalog_hit_is_flip_h():
    train = (
        arc.Pair(input=g_prod(SRC_2X3), output=g_prod(GOLDEN_FLIP_H_2X3)),
    )
    catalog = (
        arc.Program(ops=(_op_prod(arc.OpKind.ROT90),)),
        arc.Program(ops=(_op_prod(arc.OpKind.FLIP_H),)),
    )
    got = arc.synthesize(train, catalog)
    exp = ref.synthesize(
        (ref.Pair(input=g_ref(SRC_2X3), output=g_ref(GOLDEN_FLIP_H_2X3)),),
        (
            ref.Program(ops=(_op_ref(ref.OpKind.ROT90),)),
            ref.Program(ops=(_op_ref(ref.OpKind.FLIP_H),)),
        ),
    )
    assert got is not None
    assert exp is not None
    _assert_program(got, exp)
    assert got.ops[0].kind == arc.OpKind.FLIP_H
    assert got.ops[0].kind == "flip_h"


def test_synthesize_returns_none_when_nothing_fits():
    train = (
        arc.Pair(input=g_prod(SRC_2X3), output=g_prod(GOLDEN_FLIP_H_2X3)),
    )
    catalog = (arc.Program(ops=(_op_prod(arc.OpKind.ROT90),)),)
    got = arc.synthesize(train, catalog)
    exp = ref.synthesize(
        (ref.Pair(input=g_ref(SRC_2X3), output=g_ref(GOLDEN_FLIP_H_2X3)),),
        (ref.Program(ops=(_op_ref(ref.OpKind.ROT90),)),),
    )
    assert got is None
    assert exp is None


def test_synthesize_returns_first_when_several_fit():
    train = (arc.Pair(input=g_prod(SRC_1X1), output=g_prod(SRC_1X1)),)
    catalog = (
        arc.Program(ops=(_op_prod(arc.OpKind.ROT90),)),
        arc.Program(ops=(_op_prod(arc.OpKind.FLIP_H),)),
    )
    got = arc.synthesize(train, catalog)
    exp = ref.synthesize(
        (ref.Pair(input=g_ref(SRC_1X1), output=g_ref(SRC_1X1)),),
        (
            ref.Program(ops=(_op_ref(ref.OpKind.ROT90),)),
            ref.Program(ops=(_op_ref(ref.OpKind.FLIP_H),)),
        ),
    )
    assert got is not None
    assert exp is not None
    _assert_program(got, exp)
    assert got.ops[0].kind == arc.OpKind.ROT90


def test_synthesize_empty_train_raises():
    catalog = (arc.Program(ops=()),)
    with pytest.raises(arc.ArcError):
        arc.synthesize((), catalog)
    with pytest.raises(ref.ArcError):
        ref.synthesize((), (ref.Program(ops=()),))


def test_synthesize_empty_catalog_raises():
    train = (arc.Pair(input=g_prod(SRC_1X1), output=g_prod(SRC_1X1)),)
    with pytest.raises(arc.ArcError):
        arc.synthesize(train, ())
    with pytest.raises(ref.ArcError):
        ref.synthesize(
            (ref.Pair(input=g_ref(SRC_1X1), output=g_ref(SRC_1X1)),),
            (),
        )


# ---------------------------------------------------------------------------
# vote
# ---------------------------------------------------------------------------


def test_vote_majority_two_a_one_b():
    a, b = g_prod(GOLDEN_VOTE_A), g_prod(GOLDEN_VOTE_B)
    got = arc.vote((a, b, a))
    exp = ref.vote((g_ref(GOLDEN_VOTE_A), g_ref(GOLDEN_VOTE_B), g_ref(GOLDEN_VOTE_A)))
    _assert_grid(got, exp, GOLDEN_VOTE_A)


def test_vote_tie_keeps_earliest():
    a, b = g_prod(GOLDEN_VOTE_A), g_prod(GOLDEN_VOTE_B)
    got = arc.vote((a, b))
    exp = ref.vote((g_ref(GOLDEN_VOTE_A), g_ref(GOLDEN_VOTE_B)))
    _assert_grid(got, exp, GOLDEN_VOTE_A)


def test_vote_later_majority_overtakes():
    a, b = g_prod(GOLDEN_VOTE_A), g_prod(GOLDEN_VOTE_B)
    got = arc.vote((b, a, a))
    exp = ref.vote((g_ref(GOLDEN_VOTE_B), g_ref(GOLDEN_VOTE_A), g_ref(GOLDEN_VOTE_A)))
    _assert_grid(got, exp, GOLDEN_VOTE_A)


def test_vote_shape_mismatch_does_not_tally():
    a = g_prod(((1, 2), (3, 4)))
    b = g_prod(((1, 2, 3),))
    got = arc.vote((a, b, b))
    exp = ref.vote((g_ref(((1, 2), (3, 4))), g_ref(((1, 2, 3),)), g_ref(((1, 2, 3),))))
    _assert_grid(got, exp, ((1, 2, 3),))


def test_vote_empty_raises():
    with pytest.raises(arc.ArcError):
        arc.vote(())
    with pytest.raises(ref.ArcError):
        ref.vote(())


# ---------------------------------------------------------------------------
# ensemble_predict
# ---------------------------------------------------------------------------


def test_ensemble_identity_program_restores_nonsymmetric_grid():
    program = arc.Program(ops=())
    got = arc.ensemble_predict(program, g_prod(SRC_2X3), (0, 1, 4))
    exp = ref.ensemble_predict(ref.Program(ops=()), g_ref(SRC_2X3), (0, 1, 4))
    _assert_grid(got, exp, SRC_2X3)


def test_ensemble_single_identity_dihedral():
    program = arc.Program(ops=(_op_prod(arc.OpKind.IDENTITY),))
    got = arc.ensemble_predict(program, g_prod(SRC_2X3), (0,))
    exp = ref.ensemble_predict(
        ref.Program(ops=(_op_ref(ref.OpKind.IDENTITY),)),
        g_ref(SRC_2X3),
        (0,),
    )
    _assert_grid(got, exp, SRC_2X3)


def test_ensemble_empty_dihedrals_raises():
    with pytest.raises(arc.ArcError):
        arc.ensemble_predict(arc.Program(ops=()), g_prod(SRC_1X1), ())
    with pytest.raises(ref.ArcError):
        ref.ensemble_predict(ref.Program(ops=()), g_ref(SRC_1X1), ())


@pytest.mark.parametrize("bad", [(-1,), (8,), (0, 8), (9, 0)])
def test_ensemble_out_of_range_dihedral_raises(bad: tuple[int, ...]):
    with pytest.raises(arc.ArcError):
        arc.ensemble_predict(arc.Program(ops=()), g_prod(SRC_1X1), bad)
    with pytest.raises(ref.ArcError):
        ref.ensemble_predict(ref.Program(ops=()), g_ref(SRC_1X1), bad)


# ---------------------------------------------------------------------------
# sidecar_hook
# ---------------------------------------------------------------------------


def test_sidecar_hook_golden():
    got = arc.sidecar_hook("t", "l")
    exp = ref.sidecar_hook("t", "l")
    assert got.task_id == exp.task_id == "t"
    assert got.lora_id == exp.lora_id == "l"
    assert isinstance(got, arc.SidecarHook)


def test_sidecar_hook_nonempty_ids():
    got = arc.sidecar_hook("arc-001", "lora-7")
    exp = ref.sidecar_hook("arc-001", "lora-7")
    assert got.task_id == exp.task_id == "arc-001"
    assert got.lora_id == exp.lora_id == "lora-7"


def test_sidecar_hook_empty_task_id_raises():
    with pytest.raises(arc.ArcError):
        arc.sidecar_hook("", "l")
    with pytest.raises(ref.ArcError):
        ref.sidecar_hook("", "l")


def test_sidecar_hook_empty_lora_id_raises():
    with pytest.raises(arc.ArcError):
        arc.sidecar_hook("t", "")
    with pytest.raises(ref.ArcError):
        ref.sidecar_hook("t", "")


def test_sidecar_hook_both_empty_raises():
    with pytest.raises(arc.ArcError):
        arc.sidecar_hook("", "")
    with pytest.raises(ref.ArcError):
        ref.sidecar_hook("", "")


# ---------------------------------------------------------------------------
# properties: production matches the independent reference
# ---------------------------------------------------------------------------


def test_property_all_dihedrals_match_reference_on_small_grids():
    grids = (
        [[0]],
        [[1, 2, 3], [4, 5, 6]],
        [[9, 8], [7, 6], [5, 4]],
        [[2, 2, 2, 2]],
        [[3], [3], [3]],
        [[0, 1], [2, 3]],
    )
    for rows in grids:
        prod = g_prod(rows)
        reference = g_ref(rows)
        for index in range(arc.N_DIHEDRAL):
            got = arc.apply_dihedral(prod, index)
            exp = ref.apply_dihedral(reference, index)
            _assert_grid(got, exp)
            assert len(got.cells) == len(exp.cells)
            assert len(got.cells[0]) == len(exp.cells[0])


def test_property_color_perm_then_inverse_restores():
    src_rows = [[0, 1, 2], [3, 4, 5]]
    inverse = tuple(CYCLE.index(c) for c in range(arc.N_COLORS))
    got = arc.apply_color_perm(
        arc.apply_color_perm(g_prod(src_rows), CYCLE),
        inverse,
    )
    exp = ref.apply_color_perm(
        ref.apply_color_perm(g_ref(src_rows), CYCLE),
        inverse,
    )
    _assert_grid(got, exp, tuple(tuple(r) for r in src_rows))


def test_property_vote_matches_reference():
    a = ((1,),)
    b = ((2, 2),)
    c = ((1,),)
    cases = (
        (a,),
        (a, b),
        (b, a),
        (a, b, c),
        (b, b, a),
        (a, a, b, b),
    )
    for rows_list in cases:
        got = arc.vote(tuple(g_prod(rows) for rows in rows_list))
        exp = ref.vote(tuple(g_ref(rows) for rows in rows_list))
        _assert_grid(got, exp)
