"""Independent I8 reference: ARC-AGI TTT sidecar, grid DSL, executor (spec 10, 15.5).

Plain Python, slow and obvious. Does not import production ``arc``, ``synth``,
or ``sglang_fork``. Enums, dataclasses, and constants are a local mirror so
tests can compare field-by-field without sharing code. Production must never
import ``tests/``.

Pinned rules
------------
make_grid
    Empty, jagged, height/width outside [MIN_GRID_SIZE, MAX_GRID_SIZE], or a
    color not in 0..9 -> ArcError. Cells frozen as tuple-of-tuples of int.

apply_dihedral
    Index in 0..N_DIHEDRAL (0..8 exclusive) else ArcError. After any dihedral
    the grid still validates. rot90 / rot270 / transpose / anti-transpose
    swap H/W. D4 index order matches E3:
        0 identity
        1 rot90 clockwise. ``[[1,2,3],[4,5,6]]`` -> ``[[4,1],[5,2],[6,3]]``.
        2 rot180
        3 rot270 clockwise
        4 flip left-right (horizontal)
        5 flip up-down (vertical)
        6 transpose (main diagonal)
        7 anti-transpose (anti-diagonal): transpose then rot180.

inverse_dihedral
    0->0, 1->3, 2->2, 3->1, 4->4, 5->5, 6->6, 7->7. Out of range -> ArcError.
    inverse(inverse(k)) == k. apply(inverse(k), apply(k, g)) == g.

apply_color_perm
    perm[c] is the new color of c. Length 10, each of 0..9 once. Color 0 is
    not special. Bad perm -> ArcError.

execute
    Ops left to right. Empty program is identity. RECOLOR requires src and
    dst in 0..9; missing or out of range -> ArcError. src==dst is a no-op.
    Dihedral ops ignore src/dst.

consistent
    True iff execute(program, pair.input) equals pair.output (cell-wise) for
    every pair. Empty train -> ArcError.

synthesize
    First catalog program consistent with every train pair, else None.
    Empty train or empty catalog -> ArcError.

vote
    Most frequent whole grid (equality of cells). Ties keep the earliest.
    Empty -> ArcError. Shape mismatch is allowed; those candidates just fail
    to tally together.

ensemble_predict
    For each k in dihedrals: apply k, execute program, apply inverse(k);
    vote the results. Empty dihedrals -> ArcError. Out-of-range k -> ArcError.

sidecar_hook
    Both ids non-empty strings; else ArcError. Returns SidecarHook.
    Does not train.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ArcError(ValueError):
    """A grid, program, or sidecar id violates a spec 10 invariant."""


N_COLORS = 10
N_DIHEDRAL = 8
MIN_GRID_SIZE = 1
MAX_GRID_SIZE = 30

DIHEDRAL_NAMES: tuple[str, ...] = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "transpose",
    "anti_transpose",
)

# inverse_dihedral(k) for k in 0..8.
_INVERSE_DIHEDRAL: tuple[int, ...] = (0, 3, 2, 1, 4, 5, 6, 7)


class OpKind(StrEnum):
    """Closed grid DSL. Dihedral ops plus recolor."""

    IDENTITY = "identity"
    ROT90 = "rot90"
    ROT180 = "rot180"
    ROT270 = "rot270"
    FLIP_H = "flip_h"
    FLIP_V = "flip_v"
    TRANSPOSE = "transpose"
    ANTI_TRANSPOSE = "anti_transpose"
    RECOLOR = "recolor"


@dataclass(frozen=True)
class Grid:
    """Rectangular palette grid. Cells are ints in ``0..N_COLORS``."""

    cells: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Pair:
    """One demonstration or test input/output pair."""

    input: Grid
    output: Grid


@dataclass(frozen=True)
class Op:
    """One DSL instruction. ``src``/``dst`` are set only for RECOLOR."""

    kind: OpKind
    src: int | None = None
    dst: int | None = None


@dataclass(frozen=True)
class Program:
    """Composition of ops, applied left to right."""

    ops: tuple[Op, ...]


@dataclass(frozen=True)
class SidecarHook:
    """Ids I4 ``register_ttt`` hot-loads. This module does not train."""

    task_id: str
    lora_id: str


_KIND_TO_INDEX: dict[OpKind, int] = {
    OpKind.IDENTITY: 0,
    OpKind.ROT90: 1,
    OpKind.ROT180: 2,
    OpKind.ROT270: 3,
    OpKind.FLIP_H: 4,
    OpKind.FLIP_V: 5,
    OpKind.TRANSPOSE: 6,
    OpKind.ANTI_TRANSPOSE: 7,
}


def _cells(grid: Grid) -> tuple[tuple[int, ...], ...]:
    """Normalize grid cells to a tuple-of-tuples of int."""
    return tuple(tuple(int(c) for c in row) for row in grid.cells)


def make_grid(cells: list[list[int]]) -> Grid:
    """Validate and freeze a grid. Empty, jagged, size outside
    ``[MIN_GRID_SIZE, MAX_GRID_SIZE]``, or a color outside ``0..N_COLORS``
    raises ArcError.
    """
    rows = list(cells)
    if not rows:
        raise ArcError("empty grid")
    height = len(rows)
    width = len(rows[0])
    if height < MIN_GRID_SIZE or height > MAX_GRID_SIZE:
        raise ArcError(f"height {height} outside [{MIN_GRID_SIZE}, {MAX_GRID_SIZE}]")
    if width < MIN_GRID_SIZE or width > MAX_GRID_SIZE:
        raise ArcError(f"width {width} outside [{MIN_GRID_SIZE}, {MAX_GRID_SIZE}]")
    frozen: list[tuple[int, ...]] = []
    for row in rows:
        if len(row) != width:
            raise ArcError("jagged grid")
        frozen_row: list[int] = []
        for color in row:
            if not isinstance(color, int) or color < 0 or color >= N_COLORS:
                raise ArcError(f"color {color} outside 0..{N_COLORS}")
            frozen_row.append(int(color))
        frozen.append(tuple(frozen_row))
    return Grid(cells=tuple(frozen))


def apply_dihedral(grid: Grid, index: int) -> Grid:
    """Apply D4 action ``index`` in ``0..N_DIHEDRAL``. ArcError otherwise."""
    if not isinstance(index, int) or index < 0 or index >= N_DIHEDRAL:
        raise ArcError(f"dihedral index {index} out of range")
    src = _cells(grid)
    height = len(src)
    width = len(src[0])
    if index == 0:

        def cell(r: int, c: int) -> int:
            return src[r][c]

        out_h, out_w = height, width
    elif index == 1:
        # rot90 clockwise: [[1,2,3],[4,5,6]] -> [[4,1],[5,2],[6,3]]
        def cell(r: int, c: int) -> int:
            return src[height - 1 - c][r]

        out_h, out_w = width, height
    elif index == 2:

        def cell(r: int, c: int) -> int:
            return src[height - 1 - r][width - 1 - c]

        out_h, out_w = height, width
    elif index == 3:

        def cell(r: int, c: int) -> int:
            return src[c][width - 1 - r]

        out_h, out_w = width, height
    elif index == 4:

        def cell(r: int, c: int) -> int:
            return src[r][width - 1 - c]

        out_h, out_w = height, width
    elif index == 5:

        def cell(r: int, c: int) -> int:
            return src[height - 1 - r][c]

        out_h, out_w = height, width
    elif index == 6:

        def cell(r: int, c: int) -> int:
            return src[c][r]

        out_h, out_w = width, height
    else:
        # anti-transpose: transpose then rot180.
        def cell(r: int, c: int) -> int:
            return src[height - 1 - c][width - 1 - r]

        out_h, out_w = width, height
    return make_grid([[cell(r, c) for c in range(out_w)] for r in range(out_h)])


def inverse_dihedral(index: int) -> int:
    """D4 inverse of ``index``. ArcError if ``index`` is out of range.
    Rot90 inverts to rot270; reflections and 180 are self-inverse.
    """
    if not isinstance(index, int) or index < 0 or index >= N_DIHEDRAL:
        raise ArcError(f"dihedral index {index} out of range")
    return _INVERSE_DIHEDRAL[index]


def apply_color_perm(grid: Grid, perm: tuple[int, ...]) -> Grid:
    """``perm[c]`` is the new color of ``c``. Must be a permutation of
    ``0..N_COLORS``. ArcError otherwise.
    """
    p = [int(x) for x in perm]
    if len(p) != N_COLORS:
        raise ArcError("color permutation is not a permutation of 0..10")
    seen = [False] * N_COLORS
    for value in p:
        if value < 0 or value >= N_COLORS or seen[value]:
            raise ArcError("color permutation is not a permutation of 0..10")
        seen[value] = True
    src = _cells(grid)
    return make_grid([[p[color] for color in row] for row in src])


def _apply_op(op: Op, grid: Grid) -> Grid:
    if op.kind == OpKind.RECOLOR:
        src, dst = op.src, op.dst
        if src is None or dst is None:
            raise ArcError("recolor requires src and dst")
        if src not in range(N_COLORS) or dst not in range(N_COLORS):
            raise ArcError("recolor src/dst out of range")
        cells = _cells(grid)
        return make_grid([[dst if color == src else color for color in row] for row in cells])
    try:
        index = _KIND_TO_INDEX[op.kind]
    except KeyError:
        index = None
        for kind, mapped in _KIND_TO_INDEX.items():
            if op.kind == kind:
                index = mapped
                break
        if index is None:
            raise ArcError(f"unknown op kind {op.kind}") from None
    return apply_dihedral(grid, index)


def execute(program: Program, grid: Grid) -> Grid:
    """Run ``program.ops`` left to right. Empty program is identity.
    RECOLOR with missing or out-of-range src/dst raises ArcError.
    Unknown kinds cannot appear (OpKind is closed).
    """
    current = make_grid([list(row) for row in _cells(grid)])
    for op in program.ops:
        current = _apply_op(op, current)
    return current


def consistent(program: Program, train: tuple[Pair, ...]) -> bool:
    """True iff execute(program, pair.input) equals pair.output for every
    pair. Empty train raises ArcError.
    """
    if not train:
        raise ArcError("empty train")
    for pair in train:
        if _cells(execute(program, pair.input)) != _cells(pair.output):
            return False
    return True


def synthesize(train: tuple[Pair, ...], catalog: tuple[Program, ...]) -> Program | None:
    """First catalog program consistent with every train pair, else None.
    Empty train or empty catalog raises ArcError.
    """
    if not train:
        raise ArcError("empty train")
    if not catalog:
        raise ArcError("empty catalog")
    for program in catalog:
        if consistent(program, train):
            return program
    return None


def vote(grids: tuple[Grid, ...]) -> Grid:
    """Most frequent whole grid. Ties keep the earliest. Empty raises ArcError.
    Shape mismatch across candidates is allowed; they just fail to tally
    together.
    """
    if not grids:
        raise ArcError("empty grids")
    counts: dict[tuple[tuple[int, ...], ...], int] = {}
    order: list[tuple[tuple[int, ...], ...]] = []
    for grid in grids:
        key = _cells(grid)
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += 1
    best = order[0]
    best_count = counts[best]
    for key in order[1:]:
        if counts[key] > best_count:
            best = key
            best_count = counts[key]
    return Grid(cells=best)


def ensemble_predict(
    program: Program,
    test_input: Grid,
    dihedrals: tuple[int, ...],
) -> Grid:
    """For each dihedral k: apply k, execute program, apply inverse(k).
    Vote the results. Empty dihedrals raises ArcError.
    """
    if not dihedrals:
        raise ArcError("empty dihedrals")
    candidates: list[Grid] = []
    for index in dihedrals:
        augmented = apply_dihedral(test_input, index)
        predicted = execute(program, augmented)
        restored = apply_dihedral(predicted, inverse_dihedral(index))
        candidates.append(restored)
    return vote(tuple(candidates))


def sidecar_hook(task_id: str, lora_id: str) -> SidecarHook:
    """Ids for I4 register_ttt. Both non-empty. ArcError otherwise.
    Does not train a LoRA (JAX sidecar is a later GPU path).
    """
    if not task_id or not lora_id:
        raise ArcError("task_id and lora_id must be non-empty")
    return SidecarHook(task_id=task_id, lora_id=lora_id)
