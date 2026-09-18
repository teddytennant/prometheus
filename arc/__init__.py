"""ARC-AGI TTT sidecar, grid DSL, and executor (spec 10, 15.5 I8).

CPU contract: grids, D4 augmentations, a closed DSL, consistency against
demonstration pairs, majority vote under inverse augmentations, and the
sidecar LoRA ids that I4 ``register_ttt`` hot-loads. JAX fine-tuning of
the LoRA is out of this module; eval on ARC-AGI-1 public is the later
gate. Does not import ``synth`` or ``sglang_fork``.
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

# D4 index order matches E3: 0 identity, 1 rot90 CW, 2 rot180, 3 rot270 CW,
# 4 flip left-right, 5 flip up-down, 6 transpose, 7 anti-transpose.
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


_INVERSE_DIHEDRAL: tuple[int, ...] = (0, 3, 2, 1, 4, 5, 6, 7)

_KIND_TO_DIHEDRAL: dict[OpKind, int] = {
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
    src = [list(row) for row in _cells(grid)]
    if index == 0:
        out = src
    elif index == 1:
        # rot90 CW: [[1,2,3],[4,5,6]] -> [[4,1],[5,2],[6,3]]
        out = [list(row) for row in zip(*reversed(src), strict=True)]
    elif index == 2:
        out = [list(reversed(row)) for row in reversed(src)]
    elif index == 3:
        out = [list(row) for row in reversed(tuple(zip(*src, strict=True)))]
    elif index == 4:
        out = [list(reversed(row)) for row in src]
    elif index == 5:
        out = list(reversed(src))
    elif index == 6:
        out = [list(row) for row in zip(*src, strict=True)]
    else:
        # anti-transpose = transpose then rot180
        transposed = [list(row) for row in zip(*src, strict=True)]
        out = [list(reversed(row)) for row in reversed(transposed)]
    return make_grid(out)


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
        return make_grid(
            [[dst if color == src else color for color in row] for row in cells]
        )
    try:
        index = _KIND_TO_DIHEDRAL[op.kind]
    except KeyError as exc:
        raise ArcError(f"unknown op kind {op.kind}") from exc
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
