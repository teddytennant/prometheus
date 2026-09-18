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


def make_grid(cells: list[list[int]]) -> Grid:
    """Validate and freeze a grid. Empty, jagged, size outside
    ``[MIN_GRID_SIZE, MAX_GRID_SIZE]``, or a color outside ``0..N_COLORS``
    raises ArcError.
    """
    raise NotImplementedError("I8 make_grid")


def apply_dihedral(grid: Grid, index: int) -> Grid:
    """Apply D4 action ``index`` in ``0..N_DIHEDRAL``. ArcError otherwise."""
    raise NotImplementedError("I8 apply_dihedral")


def inverse_dihedral(index: int) -> int:
    """D4 inverse of ``index``. ArcError if ``index`` is out of range.
    Rot90 inverts to rot270; reflections and 180 are self-inverse.
    """
    raise NotImplementedError("I8 inverse_dihedral")


def apply_color_perm(grid: Grid, perm: tuple[int, ...]) -> Grid:
    """``perm[c]`` is the new color of ``c``. Must be a permutation of
    ``0..N_COLORS``. ArcError otherwise.
    """
    raise NotImplementedError("I8 apply_color_perm")


def execute(program: Program, grid: Grid) -> Grid:
    """Run ``program.ops`` left to right. Empty program is identity.
    RECOLOR with missing or out-of-range src/dst raises ArcError.
    Unknown kinds cannot appear (OpKind is closed).
    """
    raise NotImplementedError("I8 execute")


def consistent(program: Program, train: tuple[Pair, ...]) -> bool:
    """True iff execute(program, pair.input) equals pair.output for every
    pair. Empty train raises ArcError.
    """
    raise NotImplementedError("I8 consistent")


def synthesize(train: tuple[Pair, ...], catalog: tuple[Program, ...]) -> Program | None:
    """First catalog program consistent with every train pair, else None.
    Empty train or empty catalog raises ArcError.
    """
    raise NotImplementedError("I8 synthesize")


def vote(grids: tuple[Grid, ...]) -> Grid:
    """Most frequent whole grid. Ties keep the earliest. Empty raises ArcError.
    Shape mismatch across candidates is allowed; they just fail to tally
    together.
    """
    raise NotImplementedError("I8 vote")


def ensemble_predict(
    program: Program,
    test_input: Grid,
    dihedrals: tuple[int, ...],
) -> Grid:
    """For each dihedral k: apply k, execute program, apply inverse(k).
    Vote the results. Empty dihedrals raises ArcError.
    """
    raise NotImplementedError("I8 ensemble_predict")


def sidecar_hook(task_id: str, lora_id: str) -> SidecarHook:
    """Ids for I4 register_ttt. Both non-empty. ArcError otherwise.
    Does not train a LoRA (JAX sidecar is a later GPU path).
    """
    raise NotImplementedError("I8 sidecar_hook")
