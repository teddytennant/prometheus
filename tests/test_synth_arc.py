"""Implementer-facing tests for E3 procedural ARC-like grids.

Every call into a stubbed E3 function hits ``NotImplementedError`` today, so
this file must be red. After E3 is implemented, production must match
``tests.reference.arc``. Production must never import ``tests/``.
"""

from __future__ import annotations

import pytest

from synth import (
    DEFAULT_N_TRAIN,
    FAMILY_KINDS,
    MAX_GRID_SIZE,
    MIN_GRID_SIZE,
    N_COLORS,
    N_DIHEDRAL,
    Dihedral,
    DiversityStats,
    FamilyKind,
    FamilySpec,
    Grid,
    Pair,
    ProceduralCorpus,
    SynthError,
    Task,
    apply_color_perm,
    apply_dihedral,
    augment_pair,
    augment_task,
    diversity_stats,
    family_from_spec,
    sample_family_specs,
    tokenize_grid,
)
from tests.reference import arc as ref

# Hand-computed DiversityStats for two tasks that share every grid (collision
# 0.5) but differ in family_id / seed. Color counts and shapes are tallied
# from the cells below, not from generate.
#
# Each task:
#   train input  [[1, 0], [0, 0]]  colors 0x3, 1x1  shape (2, 2)
#   train output [[0, 1], [0, 0]]  colors 0x3, 1x1  shape (2, 2)
#   test input   [[2]]             color  2x1       shape (1, 1)
#   test output  [[2]]             color  2x1       shape (1, 1)
# Two copies: 0:12, 1:4, 2:4. unique_tasks=1, unique_shapes=2.
GOLDEN_DIVERSITY = DiversityStats(
    n_tasks=2,
    n_families=2,
    unique_test_inputs=1,
    unique_test_outputs=1,
    unique_tasks=1,
    color_histogram=(12, 4, 4, 0, 0, 0, 0, 0, 0, 0),
    unique_shapes=2,
    collision_rate=0.5,
)

ROT90_SRC = [[1, 2, 3], [4, 5, 6]]
DIHEDRAL_GOLDENS = {
    Dihedral.IDENTITY: [[1, 2, 3], [4, 5, 6]],
    Dihedral.ROT90: [[4, 1], [5, 2], [6, 3]],
    Dihedral.ROT180: [[6, 5, 4], [3, 2, 1]],
    Dihedral.ROT270: [[3, 6], [2, 5], [1, 4]],
    Dihedral.FLIP_H: [[3, 2, 1], [6, 5, 4]],
    Dihedral.FLIP_V: [[4, 5, 6], [1, 2, 3]],
    Dihedral.TRANSPOSE: [[1, 4], [2, 5], [3, 6]],
    Dihedral.ANTI_TRANSPOSE: [[6, 3], [5, 2], [4, 1]],
}

IDENTITY_PERM = list(range(N_COLORS))
SWAP01 = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]
CYCLE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]


class FakeTokenizer:
    def encode_grid(self, cells):
        return [int(c) for c in cells]


class RecordingTokenizer:
    def __init__(self) -> None:
        self.seen: list[list[int]] | None = None

    def encode_grid(self, cells):
        self.seen = list(cells)
        return [100 + int(c) for c in cells]


def g(rows: list[list[int]]) -> Grid:
    return Grid(cells=[list(row) for row in rows])


def cells_of(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid.cells]


def kind_specs() -> list[FamilySpec]:
    return [
        FamilySpec.make(FamilyKind.TRANSLATE, {"dx": 1, "dy": -1, "bg": 0}),
        FamilySpec.make(FamilyKind.RECOLOR, {"src": 1, "dst": 2}),
        FamilySpec.make(FamilyKind.CROP, {"bg": 0}),
        FamilySpec.make(FamilyKind.TILE, {"nx": 2, "ny": 2}),
        FamilySpec.make(FamilyKind.GRAVITY, {"dir": 0, "bg": 0}),
        FamilySpec.make(FamilyKind.MIRROR, {"dihedral": 1}),
        FamilySpec.make(FamilyKind.SCALE, {"factor": 2}),
        FamilySpec.make(FamilyKind.BORDER, {"color": 9, "width": 1}),
    ]


def golden_tasks() -> list[Task]:
    return [
        Task(
            family_id="translate:bg=0,dx=1,dy=0",
            seed=7,
            train=[
                Pair(input=g([[1, 0], [0, 0]]), output=g([[0, 1], [0, 0]])),
            ],
            test=Pair(input=g([[2]]), output=g([[2]])),
        ),
        Task(
            family_id="mirror:dihedral=0",
            seed=8,
            train=[
                Pair(input=g([[1, 0], [0, 0]]), output=g([[0, 1], [0, 0]])),
            ],
            test=Pair(input=g([[2]]), output=g([[2]])),
        ),
    ]


# --- already-implemented surface (may pass on the stub) ---


def test_constants():
    assert N_COLORS == 10
    assert MIN_GRID_SIZE == 1
    assert MAX_GRID_SIZE == 30
    assert N_DIHEDRAL == 8
    assert DEFAULT_N_TRAIN == 3
    assert len(Dihedral) == 8
    assert tuple(FAMILY_KINDS) == tuple(FamilyKind)


def test_grid_validation_and_flatten():
    grid = g([[1, 2], [3, 4]])
    assert grid.rows() == 2
    assert grid.cols() == 2
    assert grid.flatten() == [1, 2, 3, 4]
    with pytest.raises(SynthError, match="empty grid"):
        Grid(cells=[])
    with pytest.raises(SynthError, match="jagged grid"):
        Grid(cells=[[1, 2], [3]])
    with pytest.raises(SynthError, match="ARC color 10 out of range"):
        Grid(cells=[[10]])
    with pytest.raises(SynthError, match="grid size 31x1 out of range"):
        Grid(cells=[[0] for _ in range(31)])


def test_family_spec_id_sorts_keys():
    spec = FamilySpec.make(FamilyKind.TRANSLATE, {"dy": 2, "bg": 0, "dx": -1})
    assert spec.family_id() == "translate:bg=0,dx=-1,dy=2"
    assert FamilySpec.make(FamilyKind.CROP, {"bg": 7}).family_id() == "crop:bg=7"


def test_dihedral_index_roundtrip():
    for i, d in enumerate(Dihedral):
        assert d.index() == i
        assert Dihedral.from_index(i) is d
    with pytest.raises(SynthError, match="dihedral index 8 out of range"):
        Dihedral.from_index(8)


# --- apply_dihedral ---


def test_rot90_clockwise_golden():
    got = apply_dihedral(g(ROT90_SRC), Dihedral.ROT90)
    assert cells_of(got) == DIHEDRAL_GOLDENS[Dihedral.ROT90]


def test_all_eight_dihedrals_on_2x3_golden():
    src = g(ROT90_SRC)
    for d, expected in DIHEDRAL_GOLDENS.items():
        assert cells_of(apply_dihedral(src, d)) == expected


def test_apply_dihedral_matches_reference_on_several_grids():
    grids = [
        g([[7]]),
        g([[1, 2, 3]]),
        g([[1], [2], [3]]),
        g(ROT90_SRC),
        g([[1, 2], [3, 4]]),
        g([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
        g([[c % N_COLORS] for c in range(MAX_GRID_SIZE)]),
        g([list(range(N_COLORS)) + [0] * (MAX_GRID_SIZE - N_COLORS)]),
    ]
    for grid in grids:
        for d in Dihedral:
            got = apply_dihedral(grid, d)
            expected = ref.apply_dihedral(grid, d)
            assert cells_of(got) == cells_of(expected)
            Grid(cells=cells_of(got))


def test_dihedral_swaps_height_and_width():
    src = g([[1, 2, 3], [4, 5, 6]])
    swapped = {Dihedral.ROT90, Dihedral.ROT270, Dihedral.TRANSPOSE, Dihedral.ANTI_TRANSPOSE}
    for d in Dihedral:
        out = apply_dihedral(src, d)
        if d in swapped:
            assert (out.rows(), out.cols()) == (3, 2)
        else:
            assert (out.rows(), out.cols()) == (2, 3)


# --- apply_color_perm ---


def test_color_perm_identity_and_swap():
    src = g([[0, 1, 2], [3, 0, 9]])
    assert cells_of(apply_color_perm(src, IDENTITY_PERM)) == cells_of(src)
    assert cells_of(apply_color_perm(src, SWAP01)) == [[1, 0, 2], [3, 1, 9]]
    # Color 0 is not special: cycling sends 0 -> 1 and 9 -> 0.
    assert cells_of(apply_color_perm(g([[0, 9]]), CYCLE)) == [[1, 0]]


def test_color_perm_matches_reference():
    src = g([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    for perm in (IDENTITY_PERM, SWAP01, CYCLE, [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]):
        assert cells_of(apply_color_perm(src, perm)) == cells_of(ref.apply_color_perm(src, perm))


def test_bad_color_perm_errors():
    src = g([[1]])
    bads = [
        [0, 1],
        list(range(11)),
        [0, 0, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
        [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ]
    for perm in bads:
        with pytest.raises(SynthError, match="color permutation is not a permutation of 0..10"):
            apply_color_perm(src, perm)


# --- augment ---


def test_augment_pair_same_transform_on_both_grids():
    pair = Pair(input=g(ROT90_SRC), output=g([[9, 0], [1, 2]]))
    got = augment_pair(pair, Dihedral.ROT90, SWAP01)
    expected = ref.augment_pair(pair, Dihedral.ROT90, SWAP01)
    assert cells_of(got.input) == cells_of(expected.input)
    assert cells_of(got.output) == cells_of(expected.output)


def test_augment_task_preserves_family_id_and_seed():
    task = Task(
        family_id="recolor:dst=2,src=1",
        seed=99,
        train=[Pair(input=g([[1, 0]]), output=g([[2, 0]]))],
        test=Pair(input=g([[1]]), output=g([[2]])),
    )
    got = augment_task(task, Dihedral.FLIP_H, CYCLE)
    expected = ref.augment_task(task, Dihedral.FLIP_H, CYCLE)
    assert got.family_id == "recolor:dst=2,src=1"
    assert got.seed == 99
    assert got.family_id == expected.family_id
    assert got.seed == expected.seed
    assert cells_of(got.test.input) == cells_of(expected.test.input)
    assert cells_of(got.test.output) == cells_of(expected.test.output)
    assert len(got.train) == 1
    assert cells_of(got.train[0].input) == cells_of(expected.train[0].input)
    assert cells_of(got.train[0].output) == cells_of(expected.train[0].output)


# --- tokenize_grid ---


def test_tokenize_grid_fake_returns_cells_as_ints():
    grid = g([[1, 2], [3, 4]])
    assert tokenize_grid(FakeTokenizer(), grid) == [1, 2, 3, 4]
    assert tokenize_grid(FakeTokenizer(), grid) == ref.tokenize_grid(FakeTokenizer(), grid)


def test_tokenize_grid_passes_row_major_flatten():
    rec = RecordingTokenizer()
    grid = g([[1, 2], [3, 4]])
    assert tokenize_grid(rec, grid) == [101, 102, 103, 104]
    assert rec.seen == [1, 2, 3, 4]


# --- family_from_spec / generate ---


def test_family_from_spec_rejects_invalid_params():
    bads = [
        FamilySpec.make(FamilyKind.TRANSLATE, {"dx": 1, "dy": 0}),
        FamilySpec.make(FamilyKind.TRANSLATE, {"dx": 1, "dy": 0, "bg": 10}),
        FamilySpec.make(FamilyKind.RECOLOR, {"src": 1, "dst": 1}),
        FamilySpec.make(FamilyKind.RECOLOR, {"src": 1, "dst": 2, "extra": 0}),
        FamilySpec.make(FamilyKind.CROP, {}),
        FamilySpec.make(FamilyKind.TILE, {"nx": 0, "ny": 1}),
        FamilySpec.make(FamilyKind.TILE, {"nx": 31, "ny": 1}),
        FamilySpec.make(FamilyKind.GRAVITY, {"dir": 4, "bg": 0}),
        FamilySpec.make(FamilyKind.MIRROR, {"dihedral": 8}),
        FamilySpec.make(FamilyKind.SCALE, {"factor": 4}),
        FamilySpec.make(FamilyKind.BORDER, {"color": 0, "width": 3}),
        FamilySpec.make(FamilyKind.BORDER, {"color": 10, "width": 1}),
    ]
    for spec in bads:
        with pytest.raises(SynthError, match="unknown family"):
            family_from_spec(spec)


def test_generate_matches_reference_for_all_kinds():
    for spec in kind_specs():
        fam = family_from_spec(spec)
        assert fam.id() == spec.family_id()
        assert fam.kind() == spec.kind
        got = fam.generate(0)
        expected = ref.family_from_spec(spec).generate(0)
        assert got == expected
        assert len(got.train) == DEFAULT_N_TRAIN
        assert got.seed == 0
        assert got.family_id == spec.family_id()


def test_generate_deterministic_and_seed_sensitive():
    spec = kind_specs()[0]
    fam = family_from_spec(spec)
    a = fam.generate(123)
    b = fam.generate(123)
    c = fam.generate(124)
    assert a == b
    assert a != c
    assert a == ref.generate_task(spec, 123, DEFAULT_N_TRAIN)


def test_generate_valid_grids_and_held_out_test():
    for spec in kind_specs():
        task = family_from_spec(spec).generate(7)
        assert len(task.train) >= 1
        train_keys = []
        for pair in task.train:
            Grid(cells=cells_of(pair.input))
            Grid(cells=cells_of(pair.output))
            train_keys.append((cells_of(pair.input), cells_of(pair.output)))
            assert pair.output == ref.apply_family(pair.input, spec)
        Grid(cells=cells_of(task.test.input))
        Grid(cells=cells_of(task.test.output))
        test_key = (cells_of(task.test.input), cells_of(task.test.output))
        assert test_key not in train_keys
        assert task.test.output == ref.apply_family(task.test.input, spec)


def test_translate_wraps_around():
    spec = FamilySpec.make(FamilyKind.TRANSLATE, {"dx": 1, "dy": 0, "bg": 0})
    task = family_from_spec(spec).generate(3)
    for pair in [*task.train, task.test]:
        # Wrap, not clip: moving off the right edge reappears on the left.
        assert pair.output == ref.apply_family(pair.input, spec)


def test_crop_pairs_match_bounding_box_or_1x1_bg():
    # Pin: all-bg crop is 1x1 [[bg]], not an error. Production pairs must equal
    # the reference transform (bounding box of non-bg cells).
    spec = FamilySpec.make(FamilyKind.CROP, {"bg": 0})
    task = family_from_spec(spec).generate(11)
    for pair in [*task.train, task.test]:
        assert pair.output == ref.apply_family(pair.input, spec)


# --- sample_family_specs ---


def test_sample_family_specs_empty_is_empty_list():
    assert sample_family_specs(0, 0) == []
    assert sample_family_specs(0, 99) == []


def test_sample_family_specs_deterministic_distinct_covers_kinds():
    a = sample_family_specs(32, 42)
    b = sample_family_specs(32, 42)
    c = sample_family_specs(32, 43)
    assert a == b
    assert a != c
    assert a == ref.sample_family_specs(32, 42)
    ids = [s.family_id() for s in a]
    assert len(set(ids)) == 32
    n = len(ref.parameter_grid())
    full = sample_family_specs(n, 0)
    assert {s.kind for s in full} == set(FAMILY_KINDS)
    assert len({s.family_id() for s in full}) == n
    assert full == ref.sample_family_specs(n, 0)


# --- diversity_stats ---


def test_diversity_stats_empty_is_all_zeros():
    got = diversity_stats([])
    assert got == DiversityStats(
        n_tasks=0,
        n_families=0,
        unique_test_inputs=0,
        unique_test_outputs=0,
        unique_tasks=0,
        color_histogram=(0,) * 10,
        unique_shapes=0,
        collision_rate=0.0,
    )


def test_diversity_stats_hand_golden():
    got = diversity_stats(golden_tasks())
    assert got == GOLDEN_DIVERSITY
    assert got == ref.diversity_stats(golden_tasks())


def test_diversity_stats_matches_reference_on_generated_tasks():
    spec = kind_specs()[1]
    tasks = [ref.generate_task(spec, seed, 1) for seed in range(4)]
    # Production is called on the hand-built list; it must match the reference
    # even though the tasks themselves came from the reference generator.
    assert diversity_stats(tasks) == ref.diversity_stats(tasks)


# --- ProceduralCorpus.sample ---


def test_corpus_sample_empty_n_and_empty_specs():
    specs = kind_specs()[:2]
    with pytest.raises(SynthError, match="empty tasks"):
        ProceduralCorpus(specs, seed=0).sample(0)
    with pytest.raises(SynthError, match="empty tasks"):
        ProceduralCorpus([], seed=0).sample(0)
    with pytest.raises(SynthError, match="unknown family"):
        ProceduralCorpus([], seed=0).sample(1)
    with pytest.raises(SynthError, match="empty train"):
        ProceduralCorpus(specs, seed=0, n_train=0).sample(1)


def test_corpus_sample_matches_reference_and_cycles_specs():
    specs = kind_specs()[:3]
    corpus = ProceduralCorpus(specs, seed=5, n_train=2)
    got = corpus.sample(5)
    expected = ref.sample_corpus(specs, 5, 2, 5)
    assert got == expected
    assert len(got) == 5
    for i, task in enumerate(got):
        assert len(task.train) == 2
        assert task.seed == 5 + i
        assert task.family_id == specs[i % 3].family_id()
    fam_task = family_from_spec(specs[0]).generate(5)
    assert len(fam_task.train) == DEFAULT_N_TRAIN
    assert got[0] != fam_task


def test_corpus_sample_unknown_family_in_bag():
    bad = FamilySpec.make(FamilyKind.SCALE, {"factor": 1})
    with pytest.raises(SynthError, match="unknown family"):
        ProceduralCorpus([bad], seed=0).sample(1)
