"""Independent slow reference for E3 procedural ARC-like grids (spec 7.1, 10, 15.5).

This is the source of truth for dihedral / color-perm / augment / tokenize /
family generate / sample_family_specs / diversity_stats / ProceduralCorpus.sample.
Production ``synth`` must match these results on the same inputs. Production must
never import ``tests/``. Must match ``synth/tests/common/arc_ref.rs``.

Pinned rules
------------
Dihedral (D4), index 0..8
    0 identity
    1 rot90 clockwise. ``[[1,2,3],[4,5,6]]`` -> ``[[4,1],[5,2],[6,3]]``.
    2 rot180
    3 rot270 clockwise
    4 flip left-right (horizontal)
    5 flip up-down (vertical)
    6 transpose (main diagonal)
    7 anti-transpose (anti-diagonal): transpose then rot180.
    rot90 / rot270 / transpose / anti-transpose swap height and width.
    After any dihedral the grid still passes Grid validation.

Color permutation
    ``perm[c]`` is the new color of ``c``. Length 10, each of 0..9 exactly once.
    Color 0 is not special. Bad perm -> SynthError matching Rust BadPermutation.

Augmentation
    Dihedral first, then color perm. Same transform on every grid in the pair/task.
    ``family_id`` and ``seed`` unchanged.

tokenize_grid
    ``tok.encode_grid(grid.flatten())`` as a list. Tokenizer exceptions propagate.

Family params (exact keys; anything else is ``unknown family``)
    translate: dx, dy, bg. Non-bg cells move by (dx, dy) with wrap-around
        (torus). Output size equals input. bg in 0..10; dx, dy any int.
        Positive dx is right, positive dy is down.
    recolor: src, dst. Map src to dst. src != dst, both in 0..10.
    crop: bg. Bounding box of non-bg cells. All-bg input -> 1x1 ``[[bg]]``.
    tile: nx, ny. Repeat nx times horizontally and ny vertically. nx, ny in 1..30.
    gravity: dir (0 down, 1 up, 2 left, 3 right), bg. Non-bg cells pack toward
        that edge in scan order; bg fills the rest.
    mirror: dihedral (0..8). Output is apply_dihedral(input, that index).
    scale: factor 2 or 3. Cell replication. Output size = input * factor.
    border: color in 0..10, width 1 or 2. Output grows by 2*width per axis;
        interior is the input, frame is ``color``.

generate(seed) / generate_task(spec, seed, n_train)
    SplitMix64(seed). n_train == 0 -> empty train. Else sample exactly
    n_train + 1 distinct pairs (retry on collision, cap 10000). Train is the
    first n_train; test is the last. Test is not identical to any train pair.
    Family.generate uses DEFAULT_N_TRAIN=3. Task.seed is the generate seed.
    Input size: uniform 1..=min(8, max that keeps the transformed grid <= 30).
    Each cell uniform in 0..10. Same (kind, params, seed, n_train) matches Rust.

sample_family_specs(n, seed)
    n == 0 -> []. Shuffle the parameter grid with SplitMix64(seed) Fisher-Yates
    and take n. n larger than the grid -> SynthError message listed in
    TOO_MANY_SPECS. Cover all 8 kinds when n equals the grid length.

diversity_stats
    unique_tasks hashes train+test grid cells only (not family_id / seed).
    Empty tasks: all zeros, collision_rate 0.0.

ProceduralCorpus.sample(n)
    n == 0 -> empty tasks (even if specs is empty).
    Empty specs -> unknown family.
    n_train <= 0 -> empty train.
    Else task i uses specs[i % len] and seed+i with corpus n_train.
"""

from __future__ import annotations

from collections.abc import Sequence

from synth import (
    DEFAULT_N_TRAIN,
    MAX_GRID_SIZE,
    N_COLORS,
    N_DIHEDRAL,
    Dihedral,
    DiversityStats,
    FamilyKind,
    FamilySpec,
    Grid,
    Pair,
    SynthError,
    Task,
)

TOO_MANY_SPECS = "requested more family specs than the parameter grid"
DISTINCT_PAIR_FAIL = "could not sample distinct pairs"
MAX_SAMPLE_DIM = 8
MAX_PAIR_ATTEMPTS = 10000
_MASK64 = (1 << 64) - 1
_SPLITMIX_GOLDEN = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB


class SplitMix64:
    """Portable PRNG. Must match synth/tests/common/arc_ref.rs."""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK64

    def next_u64(self) -> int:
        self.state = (self.state + _SPLITMIX_GOLDEN) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * _SPLITMIX_M1) & _MASK64
        z = ((z ^ (z >> 27)) * _SPLITMIX_M2) & _MASK64
        return z ^ (z >> 31)

    def next_bounded(self, n: int) -> int:
        return self.next_u64() % n


class RefFamily:
    """Reference family. ``generate`` always uses DEFAULT_N_TRAIN."""

    def __init__(self, spec: FamilySpec) -> None:
        self._spec = spec
        self._id = spec.family_id()

    def id(self) -> str:
        return self._id

    def kind(self) -> FamilyKind:
        return self._spec.kind

    def generate(self, seed: int) -> Task:
        return generate_task(self._spec, seed, DEFAULT_N_TRAIN)


def apply_dihedral(grid: Grid, dihedral: Dihedral) -> Grid:
    src = grid.cells
    h = len(src)
    w = len(src[0])
    k = dihedral.index()
    if k == 0:
        oh, ow = h, w

        def cell(r: int, c: int) -> int:
            return src[r][c]
    elif k == 1:
        oh, ow = w, h

        def cell(r: int, c: int) -> int:
            return src[h - 1 - c][r]
    elif k == 2:
        oh, ow = h, w

        def cell(r: int, c: int) -> int:
            return src[h - 1 - r][w - 1 - c]
    elif k == 3:
        oh, ow = w, h

        def cell(r: int, c: int) -> int:
            return src[c][w - 1 - r]
    elif k == 4:
        oh, ow = h, w

        def cell(r: int, c: int) -> int:
            return src[r][w - 1 - c]
    elif k == 5:
        oh, ow = h, w

        def cell(r: int, c: int) -> int:
            return src[h - 1 - r][c]
    elif k == 6:
        oh, ow = w, h

        def cell(r: int, c: int) -> int:
            return src[c][r]
    elif k == 7:
        oh, ow = w, h

        def cell(r: int, c: int) -> int:
            return src[h - 1 - c][w - 1 - r]
    else:
        raise SynthError(f"dihedral index {k} out of range")
    return Grid(cells=[[cell(r, c) for c in range(ow)] for r in range(oh)])


def apply_color_perm(grid: Grid, perm: Sequence[int]) -> Grid:
    p = [int(x) for x in perm]
    if len(p) != N_COLORS:
        raise SynthError("color permutation is not a permutation of 0..10")
    seen = [False] * N_COLORS
    for v in p:
        if v < 0 or v >= N_COLORS or seen[v]:
            raise SynthError("color permutation is not a permutation of 0..10")
        seen[v] = True
    return Grid(cells=[[p[c] for c in row] for row in grid.cells])


def augment_pair(pair: Pair, dihedral: Dihedral, perm: Sequence[int]) -> Pair:
    return Pair(
        input=apply_color_perm(apply_dihedral(pair.input, dihedral), perm),
        output=apply_color_perm(apply_dihedral(pair.output, dihedral), perm),
    )


def augment_task(task: Task, dihedral: Dihedral, perm: Sequence[int]) -> Task:
    return Task(
        family_id=task.family_id,
        seed=task.seed,
        train=[augment_pair(p, dihedral, perm) for p in task.train],
        test=augment_pair(task.test, dihedral, perm),
    )


def tokenize_grid(tok: object, grid: Grid) -> list[int]:
    return list(tok.encode_grid(grid.flatten()))  # type: ignore[attr-defined]


def _require(spec: FamilySpec, keys: set[str]) -> dict[str, int]:
    d = dict(spec.params)
    if set(d) != keys:
        raise SynthError("unknown family")
    return d


def _color(v: int) -> int:
    if v < 0 or v >= N_COLORS:
        raise SynthError("unknown family")
    return v


def validate_spec(spec: FamilySpec) -> dict[str, int]:
    """Return params if ``spec`` is a known family, else ``unknown family``."""
    kind = spec.kind
    if kind is FamilyKind.TRANSLATE:
        d = _require(spec, {"dx", "dy", "bg"})
        _color(d["bg"])
        return d
    if kind is FamilyKind.RECOLOR:
        d = _require(spec, {"src", "dst"})
        src, dst = _color(d["src"]), _color(d["dst"])
        if src == dst:
            raise SynthError("unknown family")
        return d
    if kind is FamilyKind.CROP:
        d = _require(spec, {"bg"})
        _color(d["bg"])
        return d
    if kind is FamilyKind.TILE:
        d = _require(spec, {"nx", "ny"})
        if d["nx"] < 1 or d["ny"] < 1 or d["nx"] > MAX_GRID_SIZE or d["ny"] > MAX_GRID_SIZE:
            raise SynthError("unknown family")
        return d
    if kind is FamilyKind.GRAVITY:
        d = _require(spec, {"dir", "bg"})
        if d["dir"] < 0 or d["dir"] > 3:
            raise SynthError("unknown family")
        _color(d["bg"])
        return d
    if kind is FamilyKind.MIRROR:
        d = _require(spec, {"dihedral"})
        if d["dihedral"] < 0 or d["dihedral"] >= N_DIHEDRAL:
            raise SynthError("unknown family")
        return d
    if kind is FamilyKind.SCALE:
        d = _require(spec, {"factor"})
        if d["factor"] not in (2, 3):
            raise SynthError("unknown family")
        return d
    if kind is FamilyKind.BORDER:
        d = _require(spec, {"color", "width"})
        _color(d["color"])
        if d["width"] not in (1, 2):
            raise SynthError("unknown family")
        return d
    raise SynthError("unknown family")


def _wrap(i: int, n: int) -> int:
    r = i % n
    if r < 0:
        r += n
    return r


def apply_family(grid: Grid, spec: FamilySpec) -> Grid:
    d = validate_spec(spec)
    kind = spec.kind
    if kind is FamilyKind.TRANSLATE:
        return _translate(grid, d["dx"], d["dy"], d["bg"])
    if kind is FamilyKind.RECOLOR:
        return _recolor(grid, d["src"], d["dst"])
    if kind is FamilyKind.CROP:
        return _crop(grid, d["bg"])
    if kind is FamilyKind.TILE:
        return _tile(grid, d["nx"], d["ny"])
    if kind is FamilyKind.GRAVITY:
        return _gravity(grid, d["dir"], d["bg"])
    if kind is FamilyKind.MIRROR:
        return apply_dihedral(grid, Dihedral.from_index(d["dihedral"]))
    if kind is FamilyKind.SCALE:
        return _scale(grid, d["factor"])
    return _border(grid, d["color"], d["width"])


def _translate(grid: Grid, dx: int, dy: int, bg: int) -> Grid:
    h, w = grid.rows(), grid.cols()
    out = [[bg for _ in range(w)] for _ in range(h)]
    for r, row in enumerate(grid.cells):
        for c, val in enumerate(row):
            if val != bg:
                out[_wrap(r + dy, h)][_wrap(c + dx, w)] = val
    return Grid(cells=out)


def _recolor(grid: Grid, src: int, dst: int) -> Grid:
    return Grid(cells=[[dst if c == src else c for c in row] for row in grid.cells])


def _crop(grid: Grid, bg: int) -> Grid:
    coords = [(r, c) for r, row in enumerate(grid.cells) for c, val in enumerate(row) if val != bg]
    if not coords:
        return Grid(cells=[[bg]])
    min_r = min(r for r, _ in coords)
    max_r = max(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    max_c = max(c for _, c in coords)
    return Grid(cells=[row[min_c : max_c + 1] for row in grid.cells[min_r : max_r + 1]])


def _tile(grid: Grid, nx: int, ny: int) -> Grid:
    h = grid.rows()
    out: list[list[int]] = []
    for _tr in range(ny):
        for r in range(h):
            row: list[int] = []
            for _tc in range(nx):
                row.extend(grid.cells[r])
            out.append(row)
    return Grid(cells=out)


def _gravity(grid: Grid, direction: int, bg: int) -> Grid:
    h, w = grid.rows(), grid.cols()
    out = [[bg for _ in range(w)] for _ in range(h)]
    if direction == 0:
        for c in range(w):
            objs = [grid.cells[r][c] for r in range(h) if grid.cells[r][c] != bg]
            start = h - len(objs)
            for i, val in enumerate(objs):
                out[start + i][c] = val
    elif direction == 1:
        for c in range(w):
            objs = [grid.cells[r][c] for r in range(h) if grid.cells[r][c] != bg]
            for i, val in enumerate(objs):
                out[i][c] = val
    elif direction == 2:
        for r in range(h):
            objs = [val for val in grid.cells[r] if val != bg]
            for i, val in enumerate(objs):
                out[r][i] = val
    else:
        for r in range(h):
            objs = [val for val in grid.cells[r] if val != bg]
            start = w - len(objs)
            for i, val in enumerate(objs):
                out[r][start + i] = val
    return Grid(cells=out)


def _scale(grid: Grid, factor: int) -> Grid:
    out = []
    for row in grid.cells:
        scaled = [c for c in row for _ in range(factor)]
        for _ in range(factor):
            out.append(list(scaled))
    return Grid(cells=out)


def _border(grid: Grid, color: int, width: int) -> Grid:
    h, w = grid.rows(), grid.cols()
    oh, ow = h + 2 * width, w + 2 * width
    out = [[color for _ in range(ow)] for _ in range(oh)]
    for r, row in enumerate(grid.cells):
        for c, val in enumerate(row):
            out[r + width][c + width] = val
    return Grid(cells=out)


def family_from_spec(spec: FamilySpec) -> RefFamily:
    validate_spec(spec)
    return RefFamily(spec)


def _max_in_hw(spec: FamilySpec) -> tuple[int, int]:
    d = dict(spec.params)
    kind = spec.kind
    if kind is FamilyKind.TILE:
        return MAX_GRID_SIZE // d["ny"], MAX_GRID_SIZE // d["nx"]
    if kind is FamilyKind.SCALE:
        f = d["factor"]
        return MAX_GRID_SIZE // f, MAX_GRID_SIZE // f
    if kind is FamilyKind.BORDER:
        w = d["width"]
        return MAX_GRID_SIZE - 2 * w, MAX_GRID_SIZE - 2 * w
    return MAX_GRID_SIZE, MAX_GRID_SIZE


def _cells_key(grid: Grid) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in grid.cells)


def _sample_pair(rng: SplitMix64, spec: FamilySpec) -> Pair:
    max_h, max_w = _max_in_hw(spec)
    max_h = min(MAX_SAMPLE_DIM, max_h)
    max_w = min(MAX_SAMPLE_DIM, max_w)
    h = 1 + rng.next_bounded(max_h)
    w = 1 + rng.next_bounded(max_w)
    cells = [[rng.next_bounded(N_COLORS) for _ in range(w)] for _ in range(h)]
    inp = Grid(cells=cells)
    return Pair(input=inp, output=apply_family(inp, spec))


def generate_task(spec: FamilySpec, seed: int, n_train: int) -> Task:
    validate_spec(spec)
    if n_train <= 0:
        raise SynthError("empty train")
    rng = SplitMix64(seed)
    need = n_train + 1
    pairs: list[Pair] = []
    seen: set[tuple] = set()
    attempts = 0
    while len(pairs) < need:
        attempts += 1
        if attempts > MAX_PAIR_ATTEMPTS:
            raise SynthError(DISTINCT_PAIR_FAIL)
        pair = _sample_pair(rng, spec)
        key = (_cells_key(pair.input), _cells_key(pair.output))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(pair)
    return Task(
        family_id=spec.family_id(),
        seed=seed & _MASK64,
        train=pairs[:-1],
        test=pairs[-1],
    )


def parameter_grid() -> list[FamilySpec]:
    specs: list[FamilySpec] = []
    for bg in range(N_COLORS):
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                specs.append(FamilySpec.make(FamilyKind.TRANSLATE, {"dx": dx, "dy": dy, "bg": bg}))
    for src in range(N_COLORS):
        for dst in range(N_COLORS):
            if src != dst:
                specs.append(FamilySpec.make(FamilyKind.RECOLOR, {"src": src, "dst": dst}))
    for bg in range(N_COLORS):
        specs.append(FamilySpec.make(FamilyKind.CROP, {"bg": bg}))
    for nx in range(1, 5):
        for ny in range(1, 5):
            specs.append(FamilySpec.make(FamilyKind.TILE, {"nx": nx, "ny": ny}))
    for bg in range(N_COLORS):
        for direction in range(4):
            specs.append(FamilySpec.make(FamilyKind.GRAVITY, {"dir": direction, "bg": bg}))
    for d in range(N_DIHEDRAL):
        specs.append(FamilySpec.make(FamilyKind.MIRROR, {"dihedral": d}))
    for factor in (2, 3):
        specs.append(FamilySpec.make(FamilyKind.SCALE, {"factor": factor}))
    for color in range(N_COLORS):
        for width in (1, 2):
            specs.append(FamilySpec.make(FamilyKind.BORDER, {"color": color, "width": width}))
    return specs


def sample_family_specs(n: int, seed: int) -> list[FamilySpec]:
    if n == 0:
        return []
    specs = parameter_grid()
    if n < 0 or n > len(specs):
        raise SynthError(TOO_MANY_SPECS)
    rng = SplitMix64(seed)
    for i in range(len(specs)):
        j = i + rng.next_bounded(len(specs) - i)
        specs[i], specs[j] = specs[j], specs[i]
    return specs[:n]


def diversity_stats(tasks: Sequence[Task]) -> DiversityStats:
    n_tasks = len(tasks)
    families: set[str] = set()
    test_in: set[tuple] = set()
    test_out: set[tuple] = set()
    task_keys: set[tuple] = set()
    hist = [0] * N_COLORS
    shapes: set[tuple[int, int]] = set()

    def acc(grid: Grid) -> None:
        shapes.add((grid.rows(), grid.cols()))
        for row in grid.cells:
            for c in row:
                hist[c] += 1

    for task in tasks:
        families.add(task.family_id)
        test_in.add(_cells_key(task.test.input))
        test_out.add(_cells_key(task.test.output))
        train_key = tuple((_cells_key(p.input), _cells_key(p.output)) for p in task.train)
        task_keys.add((train_key, _cells_key(task.test.input), _cells_key(task.test.output)))
        for pair in task.train:
            acc(pair.input)
            acc(pair.output)
        acc(task.test.input)
        acc(task.test.output)

    unique_tasks = len(task_keys)
    collision = 0.0 if n_tasks == 0 else 1.0 - unique_tasks / n_tasks
    return DiversityStats(
        n_tasks=n_tasks,
        n_families=len(families),
        unique_test_inputs=len(test_in),
        unique_test_outputs=len(test_out),
        unique_tasks=unique_tasks,
        color_histogram=tuple(hist),
        unique_shapes=len(shapes),
        collision_rate=collision,
    )


def sample_corpus(specs: Sequence[FamilySpec], seed: int, n_train: int, n: int) -> list[Task]:
    if n == 0:
        raise SynthError("empty tasks")
    if not specs:
        raise SynthError("unknown family")
    if n_train <= 0:
        raise SynthError("empty train")
    out: list[Task] = []
    for i in range(n):
        spec = specs[i % len(specs)]
        out.append(generate_task(spec, seed + i, n_train))
    return out
