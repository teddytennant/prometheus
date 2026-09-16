"""ARC-AGI: grids, dihedral augs, DSL + executor, TTT, env (spec 10, D6, I8)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Grid = list[list[int]]
Op = dict[str, Any]


def rotate90(grid: Grid) -> Grid:
    return [list(row) for row in zip(*reversed(grid))]


def pass_at_k(preds: list, gold, k: int = 2) -> float:
    return float(any(p == gold for p in preds[:k]))


def _copy(grid: Grid) -> Grid:
    return [list(row) for row in grid]


def _flip_h(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]


def _flip_v(grid: Grid) -> Grid:
    return [list(row) for row in reversed(grid)]


def _transpose(grid: Grid) -> Grid:
    if not grid:
        return []
    return [list(row) for row in zip(*grid, strict=False)]


def _rot180(grid: Grid) -> Grid:
    return rotate90(rotate90(grid))


def _rot270(grid: Grid) -> Grid:
    return rotate90(_rot180(grid))


def _anti(grid: Grid) -> Grid:
    return _transpose(_rot180(grid))


D4_NAMES: tuple[str, ...] = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "transpose",
    "anti",
)
_D4_FNS = (_copy, rotate90, _rot180, _rot270, _flip_h, _flip_v, _transpose, _anti)
_D4 = dict(zip(D4_NAMES, _D4_FNS, strict=True))
_D4_INV_NAME = {
    "id": "id",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_h": "flip_h",
    "flip_v": "flip_v",
    "transpose": "transpose",
    "anti": "anti",
}


def dihedral_aug(grid: Grid, *, named: bool = False):
    """Eight D4 transforms. Named form is (name, grid) pairs."""
    out = [(n, fn(grid)) for n, fn in zip(D4_NAMES, _D4_FNS, strict=True)]
    if named:
        return out
    return [g for _, g in out]


def inverse_aug(grid: Grid, name: str, perm: dict[int, int] | None = None) -> Grid:
    if name not in _D4_INV_NAME:
        raise ValueError(f"unknown transform {name}")
    g = _D4[_D4_INV_NAME[name]](grid)
    if perm:
        inv = {int(v): int(k) for k, v in perm.items()}
        g = color_permutation(g, inv)
    return g


def color_permutation(grid: Grid, perm: dict[int, int] | Sequence[int]) -> Grid:
    if not isinstance(perm, dict):
        mapping = {i: int(c) for i, c in enumerate(perm)}
    else:
        mapping = {int(k): int(v) for k, v in perm.items()}
    return [[mapping.get(v, v) for v in row] for row in grid]


def ensemble_vote(named_preds: Sequence[tuple[str, Grid]]) -> Grid:
    restored = [inverse_aug(g, n) for n, g in named_preds]
    if not restored:
        return []
    shapes = [(len(g), len(g[0]) if g else 0) for g in restored]
    h, w = Counter(shapes).most_common(1)[0][0]
    out = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            votes: dict[int, int] = {}
            for g in restored:
                if len(g) == h and (g[0] if g else []) and len(g[0]) == w:
                    votes[g[y][x]] = votes.get(g[y][x], 0) + 1
            out[y][x] = max(votes, key=votes.get) if votes else 0
    return out


@dataclass(frozen=True)
class ArcTask:
    train: list[tuple[Grid, Grid]]
    test_in: Grid
    test_out: Grid
    held_out: bool = False


def color_counts(grid: Grid) -> dict[int, int]:
    c: dict[int, int] = {}
    for row in grid:
        for v in row:
            c[v] = c.get(v, 0) + 1
    return c


def translate(grid: Grid, dy: int, dx: int, fill: int = 0) -> Grid:
    if not grid:
        return []
    h, w = len(grid), len(grid[0])
    out = [[fill] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                out[ny][nx] = grid[y][x]
    return out


def extract_objects(grid: Grid, bg: int = 0) -> list[dict[str, Any]]:
    if not grid:
        return []
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    objs: list[dict[str, Any]] = []
    for y in range(h):
        for x in range(w):
            if grid[y][x] == bg or seen[y][x]:
                continue
            color = grid[y][x]
            cells: list[tuple[int, int]] = []
            stack = [(y, x)]
            seen[y][x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            objs.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": (min(ys), min(xs), max(ys), max(xs)),
                }
            )
    return objs


def _crop_nonbg(grid: Grid, bg: int = 0) -> Grid:
    objs = extract_objects(grid, bg=bg)
    if not objs:
        return _copy(grid)
    y0 = min(o["bbox"][0] for o in objs)
    x0 = min(o["bbox"][1] for o in objs)
    y1 = max(o["bbox"][2] for o in objs)
    x1 = max(o["bbox"][3] for o in objs)
    return [row[x0 : x1 + 1] for row in grid[y0 : y1 + 1]]


def flood(grid: Grid, y: int, x: int, color: int) -> Grid:
    g = _copy(grid)
    if not g or y < 0 or x < 0 or y >= len(g) or x >= len(g[0]):
        return g
    src = g[y][x]
    if src == color:
        return g
    h, w = len(g), len(g[0])
    stack = [(y, x)]
    while stack:
        cy, cx = stack.pop()
        if g[cy][cx] != src:
            continue
        g[cy][cx] = color
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and g[ny][nx] == src:
                stack.append((ny, nx))
    return g


def apply_op(grid: Grid, op: Op) -> Grid:
    name = op.get("op")
    if name == "rotate":
        k = int(op.get("k", 1)) % 4
        g = grid
        for _ in range(k):
            g = rotate90(g)
        return g
    if name == "flip":
        axis = op.get("axis", "h")
        return _flip_h(grid) if axis == "h" else _flip_v(grid)
    if name == "translate":
        return translate(grid, int(op.get("dy", 0)), int(op.get("dx", 0)), int(op.get("fill", 0)))
    if name == "recolor":
        mapping = {int(k): int(v) for k, v in dict(op.get("mapping", {})).items()}
        return color_permutation(grid, mapping)
    if name == "extract_objects":
        return _crop_nonbg(grid, bg=int(op.get("bg", 0)))
    if name == "flood":
        return flood(grid, int(op["y"]), int(op["x"]), int(op["color"]))
    raise ValueError(f"unknown op {name}")


def execute(grid: Grid, program: Sequence[Op]) -> Grid:
    g = _copy(grid)
    for op in program:
        g = apply_op(g, op)
    return g


def _shape(g: Grid) -> tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)


def _recolor_from_pair(inp: Grid, out: Grid) -> dict[int, int] | None:
    if _shape(inp) != _shape(out):
        return None
    mapping: dict[int, int] = {}
    for y, row in enumerate(inp):
        for x, a in enumerate(row):
            b = out[y][x]
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
    return mapping


def _catalog(task: ArcTask) -> list[list[Op]]:
    cands: list[list[Op]] = [[]]
    cands.extend([[{"op": "rotate", "k": k}] for k in (1, 2, 3)])
    cands.extend([[{"op": "flip", "axis": a}] for a in ("h", "v")])
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy or dx:
                cands.append([{"op": "translate", "dy": dy, "dx": dx}])
    cands.append([{"op": "extract_objects"}])
    if task.train:
        inp0, out0 = task.train[0]
        mapping = _recolor_from_pair(inp0, out0)
        if mapping is not None:
            rec = [{"op": "recolor", "mapping": mapping}]
            cands.append(rec)
            for k in (1, 2, 3):
                cands.append([{"op": "rotate", "k": k}, rec[0]])
            for a in ("h", "v"):
                cands.append([{"op": "flip", "axis": a}, rec[0]])
        if inp0 and out0 and _shape(inp0) == _shape(out0) and inp0[0][0] != out0[0][0]:
            cands.append([{"op": "flood", "y": 0, "x": 0, "color": out0[0][0]}])
    return cands


def consistent_program(task: ArcTask) -> list[Op] | None:
    for prog in _catalog(task):
        if all(execute(inp, prog) == out for inp, out in task.train):
            return prog
    return None


def _max_hw(task: ArcTask) -> tuple[int, int]:
    hs: list[int] = []
    ws: list[int] = []
    for inp, out in task.train:
        hs += [len(inp), len(out)]
        ws += [len(inp[0]) if inp else 0, len(out[0]) if out else 0]
    hs.append(len(task.test_in))
    ws.append(len(task.test_in[0]) if task.test_in else 0)
    if task.test_out:
        hs.append(len(task.test_out))
        ws.append(len(task.test_out[0]) if task.test_out else 0)
    return (max(hs) if hs else 0, max(ws) if ws else 0)


def _pad(grid: Grid, h: int, w: int) -> np.ndarray:
    arr = np.zeros((h, w), dtype=np.float64)
    for i, row in enumerate(grid[:h]):
        for j, v in enumerate(row[:w]):
            arr[i, j] = v
    return arr.reshape(-1)


def ttt_prefix(task: ArcTask) -> Grid:
    """Fit a tiny linear probe on D4-augmented demos; predict test (CPU)."""
    if not task.train:
        return _copy(task.test_in)
    h, w = _max_hw(task)
    if h == 0 or w == 0:
        return _copy(task.test_in)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for inp, out in task.train:
        for fn in _D4_FNS:
            xs.append(_pad(fn(inp), h, w))
            ys.append(_pad(fn(out), h, w))
    x = np.stack(xs)
    y = np.stack(ys)
    w_mat, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = _pad(task.test_in, h, w) @ w_mat
    pred = np.clip(np.rint(pred), 0, 9).astype(int).reshape(h, w)
    th = len(task.test_out) if task.test_out else len(task.test_in)
    tw = len(task.test_out[0]) if task.test_out and task.test_out[0] else (
        len(task.test_in[0]) if task.test_in else 0
    )
    return pred[:th, :tw].tolist()


def _rng_grid(rng: np.random.Generator, n: int = 3) -> Grid:
    return rng.integers(1, 5, size=(n, n)).tolist()


def _apply_kind(grid: Grid, kind: str) -> Grid:
    if kind == "rotate":
        return rotate90(grid)
    if kind == "flip":
        return _flip_h(grid)
    if kind == "recolor":
        return color_permutation(grid, {1: 2})
    if kind == "identity":
        return _copy(grid)
    raise ValueError(f"unknown kind {kind}")


_GEN_OFFSET = 0
_EVAL_OFFSET = 10_000_019


def generate_task(seed: int = 0, kind: str = "rotate") -> ArcTask:
    """Procedural demos. Never emits held-out eval tasks."""
    rng = np.random.default_rng(_GEN_OFFSET + int(seed))
    train = []
    for _ in range(3):
        g = _rng_grid(rng)
        train.append((g, _apply_kind(g, kind)))
    test = _rng_grid(rng)
    return ArcTask(train, test, _apply_kind(test, kind), held_out=False)


def held_out_task(seed: int = 0, kind: str = "rotate") -> ArcTask:
    """Disjoint eval pool; generators never mix these in."""
    rng = np.random.default_rng(_EVAL_OFFSET + int(seed))
    train = []
    for _ in range(3):
        g = _rng_grid(rng)
        train.append((g, _apply_kind(g, kind)))
    test = _rng_grid(rng)
    return ArcTask(train, test, _apply_kind(test, kind), held_out=True)


@dataclass
class ArcEnv:
    task: ArcTask
    frame: Grid = field(init=False)
    seen: set[tuple[tuple[int, ...], ...]] = field(default_factory=set)
    t: int = 0

    def __post_init__(self) -> None:
        self.frame = _copy(self.task.test_in)

    def _key(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(row) for row in self.frame)

    def step(self, action: dict) -> tuple[Grid, float, dict[str, Any]]:
        name = action.get("op") or action.get("type")
        if name == "paint":
            y, x, c = int(action["y"]), int(action["x"]), int(action["color"])
            if 0 <= y < len(self.frame) and 0 <= x < len(self.frame[0]):
                self.frame = _copy(self.frame)
                self.frame[y][x] = c
        elif name == "submit":
            pass
        elif name:
            self.frame = apply_op(self.frame, action)
        self.t += 1
        key = self._key()
        bonus = 0.0 if key in self.seen else 0.01
        self.seen.add(key)
        reward = 1.0 if self.frame == self.task.test_out else 0.0
        done = reward == 1.0 or name == "submit"
        return _copy(self.frame), reward, {"bonus": bonus, "done": done, "t": self.t}
