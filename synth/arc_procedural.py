"""re-ARC style procedural families (spec 7.1 E3)."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

Grid = list[list[int]]
Transform = Callable[[Grid], Grid]


def _grid(rng: random.Random, h: int, w: int, colors: int) -> Grid:
    return [[rng.randint(0, colors - 1) for _ in range(w)] for _ in range(h)]


def _rotate90(g: Grid) -> Grid:
    return [list(row) for row in zip(*g[::-1])]


def _translate(g: Grid, dy: int, dx: int, fill: int = 0) -> Grid:
    h, w = len(g), len(g[0])
    out = [[fill] * w for _ in range(h)]
    for i, row in enumerate(g):
        for j, v in enumerate(row):
            ni, nj = i + dy, j + dx
            if 0 <= ni < h and 0 <= nj < w:
                out[ni][nj] = v
    return out


def _color_map(g: Grid, perm: list[int]) -> Grid:
    return [[perm[v] for v in row] for row in g]


def _scale(g: Grid, k: int) -> Grid:
    out: Grid = []
    for row in g:
        scaled: list[int] = []
        for v in row:
            scaled.extend([v] * k)
        for _ in range(k):
            out.append(list(scaled))
    return out


def _flood_fill(g: Grid, sr: int, sc: int, new_c: int) -> Grid:
    out = [list(row) for row in g]
    h, w = len(out), len(out[0])
    old = out[sr][sc]
    if old == new_c:
        return out
    stack = [(sr, sc)]
    seen = {(sr, sc)}
    while stack:
        i, j = stack.pop()
        out[i][j] = new_c
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and (ni, nj) not in seen and out[ni][nj] == old:
                seen.add((ni, nj))
                stack.append((ni, nj))
    return out


def _make_transform(family: str, rng: random.Random) -> Transform:
    if family == "rotate":
        k = rng.randint(1, 3)

        def xf(g: Grid) -> Grid:
            out = g
            for _ in range(k):
                out = _rotate90(out)
            return out

        return xf
    if family == "translate":
        dy, dx = rng.choice((-1, 0, 1)), rng.choice((-1, 0, 1))

        def xf(g: Grid) -> Grid:
            return _translate(g, dy, dx)

        return xf
    if family == "color_map":
        perm = list(range(10))
        rng.shuffle(perm)

        def xf(g: Grid) -> Grid:
            return _color_map(g, perm)

        return xf
    if family == "scale":

        def xf(g: Grid) -> Grid:
            return _scale(g, 2)

        return xf
    if family == "flood_fill":
        new_c = rng.randint(0, 9)

        def xf(g: Grid) -> Grid:
            return _flood_fill(g, 0, 0, new_c)

        return xf
    raise ValueError(family)


def arc_procedural(n: int = 100, seed: int = 0) -> list[dict[str, Any]]:
    """Train pairs + test; one shared transform per item."""
    rng = random.Random(seed)
    families = ("rotate", "translate", "color_map", "flood_fill", "scale")
    items: list[dict[str, Any]] = []
    for i in range(n):
        family = families[i % len(families)]
        xf = _make_transform(family, rng)
        h, w = rng.randint(2, 4), rng.randint(2, 4)
        colors = rng.randint(2, 4)
        train = []
        for _ in range(2):
            g = _grid(rng, h, w, colors)
            train.append((g, xf(g)))
        tg = _grid(rng, h, w, colors)
        items.append({"family": family, "train": train, "test": (tg, xf(tg)), "id": i})
    return items
