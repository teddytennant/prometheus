"""ARC-AGI-2 grids, pass@2, latent-first (spec 10, 15.5 D6)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArcTask:
    train: list[tuple[list[list[int]], list[list[int]]]]
    test_in: list[list[int]]
    test_out: list[list[int]]


def rotate90(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in zip(*grid[::-1])]


def pass_at_k(preds: list[list[list[int]]], gold: list[list[int]], k: int = 2) -> bool:
    return any(p == gold for p in preds[:k])


def color_counts(grid: list[list[int]]) -> dict[int, int]:
    out: dict[int, int] = {}
    for row in grid:
        for v in row:
            out[v] = out.get(v, 0) + 1
    return out
