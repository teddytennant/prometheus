"""SDC hash and spike-skip on real parameter shards (spec 5.4, 16.2 V4)."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def shard_hash(arr: Any) -> str:
    buf = np.ascontiguousarray(np.asarray(arr)).tobytes()
    return hashlib.sha256(buf).hexdigest()


def detect_flip(original: Any, candidate: Any) -> bool:
    return shard_hash(original) != shard_hash(candidate)


def skip_shard(shards: list[Any], bad: int) -> list[Any]:
    return [s for i, s in enumerate(shards) if i != bad]


def flatten_leaves(tree: Any) -> list[np.ndarray]:
    leaves: list[np.ndarray] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)
        else:
            leaves.append(np.asarray(obj))

    walk(tree)
    return leaves
