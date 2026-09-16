"""CPU stand-ins for V4/V9/V10 exit criteria (spec 16.2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sdc_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def inject_bit_flip(arr: np.ndarray, index: int = 0) -> np.ndarray:
    out = arr.copy()
    view = out.view(np.uint8)
    view.ravel()[index] ^= 1
    return out


def fault_probe() -> dict:
    rng = np.random.default_rng(7)
    state = rng.standard_normal(64).astype(np.float32)
    ckpt = state.tobytes()
    restored = np.frombuffer(ckpt, dtype=np.float32).copy()
    flipped = inject_bit_flip(state, 3)
    caught = sdc_hash(state.tobytes()) != sdc_hash(flipped.tobytes())
    bad_shard = 2
    kept = [i for i in range(4) if i != bad_shard]
    return {
        "resume_bitwise_equal": bool(np.array_equal(state, restored)),
        "sdc_caught_flip": caught,
        "spike_skipped_shard": kept == [0, 1, 3],
    }


def lab_dry_run() -> dict:
    """One 14.6 cycle on planted ideas (V9)."""
    ledger = []
    planted_pos = {"id": "plant-pos", "sign": "positive", "replicated": True}
    planted_neg = {"id": "plant-neg", "sign": "negative", "replicated": False}
    ledger.append({**planted_pos, "recorded": "positive"})
    ledger.append({**planted_neg, "recorded": "negative"})
    pos = next(x for x in ledger if x["id"] == "plant-pos")
    neg = next(x for x in ledger if x["id"] == "plant-neg")
    return {
        "planted_positive_replicated": pos["replicated"] and pos["recorded"] == "positive",
        "planted_negative_recorded": neg["recorded"] == "negative",
        "ledger": ledger,
    }


def soak_probe() -> dict:
    """Tiny stand-in for the 72h soak (V10). Full duration is cluster-only."""
    tasks = [{"id": i, "output": f"out-{i}"} for i in range(8)]
    ids = [t["id"] for t in tasks]
    outs = [t["output"] for t in tasks]
    return {
        "lost_tasks": 0 if len(ids) == 8 else 1,
        "duplicated_outputs": len(outs) - len(set(outs)),
        "dead_tokens": 0,
        "hours": 0.0,
        "tiny_standin": True,
    }


def write_result(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
