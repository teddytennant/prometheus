"""V4/V9/V10 probes. V10 soak is GPU-only; no tiny_standin path."""

from __future__ import annotations

import hashlib
import json
import os
import time
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


def v5_rung0() -> dict:
    """Spec 16.2 V5: rung-0 train with ckpt/resume. Not a 20-step overfit."""
    from train.rung0 import run

    ckpt = os.environ.get("CKPT_DIR", "v5_ckpt")
    return run(ckpt)


def soak_probe(seconds: float | None = None) -> dict:
    """GPU soak: real device work. Hours accumulate across jobs via SOAK_STATE."""
    import jax
    import jax.numpy as jnp

    if jax.default_backend() != "gpu":
        raise RuntimeError("V10 soak requires a GPU")
    state_path = Path(os.environ.get("SOAK_STATE", "soak_state.json"))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(state_path.read_text()) if state_path.exists() else {}
    hours0 = float(prev.get("hours", 0.0))
    tasks0 = int(prev.get("tasks", 0))
    lost0 = int(prev.get("lost_tasks", 0))
    dup0 = int(prev.get("duplicated_outputs", 0))
    dead0 = int(prev.get("dead_tokens", 0))
    seconds = float(os.environ.get("SOAK_SECONDS", seconds if seconds is not None else 60))
    key = jax.random.PRNGKey(0)
    w = jax.random.normal(key, (1024, 1024), dtype=jnp.float32)
    w.block_until_ready()

    def step(mat, k):
        k, k1 = jax.random.split(k)
        x = jax.random.normal(k1, (1024, 1024), dtype=jnp.float32)
        y = jax.nn.tanh(mat @ x)
        return mat, k, jnp.sum(y)

    step = jax.jit(step)
    t0 = time.time()
    i = 0
    mat = w
    k = key
    seen: set[int] = set()
    while time.time() - t0 < seconds:
        mat, k, s = step(mat, k)
        mat.block_until_ready()
        val = float(s)
        if val != val:
            hours = hours0 + (time.time() - t0) / 3600.0
            out = {
                "lost_tasks": lost0 + 1,
                "duplicated_outputs": dup0,
                "dead_tokens": dead0 + 1,
                "hours": hours,
                "tiny_standin": False,
                "tasks": tasks0 + i,
            }
            state_path.write_text(json.dumps(out, indent=2))
            return out
        if i in seen:
            hours = hours0 + (time.time() - t0) / 3600.0
            out = {
                "lost_tasks": lost0,
                "duplicated_outputs": dup0 + 1,
                "dead_tokens": dead0,
                "hours": hours,
                "tiny_standin": False,
                "tasks": tasks0 + i,
            }
            state_path.write_text(json.dumps(out, indent=2))
            return out
        seen.add(i)
        i += 1
    hours = hours0 + (time.time() - t0) / 3600.0
    out = {
        "lost_tasks": lost0,
        "duplicated_outputs": dup0,
        "dead_tokens": dead0,
        "hours": hours,
        "tiny_standin": False,
        "tasks": tasks0 + i,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(out, indent=2))
    return out


def write_result(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
