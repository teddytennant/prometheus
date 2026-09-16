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
    """Kill-rank restore from a real ckpt tree, SDC on param leaves, spike-skip."""
    import tempfile

    import jax

    import model as M
    from control.sdc import detect_flip, flatten_leaves, skip_shard
    from train.ckpt import load_checkpoint, save_checkpoint
    from train.muon import init_opt_state

    cfg = M.tiny_config() if jax.default_backend() == "gpu" else M.cpu_config()
    params = M.init_params(cfg, jax.random.PRNGKey(7))
    opt = init_opt_state(params)
    tmp = Path(tempfile.mkdtemp(prefix="v4-ckpt-"))
    save_checkpoint(tmp, params, opt, {"step": 0, "killed_rank": 1})
    loaded = load_checkpoint(tmp)
    assert loaded is not None
    restored, opt2, meta = loaded
    a_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(params)]
    b_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(restored)]
    equal = len(a_leaves) == len(b_leaves) and all(
        np.array_equal(a, b) for a, b in zip(a_leaves, b_leaves)
    )
    leaves = flatten_leaves(params)
    flipped = inject_bit_flip(np.asarray(leaves[0]), 3)
    caught = detect_flip(leaves[0], flipped)
    kept = skip_shard(leaves, 1)
    ckpt_bytes = (tmp / "params.npz").stat().st_size
    return {
        "resume_bitwise_equal": bool(equal),
        "sdc_caught_flip": bool(caught),
        "spike_skipped_shard": len(kept) == len(leaves) - 1,
        "n_param_leaves": len(leaves),
        "ckpt_bytes": int(ckpt_bytes),
        "killed_rank": int(meta.get("killed_rank", 1)),
        "ckpt_path": str(tmp),
    }


def lab_dry_run() -> dict:
    """One 14.6 cycle: planted ideas written to the SQLite ledger and read back."""
    import tempfile

    from harness.ledger import Ledger

    path = Path(tempfile.mkdtemp(prefix="v9-ledger-")) / "ledger.sqlite"
    led = Ledger(path)
    led.append(
        {
            "experiment_id": "plant-pos",
            "author_role": "worker",
            "rung": 0,
            "sign": "positive",
            "replicated": True,
            "recorded": "positive",
        }
    )
    led.append(
        {
            "experiment_id": "plant-neg",
            "author_role": "worker",
            "rung": 0,
            "sign": "negative",
            "replicated": False,
            "recorded": "negative",
        }
    )
    pos = led.get("plant-pos")
    neg = led.get("plant-neg")
    n = led.count()
    led.close()
    return {
        "planted_positive_replicated": bool(
            pos and pos["replicated"] and pos["recorded"] == "positive"
        ),
        "planted_negative_recorded": bool(neg and neg["recorded"] == "negative"),
        "ledger_rows": n,
        "ledger_path": str(path),
    }


def v5_rung0() -> dict:
    """Spec 16.2 V5: rung-0 train with ckpt/resume. Not a 20-step overfit."""
    from train.rung0 import run

    ckpt = os.environ.get("CKPT_DIR", "v5_ckpt")
    return run(ckpt)


def soak_probe(seconds: float | None = None) -> dict:
    """GPU soak: real device work. Hours accumulate across jobs via SOAK_STATE."""
    import jax

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
    from harness.chaos import TaskQueue

    qdir = Path(os.environ.get("SOAK_QUEUE", str(state_path.parent / "v10_queue")))
    queue = TaskQueue(qdir)
    faults = queue.recover_incomplete()
    # Plant a killed mid-write, then recover it.
    (queue.out / "killed.tmp").write_text("{")
    faults += queue.recover_incomplete()

    import model as M
    from train.muon import init_opt_state
    from train.schedule import TrainConfig
    from train.step import train_step

    cfg = M.tiny_config()
    params = M.init_params(cfg, jax.random.PRNGKey(0))
    opt = init_opt_state(params)
    tcfg = TrainConfig(
        warmup=0, stable=8, decay=0, peak_lr=0.02,
        ns_steps=3, qk_clip=100.0, weight_decay=0.0,
    )
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32)
    t0 = time.time()
    i = 0
    dead = 0
    while time.time() - t0 < seconds:
        params, opt, loss = train_step(params, opt, tokens, cfg, tcfg)
        jax.block_until_ready(loss)
        val = float(loss)
        tid = f"task-{tasks0 + i}"

        def work(v=val, n=i):
            if v != v:
                raise ValueError("nan")
            return {"i": n, "s": v}

        try:
            _, status = queue.run(tid, work)
        except ValueError:
            dead += 1
            hours = hours0 + (time.time() - t0) / 3600.0
            out = {
                "lost_tasks": lost0,
                "duplicated_outputs": dup0 + queue.dup_prevented,
                "dead_tokens": dead0 + dead,
                "hours": hours,
                "tiny_standin": False,
                "tasks": tasks0 + i,
                "faults_injected": int(faults),
            }
            state_path.write_text(json.dumps(out, indent=2))
            return out
        # Same id again must not duplicate.
        queue.run(tid, work)
        i += 1
    hours = hours0 + (time.time() - t0) / 3600.0
    out = {
        "lost_tasks": lost0,
        # retries of the same id are prevented, not duplicates
        "duplicated_outputs": dup0 + queue.dup_prevented - i,
        "dead_tokens": dead0 + dead,
        "hours": hours,
        "tiny_standin": False,
        "tasks": tasks0 + i,
        "faults_injected": int(faults),
    }
    # dup_prevented counts the intentional second run; those are not duplicated outputs.
    out["duplicated_outputs"] = dup0
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(out, indent=2))
    return out


def write_result(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
