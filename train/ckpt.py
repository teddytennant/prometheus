"""Checkpoint save/load for rung-0 (spec 16.2 V5). Real files, not a fake bool."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np

from train.muon import OptState


def save_checkpoint(path: str | Path, params: Any, opt: OptState, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    p_leaves, p_def = jax.tree_util.tree_flatten(params)
    m_leaves, m_def = jax.tree_util.tree_flatten(opt.momentum)
    np.savez_compressed(path / "params.npz", *[np.asarray(x) for x in p_leaves])
    np.savez_compressed(path / "momentum.npz", *[np.asarray(x) for x in m_leaves])
    with (path / "tree.pkl").open("wb") as fh:
        pickle.dump({"p_def": p_def, "m_def": m_def}, fh)
    payload = {**meta, "opt_step": int(opt.step), "n_param_leaves": len(p_leaves)}
    (path / "meta.json").write_text(json.dumps(payload, indent=2, default=str))


def _leaves(npz) -> list[np.ndarray]:
    keys = sorted(npz.files, key=lambda k: int(k.split("_")[-1]))
    return [npz[k] for k in keys]


def load_checkpoint(path: str | Path) -> tuple[Any, OptState, dict[str, Any]] | None:
    path = Path(path)
    if not (path / "params.npz").exists() or not (path / "tree.pkl").exists():
        return None
    with (path / "tree.pkl").open("rb") as fh:
        defs = pickle.load(fh)
    pfile = np.load(path / "params.npz")
    params = jax.tree_util.tree_unflatten(defs["p_def"], _leaves(pfile))
    mom: Any = {}
    if (path / "momentum.npz").exists():
        mfile = np.load(path / "momentum.npz")
        mom = jax.tree_util.tree_unflatten(defs["m_def"], _leaves(mfile))
    meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}
    opt = OptState(step=int(meta.get("opt_step", 0)), momentum=mom)
    return params, opt, meta
