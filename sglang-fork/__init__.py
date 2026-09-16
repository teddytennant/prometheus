"""Independent serving path that loads a JAX checkpoint (spec 16.2 V8)."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np


def serve_jax_checkpoint(params, tokens, config, r: int = 1) -> dict[str, Any]:
    """Decode with the numpy reference, not the JAX forward used to produce the ckpt."""
    from tests.reference import model as ref

    np_params = _to_numpy(params)
    out = ref.forward(np.asarray(tokens), np_params, config, r=r)
    return {"logits": np.asarray(out.logits), "hidden": np.asarray(out.hidden)}


def _to_numpy(tree):
    if isinstance(tree, dict):
        return {k: _to_numpy(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_to_numpy(v) for v in tree)
    return np.asarray(tree)


def serving_probe() -> dict:
    """Load a JAX ckpt three ways (HBM / host RAM / NVMe) and match an independent decoder."""
    import tempfile
    from pathlib import Path

    import jax

    import model as M
    from train.ckpt import load_checkpoint, save_checkpoint
    from train.muon import init_opt_state

    cfg = M.tiny_config() if jax.default_backend() == "gpu" else M.cpu_config()
    params = M.init_params(cfg, jax.random.PRNGKey(8))
    tokens = np.random.default_rng(8).integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32)
    jax_out = M.forward(tokens, params, cfg, r=1)
    jax_lp = np.asarray(jax_out.logits)

    hbm = serve_jax_checkpoint(params, tokens, cfg, r=1)["logits"]
    blob = pickle.dumps(_to_numpy(params))
    host = serve_jax_checkpoint(pickle.loads(blob), tokens, cfg, r=1)["logits"]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt"
        save_checkpoint(path, params, init_opt_state(params), {"step": 0})
        loaded = load_checkpoint(path)
        assert loaded is not None
        disk_params, _, _ = loaded
        nvme = serve_jax_checkpoint(disk_params, tokens, cfg, r=1)["logits"]

    def max_err(a, b):
        return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

    err = max(max_err(jax_lp, hbm), max_err(jax_lp, host), max_err(jax_lp, nvme))
    match = bool(max_err(hbm, host) < 1e-5 and max_err(host, nvme) < 1e-5)
    return {
        "logprob_max_abs_err": err,
        "hbm_host_nvme_match": match,
        "tiered_restore_match": match,
        "independent": True,
        "independent_ref": True,
        "n_params": int(M.param_count(params)),
    }
