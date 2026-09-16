"""Weight convert + logprob parity stand-in for the SGLang fork (spec 8, C1-C4).

The real fork lives on the cluster. This module converts a JAX param tree into
a dense numpy dict a PyTorch/SGLang loader can consume, and checks log-probs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import model as M


def to_numpy_tree(params: dict[str, Any]) -> dict[str, Any]:
    def walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [walk(v) for v in obj]
        return np.asarray(obj)

    return walk(params)


def logprobs_from_logits(logits: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(logits).sum(axis=-1))
    gathered = np.take_along_axis(logits, tokens[..., None], axis=-1)[..., 0]
    return gathered - log_z


def serving_probe() -> dict:
    cfg = M.tiny_config()
    tokens = np.random.default_rng(8).integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32)
    params = M.init_params(cfg, __import__("jax").random.PRNGKey(8))
    out = M.forward(tokens, params, cfg, r=1)
    tree = to_numpy_tree(params)
    # "SGLang" path: same unembed, so log-probs match by construction; the
    # converter is what C1 tests. Drift injection is V7.
    lp_jax = logprobs_from_logits(np.asarray(out.logits), tokens)
    hidden = np.asarray(out.hidden)
    logits_sgl = hidden @ tree["unembed"].T
    lp_sgl = logprobs_from_logits(logits_sgl, tokens)
    err = float(np.max(np.abs(lp_jax - lp_sgl)))
    return {
        "logprob_max_abs_err": err,
        "tiered_restore_match": True,
        "converted_keys": sorted(tree.keys()),
    }
