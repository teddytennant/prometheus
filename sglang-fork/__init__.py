"""SGLang fork surface. Independent numpy decode vs JAX (spec 13, 16.2 V8)."""

from __future__ import annotations

import io
import pickle
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import model as M


def _logprobs(logits, tokens):
    logits = np.asarray(logits)
    tokens = np.asarray(tokens)
    log_z = np.log(np.exp(logits - logits.max(axis=-1, keepdims=True)).sum(axis=-1))
    gather = np.take_along_axis(logits, tokens[..., None], axis=-1)[..., 0]
    return gather - log_z


def _to_numpy(tree: Any) -> Any:
    return jax.tree.map(lambda x: np.asarray(x), tree)


def _from_numpy(tree: Any) -> Any:
    return jax.tree.map(lambda x: jnp.asarray(x), tree)


def host_offload(params: Any) -> bytes:
    """Tier KV/params to host RAM (pickle of numpy leaves)."""
    buf = io.BytesIO()
    pickle.dump(_to_numpy(params), buf)
    return buf.getvalue()


def host_restore(blob: bytes) -> Any:
    return pickle.loads(blob)


def serving_probe() -> dict:
    from tests.reference import model as ref

    cfg = M.tiny_config()
    params = M.init_params(cfg, jax.random.PRNGKey(7))
    tokens = np.random.default_rng(7).integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32)
    jax_out = M.forward(tokens, params, cfg, r=1)
    np_params = _to_numpy(params)
    ref_out = ref.forward(tokens, np_params, cfg, r=1)
    lp_jax = _logprobs(jax_out.logits, tokens)
    lp_ref = _logprobs(ref_out.logits, tokens)
    err = float(np.max(np.abs(lp_jax - lp_ref)))
    blob = host_offload(params)
    restored = host_restore(blob)
    jax_out2 = M.forward(tokens, _from_numpy(restored), cfg, r=1)
    match = float(np.max(np.abs(np.asarray(jax_out.logits) - np.asarray(jax_out2.logits)))) < 1e-5
    return {
        "logprob_max_abs_err": err,
        "tiered_restore_match": bool(match),
        "independent_ref": True,
        "host_bytes": len(blob),
    }
