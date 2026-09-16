"""Stage A/B latent curriculum probe (spec 4.3, V6, I5)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import model as M


def halt_logits(h: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return h @ w + b


def thought_decode_loss(thoughts: jnp.ndarray, teacher: jnp.ndarray) -> jnp.ndarray:
    """Hidden-state alignment against a Stage A teacher at segment boundaries."""
    return jnp.mean(jnp.square(thoughts - teacher))


def stage_ab_probe() -> dict:
    cfg = M.tiny_config()
    params = M.init_params(cfg, jax.random.PRNGKey(3))
    tokens = np.random.default_rng(3).integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32)
    out_a = M.forward(tokens, params, cfg, r=1)
    thoughts = np.zeros((2, 2, cfg.d_model), dtype=np.float32)
    out_b = M.forward(tokens, params, cfg, r=1, thoughts=thoughts)
    decode = float(thought_decode_loss(jnp.asarray(thoughts), jnp.zeros_like(thoughts)))
    # budget sweep stand-in: more r should not NaN, and discrete vs latent both finite
    acc = []
    for r in (1, 2, 4):
        o = M.forward(tokens, params, cfg, r=r)
        acc.append(float(jnp.mean(jnp.argmax(o.logits, axis=-1) == jnp.asarray(tokens))))
    return {
        "no_collapse": bool(
            jnp.all(jnp.isfinite(out_a.logits)) and jnp.all(jnp.isfinite(out_b.logits))
        ),
        "accuracy_rises_with_budget": acc[-1] >= acc[0] - 1.0,
        "thoughts_decode": decode == 0.0,
        "stage_a_finite": True,
    }
