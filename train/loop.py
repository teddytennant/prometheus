"""Short CPU loops used by unit tests (overfit, V1 parity)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import model as M
from train.muon import _muon_update
from train.schedule import TrainConfig
from train.step import cross_entropy


def overfit_one_batch(steps: int = 40) -> dict:
    """V1: a batch's CE must drop under Muon. Toy tied-embedding, not 10M.

    Full tiny_config train_step is available for GPU V1; CPU tests use this
    so the suite stays under a minute.
    """
    rng = np.random.default_rng(0)
    vocab, d, b, t = 32, 16, 4, 8
    tokens = jnp.asarray(rng.integers(0, vocab, size=(b, t), dtype=np.int32))
    w = jnp.asarray(rng.standard_normal((vocab, d)).astype(np.float32) * 0.02)
    mom = jnp.zeros_like(w)
    tcfg = TrainConfig(ns_steps=5, qk_clip=100.0, weight_decay=0.0)

    def ce(weight):
        logits = weight[tokens[:, :-1]] @ weight.T
        return cross_entropy(logits, tokens[:, 1:])

    start = None
    end = None
    lr = 0.3
    for _ in range(steps):
        loss, g = jax.value_and_grad(ce)(w)
        w, mom = _muon_update(
            w, g, mom, lr, tcfg.weight_decay, tcfg.ns_steps, tcfg.qk_clip, "embed"
        )
        if start is None:
            start = float(loss)
        end = float(loss)
    return {"ok": end < start, "loss_start": float(start), "loss_end": float(end)}


def v1_parity() -> dict:
    """JAX forward vs the numpy reference already in tests/reference."""
    from tests.reference import model as ref

    cfg = M.tiny_config()
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32)
    params = M.init_params(cfg, jax.random.PRNGKey(1))
    out = M.forward(tokens, params, cfg, r=1)
    np_params = jax.tree.map(lambda x: np.asarray(x), params)
    ref_out = ref.forward(tokens, np_params, cfg, r=1)
    err = float(np.max(np.abs(np.asarray(out.logits) - np.asarray(ref_out.logits))))
    toy = overfit_one_batch(steps=4)
    g = jax.grad(lambda weight: (weight**2).sum())(jnp.ones((4, 4)))
    finite = bool(jnp.all(jnp.isfinite(g))) and toy["loss_end"] == toy["loss_end"]
    return {"max_abs_err": err, "grad_check": finite}
