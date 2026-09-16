"""MuonClip, WSD, train step (spec 5, A2)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from train import TrainConfig, newton_schulz, overfit_one_batch, qk_clip, wsd_lr


def test_newton_schulz_near_orthogonal():
    rng = np.random.default_rng(0)
    g = rng.standard_normal((8, 8)).astype(np.float32)
    q = np.asarray(newton_schulz(jnp.asarray(g), steps=5))
    u, _, vt = np.linalg.svd(g, full_matrices=False)
    polar = u @ vt
    assert float(np.max(np.abs(q - polar))) < 0.3
    gn = g / np.linalg.norm(g)

    def off(m):
        gram = m @ m.T
        return float(np.max(np.abs(gram - np.eye(8))))

    assert off(q) < off(gn)


def test_wsd_phases():
    cfg = TrainConfig(warmup=10, stable=20, decay=10, peak_lr=1.0)
    assert wsd_lr(0, cfg) == 0.0
    assert abs(wsd_lr(10, cfg) - 1.0) < 1e-9
    assert abs(wsd_lr(25, cfg) - 1.0) < 1e-9
    assert wsd_lr(40, cfg) == 0.0


def test_qk_clip_caps_norm():
    x = jnp.ones((4, 4)) * 50
    y = qk_clip(x, 10.0)
    assert float(jnp.linalg.norm(y)) <= 10.0 + 1e-4


def test_overfit_one_batch():
    result = overfit_one_batch(steps=25)
    assert result["ok"], result
