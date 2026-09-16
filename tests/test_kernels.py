"""Kernel reference vs model (A3)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import kernels
import model as M


def test_linear_attention_matches_model():
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.standard_normal((2, 4, 2, 8)).astype(np.float32))
    k = jnp.asarray(rng.standard_normal((2, 4, 2, 8)).astype(np.float32))
    v = jnp.asarray(rng.standard_normal((2, 4, 2, 8)).astype(np.float32))
    a, _ = kernels.linear_attention(q, k, v)
    b, _ = M.linear_attention(q, k, v)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_precision_probe_finite():
    r = kernels.precision_probe()
    assert r["nvfp4_numerics_ok"]
    assert r["fp8_vs_bf16_rel"] >= 0
