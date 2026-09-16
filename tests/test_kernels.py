"""Kernel probes (A3) plus invertibility and FP8 bounds."""

from __future__ import annotations

import numpy as np

import kernels
import model as M
from kernels import combine, dispatch, fake_quant_fp8, precision_probe, two_precision_train
from kernels.ep import reconstruct_from_dispatch
from kernels.quant import fp8_block_scales, nvfp4_pack, nvfp4_unpack


def test_linear_attention_matches_model():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((2, 4, 2, 8)).astype(np.float32)
    k = rng.standard_normal((2, 4, 2, 8)).astype(np.float32)
    v = rng.standard_normal((2, 4, 2, 8)).astype(np.float32)
    a, _ = kernels.linear_attention(q, k, v)
    b, _ = M.linear_attention(q, k, v)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_precision_probe_finite():
    p = precision_probe()
    assert np.isfinite(p["fp8_vs_bf16_rel"])
    assert p["nvfp4_numerics_ok"]


def test_two_precision_train_runs():
    r = two_precision_train(n_steps=8, seed=0)
    assert r["n_steps"] == 8
    assert np.isfinite(r["fp8_vs_bf16_rel"])
    assert r["nvfp4_numerics_ok"]
    assert r["n_params"] > 0


def test_dispatch_combine_invertible():
    tokens = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
    ids = np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int32)
    buckets = dispatch(tokens, ids, 3)
    recon = reconstruct_from_dispatch(buckets, ids, 8, 3)
    np.testing.assert_allclose(recon, tokens)
    expert_out = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    weights = np.array([0.5, 1.0, 0.25], dtype=np.float32)
    eids = np.array([0, 1, 2], dtype=np.int32)
    got = combine(expert_out, weights, eids)
    want = expert_out * weights[:, None]
    np.testing.assert_allclose(got, want)


def test_fp8_vs_bf16_relative_error_bound():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((16, 16)).astype(np.float32)
    q = np.asarray(fake_quant_fp8(x))
    rel = float(np.linalg.norm(q - x) / (np.linalg.norm(x) + 1e-8))
    assert np.isfinite(rel)
    assert rel < 0.15
    scales = np.asarray(fp8_block_scales(x, block_size=32))
    assert scales.size >= 1
    packed = nvfp4_pack(np.arange(16, dtype=np.uint8))
    assert np.array_equal(nvfp4_unpack(packed, 16), np.arange(16, dtype=np.uint8))
