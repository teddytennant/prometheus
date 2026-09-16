"""New kernel APIs: chunked delta-rule, MLA+RoPE, EP all-to-all, per-block FP8."""

from __future__ import annotations

import numpy as np

from kernels import (
    all_to_all,
    apply_rope,
    fake_quant_fp8,
    linear_attention,
    mla_attention,
    nvfp4_pack,
    nvfp4_roundtrip,
    nvfp4_unpack,
    partition_by_rank,
    two_precision_train,
)
from kernels.ep import expert_home
from kernels.quant import FP8_BLOCK


def test_chunked_linear_attention_matches_unchunked():
    rng = np.random.default_rng(2)
    q = rng.standard_normal((2, 9, 2, 4)).astype(np.float32)
    k = rng.standard_normal((2, 9, 2, 4)).astype(np.float32)
    v = rng.standard_normal((2, 9, 2, 4)).astype(np.float32)
    a, sa = linear_attention(q, k, v)
    b, sb = linear_attention(q, k, v, chunk_size=3)
    np.testing.assert_allclose(a, b, atol=1e-5)
    np.testing.assert_allclose(sa, sb, atol=1e-5)


def test_mla_qk_norm_and_rope():
    rng = np.random.default_rng(3)
    q = rng.standard_normal((1, 4, 2, 8)).astype(np.float32)
    k_nope = rng.standard_normal((1, 4, 1, 4)).astype(np.float32)
    k_rope = rng.standard_normal((1, 4, 1, 4)).astype(np.float32)
    y = mla_attention(q, k_nope, k_rope, qk_norm=True)
    y2 = mla_attention(q, k_nope, k_rope, qk_norm=False)
    assert y.shape == (1, 4, 2, 4)
    assert not np.allclose(y, y2, atol=1e-6)
    pos = np.arange(4, dtype=np.float32)
    y3 = mla_attention(q, k_nope, k_rope, qk_norm=True, positions=pos)
    assert y3.shape == y.shape
    rot = apply_rope(q, pos)
    assert rot.shape == q.shape
    assert not np.allclose(rot, q)


def test_all_to_all_is_involution():
    n = 3
    send = [[np.array([src * 10 + dst], dtype=np.int32) for dst in range(n)] for src in range(n)]
    back = all_to_all(all_to_all(send))
    for i in range(n):
        for j in range(n):
            np.testing.assert_array_equal(back[i][j], send[i][j])
    tokens = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    ids = np.array([0, 3, 1, 4, 2, 5], dtype=np.int32)
    n_ep = 2
    parts = partition_by_rank(tokens, ids, n_ep)
    assert len(parts) == n_ep
    homes = [expert_home(int(e), n_ep) for e in ids]
    assert homes[0] != homes[1]


def test_fp8_per_block_keeps_small_block():
    x = np.zeros((64,), dtype=np.float32)
    x[:32] = 1.0e6
    x[32:] = 0.25
    q = np.asarray(fake_quant_fp8(x, block_size=FP8_BLOCK))
    assert np.max(np.abs(q[32:] - 0.25)) < 0.05
    assert q[0] > 1.0e5


def test_nvfp4_nibble_roundtrip_codes():
    codes = np.array([0, 1, 7, 8, 15, 3, 4, 9], dtype=np.uint8)
    packed = nvfp4_pack(codes)
    assert packed.dtype == np.uint8
    assert packed.size == 4
    assert np.array_equal(nvfp4_unpack(packed, codes.size), codes)
    x = np.linspace(-4, 4, 32, dtype=np.float32)
    y = np.asarray(nvfp4_roundtrip(x))
    assert y.shape == x.shape
    assert np.isfinite(y).all()


def test_two_precision_marks_fp8_linears_bf16_router():
    r = two_precision_train(n_steps=8, seed=1)
    assert r["fp8_linears"] is True
    assert r["bf16_router_norm"] is True
    assert r["fp8_vs_bf16_rel"] >= 0.0
