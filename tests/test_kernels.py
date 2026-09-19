"""Implementer-facing tests for the ``kernels`` public API (spec 5.1, A3).

These import production ``kernels``. Constants, enums, and dataclasses are real
and membership tests may pass against the stub. Every test that *calls*
``chunked_delta_rule``, ``chunked_delta_rule_fwd``, ``chunked_delta_rule_bwd``,
``fp8_quantize``, ``fp8_dequantize``, ``fp8_linear``, ``fp8_linear_fwd``,
``fp8_linear_bwd``, ``ep_dispatch``, or ``ep_combine`` must FAIL on the stub.

The numpy reference (``tests.reference.kernels``) is the gated delta-rule
recurrence with no chunking, E4M3FN per-block abs-max fake-quant, and
node-limited EP dispatch. Production must match it.

Groups:
- Constants: DEFAULT_CHUNK 64, DEFAULT_FP8_BLOCK 128, MAX_RACKS 4, DType,
  LinearAttnConfig / Fp8Meta / DispatchMeta / KernelError.
- Linear attention: vs-reference, causality, state carry, chunk-size
  independence, shapes/dtypes, finite-difference grads, custom_vjp, goldens,
  KernelError on bad shapes.
- FP8: roundtrip bound, vs-reference, vs FP32 matmul (documented tolerance),
  backward uses forward scales (STE), jax.custom_vjp / jax.grad STE
  (matches fp8_linear_bwd residual, not scale-path autodiff), goldens,
  KernelError on bad shapes.
- EP: combine(dispatch(x)) weighted identity, rack span > MAX_RACKS,
  padding shape, goldens, residual is opaque.
- GPU: V1 linear-attn parity, V3 FP8 parity and jax.grad STE (pytest marker ``gpu``).
"""

from __future__ import annotations

import numpy as np
import pytest

import kernels
from tests.reference import kernels as ref

TOL = dict(rtol=1e-5, atol=1e-5)
FD_TOL = dict(rtol=1e-3, atol=1e-3)
# E4M3FN 3-bit mantissa: max rounding error relative to block amax is ~16/448.
FP8_ROUNDTRIP_AMAX_FRAC = 0.05
# Fake-quant matmul vs exact FP32: cosine stays > 0.99; use a generous abs floor
# because relative error blows up on near-zero outputs.
FP8_VS_FP32 = dict(rtol=0.05, atol=0.2)


def _np(x):
    return np.asarray(x)


def _require_gpu():
    jax = pytest.importorskip("jax")
    gpus = [d for d in jax.devices() if d.platform in ("gpu", "cuda", "tpu")]
    if not gpus:
        pytest.skip("V-stage GPU test requires a GPU device")
    return jax


def _jax():
    jax = pytest.importorskip("jax")
    return jax, jax.numpy


def _delta_inputs(rng, batch=2, seq=5, heads=2, dim=4):
    q = rng.standard_normal((batch, seq, heads, dim)).astype(np.float32)
    k = rng.standard_normal((batch, seq, heads, dim)).astype(np.float32)
    v = rng.standard_normal((batch, seq, heads, dim)).astype(np.float32)
    beta = (0.35 + 0.30 * rng.random((batch, seq, heads))).astype(np.float32)
    return q, k, v, beta


def _unit(rng, shape):
    d = rng.standard_normal(shape).astype(np.float32)
    d /= np.linalg.norm(d) + 1e-12
    return d


def _dispatch_meta(expert_ids, probs, racks, n_experts, max_racks=None):
    if max_racks is None:
        max_racks = kernels.MAX_RACKS
    return kernels.DispatchMeta(
        expert_ids=np.asarray(expert_ids, dtype=np.int32),
        probs=np.asarray(probs, dtype=np.float32),
        racks=np.asarray(racks, dtype=np.int32),
        n_experts=int(n_experts),
        max_racks=int(max_racks),
    )


# ---------------------------------------------------------------------------
# Constants / dataclasses (may pass on the stub)
# ---------------------------------------------------------------------------


def test_default_chunk_is_64():
    assert kernels.DEFAULT_CHUNK == 64


def test_default_fp8_block_is_128():
    assert kernels.DEFAULT_FP8_BLOCK == 128


def test_max_racks_is_4():
    assert kernels.MAX_RACKS == 4


def test_dtype_enum_values():
    assert kernels.DType.FP32 == "fp32"
    assert kernels.DType.BF16 == "bf16"
    assert kernels.DType.FP8 == "fp8"
    assert kernels.DType.NVFP4 == "nvfp4"


def test_linear_attn_config_defaults():
    cfg = kernels.LinearAttnConfig()
    assert cfg.chunk == 64
    assert cfg.eps == pytest.approx(1e-6)


def test_kernel_error_is_value_error():
    assert issubclass(kernels.KernelError, ValueError)
    err = kernels.KernelError("shape")
    assert isinstance(err, ValueError)


def test_fp8_meta_defaults():
    meta = kernels.Fp8Meta(q=np.zeros((1,), dtype=np.int8), scale=np.ones((1,), dtype=np.float32))
    assert meta.block == 128
    assert meta.dtype == kernels.DType.FP8


def test_dispatch_meta_default_max_racks():
    meta = kernels.DispatchMeta(
        expert_ids=np.zeros((1, 1), dtype=np.int32),
        probs=np.ones((1, 1), dtype=np.float32),
        racks=np.zeros((1,), dtype=np.int32),
        n_experts=1,
    )
    assert meta.max_racks == 4


# ---------------------------------------------------------------------------
# Gated delta-rule / chunked_delta_rule
# ---------------------------------------------------------------------------


def test_chunked_delta_rule_matches_reference():
    rng = np.random.default_rng(10)
    q, k, v, beta = _delta_inputs(rng)
    out, state = kernels.chunked_delta_rule(q, k, v, beta)
    exp_out, exp_state = ref.gated_delta_rule(q, k, v, beta)
    np.testing.assert_allclose(_np(out), exp_out, **TOL)
    np.testing.assert_allclose(_np(state), exp_state, **TOL)


def test_chunked_delta_rule_golden_orthogonal_keys():
    # S_0=0, β=1, orthonormal k, so o_t = v_t and S accumulates k v^T.
    q = np.array([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=np.float32)
    k = np.array([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=np.float32)
    v = np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=np.float32)
    beta = np.array([[[1.0], [1.0]]], dtype=np.float32)
    out, state = kernels.chunked_delta_rule(q, k, v, beta)
    np.testing.assert_allclose(
        _np(out), np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=np.float32), **TOL
    )
    np.testing.assert_allclose(
        _np(state), np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32), **TOL
    )


def test_chunked_delta_rule_golden_beta_half():
    q = np.array([[[[1.0, 0.0]]]], dtype=np.float32)
    k = np.array([[[[1.0, 0.0]]]], dtype=np.float32)
    v = np.array([[[[2.0, 4.0]]]], dtype=np.float32)
    beta = np.array([[[0.5]]], dtype=np.float32)
    out, state = kernels.chunked_delta_rule(q, k, v, beta)
    np.testing.assert_allclose(_np(out), np.array([[[[1.0, 2.0]]]], dtype=np.float32), **TOL)
    np.testing.assert_allclose(
        _np(state), np.array([[[[1.0, 2.0], [0.0, 0.0]]]], dtype=np.float32), **TOL
    )


def test_chunked_delta_rule_golden_nonzero_initial_state():
    state0 = np.eye(2, dtype=np.float32)[None, None]
    q = np.array([[[[1.0, 0.0]]]], dtype=np.float32)
    k = np.array([[[[0.0, 1.0]]]], dtype=np.float32)
    v = np.array([[[[0.0, 0.0]]]], dtype=np.float32)
    beta = np.array([[[1.0]]], dtype=np.float32)
    out, state = kernels.chunked_delta_rule(q, k, v, beta, state=state0)
    np.testing.assert_allclose(_np(out), np.array([[[[1.0, 0.0]]]], dtype=np.float32), **TOL)
    np.testing.assert_allclose(
        _np(state), np.array([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=np.float32), **TOL
    )


def test_chunked_delta_rule_is_causal():
    rng = np.random.default_rng(11)
    q, k, v, beta = _delta_inputs(rng, seq=6)
    out, _ = kernels.chunked_delta_rule(q, k, v, beta)
    t = 2
    q2, k2, v2, beta2 = q.copy(), k.copy(), v.copy(), beta.copy()
    q2[:, t + 1 :] += 3.0
    k2[:, t + 1 :] -= 1.5
    v2[:, t + 1 :] += 4.0
    beta2[:, t + 1 :] = 0.9
    out2, _ = kernels.chunked_delta_rule(q2, k2, v2, beta2)
    np.testing.assert_allclose(_np(out)[:, : t + 1], _np(out2)[:, : t + 1], **TOL)


def test_chunked_delta_rule_state_carry_matches_oneshot():
    rng = np.random.default_rng(12)
    q, k, v, beta = _delta_inputs(rng, seq=7)
    split = 3
    out_a, st_a = kernels.chunked_delta_rule(
        q[:, :split], k[:, :split], v[:, :split], beta[:, :split]
    )
    out_b, st_b = kernels.chunked_delta_rule(
        q[:, split:], k[:, split:], v[:, split:], beta[:, split:], state=st_a
    )
    out_full, st_full = kernels.chunked_delta_rule(q, k, v, beta)
    got = np.concatenate([_np(out_a), _np(out_b)], axis=1)
    np.testing.assert_allclose(got, _np(out_full), **TOL)
    np.testing.assert_allclose(_np(st_b), _np(st_full), **TOL)
    exp_out, exp_st = ref.gated_delta_rule(q, k, v, beta)
    np.testing.assert_allclose(_np(out_full), exp_out, **TOL)
    np.testing.assert_allclose(_np(st_full), exp_st, **TOL)


def test_chunked_delta_rule_two_chunk_sizes_match_reference():
    rng = np.random.default_rng(13)
    q, k, v, beta = _delta_inputs(rng, seq=10, dim=3)
    out_a, st_a = kernels.chunked_delta_rule(
        q, k, v, beta, config=kernels.LinearAttnConfig(chunk=4)
    )
    out_b, st_b = kernels.chunked_delta_rule(
        q, k, v, beta, config=kernels.LinearAttnConfig(chunk=7)
    )
    exp_out, exp_st = ref.gated_delta_rule(q, k, v, beta)
    np.testing.assert_allclose(_np(out_a), exp_out, **TOL)
    np.testing.assert_allclose(_np(out_b), exp_out, **TOL)
    np.testing.assert_allclose(_np(st_a), exp_st, **TOL)
    np.testing.assert_allclose(_np(st_b), exp_st, **TOL)


def test_chunked_delta_rule_seq_not_multiple_of_default_chunk():
    rng = np.random.default_rng(14)
    q, k, v, beta = _delta_inputs(rng, seq=20, heads=1, dim=3)
    assert 20 % kernels.DEFAULT_CHUNK != 0
    out, state = kernels.chunked_delta_rule(q, k, v, beta)
    exp_out, exp_st = ref.gated_delta_rule(q, k, v, beta)
    np.testing.assert_allclose(_np(out), exp_out, **TOL)
    np.testing.assert_allclose(_np(state), exp_st, **TOL)


def test_chunked_delta_rule_shapes_and_dtype():
    rng = np.random.default_rng(15)
    q, k, v, beta = _delta_inputs(rng, batch=3, seq=4, heads=2, dim=5)
    out, state = kernels.chunked_delta_rule(q, k, v, beta)
    assert _np(out).shape == (3, 4, 2, 5)
    assert _np(state).shape == (3, 2, 5, 5)
    assert _np(out).dtype == np.float32
    assert _np(state).dtype == np.float32


def test_chunked_delta_rule_none_state_equals_zeros():
    rng = np.random.default_rng(16)
    q, k, v, beta = _delta_inputs(rng, batch=1, seq=3, heads=1, dim=3)
    zeros = np.zeros((1, 1, 3, 3), dtype=np.float32)
    out_a, st_a = kernels.chunked_delta_rule(q, k, v, beta, state=None)
    out_b, st_b = kernels.chunked_delta_rule(q, k, v, beta, state=zeros)
    np.testing.assert_allclose(_np(out_a), _np(out_b), **TOL)
    np.testing.assert_allclose(_np(st_a), _np(st_b), **TOL)


def test_chunked_delta_rule_output_linear_in_v_and_state_independent_of_q():
    rng = np.random.default_rng(17)
    q, k, v, beta = _delta_inputs(rng, batch=1, seq=4, heads=1, dim=3)
    out1, st1 = kernels.chunked_delta_rule(q, k, v, beta)
    out2, st2 = kernels.chunked_delta_rule(q, k, 2.0 * v, beta)
    np.testing.assert_allclose(_np(out2), 2.0 * _np(out1), **TOL)
    np.testing.assert_allclose(_np(st2), 2.0 * _np(st1), **TOL)
    _, st_q = kernels.chunked_delta_rule(q + 1.25, k, v, beta)
    np.testing.assert_allclose(_np(st_q), _np(st1), **TOL)


def test_chunked_delta_rule_fd_grads_qkv_beta():
    rng = np.random.default_rng(18)
    q, k, v, beta = _delta_inputs(rng, batch=1, seq=3, heads=2, dim=3)
    go = np.ones((1, 3, 2, 3), dtype=np.float32)
    gs = np.zeros((1, 2, 3, 3), dtype=np.float32)
    gq, gk, gv, gbeta, _gst = ref.gated_delta_rule_vjp(q, k, v, beta, go, gs, None)
    eps = 1e-3

    def prod_sum(qq, kk, vv, bb):
        o, _ = kernels.chunked_delta_rule(qq, kk, vv, bb)
        return float(np.sum(_np(o)))

    for name, x, gx in (("q", q, gq), ("k", k, gk), ("v", v, gv), ("beta", beta, gbeta)):
        d = _unit(rng, x.shape)
        if name == "q":
            fd = (prod_sum(x + eps * d, k, v, beta) - prod_sum(x - eps * d, k, v, beta)) / (2 * eps)
        elif name == "k":
            fd = (prod_sum(q, x + eps * d, v, beta) - prod_sum(q, x - eps * d, v, beta)) / (2 * eps)
        elif name == "v":
            fd = (prod_sum(q, k, x + eps * d, beta) - prod_sum(q, k, x - eps * d, beta)) / (2 * eps)
        else:
            fd = (prod_sum(q, k, v, x + eps * d) - prod_sum(q, k, v, x - eps * d)) / (2 * eps)
        analytic = float(np.sum(gx * d))
        np.testing.assert_allclose(fd, analytic, **FD_TOL)


def test_chunked_delta_rule_fwd_matches_reference():
    rng = np.random.default_rng(19)
    q, k, v, beta = _delta_inputs(rng, seq=6)
    cfg = kernels.LinearAttnConfig(chunk=64)
    (out, state), _residual = kernels.chunked_delta_rule_fwd(q, k, v, beta, None, cfg)
    exp_out, exp_st = ref.gated_delta_rule(q, k, v, beta)
    np.testing.assert_allclose(_np(out), exp_out, **TOL)
    np.testing.assert_allclose(_np(state), exp_st, **TOL)


def test_chunked_delta_rule_bwd_matches_reference_vjp():
    rng = np.random.default_rng(20)
    q, k, v, beta = _delta_inputs(rng, batch=1, seq=4, heads=1, dim=4)
    state0 = rng.standard_normal((1, 1, 4, 4)).astype(np.float32) * 0.05
    cfg = kernels.LinearAttnConfig(chunk=2)
    (out, ns), residual = kernels.chunked_delta_rule_fwd(q, k, v, beta, state0, cfg)
    go = rng.standard_normal(_np(out).shape).astype(np.float32)
    gs = rng.standard_normal(_np(ns).shape).astype(np.float32) * 0.05
    grads = kernels.chunked_delta_rule_bwd(residual, (go, gs))
    assert len(grads) == 5
    gq, gk, gv, gbeta, gst = grads
    exp = ref.gated_delta_rule_vjp(q, k, v, beta, go, gs, state0)
    np.testing.assert_allclose(_np(gq), exp[0], **TOL)
    np.testing.assert_allclose(_np(gk), exp[1], **TOL)
    np.testing.assert_allclose(_np(gv), exp[2], **TOL)
    np.testing.assert_allclose(_np(gbeta), exp[3], **TOL)
    np.testing.assert_allclose(_np(gst), exp[4], **TOL)


def test_chunked_delta_rule_bad_shapes_raise_kernel_error():
    rng = np.random.default_rng(21)
    q, k, v, beta = _delta_inputs(rng, batch=1, seq=3, heads=1, dim=4)
    with pytest.raises(kernels.KernelError):
        kernels.chunked_delta_rule(q[:, :, :, :3], k, v, beta)
    with pytest.raises(kernels.KernelError):
        kernels.chunked_delta_rule(q, k, v, beta[:, :2])
    with pytest.raises(kernels.KernelError):
        kernels.chunked_delta_rule(q, k, v, beta, state=np.zeros((1, 1, 3, 3), dtype=np.float32))


def test_chunked_delta_rule_negative_beta_raises_kernel_error():
    rng = np.random.default_rng(22)
    q, k, v, beta = _delta_inputs(rng, batch=1, seq=2, heads=1, dim=2)
    beta = beta.copy()
    beta[0, 0, 0] = -0.1
    with pytest.raises(kernels.KernelError):
        kernels.chunked_delta_rule(q, k, v, beta)


# ---------------------------------------------------------------------------
# FP8
# ---------------------------------------------------------------------------


def test_fp8_quantize_dequantize_roundtrip_error_bounded():
    rng = np.random.default_rng(30)
    x = rng.standard_normal((4, 32)).astype(np.float32)
    block = 8
    meta = kernels.fp8_quantize(x, block=block)
    dq = _np(kernels.fp8_dequantize(meta))
    n_blocks = 32 // block
    err = np.abs(x - dq).reshape(4, n_blocks, block)
    amax = np.max(np.abs(x).reshape(4, n_blocks, block), axis=-1)
    max_err = err.max(axis=-1)
    np.testing.assert_array_less(max_err, FP8_ROUNDTRIP_AMAX_FRAC * np.maximum(amax, 1e-8) + 1e-8)
    exp = ref.fp8_quantize(x, block=block)
    np.testing.assert_allclose(_np(meta.scale), exp.scale, **TOL)
    np.testing.assert_allclose(dq, ref.fp8_dequantize(exp), **TOL)


def test_fp8_quantize_golden_e4m3_bits():
    x = np.array([[0.0, 1.0, 2.0, -2.0]], dtype=np.float32)
    meta = kernels.fp8_quantize(x, block=4)
    # amax=2 → scale=2/448; scaled values 0, 224, 448, -448 are exact E4M3FN.
    np.testing.assert_allclose(_np(meta.scale), np.array([[2.0 / 448.0]], dtype=np.float32), **TOL)
    np.testing.assert_array_equal(
        _np(meta.q).astype(np.int8), np.array([[0, 118, 126, -2]], dtype=np.int8)
    )
    dq = _np(kernels.fp8_dequantize(meta))
    np.testing.assert_allclose(dq, x, **TOL)
    assert int(meta.block) == 4


def test_fp8_quantize_golden_full_scale():
    x = np.array([[448.0, 224.0, 0.0]], dtype=np.float32)
    meta = kernels.fp8_quantize(x, block=3)
    np.testing.assert_allclose(_np(meta.scale), np.array([[1.0]], dtype=np.float32), **TOL)
    np.testing.assert_array_equal(
        _np(meta.q).astype(np.uint8), np.array([[126, 118, 0]], dtype=np.uint8)
    )
    np.testing.assert_allclose(_np(kernels.fp8_dequantize(meta)), x, **TOL)


def test_fp8_dequantize_constructed_zeros():
    meta = kernels.Fp8Meta(
        q=np.zeros((2, 4), dtype=np.int8),
        scale=np.ones((2, 1), dtype=np.float32),
        block=4,
        dtype=kernels.DType.FP8,
    )
    dq = _np(kernels.fp8_dequantize(meta))
    np.testing.assert_allclose(dq, np.zeros((2, 4), dtype=np.float32), **TOL)


def test_fp8_quantize_zeros_scale_is_one():
    x = np.zeros((3, 16), dtype=np.float32)
    meta = kernels.fp8_quantize(x, block=8)
    np.testing.assert_allclose(_np(meta.scale), np.ones((3, 2), dtype=np.float32), **TOL)
    np.testing.assert_allclose(_np(kernels.fp8_dequantize(meta)), x, **TOL)


def test_fp8_quantize_block_remainder():
    rng = np.random.default_rng(31)
    x = rng.standard_normal((2, 10)).astype(np.float32)
    meta = kernels.fp8_quantize(x, block=8)
    assert _np(meta.q).shape == (2, 10)
    assert _np(meta.scale).shape == (2, 2)
    dq = _np(kernels.fp8_dequantize(meta))
    assert dq.shape == (2, 10)
    exp = ref.fp8_dequantize(ref.fp8_quantize(x, block=8))
    np.testing.assert_allclose(dq, exp, **TOL)


def test_fp8_linear_matches_reference():
    rng = np.random.default_rng(32)
    x = rng.standard_normal((6, 16)).astype(np.float32)
    w = rng.standard_normal((8, 16)).astype(np.float32)
    got = kernels.fp8_linear(x, w, block=8)
    exp = ref.fp8_linear(x, w, block=8)
    np.testing.assert_allclose(_np(got), exp, **TOL)
    assert _np(got).shape == (6, 8)
    assert _np(got).dtype == np.float32


def test_fp8_linear_vs_fp32_matmul_documented_tolerance():
    """y_fp8 ≈ x @ w^T. Tolerance: rtol=0.05, atol=0.2 on N(0, 0.5) in=32 block=16.

    Cosine similarity of the two matmuls stays above 0.99 for these shapes;
    abs floor covers near-zero outputs where relative error is meaningless.
    """
    rng = np.random.default_rng(33)
    x = (0.5 * rng.standard_normal((8, 32))).astype(np.float32)
    w = (0.5 * rng.standard_normal((16, 32))).astype(np.float32)
    y8 = _np(kernels.fp8_linear(x, w, block=16))
    y32 = x @ w.T
    np.testing.assert_allclose(y8, y32, **FP8_VS_FP32)
    denom = np.linalg.norm(y8) * np.linalg.norm(y32) + 1e-12
    cos = float(np.vdot(y8.ravel(), y32.ravel()) / denom)
    assert cos > 0.99


def test_fp8_linear_golden():
    x = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    w = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    y = _np(kernels.fp8_linear(x, w, block=4))
    # 3.0 quantises to 20/7 under scale 4/448; 1+2+20/7+4 = 9.857142857...
    np.testing.assert_allclose(y, np.array([[7.0 + 20.0 / 7.0]], dtype=np.float32), **TOL)


def test_fp8_linear_batched_leading_dims():
    rng = np.random.default_rng(34)
    x = rng.standard_normal((2, 3, 8)).astype(np.float32)
    w = rng.standard_normal((5, 8)).astype(np.float32)
    y = kernels.fp8_linear(x, w, block=8)
    assert _np(y).shape == (2, 3, 5)
    np.testing.assert_allclose(_np(y), ref.fp8_linear(x, w, block=8), **TOL)


def test_fp8_linear_fwd_bwd_uses_forward_scales():
    """Backward is STE through the forward dequantised tensors; scales are frozen.

    fp8_linear_bwd(residual, g) does not re-quantize. dL/dx = g @ w_hat,
    dL/dw = g^T @ x_hat with x_hat, w_hat from the forward residual.
    """
    rng = np.random.default_rng(35)
    x = rng.standard_normal((4, 12)).astype(np.float32)
    w = rng.standard_normal((6, 12)).astype(np.float32)
    y, residual = kernels.fp8_linear_fwd(x, w, 6)
    np.testing.assert_allclose(_np(y), ref.fp8_linear(x, w, block=6), **TOL)
    g = rng.standard_normal(_np(y).shape).astype(np.float32)
    gx, gw = kernels.fp8_linear_bwd(residual, g)
    exp_y, exp_resid = ref.fp8_linear_fwd(x, w, 6)
    exp_gx, exp_gw = ref.fp8_linear_bwd(exp_resid, g)
    del exp_y
    np.testing.assert_allclose(_np(gx), exp_gx, **TOL)
    np.testing.assert_allclose(_np(gw), exp_gw, **TOL)
    assert _np(gx).shape == x.shape
    assert _np(gw).shape == w.shape


def test_fp8_linear_bad_shapes_raise_kernel_error():
    x = np.ones((4, 8), dtype=np.float32)
    w = np.ones((3, 7), dtype=np.float32)
    with pytest.raises(kernels.KernelError):
        kernels.fp8_linear(x, w, block=8)
    with pytest.raises(kernels.KernelError):
        kernels.fp8_quantize(x, block=0)


# ---------------------------------------------------------------------------
# FP8 jax.custom_vjp / STE through jax.grad(fp8_linear)
# ---------------------------------------------------------------------------


def test_fp8_linear_is_registered_jax_custom_vjp():
    """fp8_linear must be jax.custom_vjp so jax.grad is STE, not discrete quantize."""
    jax, jnp = _jax()

    @jax.custom_vjp
    def _probe(z):
        return z

    def _probe_fwd(z):
        return z, None

    def _probe_bwd(_res, g):
        return (g,)

    _probe.defvjp(_probe_fwd, _probe_bwd)

    assert type(kernels.fp8_linear) is type(_probe), (
        "fp8_linear must be decorated with jax.custom_vjp so backward reuses "
        "forward FP8 scales (STE); it must not be a plain function"
    )
    assert hasattr(kernels.fp8_linear, "defvjp")

    x = jnp.asarray(np.ones((2, 4), dtype=np.float32))
    w = jnp.asarray(np.ones((3, 4), dtype=np.float32))
    gx = jax.grad(lambda xx: jnp.sum(kernels.fp8_linear(xx, w, block=4)))(x)
    assert gx.shape == x.shape
    assert gx.dtype == jnp.float32
    assert np.isfinite(_np(gx)).all()


def test_fp8_linear_jax_grad_matches_fp8_linear_bwd_ste():
    """jax.grad(fp8_linear) equals fp8_linear_bwd(residual, g) from fp8_linear_fwd."""
    jax, jnp = _jax()
    rng = np.random.default_rng(21)
    block = 8
    x = rng.standard_normal((2, 3, 8)).astype(np.float32)
    w = rng.standard_normal((5, 8)).astype(np.float32)
    x_j = jnp.asarray(x)
    w_j = jnp.asarray(w)

    y_fwd, residual = kernels.fp8_linear_fwd(x, w, block)
    g = rng.standard_normal(np.shape(y_fwd)).astype(np.float32)
    gx_bwd, gw_bwd = kernels.fp8_linear_bwd(residual, g)

    def _apply(xx, ww):
        return kernels.fp8_linear(xx, ww, block=block)

    y_vjp, vjp_fn = jax.vjp(_apply, x_j, w_j)
    gx_vjp, gw_vjp = vjp_fn(jnp.asarray(g))
    np.testing.assert_allclose(_np(y_vjp), _np(y_fwd), **TOL)
    np.testing.assert_allclose(_np(gx_vjp), _np(gx_bwd), **TOL)
    np.testing.assert_allclose(_np(gw_vjp), _np(gw_bwd), **TOL)
    assert gx_vjp.shape == x.shape
    assert gw_vjp.shape == w.shape
    assert gx_vjp.dtype == jnp.float32
    assert gw_vjp.dtype == jnp.float32

    def _sum_linear(xx, ww):
        return jnp.sum(kernels.fp8_linear(xx, ww, block=block))

    gx_grad, gw_grad = jax.grad(_sum_linear, argnums=(0, 1))(x_j, w_j)
    g_ones = np.ones_like(_np(y_fwd), dtype=np.float32)
    gx_ones, gw_ones = kernels.fp8_linear_bwd(residual, g_ones)
    np.testing.assert_allclose(_np(gx_grad), _np(gx_ones), **TOL)
    np.testing.assert_allclose(_np(gw_grad), _np(gw_ones), **TOL)

    gx_first = jax.grad(
        lambda xx: jnp.sum(kernels.fp8_linear(xx, w_j, block=block))
    )(x_j)
    np.testing.assert_allclose(_np(gx_first), _np(gx_ones), **TOL)


def test_fp8_linear_jax_grad_ste_matches_dequantized_linear():
    """STE through jax.grad(fp8_linear) equals FP32 linear VJP on frozen fake-quant."""
    jax, jnp = _jax()
    rng = np.random.default_rng(22)
    block = 8
    x = rng.standard_normal((4, 8)).astype(np.float32)
    w = rng.standard_normal((6, 8)).astype(np.float32)
    y = ref.fp8_linear(x, w, block=block)
    g = rng.standard_normal(y.shape).astype(np.float32)

    gx_ref, gw_ref = ref.fp8_linear_vjp(x, w, g, block=block)
    x_hat = ref.fp8_dequantize(ref.fp8_quantize(x, block=block))
    w_hat = ref.fp8_dequantize(ref.fp8_quantize(w, block=block))
    gx_fp32 = np.matmul(g, w_hat)
    gw_fp32 = np.matmul(g.T, x_hat)
    np.testing.assert_allclose(gx_ref, gx_fp32, **TOL)
    np.testing.assert_allclose(gw_ref, gw_fp32, **TOL)

    def _apply(xx, ww):
        return kernels.fp8_linear(xx, ww, block=block)

    _, vjp_fn = jax.vjp(_apply, jnp.asarray(x), jnp.asarray(w))
    gx_j, gw_j = vjp_fn(jnp.asarray(g))
    np.testing.assert_allclose(_np(gx_j), gx_fp32, **TOL)
    np.testing.assert_allclose(_np(gw_j), gw_fp32, **TOL)

    d = rng.standard_normal(x_hat.shape).astype(np.float32)
    d /= np.linalg.norm(d) + np.float32(1e-12)
    eps = 1e-3

    def _frozen_sum(xh):
        return float(np.sum(np.matmul(xh, w_hat.T)))

    fd = (_frozen_sum(x_hat + eps * d) - _frozen_sum(x_hat - eps * d)) / (2 * eps)
    g_ones = np.ones_like(y, dtype=np.float32)
    gx_ones, _ = ref.fp8_linear_vjp(x, w, g_ones, block=block)
    analytic = float(np.sum(gx_ones * d))
    np.testing.assert_allclose(fd, analytic, **FD_TOL)


def test_fp8_linear_jax_array_bad_shapes_raise_kernel_error():
    """KernelError still fires on the public function with jax.Array inputs."""
    _, jnp = _jax()
    x = jnp.asarray(np.ones((4, 8), dtype=np.float32))
    w_bad = jnp.asarray(np.ones((3, 7), dtype=np.float32))
    with pytest.raises(kernels.KernelError):
        kernels.fp8_linear(x, w_bad, block=8)
    w_ok = jnp.asarray(np.ones((3, 8), dtype=np.float32))
    with pytest.raises(kernels.KernelError):
        kernels.fp8_linear(x, w_ok, block=0)
    w_3d = jnp.asarray(np.ones((3, 5, 8), dtype=np.float32))
    with pytest.raises(kernels.KernelError):
        kernels.fp8_linear(x, w_3d, block=8)


def test_fp8_linear_jax_grad_is_ste_not_quantized_scale_path():
    """jax.grad must be STE, not autodiff through per-block scale / quantize.

    Golden: x = w = [1, 2, 3, 4], block=4. 3.0 quantises to 20/7, so STE
    d(sum y)/dx = x_hat = [1, 2, 20/7, 4]. Differentiating live scales
    concentrates gradient on the amax element; true FP32 would keep 3.0.
    Either bug fails this check.
    """
    jax, jnp = _jax()
    block = 4
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    w = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    y = ref.fp8_linear(x, w, block=block)
    g = np.ones_like(y, dtype=np.float32)

    gx_ste, gw_ste = ref.fp8_linear_vjp(x, w, g, block=block)
    gx_scale, gw_scale = ref.fp8_linear_scale_path_grad(x, w, g, block=block)
    np.testing.assert_allclose(gx_ste[0, 2], np.float32(20.0 / 7.0), **TOL)
    np.testing.assert_allclose(gw_ste[0, 2], np.float32(20.0 / 7.0), **TOL)
    assert np.max(np.abs(gx_ste - gx_scale)) > 1e-3
    assert np.max(np.abs(gw_ste - gw_scale)) > 1e-3
    assert not np.allclose(gx_ste[0, 2], np.float32(3.0), atol=1e-5)

    def _sum_linear(xx, ww):
        return jnp.sum(kernels.fp8_linear(xx, ww, block=block))

    gx_j, gw_j = jax.grad(_sum_linear, argnums=(0, 1))(jnp.asarray(x), jnp.asarray(w))
    np.testing.assert_allclose(_np(gx_j), gx_ste, **TOL)
    np.testing.assert_allclose(_np(gw_j), gw_ste, **TOL)
    assert not np.allclose(_np(gx_j), gx_scale, **TOL)
    assert not np.allclose(_np(gw_j), gw_scale, **TOL)


def test_fp8_linear_jax_grad_not_finite_diff_of_discrete_quant():
    """A locally constant quantized forward has FD ~ 0; STE / jax.grad must not.

    Perturbing a non-amax element inside an E4M3 bin leaves y unchanged, so
    the true derivative of discrete quantize is 0 there. Differentiating the
    quantize/scale path would miss the STE cotangent g @ w_hat.
    """
    jax, jnp = _jax()
    block = 4
    x = np.array([[8.0, 1.0, 0.5, 0.25]], dtype=np.float32)
    w = np.array([[1.3, 1.7, 2.1, 0.9]], dtype=np.float32)
    y0 = ref.fp8_linear(x, w, block=block)
    x_eps = x.copy()
    x_eps[0, 3] += np.float32(1e-3)
    y1 = ref.fp8_linear(x_eps, w, block=block)
    np.testing.assert_array_equal(y0, y1)

    g = np.ones_like(y0, dtype=np.float32)
    gx_ste, _ = ref.fp8_linear_vjp(x, w, g, block=block)
    assert abs(float(gx_ste[0, 3])) > 0.1

    gx_j = jax.grad(
        lambda xx: jnp.sum(kernels.fp8_linear(xx, jnp.asarray(w), block=block))
    )(jnp.asarray(x))
    np.testing.assert_allclose(_np(gx_j), gx_ste, **TOL)
    assert abs(float(_np(gx_j)[0, 3])) > 0.1


# ---------------------------------------------------------------------------
# Expert-parallel dispatch / combine
# ---------------------------------------------------------------------------


def test_ep_dispatch_combine_identity_roundtrip():
    tokens = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    meta = _dispatch_meta(
        expert_ids=[[0], [1], [0]],
        probs=[[1.0], [1.0], [1.0]],
        racks=[0, 1],
        n_experts=2,
    )
    dispatched, residual = kernels.ep_dispatch(tokens, meta)
    combined = kernels.ep_combine(dispatched, meta, residual)
    np.testing.assert_allclose(_np(combined), tokens, **TOL)


def test_ep_combine_inverts_dispatch_weighted():
    tokens = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    meta = _dispatch_meta(
        expert_ids=[[0, 1], [1, 0]],
        probs=[[0.25, 0.75], [0.5, 0.5]],
        racks=[0, 0],
        n_experts=2,
    )
    dispatched, residual = kernels.ep_dispatch(tokens, meta)
    combined = kernels.ep_combine(dispatched, meta, residual)
    np.testing.assert_allclose(_np(combined), tokens, **TOL)


def test_ep_combine_scales_by_prob_when_not_one():
    tokens = np.array([[2.0, 4.0]], dtype=np.float32)
    meta = _dispatch_meta(
        expert_ids=[[0]],
        probs=[[0.5]],
        racks=[0],
        n_experts=1,
    )
    dispatched, residual = kernels.ep_dispatch(tokens, meta)
    combined = kernels.ep_combine(dispatched, meta, residual)
    np.testing.assert_allclose(_np(combined), 0.5 * tokens, **TOL)


def test_ep_dispatch_golden_layout_and_padding():
    tokens = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    meta = _dispatch_meta(
        expert_ids=[[0], [1], [0]],
        probs=[[1.0], [1.0], [1.0]],
        racks=[0, 1],
        n_experts=2,
    )
    dispatched, residual = kernels.ep_dispatch(tokens, meta)
    # Token order within an expert is increasing token index. max_per_expert=2.
    exp = np.array(
        [[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [0.0, 0.0]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(_np(dispatched), exp, **TOL)
    assert _np(dispatched).shape == (2, 2, 2)
    combined = kernels.ep_combine(dispatched, meta, residual)
    np.testing.assert_allclose(_np(combined), tokens, **TOL)


def test_ep_dispatch_shapes_pad_to_max_per_expert():
    rng = np.random.default_rng(40)
    n_tokens, d_model, n_experts, top_k = 7, 5, 4, 2
    tokens = rng.standard_normal((n_tokens, d_model)).astype(np.float32)
    expert_ids = rng.integers(0, n_experts, size=(n_tokens, top_k))
    probs = rng.random((n_tokens, top_k)).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    racks = np.arange(n_experts) % 3
    meta = _dispatch_meta(expert_ids, probs, racks, n_experts)
    dispatched, residual = kernels.ep_dispatch(tokens, meta)
    counts = np.bincount(np.asarray(expert_ids).ravel(), minlength=n_experts)
    max_per = int(counts.max())
    assert _np(dispatched).shape == (n_experts, max_per, d_model)
    combined = kernels.ep_combine(dispatched, meta, residual)
    np.testing.assert_allclose(_np(combined), tokens, **TOL)


def test_ep_dispatch_raises_when_racks_span_exceeds_max_racks():
    tokens = np.ones((1, 3), dtype=np.float32)
    meta = _dispatch_meta(
        expert_ids=[[0, 1, 2, 3, 4]],
        probs=[[0.2, 0.2, 0.2, 0.2, 0.2]],
        racks=[0, 1, 2, 3, 4],
        n_experts=5,
        max_racks=kernels.MAX_RACKS,
    )
    with pytest.raises(kernels.KernelError):
        kernels.ep_dispatch(tokens, meta)


def test_ep_dispatch_exactly_max_racks_ok():
    tokens = np.array([[7.0, 8.0, 9.0]], dtype=np.float32)
    meta = _dispatch_meta(
        expert_ids=[[0, 1, 2, 3]],
        probs=[[0.25, 0.25, 0.25, 0.25]],
        racks=[0, 1, 2, 3],
        n_experts=4,
        max_racks=4,
    )
    dispatched, residual = kernels.ep_dispatch(tokens, meta)
    combined = kernels.ep_combine(dispatched, meta, residual)
    np.testing.assert_allclose(_np(combined), tokens, **TOL)


def test_ep_dispatch_duplicate_racks_count_once():
    tokens = np.array([[1.0, 1.0]], dtype=np.float32)
    meta = _dispatch_meta(
        expert_ids=[[0, 1, 2, 3, 4]],
        probs=[[0.2] * 5],
        racks=[0, 0, 1, 2, 3],
        n_experts=5,
        max_racks=4,
    )
    dispatched, residual = kernels.ep_dispatch(tokens, meta)
    combined = kernels.ep_combine(dispatched, meta, residual)
    np.testing.assert_allclose(_np(combined), tokens, **TOL)


def test_ep_dispatch_bad_token_meta_shapes_raise():
    tokens = np.ones((3, 4), dtype=np.float32)
    meta = _dispatch_meta(
        expert_ids=[[0], [1]],
        probs=[[1.0], [1.0]],
        racks=[0, 1],
        n_experts=2,
    )
    with pytest.raises(kernels.KernelError):
        kernels.ep_dispatch(tokens, meta)


def test_ep_dispatch_out_of_range_expert_raises():
    tokens = np.ones((1, 2), dtype=np.float32)
    meta = _dispatch_meta(
        expert_ids=[[9]],
        probs=[[1.0]],
        racks=[0, 1],
        n_experts=2,
    )
    with pytest.raises(kernels.KernelError):
        kernels.ep_dispatch(tokens, meta)


# ---------------------------------------------------------------------------
# V1 / V3 GPU (skipped on CPU / without a GPU)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_v1_gpu_chunked_delta_rule_matches_reference():
    """V1: GPU linear-attention kernel vs numpy per-timestep recurrence, 1e-5."""
    _require_gpu()
    rng = np.random.default_rng(50)
    q, k, v, beta = _delta_inputs(rng, batch=2, seq=16, heads=2, dim=8)
    out, state = kernels.chunked_delta_rule(
        q, k, v, beta, config=kernels.LinearAttnConfig(chunk=64)
    )
    exp_out, exp_st = ref.gated_delta_rule(q, k, v, beta)
    np.testing.assert_allclose(_np(out), exp_out, **TOL)
    np.testing.assert_allclose(_np(state), exp_st, **TOL)


@pytest.mark.gpu
def test_v3_gpu_fp8_linear_matches_reference():
    """V3: GPU FP8 linear vs numpy E4M3FN fake-quant reference, 1e-5."""
    _require_gpu()
    rng = np.random.default_rng(51)
    x = rng.standard_normal((16, 128)).astype(np.float32)
    w = rng.standard_normal((32, 128)).astype(np.float32)
    got = kernels.fp8_linear(x, w, block=kernels.DEFAULT_FP8_BLOCK)
    exp = ref.fp8_linear(x, w, block=kernels.DEFAULT_FP8_BLOCK)
    np.testing.assert_allclose(_np(got), exp, **TOL)


@pytest.mark.gpu
def test_v3_gpu_fp8_linear_jax_grad_matches_ste():
    """V3: GPU jax.grad(fp8_linear) matches STE VJP (forward scales, not differentiated)."""
    jax = _require_gpu()
    jnp = jax.numpy
    rng = np.random.default_rng(52)
    block = kernels.DEFAULT_FP8_BLOCK
    x = rng.standard_normal((8, 128)).astype(np.float32)
    w = rng.standard_normal((16, 128)).astype(np.float32)
    g = rng.standard_normal((8, 16)).astype(np.float32)
    gx_ref, gw_ref = ref.fp8_linear_vjp(x, w, g, block=block)

    def _apply(xx, ww):
        return kernels.fp8_linear(xx, ww, block=block)

    _, vjp_fn = jax.vjp(_apply, jnp.asarray(x), jnp.asarray(w))
    gx_j, gw_j = vjp_fn(jnp.asarray(g))
    np.testing.assert_allclose(_np(gx_j), gx_ref, **TOL)
    np.testing.assert_allclose(_np(gw_j), gw_ref, **TOL)
