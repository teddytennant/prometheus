"""Implementer-facing tests for the ``kernels`` public API (spec 5.1, A3).

These import production ``kernels``. Constants, enums, and dataclasses are real
and membership tests may pass against the stub. Every test that *calls*
``chunked_delta_rule``, ``chunked_delta_rule_fwd``, ``chunked_delta_rule_bwd``,
``fp8_quantize``, ``fp8_dequantize``, ``fp8_linear``, ``fp8_linear_fwd``,
or ``fp8_linear_bwd`` must FAIL on the stub. Forward ``ep_dispatch`` /
``ep_combine`` are the fused-permutation ``jax.custom_vjp`` objects (eager
``jax.grad`` matches the scatter VJPs). ``jax.jit(ep_dispatch)`` /
``jax.jit(ep_combine)`` and composed jit must FAIL until ``DispatchMeta``
is a ``jax.tree_util`` registered dataclass (array fields as data, ints as
meta) and the traced path does not treat ``DispatchMeta`` as an abstract
array.

The numpy reference (``tests.reference.kernels``) is the gated delta-rule
recurrence with no chunking, E4M3FN per-block abs-max fake-quant, and
node-limited EP dispatch. Production must match it.

Groups:
- Constants: DEFAULT_CHUNK 64, DEFAULT_FP8_BLOCK 128, MAX_RACKS 4, DType,
  LinearAttnConfig / Fp8Meta / DispatchMeta / KernelError.
- Linear attention: vs-reference, causality, state carry, chunk-size
  independence, shapes/dtypes, finite-difference grads, goldens,
  KernelError on bad shapes, jax.custom_vjp / jax.grad equals
  chunked_delta_rule_bwd and the fused reverse-state VJP (not a wrapper).
- FP8: roundtrip bound, vs-reference, vs FP32 matmul (documented tolerance),
  backward uses forward scales (STE), jax.custom_vjp / jax.grad STE
  (matches fp8_linear_bwd residual, not scale-path autodiff), goldens,
  KernelError on bad shapes.
- EP: combine(dispatch(x)) weighted identity, rack span > MAX_RACKS,
  padding shape, goldens, residual is opaque. Public ``ep_dispatch`` /
  ``ep_combine`` must be the ``jax.custom_vjp`` object itself (not a
  wrapper); ``jax.grad`` / ``jax.vjp`` through dispatched matches the
  residual scatter VJP, through combined the weighted-scatter VJP, 1e-5.
  ``DispatchMeta`` is a registered pytree (array fields are leaves);
  ``jax.jit(ep_dispatch)`` / ``jax.jit(ep_combine)`` match eager at 1e-5;
  composed jit(dispatch-then-combine) identity; ``jax.grad`` through
  those jitted primitives matches the scatter VJPs.
- GPU: V1 linear-attn parity and jax.grad reverse-state VJP, V3 FP8 parity
  and jax.grad STE (pytest marker ``gpu``). CPU tests cover jax.grad.
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


def _naive_gated_delta_scan(q, k, v, beta, state):
    """Default-loop ``jax.lax.scan`` of the gated delta-rule (not the fused reverse kernel)."""
    jax, jnp = _jax()

    def step(s, ins):
        qt, kt, vt, bt = ins
        btexp = bt[..., None, None]
        k_s = jnp.einsum("bhd,bhde->bhe", kt, s)
        s_new = (
            s
            - btexp * jnp.einsum("bhd,bhe->bhde", kt, k_s)
            + btexp * jnp.einsum("bhd,bhe->bhde", kt, vt)
        )
        o = jnp.einsum("bhd,bhde->bhe", qt, s_new)
        return s_new, o

    ins = (
        jnp.swapaxes(q, 0, 1),
        jnp.swapaxes(k, 0, 1),
        jnp.swapaxes(v, 0, 1),
        jnp.swapaxes(beta, 0, 1),
    )
    ns, out = jax.lax.scan(step, state, ins)
    return jnp.swapaxes(out, 0, 1), ns


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


def _ep_vjp_layouts(rng):
    """Layouts covering top_k>1, padded experts, unused slots, duplicate ids."""
    cases = []
    tokens = rng.standard_normal((3, 4)).astype(np.float32)
    cases.append(
        (
            "pad_topk1",
            tokens,
            _dispatch_meta(
                expert_ids=[[0], [1], [0]],
                probs=[[1.0], [1.0], [1.0]],
                racks=[0, 1],
                n_experts=2,
            ),
        )
    )
    tokens = rng.standard_normal((4, 3)).astype(np.float32)
    probs = rng.random((4, 2)).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    cases.append(
        (
            "topk2_pad",
            tokens,
            _dispatch_meta(
                expert_ids=[[0, 1], [0, 2], [1, 2], [0, 1]],
                probs=probs,
                racks=[0, 0, 1],
                n_experts=3,
            ),
        )
    )
    tokens = rng.standard_normal((3, 2)).astype(np.float32)
    cases.append(
        (
            "unused_expert",
            tokens,
            _dispatch_meta(
                expert_ids=[[0], [0], [1]],
                probs=[[1.0], [1.0], [1.0]],
                racks=[0, 1, 2],
                n_experts=3,
            ),
        )
    )
    tokens = rng.standard_normal((2, 5)).astype(np.float32)
    cases.append(
        (
            "dup_expert_topk2",
            tokens,
            _dispatch_meta(
                expert_ids=[[0, 0], [1, 0]],
                probs=[[0.4, 0.6], [0.3, 0.7]],
                racks=[0, 1],
                n_experts=2,
            ),
        )
    )
    return cases


def _pytree_leaves_contain_array(leaves, arr) -> bool:
    """True if ``arr`` appears as a pytree leaf (value + shape), not buried in aux."""
    target = np.asarray(arr)
    for leaf in leaves:
        if type(leaf) is kernels.DispatchMeta or not hasattr(leaf, "shape"):
            continue
        got = _np(leaf)
        if got.shape == target.shape and np.array_equal(got, target):
            return True
    return False


def _assert_residual_indices_equal(got, exp, err_msg=""):
    """token_index / k_index match eager; pad slots stay -1."""
    got_t = _np(got.token_index)
    got_k = _np(got.k_index)
    exp_t = _np(exp.token_index)
    exp_k = _np(exp.k_index)
    np.testing.assert_array_equal(got_t, exp_t, err_msg=err_msg)
    np.testing.assert_array_equal(got_k, exp_k, err_msg=err_msg)
    assert np.all(got_t[exp_t < 0] == -1), err_msg
    assert np.all(got_k[exp_k < 0] == -1), err_msg


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
# chunked_delta_rule jax.custom_vjp / fused reverse-state through jax.grad
# ---------------------------------------------------------------------------


def test_chunked_delta_rule_is_registered_jax_custom_vjp():
    """chunked_delta_rule must be jax.custom_vjp, not a wrapper around one."""
    jax, jnp = _jax()

    @jax.custom_vjp
    def _probe(z):
        return z

    def _probe_fwd(z):
        return z, None

    def _probe_bwd(_res, g):
        return (g,)

    _probe.defvjp(_probe_fwd, _probe_bwd)

    assert type(kernels.chunked_delta_rule) is type(_probe), (
        "chunked_delta_rule must be decorated with jax.custom_vjp so jax.grad "
        "is the fused reverse-state kernel; a plain function wrapping an inner "
        "custom_vjp is not enough"
    )
    assert hasattr(kernels.chunked_delta_rule, "defvjp")

    rng = np.random.default_rng(60)
    q, k, v, beta = _delta_inputs(rng, batch=1, seq=3, heads=1, dim=2)
    state = rng.standard_normal((1, 1, 2, 2)).astype(np.float32) * 0.05
    qj, kj, vj, bj, sj = (jnp.asarray(x) for x in (q, k, v, beta, state))
    gq = jax.grad(lambda qq: jnp.sum(kernels.chunked_delta_rule(qq, kj, vj, bj, sj)[0]))(qj)
    assert gq.shape == q.shape
    assert gq.dtype == jnp.float32
    assert np.isfinite(_np(gq)).all()


def test_chunked_delta_rule_jax_grad_matches_chunked_delta_rule_bwd():
    """jax.grad / jax.vjp through the outputs equals chunked_delta_rule_bwd.

    Covers q, k, v, beta, and state. config is not differentiated.
    """
    jax, jnp = _jax()
    rng = np.random.default_rng(61)
    q, k, v, beta = _delta_inputs(rng, batch=2, seq=5, heads=2, dim=3)
    state = rng.standard_normal((2, 2, 3, 3)).astype(np.float32) * 0.05
    cfg = kernels.LinearAttnConfig(chunk=2)
    (y_fwd, ns_fwd), residual = kernels.chunked_delta_rule_fwd(q, k, v, beta, state, cfg)
    go = rng.standard_normal(_np(y_fwd).shape).astype(np.float32)
    gs = rng.standard_normal(_np(ns_fwd).shape).astype(np.float32) * 0.05
    bwd = kernels.chunked_delta_rule_bwd(residual, (go, gs))
    gq_bwd, gk_bwd, gv_bwd, gbeta_bwd, gst_bwd = bwd

    qj, kj, vj, bj, sj = (jnp.asarray(x) for x in (q, k, v, beta, state))
    goj, gsj = jnp.asarray(go), jnp.asarray(gs)

    def _apply(qq, kk, vv, bb, st):
        return kernels.chunked_delta_rule(qq, kk, vv, bb, st, config=cfg)

    (y_vjp, ns_vjp), vjp_fn = jax.vjp(_apply, qj, kj, vj, bj, sj)
    gq_vjp, gk_vjp, gv_vjp, gbeta_vjp, gst_vjp = vjp_fn((goj, gsj))
    np.testing.assert_allclose(_np(y_vjp), _np(y_fwd), **TOL)
    np.testing.assert_allclose(_np(ns_vjp), _np(ns_fwd), **TOL)
    np.testing.assert_allclose(_np(gq_vjp), _np(gq_bwd), **TOL)
    np.testing.assert_allclose(_np(gk_vjp), _np(gk_bwd), **TOL)
    np.testing.assert_allclose(_np(gv_vjp), _np(gv_bwd), **TOL)
    np.testing.assert_allclose(_np(gbeta_vjp), _np(gbeta_bwd), **TOL)
    np.testing.assert_allclose(_np(gst_vjp), _np(gst_bwd), **TOL)
    assert gq_vjp.shape == q.shape and gk_vjp.shape == k.shape
    assert gv_vjp.shape == v.shape and gbeta_vjp.shape == beta.shape
    assert gst_vjp.shape == state.shape
    assert gq_vjp.dtype == jnp.float32
    assert gk_vjp.dtype == jnp.float32
    assert gv_vjp.dtype == jnp.float32
    assert gbeta_vjp.dtype == jnp.float32
    assert gst_vjp.dtype == jnp.float32

    def _sum_both(qq, kk, vv, bb, st):
        out, ns = kernels.chunked_delta_rule(qq, kk, vv, bb, st, config=cfg)
        return jnp.sum(out) + jnp.sum(ns)

    ones_o = np.ones_like(_np(y_fwd), dtype=np.float32)
    ones_s = np.ones_like(_np(ns_fwd), dtype=np.float32)
    ones_bwd = kernels.chunked_delta_rule_bwd(residual, (ones_o, ones_s))
    grads = jax.grad(_sum_both, argnums=(0, 1, 2, 3, 4))(qj, kj, vj, bj, sj)
    for got, exp in zip(grads, ones_bwd, strict=True):
        np.testing.assert_allclose(_np(got), _np(exp), **TOL)

    for i, expected in enumerate(ones_bwd):
        gi = jax.grad(_sum_both, argnums=i)(qj, kj, vj, bj, sj)
        np.testing.assert_allclose(_np(gi), _np(expected), **TOL)

    def _apply_none_state(qq, kk, vv, bb):
        return kernels.chunked_delta_rule(qq, kk, vv, bb, state=None, config=cfg)

    (y_ns, ns_ns), residual_ns = kernels.chunked_delta_rule_fwd(q, k, v, beta, None, cfg)
    bwd_ns = kernels.chunked_delta_rule_bwd(residual_ns, (go, gs))
    (y_none, ns_none), vjp_none = jax.vjp(_apply_none_state, qj, kj, vj, bj)
    gq_n, gk_n, gv_n, gbeta_n = vjp_none((goj, gsj))
    np.testing.assert_allclose(_np(y_none), _np(y_ns), **TOL)
    np.testing.assert_allclose(_np(ns_none), _np(ns_ns), **TOL)
    np.testing.assert_allclose(_np(gq_n), _np(bwd_ns[0]), **TOL)
    np.testing.assert_allclose(_np(gk_n), _np(bwd_ns[1]), **TOL)
    np.testing.assert_allclose(_np(gv_n), _np(bwd_ns[2]), **TOL)
    np.testing.assert_allclose(_np(gbeta_n), _np(bwd_ns[3]), **TOL)


def test_chunked_delta_rule_jax_grad_matches_fused_reverse_state_vjp():
    """Backward is the fused reverse-state kernel, not default loop autodiff.

    jax.grad(chunked_delta_rule) must match ``gated_delta_rule_vjp`` and
    ``chunked_delta_rule_bwd``. A naive ``jax.lax.scan`` autodiff of the same
    recurrence is compared; if it disagrees with the reverse-state kernel,
    production must follow the reverse-state kernel.
    """
    jax, jnp = _jax()
    rng = np.random.default_rng(62)
    q, k, v, beta = _delta_inputs(rng, batch=2, seq=4, heads=2, dim=3)
    state = rng.standard_normal((2, 2, 3, 3)).astype(np.float32) * 0.05
    cfg = kernels.LinearAttnConfig(chunk=3)
    (y_fwd, ns_fwd), residual = kernels.chunked_delta_rule_fwd(q, k, v, beta, state, cfg)
    go = rng.standard_normal(_np(y_fwd).shape).astype(np.float32)
    gs = rng.standard_normal(_np(ns_fwd).shape).astype(np.float32) * 0.05
    bwd = kernels.chunked_delta_rule_bwd(residual, (go, gs))
    ref_grads = ref.gated_delta_rule_vjp(q, k, v, beta, go, gs, state)

    qj, kj, vj, bj, sj = (jnp.asarray(x) for x in (q, k, v, beta, state))
    goj, gsj = jnp.asarray(go), jnp.asarray(gs)

    def _apply(qq, kk, vv, bb, st):
        return kernels.chunked_delta_rule(qq, kk, vv, bb, st, config=cfg)

    _, vjp_fn = jax.vjp(_apply, qj, kj, vj, bj, sj)
    prod_grads = vjp_fn((goj, gsj))
    for got, exp_bwd, exp_ref in zip(prod_grads, bwd, ref_grads, strict=True):
        np.testing.assert_allclose(_np(got), _np(exp_bwd), **TOL)
        np.testing.assert_allclose(_np(got), exp_ref, **TOL)

    _, naive_vjp = jax.vjp(_naive_gated_delta_scan, qj, kj, vj, bj, sj)
    naive_grads = naive_vjp((goj, gsj))
    naive_agrees = all(
        np.allclose(_np(n), r, **TOL) for n, r in zip(naive_grads, ref_grads, strict=True)
    )
    if not naive_agrees:
        disagreed = [
            not np.allclose(_np(p), _np(n), **TOL)
            for p, n in zip(prod_grads, naive_grads, strict=True)
        ]
        assert any(disagreed), (
            "naive jax.lax.scan autodiff disagrees with the fused reverse-state "
            "VJP; jax.grad(chunked_delta_rule) must follow the fused kernel"
        )


def test_chunked_delta_rule_jax_array_bad_shapes_raise_kernel_error():
    """KernelError on bad shapes still holds when calling the public name with jax arrays."""
    _, jnp = _jax()
    rng = np.random.default_rng(63)
    q, k, v, beta = _delta_inputs(rng, batch=1, seq=3, heads=1, dim=4)
    qj, kj, vj, bj = (jnp.asarray(x) for x in (q, k, v, beta))
    with pytest.raises(kernels.KernelError):
        kernels.chunked_delta_rule(qj[:, :, :, :3], kj, vj, bj)
    with pytest.raises(kernels.KernelError):
        kernels.chunked_delta_rule(qj, kj, vj, bj[:, :2])
    bad_state = jnp.asarray(np.zeros((1, 1, 3, 3), dtype=np.float32))
    with pytest.raises(kernels.KernelError):
        kernels.chunked_delta_rule(qj, kj, vj, bj, state=bad_state)


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


def test_ep_dispatch_is_registered_jax_custom_vjp():
    """Public ``ep_dispatch`` must be the ``jax.custom_vjp`` object, not a wrapper."""
    jax, jnp = _jax()

    @jax.custom_vjp
    def _probe(z):
        return z

    def _probe_fwd(z):
        return z, ()

    def _probe_bwd(_res, g):
        return (g,)

    _probe.defvjp(_probe_fwd, _probe_bwd)
    assert type(kernels.ep_dispatch) is type(_probe)
    assert hasattr(kernels.ep_dispatch, "defvjp")
    tokens = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float32)
    meta = _dispatch_meta(
        expert_ids=[[0], [1], [0]],
        probs=[[1.0], [1.0], [1.0]],
        racks=[0, 1],
        n_experts=2,
    )
    g = jax.grad(lambda t: jnp.sum(kernels.ep_dispatch(t, meta)[0]))(tokens)
    assert g.shape == tokens.shape
    np.testing.assert_allclose(_np(g), np.ones_like(_np(tokens)), **TOL)


def test_ep_combine_is_registered_jax_custom_vjp():
    """Public ``ep_combine`` must be the ``jax.custom_vjp`` object, not a wrapper."""
    jax, jnp = _jax()

    @jax.custom_vjp
    def _probe(z):
        return z

    def _probe_fwd(z):
        return z, ()

    def _probe_bwd(_res, g):
        return (g,)

    _probe.defvjp(_probe_fwd, _probe_bwd)
    assert type(kernels.ep_combine) is type(_probe)
    assert hasattr(kernels.ep_combine, "defvjp")
    tokens = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    meta = _dispatch_meta(
        expert_ids=[[0], [1], [0]],
        probs=[[1.0], [1.0], [1.0]],
        racks=[0, 1],
        n_experts=2,
    )
    dispatched, residual = kernels.ep_dispatch(tokens, meta)
    expert_out = jnp.asarray(dispatched)
    g = jax.grad(lambda eo: jnp.sum(kernels.ep_combine(eo, meta, residual)))(expert_out)
    assert g.shape == expert_out.shape
    assert _np(g).dtype == np.float32


def test_ep_dispatch_jax_grad_matches_scatter_vjp():
    """jax.grad / jax.vjp through dispatched equals scatter of cotangents via residual."""
    jax, jnp = _jax()
    rng = np.random.default_rng(60)
    for name, tokens, meta in _ep_vjp_layouts(rng):
        dispatched_ref, residual_ref = ref.ep_dispatch(tokens, meta)
        g_disp = rng.standard_normal(dispatched_ref.shape).astype(np.float32)
        exp = ref.ep_dispatch_vjp(tokens, meta, g_disp, residual_ref)
        tokens_j = jnp.asarray(tokens)
        g_disp_j = jnp.asarray(g_disp)

        def _dispatched(tok, _meta=meta):
            d, _r = kernels.ep_dispatch(tok, _meta)
            return d

        _primals, vjp_fn = jax.vjp(_dispatched, tokens_j)
        (got_vjp,) = vjp_fn(g_disp_j)
        np.testing.assert_allclose(_np(got_vjp), exp, **TOL, err_msg=name)
        assert _np(got_vjp).shape == tokens.shape
        assert _np(got_vjp).dtype == np.float32

        def _loss(tok, _meta=meta, _g=g_disp_j):
            d, _r = kernels.ep_dispatch(tok, _meta)
            return jnp.sum(d * _g)

        got_grad = jax.grad(_loss)(tokens_j)
        np.testing.assert_allclose(_np(got_grad), exp, **TOL, err_msg=name)

        # Directional finite-difference vs jax.grad (fails on the numpy stub).
        d = _unit(rng, tokens.shape)
        eps = 1e-3
        plus, _ = ref.ep_dispatch(tokens + eps * d, meta)
        minus, _ = ref.ep_dispatch(tokens - eps * d, meta)
        fd = (float(np.sum(plus * g_disp)) - float(np.sum(minus * g_disp))) / (2 * eps)
        analytic = float(np.sum(_np(got_grad) * d))
        np.testing.assert_allclose(fd, analytic, **FD_TOL, err_msg=name)


def test_ep_combine_jax_grad_matches_weighted_scatter_vjp():
    """jax.grad / jax.vjp through combined equals weighted scatter onto expert slots."""
    jax, jnp = _jax()
    rng = np.random.default_rng(61)
    for name, tokens, meta in _ep_vjp_layouts(rng):
        dispatched_ref, residual_ref = ref.ep_dispatch(tokens, meta)
        _, residual = kernels.ep_dispatch(tokens, meta)
        expert_out = rng.standard_normal(dispatched_ref.shape).astype(np.float32)
        g_comb = rng.standard_normal(tokens.shape).astype(np.float32)
        exp = ref.ep_combine_vjp(expert_out, meta, residual_ref, g_comb)
        expert_j = jnp.asarray(expert_out)
        g_comb_j = jnp.asarray(g_comb)

        def _combined(eo, _meta=meta, _residual=residual):
            return kernels.ep_combine(eo, _meta, _residual)

        _primals, vjp_fn = jax.vjp(_combined, expert_j)
        (got_vjp,) = vjp_fn(g_comb_j)
        np.testing.assert_allclose(_np(got_vjp), exp, **TOL, err_msg=name)
        assert _np(got_vjp).shape == expert_out.shape
        assert _np(got_vjp).dtype == np.float32

        def _loss(eo, _meta=meta, _residual=residual, _g=g_comb_j):
            return jnp.sum(kernels.ep_combine(eo, _meta, _residual) * _g)

        got_grad = jax.grad(_loss)(expert_j)
        np.testing.assert_allclose(_np(got_grad), exp, **TOL, err_msg=name)

        d = _unit(rng, expert_out.shape)
        eps = 1e-3
        plus = ref.ep_combine(expert_out + eps * d, meta, residual_ref)
        minus = ref.ep_combine(expert_out - eps * d, meta, residual_ref)
        fd = (float(np.sum(plus * g_comb)) - float(np.sum(minus * g_comb))) / (2 * eps)
        analytic = float(np.sum(_np(got_grad) * d))
        np.testing.assert_allclose(fd, analytic, **FD_TOL, err_msg=name)

        # Unused / padded slots must receive zero cotangent.
        token_index = np.asarray(residual_ref.token_index)
        unused = token_index < 0
        if np.any(unused):
            np.testing.assert_allclose(_np(got_grad)[unused], 0.0, **TOL, err_msg=f"{name}-unused")


# ---------------------------------------------------------------------------
# EP jax.jit / DispatchMeta pytree (must fail until meta is a registered pytree)
# ---------------------------------------------------------------------------


def test_dispatch_meta_is_registered_jax_pytree():
    """DispatchMeta must be a jax.tree_util registered dataclass so jit can take it.

    expert_ids / probs / racks are data-field array leaves (values match).
    n_experts / max_racks may live in aux_data or in leaves. A dummy register
    that treats the whole object as one leaf, or stuffs arrays into aux, fails.
    """
    jax, jnp = _jax()
    expert_ids = np.array([[0, 2], [1, 0], [2, 1]], dtype=np.int32)
    probs = np.array([[0.11, 0.89], [0.22, 0.78], [0.33, 0.67]], dtype=np.float32)
    racks = np.array([7, 8, 9], dtype=np.int32)
    meta = _dispatch_meta(expert_ids, probs, racks, n_experts=3, max_racks=4)

    leaves, treedef = jax.tree_util.tree_flatten(meta)
    assert not any(type(leaf) is kernels.DispatchMeta for leaf in leaves), (
        "DispatchMeta must not be a pytree leaf; jax.jit would treat it as an "
        "abstract array. Register it so array fields are leaves."
    )
    assert _pytree_leaves_contain_array(leaves, expert_ids), (
        "expert_ids must be a pytree data-field leaf (value match); a dummy "
        "register or aux-only flatten is not enough"
    )
    assert _pytree_leaves_contain_array(leaves, probs), (
        "probs must be a pytree data-field leaf (value match)"
    )
    assert _pytree_leaves_contain_array(leaves, racks), (
        "racks must be a pytree data-field leaf (value match)"
    )

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_array_equal(_np(rebuilt.expert_ids), expert_ids)
    np.testing.assert_array_equal(_np(rebuilt.probs), probs)
    np.testing.assert_array_equal(_np(rebuilt.racks), racks)
    assert int(rebuilt.n_experts) == 3
    assert int(rebuilt.max_racks) == 4

    def _double_floats(x):
        if hasattr(x, "dtype") and np.issubdtype(np.asarray(x).dtype, np.floating):
            return x * np.float32(2.0)
        return x

    mapped = jax.tree_util.tree_map(_double_floats, meta)
    assert type(mapped) is kernels.DispatchMeta
    np.testing.assert_allclose(_np(mapped.probs), np.float32(2.0) * probs, **TOL)
    np.testing.assert_array_equal(_np(mapped.expert_ids), expert_ids)
    np.testing.assert_array_equal(_np(mapped.racks), racks)
    assert int(mapped.n_experts) == 3
    assert int(mapped.max_racks) == 4

    # jit must accept DispatchMeta as a pytree argument, not an abstract array.
    jitted_meta = jax.jit(lambda m: m)(meta)
    np.testing.assert_array_equal(_np(jitted_meta.expert_ids), expert_ids)
    np.testing.assert_allclose(_np(jitted_meta.probs), probs, **TOL)
    np.testing.assert_array_equal(_np(jitted_meta.racks), racks)
    assert int(jitted_meta.n_experts) == 3
    assert int(jitted_meta.max_racks) == 4

    # Constructing DispatchMeta from traced array args must be legal.
    def _rebuild(eids, pr, rk):
        m = kernels.DispatchMeta(expert_ids=eids, probs=pr, racks=rk, n_experts=3, max_racks=4)
        return m.probs

    rebuilt_probs = jax.jit(_rebuild)(
        jnp.asarray(expert_ids), jnp.asarray(probs), jnp.asarray(racks)
    )
    np.testing.assert_allclose(_np(rebuilt_probs), probs, **TOL)


def test_ep_dispatch_jax_jit_matches_eager():
    """jax.jit(ep_dispatch)(tokens, meta) equals eager; pad slots stay -1."""
    jax, jnp = _jax()
    rng = np.random.default_rng(70)
    jitted = jax.jit(kernels.ep_dispatch)
    for name, tokens, meta in _ep_vjp_layouts(rng):
        tokens_j = jnp.asarray(tokens)
        eager_d, eager_r = kernels.ep_dispatch(tokens_j, meta)
        jit_d, jit_r = jitted(tokens_j, meta)
        np.testing.assert_allclose(_np(jit_d), _np(eager_d), **TOL, err_msg=name)
        ref_d, _ = ref.ep_dispatch(tokens, meta)
        np.testing.assert_allclose(_np(jit_d), ref_d, **TOL, err_msg=name)
        _assert_residual_indices_equal(jit_r, eager_r, err_msg=name)
        assert jit_d.shape == eager_d.shape
        assert _np(jit_d).dtype == np.float32

    # numpy tokens converted at the jit boundary (not pre-wrapped in jnp.asarray).
    tokens_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    meta_np = _dispatch_meta(
        expert_ids=[[0], [1], [0]],
        probs=[[1.0], [1.0], [1.0]],
        racks=[0, 1],
        n_experts=2,
    )
    eager_d, eager_r = kernels.ep_dispatch(tokens_np, meta_np)
    jit_d, jit_r = jitted(tokens_np, meta_np)
    np.testing.assert_allclose(_np(jit_d), _np(eager_d), **TOL, err_msg="numpy-tokens")
    _assert_residual_indices_equal(jit_r, eager_r, err_msg="numpy-tokens")

    # jax.Array meta fields must be traced (not host-copied as an abstract array).
    meta_j = kernels.DispatchMeta(
        expert_ids=jnp.asarray(meta_np.expert_ids),
        probs=jnp.asarray(meta_np.probs),
        racks=jnp.asarray(meta_np.racks),
        n_experts=int(meta_np.n_experts),
        max_racks=int(meta_np.max_racks),
    )
    jit_d_j, jit_r_j = jitted(jnp.asarray(tokens_np), meta_j)
    np.testing.assert_allclose(_np(jit_d_j), _np(eager_d), **TOL, err_msg="jax-meta")
    _assert_residual_indices_equal(jit_r_j, eager_r, err_msg="jax-meta")


def test_ep_combine_jax_jit_matches_eager():
    """jax.jit(ep_combine)(expert_out, meta, residual) equals eager / ref at 1e-5."""
    jax, jnp = _jax()
    rng = np.random.default_rng(71)
    jitted_dispatch = jax.jit(kernels.ep_dispatch)
    jitted_combine = jax.jit(kernels.ep_combine)
    for name, tokens, meta in _ep_vjp_layouts(rng):
        tokens_j = jnp.asarray(tokens)
        eager_d, eager_r = kernels.ep_dispatch(tokens_j, meta)
        g = rng.standard_normal(np.shape(_np(eager_d))).astype(np.float32)
        expert_out = _np(eager_d) + g
        expert_j = jnp.asarray(expert_out)

        eager_c = kernels.ep_combine(expert_j, meta, eager_r)
        jit_c_eager_r = jitted_combine(expert_j, meta, eager_r)
        np.testing.assert_allclose(_np(jit_c_eager_r), _np(eager_c), **TOL, err_msg=name)
        np.testing.assert_allclose(
            _np(jit_c_eager_r),
            ref.ep_combine(expert_out, meta, ref.ep_dispatch(tokens, meta)[1]),
            **TOL,
            err_msg=f"{name}-ref",
        )
        assert _np(jit_c_eager_r).dtype == np.float32
        assert jit_c_eager_r.shape == tokens.shape

        _, jit_r = jitted_dispatch(tokens_j, meta)
        jit_c_jit_r = jitted_combine(expert_j, meta, jit_r)
        np.testing.assert_allclose(_np(jit_c_jit_r), _np(eager_c), **TOL, err_msg=f"{name}-jit-res")


def test_ep_dispatch_then_combine_jax_jit_composes():
    """A single jax.jit of dispatch-then-combine must match eager (identity at mass 1)."""
    jax, jnp = _jax()
    rng = np.random.default_rng(72)

    def _roundtrip(tok, m):
        dispatched, residual = kernels.ep_dispatch(tok, m)
        return kernels.ep_combine(dispatched, m, residual)

    jitted = jax.jit(_roundtrip)
    for name, tokens, meta in _ep_vjp_layouts(rng):
        tokens_j = jnp.asarray(tokens)
        eager = _roundtrip(tokens_j, meta)
        got = jitted(tokens_j, meta)
        np.testing.assert_allclose(_np(got), _np(eager), **TOL, err_msg=name)
        mass = np.asarray(meta.probs).sum(axis=-1)
        if np.allclose(mass, 1.0, **TOL):
            np.testing.assert_allclose(_np(got), tokens, **TOL, err_msg=f"{name}-identity")
        assert got.shape == tokens.shape
        assert _np(got).dtype == np.float32

    # Non-unit routing mass: compose must run combine, not passthrough tokens.
    tokens_w = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
    meta_w = _dispatch_meta(
        expert_ids=[[0, 1], [1, 2], [0, 2]],
        probs=[[0.25, 0.25], [0.5, 0.1], [0.1, 0.2]],
        racks=[0, 0, 1],
        n_experts=3,
    )
    eager_w = _roundtrip(tokens_w, meta_w)
    got_w = jitted(jnp.asarray(tokens_w), meta_w)
    disp_w, res_w = ref.ep_dispatch(tokens_w, meta_w)
    np.testing.assert_allclose(_np(got_w), _np(eager_w), **TOL, err_msg="non-unit-mass")
    np.testing.assert_allclose(_np(got_w), ref.ep_combine(disp_w, meta_w, res_w), **TOL)
    assert not np.allclose(_np(got_w), tokens_w, atol=1e-5)


def test_ep_dispatch_jax_grad_through_jit_matches_scatter_vjp():
    """jax.grad through jax.jit(ep_dispatch) matches ref.ep_dispatch_vjp; FD vs ref."""
    jax, jnp = _jax()
    rng = np.random.default_rng(73)
    jitted = jax.jit(kernels.ep_dispatch)
    for name, tokens, meta in _ep_vjp_layouts(rng):
        dispatched_ref, residual_ref = ref.ep_dispatch(tokens, meta)
        g_disp = rng.standard_normal(dispatched_ref.shape).astype(np.float32)
        exp = ref.ep_dispatch_vjp(tokens, meta, g_disp, residual_ref)
        tokens_j = jnp.asarray(tokens)
        g_disp_j = jnp.asarray(g_disp)

        def _loss(tok, m, _g=g_disp_j):
            dispatched, _residual = jitted(tok, m)
            return jnp.sum(dispatched * _g)

        got_grad = jax.grad(_loss, argnums=0)(tokens_j, meta)
        np.testing.assert_allclose(_np(got_grad), exp, **TOL, err_msg=name)
        assert got_grad.shape == tokens.shape
        assert got_grad.dtype == jnp.float32

        d = _unit(rng, tokens.shape)
        eps = 1e-3
        plus, _ = ref.ep_dispatch(tokens + eps * d, meta)
        minus, _ = ref.ep_dispatch(tokens - eps * d, meta)
        fd = (float(np.sum(plus * g_disp)) - float(np.sum(minus * g_disp))) / (2 * eps)
        analytic = float(np.sum(_np(got_grad) * d))
        np.testing.assert_allclose(fd, analytic, **FD_TOL, err_msg=name)


def test_ep_combine_jax_grad_through_jit_matches_weighted_scatter_vjp():
    """jax.grad through jax.jit(ep_combine) matches ref.ep_combine_vjp; pad slots 0."""
    jax, jnp = _jax()
    rng = np.random.default_rng(74)
    jitted = jax.jit(kernels.ep_combine)
    for name, tokens, meta in _ep_vjp_layouts(rng):
        dispatched, residual_ref = ref.ep_dispatch(tokens, meta)
        expert_out = dispatched + rng.standard_normal(dispatched.shape).astype(np.float32)
        g_comb = rng.standard_normal((tokens.shape[0], tokens.shape[1])).astype(np.float32)
        exp = ref.ep_combine_vjp(expert_out, meta, residual_ref, g_comb)
        _, residual = kernels.ep_dispatch(tokens, meta)
        expert_j = jnp.asarray(expert_out)
        g_comb_j = jnp.asarray(g_comb)

        def _loss(eo, m, r, _g=g_comb_j):
            return jnp.sum(jitted(eo, m, r) * _g)

        got_grad = jax.grad(_loss, argnums=0)(expert_j, meta, residual)
        np.testing.assert_allclose(_np(got_grad), exp, **TOL, err_msg=name)
        assert got_grad.shape == expert_out.shape
        assert got_grad.dtype == jnp.float32

        d = _unit(rng, expert_out.shape)
        eps = 1e-3
        plus = ref.ep_combine(expert_out + eps * d, meta, residual_ref)
        minus = ref.ep_combine(expert_out - eps * d, meta, residual_ref)
        fd = (float(np.sum(plus * g_comb)) - float(np.sum(minus * g_comb))) / (2 * eps)
        analytic = float(np.sum(_np(got_grad) * d))
        np.testing.assert_allclose(fd, analytic, **FD_TOL, err_msg=name)

        token_index = np.asarray(residual_ref.token_index)
        unused = token_index < 0
        if np.any(unused):
            np.testing.assert_allclose(_np(got_grad)[unused], 0.0, **TOL, err_msg=f"{name}-unused")


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
def test_v1_gpu_chunked_delta_rule_jax_grad_matches_bwd():
    """V1: GPU jax.grad(chunked_delta_rule) matches reverse-state VJP / bwd."""
    jax = _require_gpu()
    jnp = jax.numpy
    rng = np.random.default_rng(64)
    q, k, v, beta = _delta_inputs(rng, batch=2, seq=8, heads=2, dim=4)
    state = rng.standard_normal((2, 2, 4, 4)).astype(np.float32) * 0.05
    cfg = kernels.LinearAttnConfig(chunk=64)
    (y_fwd, ns_fwd), residual = kernels.chunked_delta_rule_fwd(q, k, v, beta, state, cfg)
    go = rng.standard_normal(_np(y_fwd).shape).astype(np.float32)
    gs = rng.standard_normal(_np(ns_fwd).shape).astype(np.float32) * 0.05
    bwd = kernels.chunked_delta_rule_bwd(residual, (go, gs))
    ref_grads = ref.gated_delta_rule_vjp(q, k, v, beta, go, gs, state)

    def _apply(qq, kk, vv, bb, st):
        return kernels.chunked_delta_rule(qq, kk, vv, bb, st, config=cfg)

    _, vjp_fn = jax.vjp(
        _apply,
        jnp.asarray(q),
        jnp.asarray(k),
        jnp.asarray(v),
        jnp.asarray(beta),
        jnp.asarray(state),
    )
    prod_grads = vjp_fn((jnp.asarray(go), jnp.asarray(gs)))
    for got, exp_bwd, exp_ref in zip(prod_grads, bwd, ref_grads, strict=True):
        np.testing.assert_allclose(_np(got), _np(exp_bwd), **TOL)
        np.testing.assert_allclose(_np(got), exp_ref, **TOL)


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
