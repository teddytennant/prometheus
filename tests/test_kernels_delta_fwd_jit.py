"""A3-delta-fwd-jit oracle: ``jax.jit`` of the explicit VJP pair must round-trip ``_DeltaResidual``.

``kernels._DeltaResidual`` must be a ``jax.tree_util`` registered dataclass so
``jax.jit(chunked_delta_rule_fwd)`` / ``jax.jit(chunked_delta_rule_bwd)`` can
return and take it. Array fields ``q``, ``k``, ``v``, ``beta``, ``state0`` are
data (traced leaves). ``chunk`` (Python int) is meta (not traced). Analog of
``Fp8Meta`` / ``DispatchMeta``.

These tests must fail on the current unregistered dataclass and ``numpy.asarray``
host conversion (TypeError / "not a valid JAX type" / TracerArrayConversionError)
and pass once it is registered and traced arrays are not converted. Analog math
is unchanged (gated delta-rule recurrence). Do not copy production math into
this file; compare against eager production and ``tests.reference.kernels``.

``config`` is a host ``LinearAttnConfig``. Call
``jax.jit(chunked_delta_rule_fwd, static_argnames=('config',))``. Do not jit
``LinearAttnConfig`` as an array. Public ``chunked_delta_rule`` custom_vjp stays.
Do not mark gpu. Production must not import ``tests``. Numpy eager paths stay
covered by ``tests/test_kernels.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import kernels
from tests.reference import kernels as ref

TOL = dict(rtol=1e-5, atol=1e-5)

DATA_FIELDS = ("q", "k", "v", "beta", "state0")

# Golden from tests/test_kernels.py: S_0=0, beta=1, orthonormal k, so o_t = v_t.
GOLDEN_Q = np.array([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=np.float32)
GOLDEN_K = np.array([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=np.float32)
GOLDEN_V = np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=np.float32)
GOLDEN_BETA = np.array([[[1.0], [1.0]]], dtype=np.float32)
GOLDEN_OUT = np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=np.float32)
GOLDEN_STATE = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
GOLDEN_CHUNK = 2


def _np(x: object) -> np.ndarray:
    return np.asarray(x)


def _close(got: object, exp: object) -> None:
    """FP32 parity at 1e-5. Do not use pytest.approx on jax scalars."""
    np.testing.assert_allclose(_np(got).astype(np.float32), _np(exp).astype(np.float32), **TOL)


def _pytree_leaves_contain_array(leaves, arr) -> bool:
    """True if ``arr`` appears as a pytree leaf (value + shape), not buried in aux."""
    target = np.asarray(arr)
    for leaf in leaves:
        if type(leaf) is kernels._DeltaResidual or not hasattr(leaf, "shape"):
            continue
        got = _np(leaf)
        if got.shape == target.shape and np.array_equal(got, target):
            return True
    return False


def _delta_inputs(rng, batch=2, seq=5, heads=2, dim=4):
    q = rng.standard_normal((batch, seq, heads, dim)).astype(np.float32)
    k = rng.standard_normal((batch, seq, heads, dim)).astype(np.float32)
    v = rng.standard_normal((batch, seq, heads, dim)).astype(np.float32)
    beta = (0.35 + 0.30 * rng.random((batch, seq, heads))).astype(np.float32)
    return q, k, v, beta


def _delta_residual(q, k, v, beta, state0, *, chunk: int) -> kernels._DeltaResidual:
    return kernels._DeltaResidual(
        q=q, k=k, v=v, beta=beta, state0=state0, chunk=int(chunk)
    )


def _zeros_state(q) -> np.ndarray:
    batch, _, heads, dim = _np(q).shape
    return np.zeros((batch, heads, dim, dim), dtype=np.float32)


def _assert_residual_meta(residual: kernels._DeltaResidual, chunk: int) -> None:
    assert type(residual) is kernels._DeltaResidual
    assert type(residual.chunk) is int
    assert int(residual.chunk) == int(chunk)
    assert not isinstance(residual.chunk, jax.Array)


def _assert_fwd_shapes(out, ns, q, beta) -> None:
    qn, bn = _np(q), _np(beta)
    batch, seq, heads, dim = qn.shape
    assert _np(out).shape == (batch, seq, heads, dim)
    assert _np(ns).shape == (batch, heads, dim, dim)
    assert _np(out).dtype == np.float32
    assert _np(ns).dtype == np.float32
    assert bn.shape == (batch, seq, heads)


def _fwd_cases():
    rng = np.random.default_rng(101)
    q, k, v, beta = _delta_inputs(rng, batch=2, seq=5, heads=2, dim=4)
    state0 = rng.standard_normal((2, 2, 4, 4)).astype(np.float32) * 0.05
    q1, k1, v1, beta1 = _delta_inputs(rng, batch=1, seq=3, heads=1, dim=3)
    return (
        ("chunk2_state", q, k, v, beta, state0, kernels.LinearAttnConfig(chunk=2)),
        ("chunk4_none_state", q, k, v, beta, None, kernels.LinearAttnConfig(chunk=4)),
        ("default_chunk", q1, k1, v1, beta1, None, kernels.LinearAttnConfig()),
    )


def test_delta_residual_is_registered_jax_pytree():
    """_DeltaResidual flattens to five array leaves so jax.jit can return/take it.

    The whole object must not be one leaf, and the arrays must not be stuffed
    into aux. chunk survives unflatten as a Python int. Dummy leaves replace
    arrays. tree_map on floats doubles the arrays and leaves chunk. jax.jit
    of identity roundtrips type and values; chunk stays a Python int.
    """
    rng = np.random.default_rng(102)
    q, k, v, beta = _delta_inputs(rng, batch=1, seq=3, heads=2, dim=2)
    state0 = _zeros_state(q)
    chunk = 2
    residual = _delta_residual(q, k, v, beta, state0, chunk=chunk)

    leaves, treedef = jax.tree_util.tree_flatten(residual)
    assert not any(type(leaf) is kernels._DeltaResidual for leaf in leaves), (
        "_DeltaResidual must not be a pytree leaf; jax.jit would treat it as an "
        "abstract array. Register it so array fields are leaves."
    )
    for name, arr in (("q", q), ("k", k), ("v", v), ("beta", beta), ("state0", state0)):
        assert _pytree_leaves_contain_array(leaves, arr), (
            f"{name} must be a pytree data-field leaf (value match); a dummy "
            "register or aux-only flatten is not enough"
        )
    assert len(leaves) == 5, "only q, k, v, beta, state0 are data leaves; chunk is meta"

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    _assert_residual_meta(rebuilt, chunk)
    _close(rebuilt.q, q)
    _close(rebuilt.k, k)
    _close(rebuilt.v, v)
    _close(rebuilt.beta, beta)
    _close(rebuilt.state0, state0)

    dummy_leaves = [np.zeros_like(_np(leaf)) for leaf in leaves]
    dummy = jax.tree_util.tree_unflatten(treedef, dummy_leaves)
    _assert_residual_meta(dummy, chunk)
    np.testing.assert_array_equal(_np(dummy.q), np.zeros_like(q))
    np.testing.assert_array_equal(_np(dummy.k), np.zeros_like(k))
    np.testing.assert_array_equal(_np(dummy.v), np.zeros_like(v))
    np.testing.assert_array_equal(_np(dummy.beta), np.zeros_like(beta))
    np.testing.assert_array_equal(_np(dummy.state0), np.zeros_like(state0))

    def _double_floats(x):
        if hasattr(x, "dtype") and np.issubdtype(np.asarray(x).dtype, np.floating):
            return x * np.float32(2.0)
        return x

    mapped = jax.tree_util.tree_map(_double_floats, residual)
    _assert_residual_meta(mapped, chunk)
    _close(mapped.q, np.float32(2.0) * q)
    _close(mapped.k, np.float32(2.0) * k)
    _close(mapped.v, np.float32(2.0) * v)
    _close(mapped.beta, np.float32(2.0) * beta)
    _close(mapped.state0, np.float32(2.0) * state0)

    ident = jax.jit(lambda r: r)(residual)
    _assert_residual_meta(ident, chunk)
    _close(ident.q, q)
    _close(ident.k, k)
    _close(ident.v, v)
    _close(ident.beta, beta)
    _close(ident.state0, state0)
    assert _np(ident.q).dtype == np.float32
    assert _np(ident.k).dtype == np.float32
    assert _np(ident.v).dtype == np.float32
    assert _np(ident.beta).dtype == np.float32
    assert _np(ident.state0).dtype == np.float32


def test_delta_residual_chunk_is_meta_not_traced():
    """chunk is meta: same array shapes keep the same leaf structure; not a traced array."""
    rng = np.random.default_rng(103)
    q, k, v, beta = _delta_inputs(rng, batch=1, seq=2, heads=1, dim=2)
    state0 = _zeros_state(q)
    res_a = _delta_residual(q, k, v, beta, state0, chunk=2)
    res_b = _delta_residual(q, k, v, beta, state0, chunk=7)

    leaves_a, td_a = jax.tree_util.tree_flatten(res_a)
    leaves_b, td_b = jax.tree_util.tree_flatten(res_b)
    assert not any(type(leaf) is kernels._DeltaResidual for leaf in leaves_a)
    assert len(leaves_a) == len(leaves_b) == 5
    assert td_a.children() == td_b.children()
    assert td_a.num_leaves == td_b.num_leaves == 5
    for arr in (q, k, v, beta, state0):
        assert _pytree_leaves_contain_array(leaves_a, arr)
    for leaf in leaves_a:
        assert not isinstance(leaf, int)
        assert hasattr(leaf, "shape")

    node_cls, aux = td_a.node_data()
    assert node_cls is kernels._DeltaResidual
    aux_leaves = jax.tree_util.tree_leaves(aux)
    assert 2 in aux_leaves
    assert not any(hasattr(v, "shape") and getattr(v, "ndim", 0) > 0 for v in aux_leaves)

    ident = jax.jit(lambda r: r)(res_a)
    _assert_residual_meta(ident, 2)
    _close(ident.q, q)
    _close(ident.k, k)
    _close(ident.v, v)
    _close(ident.beta, beta)
    _close(ident.state0, state0)


def test_delta_residual_construct_from_traced_fields_under_jit():
    """Constructing _DeltaResidual from traced arrays with host chunk is legal under jit."""
    rng = np.random.default_rng(104)
    q, k, v, beta = _delta_inputs(rng, batch=1, seq=2, heads=1, dim=3)
    state0 = _zeros_state(q)
    qj, kj, vj, bj, sj = map(jnp.asarray, (q, k, v, beta, state0))
    chunk = 4

    def _rebuild(qq, kk, vv, bb, ss):
        return kernels._DeltaResidual(q=qq, k=kk, v=vv, beta=bb, state0=ss, chunk=chunk)

    got = jax.jit(_rebuild)(qj, kj, vj, bj, sj)
    _assert_residual_meta(got, chunk)
    _close(got.q, q)
    _close(got.k, k)
    _close(got.v, v)
    _close(got.beta, beta)
    _close(got.state0, state0)
    assert _np(got.q).shape == q.shape
    assert _np(got.k).shape == k.shape
    assert _np(got.v).shape == v.shape
    assert _np(got.beta).shape == beta.shape
    assert _np(got.state0).shape == state0.shape
    assert _np(got.q).dtype == np.float32
    assert _np(got.state0).dtype == np.float32


def test_chunked_delta_rule_fwd_jit_matches_eager():
    """jax.jit(chunked_delta_rule_fwd, static_argnames=('config',)) matches eager at 1e-5.

    Residual is a pytree _DeltaResidual. Also matches tests.reference.kernels.gated_delta_rule.
    Integer chunk, shapes, and dtypes are exact.
    """
    jitted = jax.jit(kernels.chunked_delta_rule_fwd, static_argnames=("config",))
    for name, q, k, v, beta, state, cfg in _fwd_cases():
        qj, kj, vj, bj = map(jnp.asarray, (q, k, v, beta))
        sj = None if state is None else jnp.asarray(state)
        eager_out, eager_res = kernels.chunked_delta_rule_fwd(qj, kj, vj, bj, sj, cfg)
        got_out, got_res = jitted(qj, kj, vj, bj, sj, cfg)
        eager_y, eager_ns = eager_out
        got_y, got_ns = got_out
        _close(got_y, eager_y)
        _close(got_ns, eager_ns)
        _assert_fwd_shapes(got_y, got_ns, q, beta)
        _assert_residual_meta(got_res, cfg.chunk)
        _close(got_res.q, eager_res.q)
        _close(got_res.k, eager_res.k)
        _close(got_res.v, eager_res.v)
        _close(got_res.beta, eager_res.beta)
        _close(got_res.state0, eager_res.state0)
        assert _np(got_res.q).shape == q.shape, name
        assert _np(got_res.beta).shape == beta.shape, name
        assert _np(got_res.state0).dtype == np.float32, name

        exp_y, exp_ns = ref.gated_delta_rule(q, k, v, beta, state=state)
        _close(got_y, exp_y)
        _close(got_ns, exp_ns)

        # numpy input converted at the jit boundary
        got_np_out, got_np_res = jitted(q, k, v, beta, state, cfg)
        _close(got_np_out[0], eager_y)
        _close(got_np_out[1], eager_ns)
        _assert_residual_meta(got_np_res, cfg.chunk)


def test_chunked_delta_rule_bwd_jit_matches_eager():
    """jax.jit(chunked_delta_rule_bwd) on a pytree residual + jax grads matches eager at 1e-5."""
    jitted = jax.jit(kernels.chunked_delta_rule_bwd)
    rng = np.random.default_rng(105)
    for name, q, k, v, beta, state, cfg in _fwd_cases():
        state0 = _zeros_state(q) if state is None else state
        (out, ns), eager_res = kernels.chunked_delta_rule_fwd(q, k, v, beta, state, cfg)
        go = rng.standard_normal(_np(out).shape).astype(np.float32)
        gs = rng.standard_normal(_np(ns).shape).astype(np.float32) * 0.05
        residual = _delta_residual(
            jnp.asarray(q),
            jnp.asarray(k),
            jnp.asarray(v),
            jnp.asarray(beta),
            jnp.asarray(state0),
            chunk=int(cfg.chunk),
        )
        goj, gsj = jnp.asarray(go), jnp.asarray(gs)
        eager = kernels.chunked_delta_rule_bwd(eager_res, (go, gs))
        got = jitted(residual, (goj, gsj))
        assert len(got) == 5, name
        for i, (g, e) in enumerate(zip(got, eager, strict=True)):
            _close(g, e)
            assert _np(g).dtype == np.float32, f"{name}-{i}"
        assert _np(got[0]).shape == q.shape, name
        assert _np(got[1]).shape == k.shape, name
        assert _np(got[2]).shape == v.shape, name
        assert _np(got[3]).shape == beta.shape, name
        assert _np(got[4]).shape == state0.shape, name

        exp = ref.gated_delta_rule_vjp(q, k, v, beta, go, gs, state0)
        for g, e in zip(got, exp, strict=True):
            _close(g, e)


def test_chunked_delta_rule_fwd_bwd_jit_composes():
    """Jitted fwd residual feeds jitted bwd; the pair matches eager at 1e-5."""
    j_fwd = jax.jit(kernels.chunked_delta_rule_fwd, static_argnames=("config",))
    j_bwd = jax.jit(kernels.chunked_delta_rule_bwd)
    rng = np.random.default_rng(106)
    q, k, v, beta = _delta_inputs(rng, batch=2, seq=5, heads=2, dim=3)
    state0 = rng.standard_normal((2, 2, 3, 3)).astype(np.float32) * 0.05
    cfg = kernels.LinearAttnConfig(chunk=2)
    qj, kj, vj, bj, sj = map(jnp.asarray, (q, k, v, beta, state0))
    (e_out, e_ns), e_res = kernels.chunked_delta_rule_fwd(qj, kj, vj, bj, sj, cfg)
    go = rng.standard_normal(_np(e_out).shape).astype(np.float32)
    gs = rng.standard_normal(_np(e_ns).shape).astype(np.float32) * 0.05
    e_grads = kernels.chunked_delta_rule_bwd(e_res, (go, gs))

    (j_out, j_ns), j_res = j_fwd(qj, kj, vj, bj, sj, cfg)
    _close(j_out, e_out)
    _close(j_ns, e_ns)
    _assert_residual_meta(j_res, cfg.chunk)
    j_grads = j_bwd(j_res, (jnp.asarray(go), jnp.asarray(gs)))
    assert len(j_grads) == 5
    for g, e in zip(j_grads, e_grads, strict=True):
        _close(g, e)

    exp_y, exp_ns = ref.gated_delta_rule(q, k, v, beta, state=state0)
    _close(j_out, exp_y)
    _close(j_ns, exp_ns)
    exp_grads = ref.gated_delta_rule_vjp(q, k, v, beta, go, gs, state0)
    for g, e in zip(j_grads, exp_grads, strict=True):
        _close(g, e)


def test_chunked_delta_rule_fwd_jit_golden_orthogonal_keys():
    """Closed-form orthogonal keys / beta=1 under jit matches the existing golden at 1e-5."""
    cfg = kernels.LinearAttnConfig(chunk=GOLDEN_CHUNK)
    qj, kj, vj, bj = map(jnp.asarray, (GOLDEN_Q, GOLDEN_K, GOLDEN_V, GOLDEN_BETA))
    jitted = jax.jit(kernels.chunked_delta_rule_fwd, static_argnames=("config",))
    (out, ns), residual = jitted(qj, kj, vj, bj, None, cfg)
    _close(out, GOLDEN_OUT)
    _close(ns, GOLDEN_STATE)
    _assert_residual_meta(residual, GOLDEN_CHUNK)
    _assert_fwd_shapes(out, ns, GOLDEN_Q, GOLDEN_BETA)

    eager_out, eager_res = kernels.chunked_delta_rule_fwd(
        qj, kj, vj, bj, None, cfg
    )
    _close(out, eager_out[0])
    _close(ns, eager_out[1])
    _close(residual.q, eager_res.q)
    _close(residual.k, eager_res.k)
    _close(residual.v, eager_res.v)
    _close(residual.beta, eager_res.beta)
    _close(residual.state0, eager_res.state0)

    exp_y, exp_ns = ref.gated_delta_rule(GOLDEN_Q, GOLDEN_K, GOLDEN_V, GOLDEN_BETA)
    _close(out, exp_y)
    _close(ns, exp_ns)

    # beta=1 identity-like: o_t = v_t and S = k v^T for orthonormal k, S_0=0.
    np.testing.assert_allclose(_np(out), GOLDEN_V, **TOL)


def test_chunked_delta_rule_public_jit_matches_eager_and_jitted_fwd():
    """Public jax.jit(chunked_delta_rule) still matches eager; also matches jitted explicit fwd.

    Closed-over config and static_argnames=('config',) both stay valid. Do not
    weaken the public custom_vjp primitive.
    """
    rng = np.random.default_rng(107)
    q, k, v, beta = _delta_inputs(rng, batch=2, seq=5, heads=2, dim=4)
    state0 = rng.standard_normal((2, 2, 4, 4)).astype(np.float32) * 0.05
    cfg = kernels.LinearAttnConfig(chunk=2)
    qj, kj, vj, bj, sj = map(jnp.asarray, (q, k, v, beta, state0))
    eager = kernels.chunked_delta_rule(qj, kj, vj, bj, sj, config=cfg)

    def _closed(qq, kk, vv, bb, ss):
        return kernels.chunked_delta_rule(qq, kk, vv, bb, ss, config=cfg)

    got_closed = jax.jit(_closed)(qj, kj, vj, bj, sj)
    _close(got_closed[0], eager[0])
    _close(got_closed[1], eager[1])
    _assert_fwd_shapes(got_closed[0], got_closed[1], q, beta)

    got_static = jax.jit(kernels.chunked_delta_rule, static_argnames=("config",))(
        qj, kj, vj, bj, sj, cfg
    )
    _close(got_static[0], eager[0])
    _close(got_static[1], eager[1])

    exp_y, exp_ns = ref.gated_delta_rule(q, k, v, beta, state=state0)
    _close(got_closed[0], exp_y)
    _close(got_closed[1], exp_ns)

    j_fwd = jax.jit(kernels.chunked_delta_rule_fwd, static_argnames=("config",))
    (fwd_y, fwd_ns), fwd_res = j_fwd(qj, kj, vj, bj, sj, cfg)
    _close(got_closed[0], fwd_y)
    _close(got_closed[1], fwd_ns)
    _assert_residual_meta(fwd_res, cfg.chunk)
    _close(fwd_res.q, q)
    _close(fwd_res.k, k)
    _close(fwd_res.v, v)
    _close(fwd_res.beta, beta)
    _close(fwd_res.state0, state0)
