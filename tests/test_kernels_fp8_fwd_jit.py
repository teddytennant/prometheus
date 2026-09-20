"""A3-fp8-fwd-jit oracle: ``jax.jit`` of ``fp8_linear_fwd`` / ``fp8_linear_bwd``.

These tests call the public ``kernels.fp8_linear_fwd`` /
``kernels.fp8_linear_bwd`` signatures with ``jnp`` arrays and a Python
``block`` int held static (``static_argnames=('block',)``). They must fail
on the current host-converting implementations (``numpy.asarray`` /
``_f32`` on tracers: ``TracerArrayConversionError`` or ``TypeError``) and
pass once the explicit VJP pair stays in JAX.

Public ``fp8_linear`` custom_vjp already jits; these tests do not replace
that. Residual is ``(x_meta, w_meta)``, two registered ``Fp8Meta`` pytrees.
``block`` is a Python int (static). Analog of A3-delta-fwd-jit.

Coverage
--------
1. jit vs eager y (1e-5). Residual is a pytree of two Fp8Meta; flatten
   leaves are arrays, not a Python object treated as an abstract array.
2. Residual q/scale match eager fp8_quantize of the same x/w and
   tests.reference.kernels.
3. jax.jit(fp8_linear_bwd)(residual, g) matches eager at 1e-5. Residual
   may come from eager or jit fwd.
4. jax.jvp / jax.grad through a wrapper that unpacks the jit fwd output
   (residual pytree in the autodiff graph) is finite. Unused autodiff
   branch finite.
5. Batched leading dims (2, 3, 8) x (5, 8).
6. STE: jit bwd does not re-quantize; scales frozen. Compare against
   eager fp8_linear_bwd / ref.fp8_linear_bwd, not a new recurrence.
7. block stays a Python int. Call with static_argnames=('block',). Do
   not jit a validate helper as an entry point.

Do not mark gpu. Expected values come from eager production and
tests/reference/, not from copying production quantize math.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import kernels
from tests.reference import kernels as ref

TOL = dict(rtol=1e-5, atol=1e-5)

BLOCK = 8


def _np(x: object) -> np.ndarray:
    return np.asarray(x)


def _j32(x: object) -> jax.Array:
    return jnp.asarray(x, dtype=jnp.float32)


def _close(got: object, exp: object, **kwargs: float) -> None:
    tol = dict(TOL)
    tol.update(kwargs)
    np.testing.assert_allclose(_np(got), _np(exp), **tol)


def _xw(
    rng: np.random.Generator, *, rows: int = 4, inn: int = 8, out: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.standard_normal((rows, inn)).astype(np.float32)
    w = rng.standard_normal((out, inn)).astype(np.float32)
    return x, w


def _jfwd():
    return jax.jit(kernels.fp8_linear_fwd, static_argnames=("block",))


def _jbwd():
    return jax.jit(kernels.fp8_linear_bwd)


def _assert_fp8_meta(meta: object, *, block: int) -> None:
    """One residual half: registered Fp8Meta, array leaves, Python-int block."""
    assert type(meta) is kernels.Fp8Meta, type(meta)
    leaves, treedef = jax.tree_util.tree_flatten(meta)
    assert treedef.num_leaves == 2, treedef
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        assert (
            isinstance(leaf, jax.core.Tracer)
            or isinstance(leaf, jax.Array)
            or isinstance(leaf, np.ndarray)
        )
        assert arr.ndim >= 1
    assert type(meta.block) is int
    assert not isinstance(meta.block, jax.Array)
    assert meta.block == block
    assert type(meta.dtype) is kernels.DType
    assert not isinstance(meta, jax.Array)


def _assert_fp8_fwd_residual(res: object, *, block: int) -> None:
    """Residual is a pytree of two Fp8Meta; flatten leaves are arrays."""
    assert type(res) is tuple, type(res)
    assert len(res) == 2
    x_meta, w_meta = res
    _assert_fp8_meta(x_meta, block=block)
    _assert_fp8_meta(w_meta, block=block)
    leaves, treedef = jax.tree_util.tree_flatten(res)
    assert treedef.num_leaves == 4, treedef
    for leaf in leaves:
        assert not hasattr(leaf, "aval") or isinstance(leaf, jax.Array)
        assert not isinstance(leaf, kernels.Fp8Meta)
    assert not isinstance(res, jax.Array)


def _assert_q_scale_match(got: object, exp: object) -> None:
    np.testing.assert_array_equal(_np(got.q).astype(np.int8), _np(exp.q).astype(np.int8))
    _close(got.scale, exp.scale)
    assert int(got.block) == int(exp.block)


# ---------------------------------------------------------------------------
# 1. jit fwd vs eager y; residual is a pytree of two Fp8Meta
# ---------------------------------------------------------------------------


def test_fp8_linear_fwd_jit_matches_eager_y() -> None:
    """jax.jit(fp8_linear_fwd, static_argnames=('block',)) matches eager y at 1e-5."""
    rng = np.random.default_rng(40)
    x, w = _xw(rng)
    xj, wj = _j32(x), _j32(w)
    y_j, res_j = _jfwd()(xj, wj, BLOCK)
    y_e, _res_e = kernels.fp8_linear_fwd(x, w, BLOCK)
    _close(y_j, y_e)
    _close(y_j, ref.fp8_linear(x, w, block=BLOCK))
    _close(y_j, ref.fp8_linear_fwd(x, w, BLOCK)[0])
    _close(y_j, kernels.fp8_linear(x, w, block=BLOCK))
    assert _np(y_j).shape == (x.shape[0], w.shape[0])
    assert _np(y_j).dtype == np.float32 or jnp.asarray(y_j).dtype == jnp.float32
    _assert_fp8_fwd_residual(res_j, block=BLOCK)


def test_fp8_linear_fwd_jit_residual_is_pytree_of_two_fp8meta() -> None:
    """Flattening the residual must not treat a Python object as an abstract array."""
    rng = np.random.default_rng(41)
    x, w = _xw(rng)
    y_j, res_j = _jfwd()(_j32(x), _j32(w), BLOCK)
    assert _np(y_j).shape == (x.shape[0], w.shape[0])
    _assert_fp8_fwd_residual(res_j, block=BLOCK)
    x_meta, w_meta = res_j
    assert type(x_meta) is kernels.Fp8Meta
    assert type(w_meta) is kernels.Fp8Meta
    leaves, _ = jax.tree_util.tree_flatten(res_j)
    assert all(not isinstance(leaf, kernels.Fp8Meta) for leaf in leaves)
    assert _np(x_meta.q).shape == x.shape
    assert _np(w_meta.q).shape == w.shape


# ---------------------------------------------------------------------------
# 2. Residual q/scale match eager / reference quantize
# ---------------------------------------------------------------------------


def test_fp8_linear_fwd_jit_residual_matches_eager_quantize() -> None:
    """Residual q/scale match eager fp8_quantize of the same x/w (and the NumPy ref)."""
    rng = np.random.default_rng(42)
    x, w = _xw(rng, rows=3, inn=12, out=4)
    block = 6
    y_j, res_j = _jfwd()(_j32(x), _j32(w), block)
    _close(y_j, kernels.fp8_linear_fwd(x, w, block)[0])
    x_meta, w_meta = res_j
    _assert_q_scale_match(x_meta, kernels.fp8_quantize(x, block=block))
    _assert_q_scale_match(w_meta, kernels.fp8_quantize(w, block=block))
    _assert_q_scale_match(x_meta, ref.fp8_quantize(x, block=block))
    _assert_q_scale_match(w_meta, ref.fp8_quantize(w, block=block))
    _eager_y, eager_res = kernels.fp8_linear_fwd(x, w, block)
    eager_x, eager_w = eager_res
    _assert_q_scale_match(x_meta, eager_x)
    _assert_q_scale_match(w_meta, eager_w)


# ---------------------------------------------------------------------------
# 3. jit bwd vs eager; residual from eager or jit fwd
# ---------------------------------------------------------------------------


def test_fp8_linear_bwd_jit_matches_eager_from_eager_residual() -> None:
    """jax.jit(fp8_linear_bwd)(eager residual, g) matches eager bwd at 1e-5."""
    rng = np.random.default_rng(43)
    x, w = _xw(rng)
    y_e, residual = kernels.fp8_linear_fwd(x, w, BLOCK)
    g = rng.standard_normal(_np(y_e).shape).astype(np.float32)
    gx_j, gw_j = _jbwd()(residual, _j32(g))
    gx_e, gw_e = kernels.fp8_linear_bwd(residual, g)
    gx_r, gw_r = ref.fp8_linear_bwd(residual, g)
    _close(gx_j, gx_e)
    _close(gw_j, gw_e)
    _close(gx_j, gx_r)
    _close(gw_j, gw_r)
    assert _np(gx_j).shape == x.shape
    assert _np(gw_j).shape == w.shape


def test_fp8_linear_bwd_jit_matches_eager_from_jit_fwd_residual() -> None:
    """jax.jit(fp8_linear_bwd)(jit-fwd residual, g) matches eager bwd at 1e-5."""
    rng = np.random.default_rng(44)
    x, w = _xw(rng)
    y_j, residual_j = _jfwd()(_j32(x), _j32(w), BLOCK)
    y_e, residual_e = kernels.fp8_linear_fwd(x, w, BLOCK)
    g = rng.standard_normal(_np(y_e).shape).astype(np.float32)
    gx_j, gw_j = _jbwd()(residual_j, _j32(g))
    gx_e, gw_e = kernels.fp8_linear_bwd(residual_e, g)
    gx_r, gw_r = ref.fp8_linear_bwd(residual_e, g)
    _close(y_j, y_e)
    _close(gx_j, gx_e)
    _close(gw_j, gw_e)
    _close(gx_j, gx_r)
    _close(gw_j, gw_r)


# ---------------------------------------------------------------------------
# 4. jax.jvp / jax.grad through a wrapper that unpacks the jit fwd residual
# ---------------------------------------------------------------------------


def test_fp8_linear_fwd_jit_jvp_grad_unpack_residual_finite() -> None:
    """jvp/grad through a wrapper that unpacks the jit fwd residual stays finite."""
    rng = np.random.default_rng(45)
    x, w = _xw(rng)
    xj, wj = _j32(x), _j32(w)
    dx = _j32(0.01 * rng.standard_normal(x.shape))
    dw = _j32(0.01 * rng.standard_normal(w.shape))
    jfwd = _jfwd()

    def wrapped(xx, ww):
        y, residual = jfwd(xx, ww, BLOCK)
        x_meta, w_meta = residual
        unused = jnp.sum(x_meta.scale) + jnp.sum(w_meta.scale)
        return y + jnp.float32(0.0) * unused

    primals_out, tangents_out = jax.jvp(wrapped, (xj, wj), (dx, dw))
    assert np.all(np.isfinite(_np(primals_out)))
    assert np.all(np.isfinite(_np(tangents_out)))
    _close(primals_out, kernels.fp8_linear_fwd(x, w, BLOCK)[0])

    def loss(xx, ww):
        return jnp.sum(wrapped(xx, ww))

    gx, gw = jax.grad(loss, argnums=(0, 1))(xj, wj)
    assert np.all(np.isfinite(_np(gx)))
    assert np.all(np.isfinite(_np(gw)))
    assert _np(gx).shape == x.shape
    assert _np(gw).shape == w.shape


def test_fp8_linear_fwd_jit_unused_autodiff_branch_finite() -> None:
    """Zero blocks would nan a log/div unused quantize branch; grads stay finite."""
    x = np.zeros((4, 8), dtype=np.float32)
    x[0, :4] = np.array([1.0, -2.0, 0.5, 0.0], dtype=np.float32)
    w = np.array(
        [[0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.25, 0.0], [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    xj, wj = _j32(x), _j32(w)
    jfwd = _jfwd()

    def wrapped(xx, ww):
        y, residual = jfwd(xx, ww, BLOCK)
        x_meta, w_meta = residual
        unused = jnp.sum(x_meta.scale) + jnp.sum(w_meta.scale)
        return jnp.sum(y) + jnp.float32(0.0) * unused

    y_j, residual = jfwd(xj, wj, BLOCK)
    assert np.all(np.isfinite(_np(y_j)))
    x_meta, w_meta = residual
    assert np.all(np.isfinite(_np(x_meta.scale)))
    assert np.all(np.isfinite(_np(w_meta.scale)))

    g = jax.grad(wrapped, argnums=(0, 1))(xj, wj)
    for gi in g:
        assert np.all(np.isfinite(_np(gi)))

    dx = _j32(np.ones_like(x))
    dw = _j32(np.ones_like(w))
    primal, tangent = jax.jvp(wrapped, (xj, wj), (dx, dw))
    assert np.isfinite(_np(primal))
    assert np.isfinite(_np(tangent))


# ---------------------------------------------------------------------------
# 5. Batched leading dims
# ---------------------------------------------------------------------------


def test_fp8_linear_fwd_jit_batched_leading_dims() -> None:
    """Existing eager shape (2, 3, 8) x (5, 8) jits for fwd and bwd."""
    rng = np.random.default_rng(34)
    x = rng.standard_normal((2, 3, 8)).astype(np.float32)
    w = rng.standard_normal((5, 8)).astype(np.float32)
    y_j, res_j = _jfwd()(_j32(x), _j32(w), BLOCK)
    y_e, res_e = kernels.fp8_linear_fwd(x, w, BLOCK)
    _close(y_j, y_e)
    _close(y_j, ref.fp8_linear(x, w, block=BLOCK))
    assert _np(y_j).shape == (2, 3, 5)
    _assert_fp8_fwd_residual(res_j, block=BLOCK)
    g = rng.standard_normal((2, 3, 5)).astype(np.float32)
    gx_j, gw_j = _jbwd()(res_j, _j32(g))
    gx_e, gw_e = kernels.fp8_linear_bwd(res_e, g)
    gx_r, gw_r = ref.fp8_linear_bwd(res_e, g)
    _close(gx_j, gx_e)
    _close(gw_j, gw_e)
    _close(gx_j, gx_r)
    _close(gw_j, gw_r)
    assert _np(gx_j).shape == x.shape
    assert _np(gw_j).shape == w.shape


# ---------------------------------------------------------------------------
# 6. STE: jit bwd does not re-quantize; scales frozen
# ---------------------------------------------------------------------------


def test_fp8_linear_bwd_jit_ste_frozen_scales() -> None:
    """jit bwd uses residual scales as frozen; it does not re-quantize x/w."""
    rng = np.random.default_rng(35)
    x, w = _xw(rng, rows=4, inn=12, out=6)
    block = 6
    y_e, residual = kernels.fp8_linear_fwd(x, w, block)
    g = rng.standard_normal(_np(y_e).shape).astype(np.float32)
    x_meta, w_meta = residual
    frozen_x = kernels.Fp8Meta(
        q=x_meta.q,
        scale=_np(x_meta.scale) * np.float32(2.0),
        block=int(x_meta.block),
        dtype=x_meta.dtype,
    )
    frozen = (frozen_x, w_meta)
    gx_j, gw_j = _jbwd()(frozen, _j32(g))
    gx_e, gw_e = kernels.fp8_linear_bwd(frozen, g)
    gx_r, gw_r = ref.fp8_linear_bwd(frozen, g)
    _close(gx_j, gx_e)
    _close(gw_j, gw_e)
    _close(gx_j, gx_r)
    _close(gw_j, gw_r)
    live_gx, live_gw = ref.fp8_linear_vjp(x, w, g, block=block)
    assert not np.allclose(_np(gx_j), live_gx, **TOL)
    assert not np.allclose(_np(gw_j), live_gw, **TOL)


# ---------------------------------------------------------------------------
# 7. block stays a Python int; static_argnames=('block',)
# ---------------------------------------------------------------------------


def test_fp8_linear_fwd_jit_block_is_python_int_static() -> None:
    """block stays a Python int. Call with static_argnames=('block',)."""
    rng = np.random.default_rng(46)
    x, w = _xw(rng, rows=2, inn=8, out=3)
    block = 4
    y_j, res_j = _jfwd()(_j32(x), _j32(w), block)
    x_meta, w_meta = res_j
    assert type(x_meta.block) is int
    assert type(w_meta.block) is int
    assert x_meta.block == block
    assert w_meta.block == block
    assert not isinstance(x_meta.block, jax.Array)
    assert not isinstance(w_meta.block, jax.Array)
    _close(y_j, kernels.fp8_linear_fwd(x, w, block)[0])
    _close(y_j, ref.fp8_linear(x, w, block=block))
    _assert_fp8_fwd_residual(res_j, block=block)
