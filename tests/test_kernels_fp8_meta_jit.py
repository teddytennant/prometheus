"""A3-fp8-meta-pytree oracle: ``jax.jit`` of FP8 primitives must round-trip ``Fp8Meta``.

``kernels.Fp8Meta`` must be a ``jax.tree_util`` registered dataclass so
``jax.jit(fp8_quantize)`` / ``jax.jit(fp8_dequantize)`` can return and take it.
Array fields ``q`` and ``scale`` are data (traced leaves). ``block`` (Python int)
and ``dtype`` (``kernels.DType``) are meta (not traced).

These tests must fail on the current unregistered dataclass (TypeError / "not a
valid JAX type" / treating the object as an abstract array) and pass once it is
registered. Quantize math is unchanged (E4M3FN per-block abs-max). Do not copy
production math into this file; compare against eager production and
``tests.reference.kernels`` where a reference exists.

``block`` is a Python int. Call ``jax.jit(fp8_quantize, static_argnames=('block',))``.
Do not jit a validate helper as an entry point. Do not mark gpu. Production must
not import ``tests``. Numpy eager paths stay covered by ``tests/test_kernels.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import kernels
from tests.reference import kernels as ref

TOL = dict(rtol=1e-5, atol=1e-5)

# Golden from tests/test_kernels.py: exact E4M3 bit patterns, one block of 4.
GOLDEN_X = np.array([[0.0, 1.0, 2.0, -2.0]], dtype=np.float32)
GOLDEN_BLOCK = 4


def _np(x: object) -> np.ndarray:
    return np.asarray(x)


def _close(got: object, exp: object) -> None:
    """FP32 parity at 1e-5. Do not use pytest.approx on jax scalars."""
    np.testing.assert_allclose(_np(got).astype(np.float32), _np(exp).astype(np.float32), **TOL)


def _pytree_leaves_contain_array(leaves, arr) -> bool:
    """True if ``arr`` appears as a pytree leaf (value + shape), not buried in aux."""
    target = np.asarray(arr)
    for leaf in leaves:
        if type(leaf) is kernels.Fp8Meta or not hasattr(leaf, "shape"):
            continue
        got = _np(leaf)
        if got.shape == target.shape and np.array_equal(got, target):
            return True
    return False


def _fp8_meta(q, scale, *, block: int = GOLDEN_BLOCK, dtype=None) -> kernels.Fp8Meta:
    return kernels.Fp8Meta(
        q=q,
        scale=scale,
        block=int(block),
        dtype=kernels.DType.FP8 if dtype is None else dtype,
    )


def _quantize_cases():
    rng = np.random.default_rng(42)
    return (
        ("golden", GOLDEN_X, GOLDEN_BLOCK),
        ("zeros", np.zeros((2, 8), dtype=np.float32), 8),
        ("remainder", rng.standard_normal((3, 10)).astype(np.float32), 8),
        ("rank3", rng.standard_normal((2, 3, 5)).astype(np.float32), 4),
    )


def test_fp8_meta_is_registered_jax_pytree():
    """Fp8Meta flattens to q/scale array leaves so jax.jit can return/take it.

    The whole object must not be one leaf, and the arrays must not be stuffed
    into aux. block and dtype survive unflatten. tree_map on floats doubles
    scale and leaves q/block/dtype. jax.jit(lambda m: m)(meta) roundtrips.
    """
    q = np.array([[0, 118, 126, -2]], dtype=np.int8)
    scale = np.array([[0.00446429]], dtype=np.float32)
    meta = _fp8_meta(q, scale, block=GOLDEN_BLOCK, dtype=kernels.DType.FP8)

    leaves, treedef = jax.tree_util.tree_flatten(meta)
    assert not any(type(leaf) is kernels.Fp8Meta for leaf in leaves), (
        "Fp8Meta must not be a pytree leaf; jax.jit would treat it as an "
        "abstract array. Register it so array fields are leaves."
    )
    assert _pytree_leaves_contain_array(leaves, q), (
        "q must be a pytree data-field leaf (value match); a dummy register "
        "or aux-only flatten is not enough"
    )
    assert _pytree_leaves_contain_array(leaves, scale), (
        "scale must be a pytree data-field leaf (value match)"
    )
    assert len(leaves) == 2, "only q and scale are data leaves; block/dtype are meta"

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is kernels.Fp8Meta
    np.testing.assert_array_equal(_np(rebuilt.q), q)
    _close(rebuilt.scale, scale)
    assert int(rebuilt.block) == GOLDEN_BLOCK
    assert rebuilt.dtype == kernels.DType.FP8

    # Arrays live in the leaves, not aux: dummy leaves must replace q/scale.
    dummy_leaves = [np.zeros_like(_np(leaf)) for leaf in leaves]
    dummy = jax.tree_util.tree_unflatten(treedef, dummy_leaves)
    assert type(dummy) is kernels.Fp8Meta
    np.testing.assert_array_equal(_np(dummy.q), np.zeros_like(q))
    np.testing.assert_array_equal(_np(dummy.scale), np.zeros_like(scale))
    assert int(dummy.block) == GOLDEN_BLOCK
    assert dummy.dtype == kernels.DType.FP8

    def _double_floats(x):
        if hasattr(x, "dtype") and np.issubdtype(np.asarray(x).dtype, np.floating):
            return x * np.float32(2.0)
        return x

    mapped = jax.tree_util.tree_map(_double_floats, meta)
    assert type(mapped) is kernels.Fp8Meta
    _close(mapped.scale, np.float32(2.0) * scale)
    np.testing.assert_array_equal(_np(mapped.q), q)
    assert int(mapped.block) == GOLDEN_BLOCK
    assert mapped.dtype == kernels.DType.FP8

    ident = jax.jit(lambda m: m)(meta)
    assert type(ident) is kernels.Fp8Meta
    np.testing.assert_array_equal(_np(ident.q), q)
    _close(ident.scale, scale)
    assert int(ident.block) == GOLDEN_BLOCK
    assert ident.dtype == kernels.DType.FP8
    assert type(ident.block) is int
    assert isinstance(ident.dtype, kernels.DType)
    assert not isinstance(ident.block, jax.Array)
    assert not isinstance(ident.dtype, jax.Array)


def test_fp8_meta_construct_from_traced_fields_under_jit():
    """Constructing Fp8Meta from traced q/scale with host block/dtype is legal under jit."""
    q = jnp.asarray(np.array([[0, 118, 126, -2]], dtype=np.int8))
    scale = jnp.asarray(np.array([[0.00446429]], dtype=np.float32))

    def _rebuild(qq, ss):
        return kernels.Fp8Meta(
            q=qq, scale=ss, block=GOLDEN_BLOCK, dtype=kernels.DType.FP8
        )

    got = jax.jit(_rebuild)(q, scale)
    assert type(got) is kernels.Fp8Meta
    np.testing.assert_array_equal(_np(got.q), _np(q))
    _close(got.scale, scale)
    assert int(got.block) == GOLDEN_BLOCK
    assert got.dtype == kernels.DType.FP8
    assert type(got.block) is int
    assert not isinstance(got.block, jax.Array)
    assert not isinstance(got.dtype, jax.Array)
    assert _np(got.q).shape == (1, 4)
    assert _np(got.scale).shape == (1, 1)
    assert _np(got.q).dtype == np.int8
    assert _np(got.scale).dtype == np.float32


def test_fp8_quantize_jit_matches_eager():
    """jax.jit(fp8_quantize, static_argnames=('block',)) matches eager at 1e-5.

    Integer q is exact. Returns a pytree Fp8Meta. Also matches the NumPy reference.
    """
    jitted = jax.jit(kernels.fp8_quantize, static_argnames=("block",))
    for name, x_np, block in _quantize_cases():
        x_j = jnp.asarray(x_np)
        eager = kernels.fp8_quantize(x_j, block=block)
        got = jitted(x_j, block=block)
        assert type(got) is kernels.Fp8Meta, name
        np.testing.assert_array_equal(_np(got.q), _np(eager.q), err_msg=name)
        _close(got.scale, eager.scale)
        assert int(got.block) == int(block), name
        assert got.dtype == kernels.DType.FP8, name
        assert _np(got.q).shape == x_np.shape, name
        assert _np(got.q).dtype == np.int8, name
        assert _np(got.scale).dtype == np.float32, name

        eager_np = kernels.fp8_quantize(x_np, block=block)
        np.testing.assert_array_equal(_np(got.q), _np(eager_np.q), err_msg=f"{name}-numpy-eager")
        _close(got.scale, eager_np.scale)

        exp = ref.fp8_quantize(x_np, block=block)
        np.testing.assert_array_equal(_np(got.q), _np(exp.q), err_msg=f"{name}-ref")
        _close(got.scale, exp.scale)

        # numpy input converted at the jit boundary.
        got_np = jitted(x_np, block=block)
        assert type(got_np) is kernels.Fp8Meta, name
        np.testing.assert_array_equal(
            _np(got_np.q), _np(eager_np.q), err_msg=f"{name}-numpy-boundary"
        )
        _close(got_np.scale, eager_np.scale)


def test_fp8_dequantize_jit_matches_eager():
    """jax.jit(fp8_dequantize)(meta) matches eager at 1e-5 on a pytree Fp8Meta of jnp arrays."""
    jitted = jax.jit(kernels.fp8_dequantize)
    for name, x_np, block in _quantize_cases():
        eager_meta = kernels.fp8_quantize(x_np, block=block)
        meta = _fp8_meta(
            jnp.asarray(eager_meta.q),
            jnp.asarray(eager_meta.scale),
            block=int(eager_meta.block),
            dtype=eager_meta.dtype,
        )
        eager = kernels.fp8_dequantize(meta)
        got = jitted(meta)
        _close(got, eager)
        assert _np(got).shape == x_np.shape, name
        assert _np(got).dtype == np.float32, name

        exp = ref.fp8_dequantize(ref.fp8_quantize(x_np, block=block))
        _close(got, exp)


def test_fp8_quantize_dequantize_jit_roundtrip_matches_eager():
    """Jitted dequant(quant(x)) matches eager and the reference at 1e-5 (E4M3FN).

    The pytree must cross the jit boundary (quant returns Fp8Meta, dequant takes
    it). A closed-over Python container inside one fused jit is not enough.
    """
    quant = jax.jit(kernels.fp8_quantize, static_argnames=("block",))
    dequant = jax.jit(kernels.fp8_dequantize)
    for name, x_np, block in _quantize_cases():
        x_j = jnp.asarray(x_np)
        eager = kernels.fp8_dequantize(kernels.fp8_quantize(x_j, block=block))
        got = dequant(quant(x_j, block=block))
        _close(got, eager)
        assert _np(got).shape == x_np.shape, name
        assert _np(got).dtype == np.float32, name

        exp = ref.fp8_dequantize(ref.fp8_quantize(x_np, block=block))
        _close(got, exp)


def test_fp8_meta_block_and_dtype_are_meta_not_traced():
    """block/dtype are meta: same array shapes keep the same leaf structure; not traced arrays."""
    q = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int8)
    scale = np.array([[0.5], [1.25]], dtype=np.float32)
    meta_a = _fp8_meta(q, scale, block=4, dtype=kernels.DType.FP8)
    meta_b = _fp8_meta(q, scale, block=16, dtype=kernels.DType.NVFP4)

    leaves_a, td_a = jax.tree_util.tree_flatten(meta_a)
    leaves_b, td_b = jax.tree_util.tree_flatten(meta_b)
    assert not any(type(leaf) is kernels.Fp8Meta for leaf in leaves_a)
    assert len(leaves_a) == len(leaves_b) == 2
    assert td_a.children() == td_b.children()
    assert td_a.num_leaves == td_b.num_leaves == 2
    assert _pytree_leaves_contain_array(leaves_a, q)
    assert _pytree_leaves_contain_array(leaves_a, scale)
    # Meta values are not array leaves.
    for leaf in leaves_a:
        assert not isinstance(leaf, (int, kernels.DType))
        assert hasattr(leaf, "shape")

    node_cls, aux = td_a.node_data()
    assert node_cls is kernels.Fp8Meta
    aux_leaves = jax.tree_util.tree_leaves(aux)
    assert GOLDEN_BLOCK in aux_leaves or 4 in aux_leaves
    assert any(v == kernels.DType.FP8 or v == "fp8" for v in aux_leaves)
    assert not any(hasattr(v, "shape") and getattr(v, "ndim", 0) > 0 for v in aux_leaves)

    ident = jax.jit(lambda m: m)(meta_a)
    assert type(ident) is kernels.Fp8Meta
    np.testing.assert_array_equal(_np(ident.q), q)
    _close(ident.scale, scale)
    assert type(ident.block) is int
    assert int(ident.block) == 4
    assert ident.dtype == kernels.DType.FP8
    assert isinstance(ident.dtype, kernels.DType)
    assert not isinstance(ident.block, jax.Array)
    assert not isinstance(ident.dtype, jax.Array)
