"""A3-ep-fwd-jit oracle: ``jax.jit`` of the explicit EP VJP pair.

These tests call the public ``kernels.ep_dispatch_fwd`` /
``kernels.ep_dispatch_bwd`` / ``kernels.ep_combine_fwd`` /
``kernels.ep_combine_bwd`` signatures with ``jnp`` arrays. They must fail
on the current ``NotImplementedError("A3-ep-fwd-jit")`` stub and on a
host-converting implementation (``numpy.asarray`` / ``_f32`` on tracers:
``TracerArrayConversionError`` or ``TypeError``). They pass once the
explicit VJP pair stays in JAX.

Public ``ep_dispatch`` / ``ep_combine`` custom_vjp already jits; these
tests do not replace that math. Dispatch residual is a pytree
``_DispatchResidual`` (``token_index`` / ``k_index`` data array leaves;
``max_per_expert`` meta, Python int). Combine residual is a pytree of
array leaves (not a Python object treated as an abstract array). Analog
of A3-fp8-fwd-jit / A3-delta-fwd-jit.

Coverage
--------
1. jit vs eager dispatched (1e-5). Residual is ``_DispatchResidual``;
   flatten leaves are arrays; ``max_per_expert`` is a Python int.
2. Dummy leaves replace arrays. jax.jit of identity roundtrips type and
   values. Constructing ``_DispatchResidual`` from traced fields is legal.
3. jax.jit(ep_dispatch_bwd)(residual, g) matches eager at 1e-5. Residual
   may come from eager or jit fwd. Compare against the slow numpy
   scatter-add, not production calling itself.
4. jax.jit(ep_combine_fwd) matches eager at 1e-5 and returns a pytree
   residual (array leaves; dummy replace; identity jit).
5. jax.jit(ep_combine_bwd)(residual, g) matches eager at 1e-5. Compare
   against the slow numpy weighted gather.
6. More than one (n_tokens, n_experts, top-k, d_model) shape. Padded
   slots (token_index -1) stay -1 and contribute nothing to bwd.
7. Grad check: dispatch_bwd vs finite-diff / explicit scatter-add of
   g_dispatched into grad_tokens; combine_bwd vs finite-diff / weighted
   gather of g_combined.
8. Shape / dtype. Golden padded layout.

Do not mark gpu. Expected values come from tests/reference/, not from
copying production ``_ep_*`` helpers. Do not jit ``gspo_dapo_loss``,
``apply_precision``, ``init_opt_state``, or ``wsd_lr``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import kernels
from tests.reference import kernels as ref

TOL = dict(rtol=1e-5, atol=1e-5)
FD_TOL = dict(rtol=1e-3, atol=1e-3)


def _np(x: object) -> np.ndarray:
    return np.asarray(x)


def _j32(x: object) -> jax.Array:
    return jnp.asarray(x, dtype=jnp.float32)


def _close(got: object, exp: object, **kwargs: float) -> None:
    tol = dict(TOL)
    tol.update(kwargs)
    np.testing.assert_allclose(_np(got), _np(exp), **tol)


def _unit(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
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


def _golden_pad_layout() -> tuple[np.ndarray, kernels.DispatchMeta]:
    tokens = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    meta = _dispatch_meta(
        expert_ids=[[0], [1], [0]],
        probs=[[1.0], [1.0], [1.0]],
        racks=[0, 1],
        n_experts=2,
    )
    return tokens, meta


def _layouts(rng: np.random.Generator):
    """More than one (n_tokens, n_experts, top-k, d_model); padded / unused / dup."""
    cases = []
    tokens, meta = _golden_pad_layout()
    cases.append(("golden_pad_topk1", tokens, meta))

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
    tokens = rng.standard_normal((7, 4)).astype(np.float32)
    probs = rng.random((7, 2)).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    cases.append(
        (
            "n7_e4_k2_d4",
            tokens,
            _dispatch_meta(
                expert_ids=[[0, 1], [1, 2], [2, 3], [0, 3], [1, 3], [0, 2], [2, 1]],
                probs=probs,
                racks=[0, 0, 1, 1],
                n_experts=4,
            ),
        )
    )
    tokens = rng.standard_normal((1, 3)).astype(np.float32)
    cases.append(
        (
            "single_token",
            tokens,
            _dispatch_meta(
                expert_ids=[[0]],
                probs=[[1.0]],
                racks=[0],
                n_experts=1,
            ),
        )
    )
    return cases


def _pytree_leaves_contain_array(leaves, arr) -> bool:
    target = np.asarray(arr)
    for leaf in leaves:
        if not hasattr(leaf, "shape"):
            continue
        got = _np(leaf)
        if got.shape == target.shape and np.array_equal(got, target):
            return True
    return False


def _assert_dispatch_residual_meta(residual, max_per_expert: int) -> None:
    assert type(residual) is kernels._DispatchResidual
    assert type(residual.max_per_expert) is int
    assert int(residual.max_per_expert) == int(max_per_expert)
    assert not isinstance(residual.max_per_expert, jax.Array)
    assert not hasattr(residual.max_per_expert, "shape")


def _assert_residual_indices_equal(got, exp, err_msg: str = "") -> None:
    got_t = _np(got.token_index)
    got_k = _np(got.k_index)
    exp_t = _np(exp.token_index)
    exp_k = _np(exp.k_index)
    np.testing.assert_array_equal(got_t, exp_t, err_msg=err_msg)
    np.testing.assert_array_equal(got_k, exp_k, err_msg=err_msg)
    assert np.all(got_t[exp_t < 0] == -1), err_msg
    assert np.all(got_k[exp_k < 0] == -1), err_msg


def _assert_pytree_array_leaves(residual, *, min_leaves: int = 1) -> tuple:
    leaves, treedef = jax.tree_util.tree_flatten(residual)
    assert len(leaves) >= min_leaves, (
        "residual must flatten to array leaves so jax.jit can return/take it"
    )
    assert not any(leaf is residual for leaf in leaves) or all(
        hasattr(leaf, "shape") for leaf in leaves
    )
    for leaf in leaves:
        assert hasattr(leaf, "shape"), (
            "combine/dispatch residual leaves must be arrays; an unregistered "
            "Python object would be treated as an abstract array under jax.jit"
        )
    return leaves, treedef


def _scatter_add_grad_tokens(g_dispatched, token_index, n_tokens: int) -> np.ndarray:
    """Independent numpy scatter-add of g_dispatched into grad_tokens."""
    g = np.asarray(g_dispatched, dtype=np.float32)
    idx = np.asarray(token_index)
    n_experts, max_per, d_model = g.shape
    grad = np.zeros((n_tokens, d_model), dtype=np.float32)
    for e in range(n_experts):
        for slot in range(max_per):
            t = int(idx[e, slot])
            if t < 0:
                continue
            grad[t] += g[e, slot]
    return grad


def _weighted_gather_grad_expert(g_combined, probs, token_index, k_index) -> np.ndarray:
    """Independent numpy weighted gather of g_combined into grad_expert_out."""
    g = np.asarray(g_combined, dtype=np.float32)
    p = np.asarray(probs, dtype=np.float32)
    idx = np.asarray(token_index)
    kid = np.asarray(k_index)
    n_experts, max_per = idx.shape
    d_model = int(g.shape[-1])
    grad = np.zeros((n_experts, max_per, d_model), dtype=np.float32)
    for e in range(n_experts):
        for slot in range(max_per):
            t = int(idx[e, slot])
            if t < 0:
                continue
            k = int(kid[e, slot])
            grad[e, slot] = p[t, k] * g[t]
    return grad


def _jdispatch_fwd():
    return jax.jit(kernels.ep_dispatch_fwd)


def _jdispatch_bwd():
    return jax.jit(kernels.ep_dispatch_bwd)


def _jcombine_fwd():
    return jax.jit(kernels.ep_combine_fwd)


def _jcombine_bwd():
    return jax.jit(kernels.ep_combine_bwd)


def _make_dispatch_residual(token_index, k_index, max_per_expert: int):
    return kernels._DispatchResidual(
        token_index=np.asarray(token_index, dtype=np.int32),
        k_index=np.asarray(k_index, dtype=np.int32),
        max_per_expert=int(max_per_expert),
    )


# ---------------------------------------------------------------------------
# Dispatch residual pytree + jit fwd
# ---------------------------------------------------------------------------


def test_ep_dispatch_fwd_residual_is_dispatch_residual_pytree():
    """_DispatchResidual flattens to token_index/k_index; max_per_expert is a Python int.

    Dummy leaves replace arrays. jax.jit of identity roundtrips type and values.
    jax.jit(ep_dispatch_fwd) must return this pytree (fails on the stub).
    """
    token_index = np.array([[0, 2], [1, -1]], dtype=np.int32)
    k_index = np.array([[0, 0], [0, -1]], dtype=np.int32)
    max_per = 2
    constructed = _make_dispatch_residual(token_index, k_index, max_per)

    leaves, treedef = jax.tree_util.tree_flatten(constructed)
    assert not any(type(leaf) is kernels._DispatchResidual for leaf in leaves), (
        "_DispatchResidual must not be a pytree leaf; jax.jit would treat it as "
        "an abstract array. Register it so array fields are leaves."
    )
    assert _pytree_leaves_contain_array(leaves, token_index)
    assert _pytree_leaves_contain_array(leaves, k_index)
    assert len(leaves) == 2, (
        "only token_index and k_index are data leaves; max_per_expert is meta"
    )
    for leaf in leaves:
        assert not isinstance(leaf, int)
        assert hasattr(leaf, "shape")

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    _assert_dispatch_residual_meta(rebuilt, max_per)
    np.testing.assert_array_equal(_np(rebuilt.token_index), token_index)
    np.testing.assert_array_equal(_np(rebuilt.k_index), k_index)

    dummy_leaves = [np.zeros_like(_np(leaf)) for leaf in leaves]
    dummy = jax.tree_util.tree_unflatten(treedef, dummy_leaves)
    _assert_dispatch_residual_meta(dummy, max_per)
    np.testing.assert_array_equal(_np(dummy.token_index), np.zeros_like(token_index))
    np.testing.assert_array_equal(_np(dummy.k_index), np.zeros_like(k_index))

    ident = jax.jit(lambda r: r)(constructed)
    _assert_dispatch_residual_meta(ident, max_per)
    np.testing.assert_array_equal(_np(ident.token_index), token_index)
    np.testing.assert_array_equal(_np(ident.k_index), k_index)

    node_cls, aux = treedef.node_data()
    assert node_cls is kernels._DispatchResidual
    aux_leaves = jax.tree_util.tree_leaves(aux)
    assert max_per in aux_leaves
    assert not any(hasattr(v, "shape") and getattr(v, "ndim", 0) > 0 for v in aux_leaves)

    def _rebuild(ti, ki):
        return kernels._DispatchResidual(
            token_index=ti, k_index=ki, max_per_expert=max_per
        )

    traced = jax.jit(_rebuild)(jnp.asarray(token_index), jnp.asarray(k_index))
    _assert_dispatch_residual_meta(traced, max_per)
    np.testing.assert_array_equal(_np(traced.token_index), token_index)
    np.testing.assert_array_equal(_np(traced.k_index), k_index)

    tokens, meta = _golden_pad_layout()
    dispatched, residual = kernels.ep_dispatch_fwd(tokens, meta)
    assert type(residual) is kernels._DispatchResidual
    _assert_dispatch_residual_meta(residual, int(_np(residual.token_index).shape[1]))
    got_leaves, _ = jax.tree_util.tree_flatten(residual)
    assert len(got_leaves) == 2
    assert _pytree_leaves_contain_array(got_leaves, residual.token_index)
    assert _pytree_leaves_contain_array(got_leaves, residual.k_index)
    ident_fwd = jax.jit(lambda r: r)(residual)
    _assert_dispatch_residual_meta(ident_fwd, residual.max_per_expert)
    _assert_residual_indices_equal(ident_fwd, residual)
    assert dispatched.shape[0] == meta.n_experts


def test_ep_dispatch_fwd_jit_matches_eager():
    """jax.jit(ep_dispatch_fwd)(tokens, meta) matches eager and the numpy reference at 1e-5."""
    rng = np.random.default_rng(201)
    jfwd = _jdispatch_fwd()
    for name, tokens, meta in _layouts(rng):
        dispatched_e, residual_e = kernels.ep_dispatch_fwd(tokens, meta)
        dispatched_j, residual_j = jfwd(_j32(tokens), meta)
        dispatched_r, residual_r = ref.ep_dispatch_fwd(tokens, meta)
        _close(dispatched_e, dispatched_r, err_msg=name)
        _close(dispatched_j, dispatched_e, err_msg=name)
        _close(dispatched_j, dispatched_r, err_msg=name)
        assert type(residual_e) is kernels._DispatchResidual, name
        assert type(residual_j) is kernels._DispatchResidual, name
        _assert_dispatch_residual_meta(residual_e, int(residual_r.max_per_expert))
        _assert_dispatch_residual_meta(residual_j, int(residual_r.max_per_expert))
        _assert_residual_indices_equal(residual_e, residual_r, err_msg=name)
        _assert_residual_indices_equal(residual_j, residual_r, err_msg=name)
        assert np.any(_np(residual_e.token_index) < 0) or name in (
            "single_token",
            "dup_expert_topk2",
        )


def test_ep_dispatch_fwd_jit_accepts_jax_array_inputs():
    """Traced tokens / meta array fields must not be converted with numpy.asarray."""
    tokens, meta = _golden_pad_layout()
    meta_j = kernels.DispatchMeta(
        expert_ids=jnp.asarray(meta.expert_ids, dtype=jnp.int32),
        probs=jnp.asarray(meta.probs, dtype=jnp.float32),
        racks=jnp.asarray(meta.racks, dtype=jnp.int32),
        n_experts=int(meta.n_experts),
        max_racks=int(meta.max_racks),
        _static_max_per=int(meta._static_max_per) if meta._static_max_per is not None else None,
    )
    dispatched_e, residual_e = kernels.ep_dispatch_fwd(_j32(tokens), meta_j)
    dispatched_j, residual_j = _jdispatch_fwd()(_j32(tokens), meta_j)
    dispatched_r, residual_r = ref.ep_dispatch_fwd(tokens, meta)
    _close(dispatched_e, dispatched_r)
    _close(dispatched_j, dispatched_e)
    _assert_residual_indices_equal(residual_e, residual_r)
    _assert_residual_indices_equal(residual_j, residual_r)
    _assert_dispatch_residual_meta(residual_j, int(residual_r.max_per_expert))


def test_ep_dispatch_fwd_shapes_and_dtypes():
    tokens, meta = _golden_pad_layout()
    n_tokens, d_model = tokens.shape
    dispatched, residual = kernels.ep_dispatch_fwd(tokens, meta)
    max_per = int(residual.max_per_expert)
    assert _np(dispatched).shape == (meta.n_experts, max_per, d_model)
    assert _np(dispatched).dtype == np.float32
    assert _np(residual.token_index).shape == (meta.n_experts, max_per)
    assert _np(residual.k_index).shape == (meta.n_experts, max_per)
    assert np.issubdtype(_np(residual.token_index).dtype, np.integer)
    assert np.issubdtype(_np(residual.k_index).dtype, np.integer)
    assert np.all(_np(residual.token_index)[_np(residual.token_index) < 0] == -1)
    occupied = _np(residual.token_index)[_np(residual.token_index) >= 0]
    assert occupied.min() >= 0
    assert occupied.max() < n_tokens
    dispatched_j, residual_j = _jdispatch_fwd()(_j32(tokens), meta)
    assert _np(dispatched_j).shape == _np(dispatched).shape
    assert _np(dispatched_j).dtype == np.float32


def test_ep_dispatch_fwd_jit_jvp_unpack_residual_finite():
    """Unpacking the jit fwd residual pytree in the autodiff graph stays finite."""
    tokens, meta = _golden_pad_layout()
    jfwd = _jdispatch_fwd()
    dt = _unit(np.random.default_rng(202), tokens.shape)

    def wrapped(tok):
        dispatched, residual = jfwd(tok, meta)
        unused = jnp.sum(jnp.asarray(residual.token_index, dtype=jnp.float32))
        unused = unused + jnp.sum(jnp.asarray(residual.k_index, dtype=jnp.float32))
        return dispatched + jnp.float32(0.0) * unused

    primals, tangents = jax.jvp(wrapped, (_j32(tokens),), (_j32(dt),))
    assert np.all(np.isfinite(_np(primals)))
    assert np.all(np.isfinite(_np(tangents)))


# ---------------------------------------------------------------------------
# Dispatch bwd
# ---------------------------------------------------------------------------


def test_ep_dispatch_bwd_jit_matches_eager():
    """jax.jit(ep_dispatch_bwd)(residual, g) matches eager and the numpy scatter-add at 1e-5."""
    rng = np.random.default_rng(203)
    jfwd = _jdispatch_fwd()
    jbwd = _jdispatch_bwd()
    for name, tokens, meta in _layouts(rng):
        dispatched_e, residual_e = kernels.ep_dispatch_fwd(tokens, meta)
        g = rng.standard_normal(_np(dispatched_e).shape).astype(np.float32)
        grad_e = kernels.ep_dispatch_bwd(residual_e, g)
        _, residual_j = jfwd(_j32(tokens), meta)
        grad_j_from_eager_res = jbwd(residual_e, _j32(g))
        grad_j_from_jit_res = jbwd(residual_j, _j32(g))
        grad_r = ref.ep_dispatch_bwd(residual_e, g)
        n_tokens = tokens.shape[0]
        scatter = _scatter_add_grad_tokens(g, residual_e.token_index, n_tokens)
        _close(grad_e, grad_r, err_msg=name)
        _close(grad_e, scatter, err_msg=name)
        _close(grad_j_from_eager_res, grad_e, err_msg=name)
        _close(grad_j_from_jit_res, grad_e, err_msg=name)
        assert _np(grad_e).shape == tokens.shape
        assert _np(grad_e).dtype == np.float32


def test_ep_dispatch_bwd_matches_numpy_scatter_add():
    """dispatch_bwd is scatter-add of g_dispatched into grad_tokens; pads contribute 0."""
    tokens, meta = _golden_pad_layout()
    dispatched, residual = kernels.ep_dispatch_fwd(tokens, meta)
    g = np.arange(np.prod(_np(dispatched).shape), dtype=np.float32).reshape(
        _np(dispatched).shape
    )
    grad = kernels.ep_dispatch_bwd(residual, g)
    scatter = _scatter_add_grad_tokens(g, residual.token_index, tokens.shape[0])
    _close(grad, scatter)
    _close(grad, ref.ep_dispatch_bwd(residual, g))
    pad = _np(residual.token_index) < 0
    assert np.any(pad)
    # Occupied slots: token 0 <- g[0,0], token 2 <- g[0,1], token 1 <- g[1,0]
    np.testing.assert_allclose(_np(grad)[0], _np(g)[0, 0], **TOL)
    np.testing.assert_allclose(_np(grad)[2], _np(g)[0, 1], **TOL)
    np.testing.assert_allclose(_np(grad)[1], _np(g)[1, 0], **TOL)


def test_ep_dispatch_bwd_finite_diff():
    rng = np.random.default_rng(204)
    _, tokens, meta = _layouts(rng)[1]  # topk2_pad
    dispatched, residual = kernels.ep_dispatch_fwd(tokens, meta)
    g = rng.standard_normal(_np(dispatched).shape).astype(np.float32)
    grad = kernels.ep_dispatch_bwd(residual, g)
    eps = np.float32(1e-3)
    d = _unit(rng, tokens.shape)
    plus, _ = ref.ep_dispatch(tokens + eps * d, meta)
    minus, _ = ref.ep_dispatch(tokens - eps * d, meta)
    fd = (float(np.sum(plus * g)) - float(np.sum(minus * g))) / (2.0 * float(eps))
    analytic = float(np.sum(_np(grad) * d))
    np.testing.assert_allclose(analytic, fd, **FD_TOL)


# ---------------------------------------------------------------------------
# Combine fwd residual pytree + jit
# ---------------------------------------------------------------------------


def test_ep_combine_fwd_jit_matches_eager():
    """jax.jit(ep_combine_fwd) matches eager and the numpy reference at 1e-5."""
    rng = np.random.default_rng(205)
    jdf = _jdispatch_fwd()
    jcf = _jcombine_fwd()
    for name, tokens, meta in _layouts(rng):
        dispatched, dres = kernels.ep_dispatch_fwd(tokens, meta)
        expert_out = rng.standard_normal(_np(dispatched).shape).astype(np.float32)
        combined_e, cres_e = kernels.ep_combine_fwd(expert_out, meta, dres)
        _, dres_j = jdf(_j32(tokens), meta)
        combined_j, cres_j = jcf(_j32(expert_out), meta, dres_j)
        combined_r, _ = ref.ep_combine_fwd(expert_out, meta, dres)
        _close(combined_e, combined_r, err_msg=name)
        _close(combined_j, combined_e, err_msg=name)
        _close(combined_j, combined_r, err_msg=name)
        assert _np(combined_e).shape == tokens.shape
        assert _np(combined_e).dtype == np.float32
        _assert_pytree_array_leaves(cres_e)
        _assert_pytree_array_leaves(cres_j)


def test_ep_combine_fwd_residual_is_pytree():
    """combine_fwd residual flattens to array leaves; dummy replace; identity jit."""
    tokens, meta = _golden_pad_layout()
    dispatched, dres = kernels.ep_dispatch_fwd(tokens, meta)
    combined, residual = kernels.ep_combine_fwd(dispatched, meta, dres)
    leaves, treedef = _assert_pytree_array_leaves(residual, min_leaves=1)
    dummy_leaves = [np.zeros_like(_np(leaf)) for leaf in leaves]
    dummy = jax.tree_util.tree_unflatten(treedef, dummy_leaves)
    dummy_leaves2, _ = jax.tree_util.tree_flatten(dummy)
    assert len(dummy_leaves2) == len(leaves)
    ident = jax.jit(lambda r: r)(residual)
    ident_leaves, ident_td = jax.tree_util.tree_flatten(ident)
    assert ident_td.num_leaves == treedef.num_leaves
    for a, b in zip(leaves, ident_leaves, strict=True):
        if np.issubdtype(_np(a).dtype, np.floating):
            np.testing.assert_allclose(_np(a), _np(b), **TOL)
        else:
            np.testing.assert_array_equal(_np(a), _np(b))
    combined_j, residual_j = _jcombine_fwd()(_j32(dispatched), meta, dres)
    _close(combined_j, combined)
    _assert_pytree_array_leaves(residual_j)
    ident_j = jax.jit(lambda r: r)(residual_j)
    _assert_pytree_array_leaves(ident_j)


def test_ep_combine_fwd_shapes_and_dtypes():
    tokens, meta = _golden_pad_layout()
    dispatched, dres = kernels.ep_dispatch_fwd(tokens, meta)
    combined, residual = kernels.ep_combine_fwd(dispatched, meta, dres)
    assert _np(combined).shape == tokens.shape
    assert _np(combined).dtype == np.float32
    combined_j, _ = _jcombine_fwd()(_j32(dispatched), meta, dres)
    assert _np(combined_j).shape == tokens.shape
    assert _np(combined_j).dtype == np.float32
    leaves, _ = jax.tree_util.tree_flatten(residual)
    assert all(hasattr(leaf, "shape") for leaf in leaves)


# ---------------------------------------------------------------------------
# Combine bwd
# ---------------------------------------------------------------------------


def test_ep_combine_bwd_jit_matches_eager():
    """jax.jit(ep_combine_bwd) matches eager and the numpy weighted gather at 1e-5."""
    rng = np.random.default_rng(206)
    jcf = _jcombine_fwd()
    jcb = _jcombine_bwd()
    for name, tokens, meta in _layouts(rng):
        dispatched, dres = kernels.ep_dispatch_fwd(tokens, meta)
        expert_out = rng.standard_normal(_np(dispatched).shape).astype(np.float32)
        _, cres_e = kernels.ep_combine_fwd(expert_out, meta, dres)
        g = rng.standard_normal(tokens.shape).astype(np.float32)
        grad_e = kernels.ep_combine_bwd(cres_e, g)
        _, cres_j = jcf(_j32(expert_out), meta, dres)
        grad_j_from_eager_res = jcb(cres_e, _j32(g))
        grad_j_from_jit_res = jcb(cres_j, _j32(g))
        grad_r = ref.ep_combine_vjp(expert_out, meta, dres, g)
        gather = _weighted_gather_grad_expert(
            g, meta.probs, dres.token_index, dres.k_index
        )
        _close(grad_e, grad_r, err_msg=name)
        _close(grad_e, gather, err_msg=name)
        _close(grad_j_from_eager_res, grad_e, err_msg=name)
        _close(grad_j_from_jit_res, grad_e, err_msg=name)
        assert _np(grad_e).shape == _np(expert_out).shape
        assert _np(grad_e).dtype == np.float32
        pad = _np(dres.token_index) < 0
        if np.any(pad):
            np.testing.assert_array_equal(_np(grad_e)[pad], 0)


def test_ep_combine_bwd_matches_weighted_gather():
    tokens, meta = _golden_pad_layout()
    dispatched, dres = kernels.ep_dispatch_fwd(tokens, meta)
    _, cres = kernels.ep_combine_fwd(dispatched, meta, dres)
    g = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
    grad = kernels.ep_combine_bwd(cres, g)
    gather = _weighted_gather_grad_expert(g, meta.probs, dres.token_index, dres.k_index)
    _close(grad, gather)
    _close(grad, ref.ep_combine_vjp(dispatched, meta, dres, g))
    # unit probs: occupied slots copy g[t]; pad is 0
    np.testing.assert_allclose(_np(grad)[0, 0], g[0], **TOL)
    np.testing.assert_allclose(_np(grad)[0, 1], g[2], **TOL)
    np.testing.assert_allclose(_np(grad)[1, 0], g[1], **TOL)
    np.testing.assert_array_equal(_np(grad)[1, 1], 0)


def test_ep_combine_bwd_finite_diff():
    rng = np.random.default_rng(207)
    _, tokens, meta = _layouts(rng)[1]
    dispatched, dres = kernels.ep_dispatch_fwd(tokens, meta)
    expert_out = rng.standard_normal(_np(dispatched).shape).astype(np.float32)
    _, cres = kernels.ep_combine_fwd(expert_out, meta, dres)
    g = rng.standard_normal(tokens.shape).astype(np.float32)
    grad = kernels.ep_combine_bwd(cres, g)
    eps = np.float32(1e-3)
    d = _unit(rng, expert_out.shape)
    plus = ref.ep_combine(expert_out + eps * d, meta, dres)
    minus = ref.ep_combine(expert_out - eps * d, meta, dres)
    fd = (float(np.sum(plus * g)) - float(np.sum(minus * g))) / (2.0 * float(eps))
    analytic = float(np.sum(_np(grad) * d))
    np.testing.assert_allclose(analytic, fd, **FD_TOL)


# ---------------------------------------------------------------------------
# Golden, roundtrip, multi-shape
# ---------------------------------------------------------------------------


def test_ep_fwd_bwd_golden_padded_slots():
    tokens, meta = _golden_pad_layout()
    dispatched, dres = kernels.ep_dispatch_fwd(tokens, meta)
    expected = np.array(
        [[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [0.0, 0.0]]], dtype=np.float32
    )
    _close(dispatched, expected)
    np.testing.assert_array_equal(_np(dres.token_index), np.array([[0, 2], [1, -1]]))
    np.testing.assert_array_equal(_np(dres.k_index), np.array([[0, 0], [0, -1]]))
    assert dres.max_per_expert == 2
    assert type(dres.max_per_expert) is int

    dispatched_j, dres_j = _jdispatch_fwd()(_j32(tokens), meta)
    _close(dispatched_j, expected)
    _assert_residual_indices_equal(dres_j, dres)

    g_disp = np.ones_like(expected, dtype=np.float32)
    grad_tokens = kernels.ep_dispatch_bwd(dres, g_disp)
    _close(grad_tokens, np.ones_like(tokens))
    _close(_jdispatch_bwd()(dres, _j32(g_disp)), grad_tokens)

    combined, cres = kernels.ep_combine_fwd(dispatched, meta, dres)
    _close(combined, tokens)
    combined_j, _ = _jcombine_fwd()(_j32(dispatched), meta, dres)
    _close(combined_j, tokens)

    g_comb = np.ones_like(tokens, dtype=np.float32)
    grad_expert = kernels.ep_combine_bwd(cres, g_comb)
    expected_ge = np.array(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]], dtype=np.float32
    )
    _close(grad_expert, expected_ge)
    _close(_jcombine_bwd()(cres, _j32(g_comb)), grad_expert)


def test_ep_fwd_pair_roundtrip_identity_when_mass_one():
    rng = np.random.default_rng(208)
    for name, tokens, meta in _layouts(rng):
        mass = _np(meta.probs).sum(axis=1)
        dispatched, dres = kernels.ep_dispatch_fwd(tokens, meta)
        combined, _ = kernels.ep_combine_fwd(dispatched, meta, dres)
        expected = tokens * mass[:, None]
        _close(combined, expected, err_msg=name)
        dispatched_j, dres_j = _jdispatch_fwd()(_j32(tokens), meta)
        combined_j, _ = _jcombine_fwd()(dispatched_j, meta, dres_j)
        _close(combined_j, expected, err_msg=name)


def test_ep_fwd_bwd_multiple_shapes():
    rng = np.random.default_rng(209)
    jdf = _jdispatch_fwd()
    jdb = _jdispatch_bwd()
    jcf = _jcombine_fwd()
    jcb = _jcombine_bwd()
    shapes_seen = set()
    for name, tokens, meta in _layouts(rng):
        shapes_seen.add(
            (
                tokens.shape[0],
                meta.n_experts,
                _np(meta.expert_ids).shape[1],
                tokens.shape[1],
            )
        )
        dispatched, dres = kernels.ep_dispatch_fwd(tokens, meta)
        g_disp = rng.standard_normal(_np(dispatched).shape).astype(np.float32)
        gt_e = kernels.ep_dispatch_bwd(dres, g_disp)
        dispatched_j, dres_j = jdf(_j32(tokens), meta)
        gt_j = jdb(dres_j, _j32(g_disp))
        _close(dispatched_j, dispatched, err_msg=name)
        _close(gt_j, gt_e, err_msg=name)
        _close(gt_e, ref.ep_dispatch_bwd(dres, g_disp), err_msg=name)
        expert_out = rng.standard_normal(_np(dispatched).shape).astype(np.float32)
        combined_e, cres_e = kernels.ep_combine_fwd(expert_out, meta, dres)
        g_comb = rng.standard_normal(tokens.shape).astype(np.float32)
        ge_e = kernels.ep_combine_bwd(cres_e, g_comb)
        combined_j, cres_j = jcf(_j32(expert_out), meta, dres_j)
        ge_j = jcb(cres_j, _j32(g_comb))
        _close(combined_j, combined_e, err_msg=name)
        _close(ge_j, ge_e, err_msg=name)
        _close(ge_e, ref.ep_combine_vjp(expert_out, meta, dres, g_comb), err_msg=name)
        pad = _np(dres.token_index) < 0
        if np.any(pad):
            np.testing.assert_array_equal(_np(ge_e)[pad], 0)
    assert len(shapes_seen) >= 4


def test_ep_fwd_bwd_does_not_host_convert_under_composed_jit():
    """Composed jit of the four public functions must not numpy.asarray tracers."""
    rng = np.random.default_rng(210)
    _, tokens, meta = _layouts(rng)[1]
    dispatched_r, dres_r = ref.ep_dispatch_fwd(tokens, meta)
    expert_out = rng.standard_normal(_np(dispatched_r).shape).astype(np.float32)
    g_disp = rng.standard_normal(_np(dispatched_r).shape).astype(np.float32)
    g_comb = rng.standard_normal(tokens.shape).astype(np.float32)

    def pipeline(tok, expert, gd, gc):
        dispatched, dres = kernels.ep_dispatch_fwd(tok, meta)
        gt = kernels.ep_dispatch_bwd(dres, gd)
        combined, cres = kernels.ep_combine_fwd(expert, meta, dres)
        ge = kernels.ep_combine_bwd(cres, gc)
        return dispatched, gt, combined, ge

    dispatched, gt, combined, ge = jax.jit(pipeline)(
        _j32(tokens), _j32(expert_out), _j32(g_disp), _j32(g_comb)
    )
    _close(dispatched, dispatched_r)
    _close(gt, ref.ep_dispatch_bwd(dres_r, g_disp))
    combined_r, _ = ref.ep_combine_fwd(expert_out, meta, dres_r)
    _close(combined, combined_r)
    _close(ge, ref.ep_combine_vjp(expert_out, meta, dres_r, g_comb))
