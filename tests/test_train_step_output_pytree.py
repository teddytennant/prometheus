"""A2-step-output-pytree oracle: ``jax.jit(train_step)`` must take a ``Batch`` and
return a pytree ``StepOutput``.

``train.Batch``, ``train.LossBreakdown``, and ``train.StepOutput`` must be
``jax.tree_util`` registered dataclasses so ``jax.jit`` can take a ``Batch`` and
return a ``StepOutput`` without unpacking fields to tuples. Analog of
``Fp8Meta`` / ``NoisyLatent`` / ``DispatchMeta``.

``Batch`` fields ``tokens``, ``loss_mask``, ``positions`` are data (arrays). No
meta. ``LossBreakdown`` fields ``ce``, ``mtp``, ``z``, ``total`` are data
(arrays). No meta. ``StepOutput`` fields ``params``, ``opt_state``, ``loss``,
``grad_norm`` are data (dicts of arrays, nested ``LossBreakdown``, array).
``lr`` (Python float) and ``step`` (Python int) are meta.

These tests must fail on the current unregistered dataclasses (TypeError /
"not a valid JAX type" / treating the object as one leaf) and pass once they
are registered. Analog math is unchanged. Do not copy production math into this
file; compare jitted ``train_step`` against eager production at 1e-5.

Python scalars ``step`` / ``lr`` and ``TrainConfig`` / ``ModelConfig`` stay
host-side as in A2-train-step-jit. Tests never register pytrees. Do not jit
``apply_precision`` / ``init_opt_state`` / ``wsd_lr`` as entry points. Do not
register ``TrainConfig``. Do not mark gpu. Production must not import
``tests``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import model
import train

TOL = dict(rtol=1e-5, atol=1e-5)
STEP = 1
HOST_LR = 2.0e-4
HOST_STEP = 7

BATCH_DATA_FIELDS = ("tokens", "loss_mask", "positions")
LOSS_DATA_FIELDS = ("ce", "mtp", "z", "total")


def _np(x: object) -> np.ndarray:
    return np.asarray(x)


def _close(got: object, exp: object) -> None:
    """FP32 parity at 1e-5. Do not use pytest.approx on jax scalars."""
    np.testing.assert_allclose(_np(got).astype(np.float32), _np(exp).astype(np.float32), **TOL)


def _leaf_matches(leaf: object, value: object, *, forbidden: tuple[type, ...]) -> bool:
    """True if ``value`` appears as a numeric pytree leaf, not a dataclass object."""
    if type(leaf) in forbidden:
        return False
    if not hasattr(leaf, "shape"):
        return False
    got = _np(leaf)
    target = _np(value)
    if got.shape != target.shape:
        return False
    if np.issubdtype(got.dtype, np.floating) or np.issubdtype(target.dtype, np.floating):
        return np.allclose(got.astype(np.float64), target.astype(np.float64), **TOL)
    return np.array_equal(got, target)


def _pytree_leaves_contain_array(leaves, arr, *, forbidden: tuple[type, ...]) -> bool:
    """True if ``arr`` appears as a pytree leaf (value + shape), not buried in aux."""
    return any(_leaf_matches(leaf, arr, forbidden=forbidden) for leaf in leaves)


def _double_floats(x: object) -> object:
    if hasattr(x, "dtype") and np.issubdtype(np.asarray(x).dtype, np.floating):
        return x * np.float32(2.0)
    return x


def _assert_no_dataclass_leaf(leaves, cls: type) -> None:
    assert not any(type(leaf) is cls for leaf in leaves), (
        f"{cls.__name__} must not be a pytree leaf; jax.jit would treat it as an "
        "abstract array. Register it so array fields are leaves."
    )


def _assert_aux_has_no_arrays(aux: object) -> None:
    aux_leaves = jax.tree_util.tree_leaves(aux)
    assert not any(hasattr(v, "shape") and getattr(v, "ndim", 0) > 0 for v in aux_leaves), (
        "array data must not be stuffed into aux"
    )


def _assert_python_meta_lr_step(obj: train.StepOutput, *, lr: float, step: int) -> None:
    assert type(obj) is train.StepOutput
    assert type(obj.lr) is float, f"lr must stay a Python float, got {type(obj.lr)}"
    assert type(obj.step) is int, f"step must stay a Python int, got {type(obj.step)}"
    assert not isinstance(obj.lr, jax.Array)
    assert not isinstance(obj.step, jax.Array)
    assert obj.lr == lr
    assert obj.step == step


def _assert_array_tree_close(got: object, exp: object) -> None:
    if isinstance(got, dict):
        assert isinstance(exp, dict)
        assert set(got) == set(exp)
        for key in got:
            _assert_array_tree_close(got[key], exp[key])
        return
    if isinstance(got, (list, tuple)):
        assert type(got) is type(exp) or isinstance(exp, (list, tuple))
        assert len(got) == len(exp)
        for a, b in zip(got, exp, strict=True):
            _assert_array_tree_close(a, b)
        return
    assert _np(got).shape == _np(exp).shape
    _close(got, exp)


def _tiny_batch(mcfg: model.ModelConfig, batch: int = 2, seq: int = 8) -> train.Batch:
    """Same construction as ``tests/test_train.py`` / ``tests/test_train_step_jit.py``."""
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, mcfg.vocab_size, size=(batch, seq), dtype=np.int32)
    loss_mask = np.ones((batch, seq), dtype=np.float32)
    loss_mask[:, 0] = 0.0
    positions = np.broadcast_to(np.arange(seq, dtype=np.int32), (batch, seq)).copy()
    return train.Batch(tokens=tokens, loss_mask=loss_mask, positions=positions)


def _as_batch(batch: train.Batch) -> train.Batch:
    return train.Batch(
        tokens=jnp.asarray(batch.tokens, dtype=jnp.int32),
        loss_mask=jnp.asarray(batch.loss_mask, dtype=jnp.float32),
        positions=jnp.asarray(batch.positions, dtype=jnp.int32),
    )


def _sample_batch() -> train.Batch:
    return train.Batch(
        tokens=np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32),
        loss_mask=np.array([[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]], dtype=np.float32),
        positions=np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32),
    )


def _sample_loss() -> train.LossBreakdown:
    return train.LossBreakdown(
        ce=np.array(1.25, dtype=np.float32),
        mtp=np.array(0.25, dtype=np.float32),
        z=np.array(0.05, dtype=np.float32),
        total=np.array(1.55, dtype=np.float32),
    )


def _sample_step_output() -> train.StepOutput:
    params = {
        "embed": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "layers": [{"W": np.array([[0.5, -0.5]], dtype=np.float32)}],
    }
    opt_state = {
        "embed": {
            "m": np.zeros((2, 2), dtype=np.float32),
            "v": np.ones((2, 2), dtype=np.float32),
        },
        "layers": [{"W": {"momentum": np.array([[0.1, 0.2]], dtype=np.float32)}}],
    }
    return train.StepOutput(
        params=params,
        opt_state=opt_state,
        loss=_sample_loss(),
        lr=HOST_LR,
        grad_norm=np.array(3.5, dtype=np.float32),
        step=HOST_STEP,
    )


def _setup() -> tuple:
    mcfg = model.tiny_config()
    tcfg = train.tiny_train_config()
    params = model.init_params(mcfg, 0)
    batch = _as_batch(_tiny_batch(mcfg))
    return mcfg, tcfg, params, batch


def _assert_step_output_match(got: train.StepOutput, eager: train.StepOutput) -> None:
    assert type(got) is train.StepOutput
    assert type(got.loss) is train.LossBreakdown
    _close(got.loss.ce, eager.loss.ce)
    _close(got.loss.mtp, eager.loss.mtp)
    _close(got.loss.z, eager.loss.z)
    _close(got.loss.total, eager.loss.total)
    _close(got.grad_norm, eager.grad_norm)
    _assert_array_tree_close(got.params, eager.params)
    _assert_array_tree_close(got.opt_state, eager.opt_state)
    assert type(got.lr) is float
    assert type(got.step) is int
    assert not isinstance(got.lr, jax.Array)
    assert not isinstance(got.step, jax.Array)
    _close(got.lr, eager.lr)
    assert int(got.step) == int(eager.step) == STEP


# ---------------------------------------------------------------------------
# 1. Each dataclass is a registered pytree
# ---------------------------------------------------------------------------


def test_batch_is_registered_jax_pytree() -> None:
    """Batch flattens to tokens/loss_mask/positions array leaves so jax.jit can take it.

    The whole object must not be one leaf, and the arrays must not be stuffed
    into aux. Dummy leaves replace data. tree_map on floats doubles loss_mask
    and leaves integer tokens/positions. jax.jit(lambda b: b)(batch) roundtrips.
    """
    batch = _sample_batch()
    forbidden = (train.Batch,)

    leaves, treedef = jax.tree_util.tree_flatten(batch)
    _assert_no_dataclass_leaf(leaves, train.Batch)
    assert len(leaves) == len(BATCH_DATA_FIELDS), (
        "Batch must flatten to one leaf per data field "
        f"(got {len(leaves)} leaves, expected {len(BATCH_DATA_FIELDS)})"
    )
    for name in BATCH_DATA_FIELDS:
        assert _pytree_leaves_contain_array(
            leaves, getattr(batch, name), forbidden=forbidden
        ), (
            f"{name} must be a pytree data-field leaf (value match); a dummy "
            "register or aux-only flatten is not enough"
        )

    node = treedef.node_data()
    assert node is not None, "Batch must be a registered pytree node, not a generic leaf"
    cls, aux = node
    assert cls is train.Batch
    assert aux == (), "Batch has no meta fields; arrays must not be stuffed into aux"
    _assert_aux_has_no_arrays(aux)

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is train.Batch
    np.testing.assert_array_equal(_np(rebuilt.tokens), batch.tokens)
    _close(rebuilt.loss_mask, batch.loss_mask)
    np.testing.assert_array_equal(_np(rebuilt.positions), batch.positions)

    dummy_leaves = [np.zeros_like(_np(leaf)) for leaf in leaves]
    dummy = jax.tree_util.tree_unflatten(treedef, dummy_leaves)
    assert type(dummy) is train.Batch
    np.testing.assert_array_equal(_np(dummy.tokens), np.zeros_like(batch.tokens))
    np.testing.assert_array_equal(_np(dummy.loss_mask), np.zeros_like(batch.loss_mask))
    np.testing.assert_array_equal(_np(dummy.positions), np.zeros_like(batch.positions))

    mapped = jax.tree_util.tree_map(_double_floats, batch)
    assert type(mapped) is train.Batch
    np.testing.assert_array_equal(_np(mapped.tokens), batch.tokens)
    _close(mapped.loss_mask, np.float32(2.0) * batch.loss_mask)
    np.testing.assert_array_equal(_np(mapped.positions), batch.positions)

    ident = jax.jit(lambda b: b)(batch)
    assert type(ident) is train.Batch
    np.testing.assert_array_equal(_np(ident.tokens), batch.tokens)
    _close(ident.loss_mask, batch.loss_mask)
    np.testing.assert_array_equal(_np(ident.positions), batch.positions)
    assert _np(ident.tokens).dtype == np.int32
    assert _np(ident.loss_mask).dtype == np.float32
    assert _np(ident.positions).dtype == np.int32
    assert _np(ident.tokens).shape == (2, 4)
    assert _np(ident.loss_mask).shape == (2, 4)
    assert _np(ident.positions).shape == (2, 4)


def test_loss_breakdown_is_registered_jax_pytree() -> None:
    """LossBreakdown flattens to ce/mtp/z/total array leaves so jax.jit can return it.

    The whole object must not be one leaf, and the arrays must not be stuffed
    into aux. Dummy leaves replace data. tree_map on floats doubles every field.
    jax.jit(lambda x: x)(loss) roundtrips type + values.
    """
    loss = _sample_loss()
    forbidden = (train.LossBreakdown,)

    leaves, treedef = jax.tree_util.tree_flatten(loss)
    _assert_no_dataclass_leaf(leaves, train.LossBreakdown)
    assert len(leaves) == len(LOSS_DATA_FIELDS), (
        "LossBreakdown must flatten to one leaf per data field "
        f"(got {len(leaves)} leaves, expected {len(LOSS_DATA_FIELDS)})"
    )
    for name in LOSS_DATA_FIELDS:
        assert _pytree_leaves_contain_array(
            leaves, getattr(loss, name), forbidden=forbidden
        ), (
            f"{name} must be a pytree data-field leaf (value match); a dummy "
            "register or aux-only flatten is not enough"
        )

    node = treedef.node_data()
    assert node is not None, (
        "LossBreakdown must be a registered pytree node, not a generic leaf"
    )
    cls, aux = node
    assert cls is train.LossBreakdown
    assert aux == (), "LossBreakdown has no meta fields; arrays must not be stuffed into aux"
    _assert_aux_has_no_arrays(aux)

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is train.LossBreakdown
    for name in LOSS_DATA_FIELDS:
        _close(getattr(rebuilt, name), getattr(loss, name))

    dummy_leaves = [np.zeros_like(_np(leaf)) for leaf in leaves]
    dummy = jax.tree_util.tree_unflatten(treedef, dummy_leaves)
    assert type(dummy) is train.LossBreakdown
    for name in LOSS_DATA_FIELDS:
        np.testing.assert_array_equal(_np(getattr(dummy, name)), np.zeros((), dtype=np.float32))

    mapped = jax.tree_util.tree_map(_double_floats, loss)
    assert type(mapped) is train.LossBreakdown
    for name in LOSS_DATA_FIELDS:
        _close(getattr(mapped, name), np.float32(2.0) * getattr(loss, name))

    ident = jax.jit(lambda x: x)(loss)
    assert type(ident) is train.LossBreakdown
    for name in LOSS_DATA_FIELDS:
        _close(getattr(ident, name), getattr(loss, name))
        assert _np(getattr(ident, name)).dtype == np.float32
        assert _np(getattr(ident, name)).shape == ()


def test_step_output_is_registered_jax_pytree() -> None:
    """StepOutput flattens to array / dict-of-array leaves so jax.jit can return it.

    The dataclass itself must not be a pytree leaf. Arrays must not be stuffed
    into aux. Dummy leaves replace data; meta (lr, step) survive unflatten.
    tree_map on floats doubles floating array leaves and leaves meta.
    jax.jit(lambda x: x)(out) roundtrips type + values. lr stays a Python
    float and step stays a Python int (not jax.Array).
    """
    out = _sample_step_output()
    forbidden = (train.StepOutput, train.LossBreakdown)

    leaves, treedef = jax.tree_util.tree_flatten(out)
    _assert_no_dataclass_leaf(leaves, train.StepOutput)
    _assert_no_dataclass_leaf(leaves, train.LossBreakdown)
    assert not any(type(leaf) is float for leaf in leaves)
    assert not any(type(leaf) is int for leaf in leaves)

    n_params = len(jax.tree_util.tree_leaves(out.params))
    n_opt = len(jax.tree_util.tree_leaves(out.opt_state))
    assert len(leaves) == n_params + n_opt + len(LOSS_DATA_FIELDS) + 1, (
        "StepOutput data leaves are params + opt_state + loss.ce/mtp/z/total + "
        f"grad_norm; lr/step are meta (got {len(leaves)} leaves)"
    )
    assert _pytree_leaves_contain_array(leaves, out.params["embed"], forbidden=forbidden)
    assert _pytree_leaves_contain_array(leaves, out.params["layers"][0]["W"], forbidden=forbidden)
    assert _pytree_leaves_contain_array(leaves, out.opt_state["embed"]["m"], forbidden=forbidden)
    assert _pytree_leaves_contain_array(leaves, out.opt_state["embed"]["v"], forbidden=forbidden)
    assert _pytree_leaves_contain_array(
        leaves, out.opt_state["layers"][0]["W"]["momentum"], forbidden=forbidden
    )
    assert _pytree_leaves_contain_array(leaves, out.grad_norm, forbidden=forbidden)
    for name in LOSS_DATA_FIELDS:
        assert _pytree_leaves_contain_array(
            leaves, getattr(out.loss, name), forbidden=forbidden
        ), f"loss.{name} must be a pytree data-field leaf of StepOutput"

    node = treedef.node_data()
    assert node is not None, "StepOutput must be a registered pytree node, not a generic leaf"
    cls, aux = node
    assert cls is train.StepOutput
    _assert_aux_has_no_arrays(aux)
    aux_leaves = jax.tree_util.tree_leaves(aux)
    assert HOST_STEP in aux_leaves, "step must live in meta/aux, not as a traced leaf"
    assert any(v == HOST_LR or np.asarray(v) == np.asarray(HOST_LR) for v in aux_leaves), (
        "lr must live in meta/aux, not as a traced leaf"
    )

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is train.StepOutput
    assert type(rebuilt.loss) is train.LossBreakdown
    _assert_array_tree_close(rebuilt.params, out.params)
    _assert_array_tree_close(rebuilt.opt_state, out.opt_state)
    for name in LOSS_DATA_FIELDS:
        _close(getattr(rebuilt.loss, name), getattr(out.loss, name))
    _close(rebuilt.grad_norm, out.grad_norm)
    _assert_python_meta_lr_step(rebuilt, lr=HOST_LR, step=HOST_STEP)

    dummy_leaves = [np.zeros_like(_np(leaf)) for leaf in leaves]
    dummy = jax.tree_util.tree_unflatten(treedef, dummy_leaves)
    assert type(dummy) is train.StepOutput
    assert type(dummy.loss) is train.LossBreakdown
    np.testing.assert_array_equal(_np(dummy.params["embed"]), np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(
        _np(dummy.params["layers"][0]["W"]), np.zeros((1, 2), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        _np(dummy.opt_state["embed"]["m"]), np.zeros((2, 2), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        _np(dummy.opt_state["embed"]["v"]), np.zeros((2, 2), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        _np(dummy.opt_state["layers"][0]["W"]["momentum"]),
        np.zeros((1, 2), dtype=np.float32),
    )
    zero_scalar = np.zeros((), dtype=np.float32)
    for name in LOSS_DATA_FIELDS:
        np.testing.assert_array_equal(_np(getattr(dummy.loss, name)), zero_scalar)
    np.testing.assert_array_equal(_np(dummy.grad_norm), np.zeros((), dtype=np.float32))
    _assert_python_meta_lr_step(dummy, lr=HOST_LR, step=HOST_STEP)

    mapped = jax.tree_util.tree_map(_double_floats, out)
    assert type(mapped) is train.StepOutput
    assert type(mapped.loss) is train.LossBreakdown
    _close(mapped.params["embed"], np.float32(2.0) * out.params["embed"])
    _close(mapped.params["layers"][0]["W"], np.float32(2.0) * out.params["layers"][0]["W"])
    _close(mapped.opt_state["embed"]["m"], np.float32(2.0) * out.opt_state["embed"]["m"])
    _close(mapped.opt_state["embed"]["v"], np.float32(2.0) * out.opt_state["embed"]["v"])
    _close(
        mapped.opt_state["layers"][0]["W"]["momentum"],
        np.float32(2.0) * out.opt_state["layers"][0]["W"]["momentum"],
    )
    for name in LOSS_DATA_FIELDS:
        _close(getattr(mapped.loss, name), np.float32(2.0) * getattr(out.loss, name))
    _close(mapped.grad_norm, np.float32(2.0) * out.grad_norm)
    _assert_python_meta_lr_step(mapped, lr=HOST_LR, step=HOST_STEP)

    ident = jax.jit(lambda x: x)(out)
    assert type(ident) is train.StepOutput
    assert type(ident.loss) is train.LossBreakdown
    _assert_array_tree_close(ident.params, out.params)
    _assert_array_tree_close(ident.opt_state, out.opt_state)
    for name in LOSS_DATA_FIELDS:
        _close(getattr(ident.loss, name), getattr(out.loss, name))
        assert _np(getattr(ident.loss, name)).dtype == np.float32
    _close(ident.grad_norm, out.grad_norm)
    _assert_python_meta_lr_step(ident, lr=HOST_LR, step=HOST_STEP)
    assert _np(ident.params["embed"]).dtype == np.float32
    assert _np(ident.grad_norm).dtype == np.float32


def test_step_output_lr_and_step_are_meta_not_traced() -> None:
    """lr/step are meta: same array shapes keep the same leaf structure; not traced."""
    out_a = _sample_step_output()
    out_b = train.StepOutput(
        params=out_a.params,
        opt_state=out_a.opt_state,
        loss=out_a.loss,
        lr=1.0e-3,
        grad_norm=out_a.grad_norm,
        step=99,
    )

    leaves_a, td_a = jax.tree_util.tree_flatten(out_a)
    leaves_b, td_b = jax.tree_util.tree_flatten(out_b)
    _assert_no_dataclass_leaf(leaves_a, train.StepOutput)
    _assert_no_dataclass_leaf(leaves_a, train.LossBreakdown)
    assert len(leaves_a) == len(leaves_b)
    assert td_a.children() == td_b.children()
    assert td_a.num_leaves == td_b.num_leaves
    for leaf in leaves_a:
        assert not isinstance(leaf, (int, float)) or hasattr(leaf, "shape")
        assert type(leaf) is not train.StepOutput
        assert hasattr(leaf, "shape")

    node_cls, aux = td_a.node_data()
    assert node_cls is train.StepOutput
    aux_leaves = jax.tree_util.tree_leaves(aux)
    assert HOST_STEP in aux_leaves
    assert 99 not in aux_leaves
    assert any(v == HOST_LR or np.asarray(v) == np.asarray(HOST_LR) for v in aux_leaves)
    _assert_aux_has_no_arrays(aux)

    ident = jax.jit(lambda x: x)(out_a)
    _assert_python_meta_lr_step(ident, lr=HOST_LR, step=HOST_STEP)
    ident_b = jax.jit(lambda x: x)(out_b)
    _assert_python_meta_lr_step(ident_b, lr=1.0e-3, step=99)


# ---------------------------------------------------------------------------
# 4. Nested: flattening StepOutput flattens LossBreakdown fields as leaves
# ---------------------------------------------------------------------------


def test_step_output_flatten_nests_loss_breakdown() -> None:
    """Flattening a StepOutput must flatten LossBreakdown.loss fields as leaves.

    ce/mtp/z/total appear as array leaves of the StepOutput tree, not as one
    LossBreakdown leaf.
    """
    out = _sample_step_output()
    forbidden = (train.StepOutput, train.LossBreakdown)
    leaves, treedef = jax.tree_util.tree_flatten(out)
    _assert_no_dataclass_leaf(leaves, train.StepOutput)
    _assert_no_dataclass_leaf(leaves, train.LossBreakdown)
    for name in LOSS_DATA_FIELDS:
        assert _pytree_leaves_contain_array(
            leaves, getattr(out.loss, name), forbidden=forbidden
        ), (
            f"loss.{name} must appear as an array leaf of the StepOutput tree, "
            "not as one LossBreakdown leaf"
        )
    n_params = len(jax.tree_util.tree_leaves(out.params))
    n_opt = len(jax.tree_util.tree_leaves(out.opt_state))
    assert len(leaves) == n_params + n_opt + 4 + 1

    node = treedef.node_data()
    assert node is not None
    cls, _aux = node
    assert cls is train.StepOutput


# ---------------------------------------------------------------------------
# 2. Construct from traced array fields under jit
# ---------------------------------------------------------------------------


def test_batch_construct_from_traced_fields_under_jit() -> None:
    """Constructing Batch from traced array fields is legal under jax.jit."""
    batch = _sample_batch()
    tokens = jnp.asarray(batch.tokens)
    loss_mask = jnp.asarray(batch.loss_mask)
    positions = jnp.asarray(batch.positions)

    def _rebuild(tok, mask, pos):
        return train.Batch(tokens=tok, loss_mask=mask, positions=pos)

    got = jax.jit(_rebuild)(tokens, loss_mask, positions)
    assert type(got) is train.Batch
    np.testing.assert_array_equal(_np(got.tokens), batch.tokens)
    _close(got.loss_mask, batch.loss_mask)
    np.testing.assert_array_equal(_np(got.positions), batch.positions)
    assert _np(got.tokens).dtype == np.int32
    assert _np(got.loss_mask).dtype == np.float32
    assert _np(got.positions).dtype == np.int32
    assert _np(got.tokens).shape == (2, 4)


def test_loss_breakdown_construct_from_traced_fields_under_jit() -> None:
    """Constructing LossBreakdown from traced array fields is legal under jax.jit."""
    loss = _sample_loss()

    def _rebuild(ce, mtp, z, total):
        return train.LossBreakdown(ce=ce, mtp=mtp, z=z, total=total)

    got = jax.jit(_rebuild)(
        jnp.asarray(loss.ce),
        jnp.asarray(loss.mtp),
        jnp.asarray(loss.z),
        jnp.asarray(loss.total),
    )
    assert type(got) is train.LossBreakdown
    for name in LOSS_DATA_FIELDS:
        _close(getattr(got, name), getattr(loss, name))
        assert _np(getattr(got, name)).dtype == np.float32
        assert _np(getattr(got, name)).shape == ()


def test_step_output_construct_from_traced_fields_under_jit() -> None:
    """Constructing StepOutput from traced arrays with host lr/step is legal under jit."""
    out = _sample_step_output()
    params = jax.tree_util.tree_map(jnp.asarray, out.params)
    opt_state = jax.tree_util.tree_map(jnp.asarray, out.opt_state)

    def _rebuild(params, opt_state, ce, mtp, z, total, grad_norm):
        return train.StepOutput(
            params=params,
            opt_state=opt_state,
            loss=train.LossBreakdown(ce=ce, mtp=mtp, z=z, total=total),
            lr=HOST_LR,
            grad_norm=grad_norm,
            step=HOST_STEP,
        )

    got = jax.jit(_rebuild)(
        params,
        opt_state,
        jnp.asarray(out.loss.ce),
        jnp.asarray(out.loss.mtp),
        jnp.asarray(out.loss.z),
        jnp.asarray(out.loss.total),
        jnp.asarray(out.grad_norm),
    )
    assert type(got) is train.StepOutput
    assert type(got.loss) is train.LossBreakdown
    _assert_array_tree_close(got.params, out.params)
    _assert_array_tree_close(got.opt_state, out.opt_state)
    for name in LOSS_DATA_FIELDS:
        _close(getattr(got.loss, name), getattr(out.loss, name))
    _close(got.grad_norm, out.grad_norm)
    _assert_python_meta_lr_step(got, lr=HOST_LR, step=HOST_STEP)


# ---------------------------------------------------------------------------
# 3. jax.jit(train_step) takes a Batch and returns a StepOutput
# ---------------------------------------------------------------------------


def test_train_step_jit_takes_batch_returns_step_output() -> None:
    """jax.jit(train_step) takes a Batch and returns a StepOutput; matches eager at 1e-5.

    static_argnums for model_config / train_config / step as in
    tests/test_train_step_jit.py. No field-tuple unpack. Array leaves (params,
    opt_state, loss.ce/mtp/z/total, grad_norm) match eager at rtol=atol=1e-5.
    lr and step match as Python scalars. Compare against eager production, not
    a second Muon implementation.
    """
    mcfg, tcfg, params, batch = _setup()

    def call(params, opt_state, batch, model_config, train_config, step):
        return train.train_step(
            params, opt_state, batch, model_config, train_config, step=step
        )

    jitted = jax.jit(call, static_argnums=(3, 4, 5))
    opt_jit = train.init_opt_state(params, tcfg)
    got = jitted(params, opt_jit, batch, mcfg, tcfg, STEP)

    opt_e = train.init_opt_state(params, tcfg)
    eager = train.train_step(params, opt_e, batch, mcfg, tcfg, step=STEP)
    _assert_step_output_match(got, eager)

    assert not np.allclose(_np(got.params["embed"]), _np(params["embed"]), atol=0.0)
    for name in LOSS_DATA_FIELDS:
        arr = _np(getattr(got.loss, name))
        assert arr.dtype == np.float32, name
        assert arr.shape == () or arr.ndim == 0, name
        assert np.isfinite(arr).all(), name
    assert _np(got.grad_norm).dtype == np.float32
    assert np.isfinite(_np(got.grad_norm)).all()
