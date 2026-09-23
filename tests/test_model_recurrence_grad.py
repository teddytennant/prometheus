"""Truncated-recurrence backward contract for ``model.forward``.

The current forward accepts ``truncated_recurrence`` and rejects bad values,
but it does not ``stop_gradient`` the carry. Gradient and carry-stop tests
are expected to fail until that cut exists. Do not weaken a passing cut test
to manufacture a failure.

Groups
------
1. Toy scan in ``tests.reference.recurrence_cut`` (not a transformer): short
   window grad differs from the full unroll and matches a hand-stopped scan,
   finite differences, jaxpr stop only when the cut index is positive.
2. Forward values, shapes, and dtypes do not depend on ``truncated_recurrence``.
3. Bad ``truncated_recurrence`` values raise ``ConfigError``.
4. ``jax.grad`` of ``sum(logits)`` and of ``z_loss`` differs when the effective
   window is shorter than ``r_used``.
5. Gradients match when ``truncated_recurrence >= r_used``.
6. Jaxpr walk: a carry-sized ``stop_gradient`` is present only when
   ``effective < r_used``. ``logsumexp`` already lowers to ``stop_gradient`` on
   a per-token shift, so a raw name check is not the cut.
7. ``jax.jit`` forward and ``jax.jit`` of ``jax.grad`` match eager.
8. ``r=None`` still returns and ``r_used >= 1``.

Batch 1, seq 4, ``tiny_config()``, one ``PRNGKey``. No GPU tests.
"""

from __future__ import annotations

import inspect
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model import (
    TRUNCATED_RECURRENCE,
    ConfigError,
    forward,
    init_params,
    tiny_config,
)
from tests.reference.recurrence_cut import truncated_scan

BATCH = 1
SEQ = 4
GRAD_RTOL = 1e-4
GRAD_ATOL = 1e-5
# Eager and jit grads match well inside 1e-4. A few embed entries differ by
# about 8e-5, so the jit match uses a slightly looser absolute tolerance than
# the short-window inequality.
JIT_GRAD_RTOL = 1e-4
JIT_GRAD_ATOL = 1e-4
FORWARD_ATOL = 1e-5
FORWARD_RTOL = 1e-5

_BUNDLE: dict[str, Any] = {}
_GRAD_CACHE: dict[tuple[int, int], tuple[Any, Any]] = {}


def _bundle() -> tuple[Any, Any, Any]:
    if "ready" not in _BUNDLE:
        config = tiny_config()
        params = init_params(config, jax.random.PRNGKey(0))
        tokens = jnp.arange(SEQ, dtype=jnp.int32)[None, :] + jnp.int32(1)
        assert tokens.shape == (BATCH, SEQ)
        assert int(tokens.min()) >= 0
        assert int(tokens.max()) < config.vocab_size
        _BUNDLE["config"] = config
        _BUNDLE["params"] = params
        _BUNDLE["tokens"] = tokens
        _BUNDLE["ready"] = True
    return _BUNDLE["config"], _BUNDLE["params"], _BUNDLE["tokens"]


def _hand_stopped(update, carry, r_used: int, truncated_recurrence: int):
    """Independent cut: two loops, stop only when the cut index is positive.

    Cut index is ``r_used - min(truncated_recurrence, r_used)``. Index 0 does
    not stop. ``update`` still runs ``r_used`` times.
    """
    if type(r_used) is not int or type(truncated_recurrence) is not int:
        raise TypeError("r_used and truncated_recurrence must be Python int")
    effective = min(truncated_recurrence, r_used)
    cut = r_used - effective
    for step in range(cut):
        carry = update(carry, step)
    if cut > 0:
        carry = jax.lax.stop_gradient(carry)
    for step in range(cut, r_used):
        carry = update(carry, step)
    return carry


def _is_atom(obj: object) -> bool:
    if obj is None or isinstance(obj, (str, bytes, int, float, bool, complex)):
        return True
    if isinstance(obj, (np.ndarray, np.generic)):
        return True
    name = type(obj).__name__
    return name in {"ArrayImpl", "Array", "Literal", "DType", "dtype"}


def _walk_jaxpr(obj: object, seen: set[int], names: list[str], stops: list[tuple]) -> None:
    """Collect primitive names and ``stop_gradient`` inputs, including nested jaxprs."""
    if _is_atom(obj):
        return
    oid = id(obj)
    if oid in seen:
        return
    eqns = None
    if not isinstance(obj, (dict, list, tuple)):
        eqns = getattr(obj, "eqns", None)
    if eqns is not None:
        seen.add(oid)
        const_ids = {id(var) for var in getattr(obj, "constvars", ()) or ()}
        for eqn in eqns:
            prim = getattr(eqn, "primitive", None)
            pname = getattr(prim, "name", None)
            if pname:
                names.append(pname)
            if pname == "stop_gradient":
                for inv in eqn.invars:
                    stops.append(_stop_record(inv, const_ids))
            _walk_jaxpr(getattr(eqn, "params", {}), seen, names, stops)
        consts = getattr(obj, "consts", None)
        if consts:
            _walk_jaxpr(consts, seen, names, stops)
        return
    if isinstance(obj, dict):
        seen.add(oid)
        for value in obj.values():
            _walk_jaxpr(value, seen, names, stops)
        return
    if isinstance(obj, (list, tuple)):
        seen.add(oid)
        for value in obj:
            _walk_jaxpr(value, seen, names, stops)
        return
    inner = getattr(obj, "jaxpr", None)
    if inner is not None and inner is not obj:
        seen.add(oid)
        _walk_jaxpr(inner, seen, names, stops)
        consts = getattr(obj, "consts", None)
        if consts:
            _walk_jaxpr(consts, seen, names, stops)


def _stop_record(inv: object, const_ids: set[int]) -> tuple:
    aval = getattr(inv, "aval", None)
    if aval is None:
        val = getattr(inv, "val", None)
        shape = tuple(np.shape(val)) if val is not None else ()
        dtype = str(getattr(val, "dtype", type(val).__name__))
        return shape, dtype, True
    shape = tuple(int(dim) for dim in aval.shape)
    return shape, str(aval.dtype), id(inv) in const_ids or type(inv).__name__ == "Literal"


def _jaxpr_facts(closed: object) -> tuple[list[str], list[tuple]]:
    names: list[str] = []
    stops: list[tuple] = []
    _walk_jaxpr(closed, set(), names, stops)
    return names, stops


def _numel(shape: tuple) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _carry_stops(stops: list[tuple], d_model: int) -> list[tuple]:
    """Stops large enough to be the hidden carry, not a per-token logsumexp shift.

    ``jax.nn.logsumexp`` lowers to ``stop_gradient`` on the max shift. That
    shift has length seq here, which is smaller than ``d_model``. A recurrence
    cut stops the carry, including hidden state of size at least ``d_model``.
    A closed-over constant is not counted.
    """
    found = []
    for shape, dtype, is_const in stops:
        if is_const:
            continue
        if _numel(shape) >= d_model:
            found.append((shape, dtype, is_const))
    return found


def _forward_jaxpr(tokens, params, config, r: int, truncated_recurrence: int):
    """Trace ``forward`` with config, r, and truncated_recurrence closed over (static)."""

    def body(tok, pars):
        return forward(
            tok,
            pars,
            config,
            r=r,
            truncated_recurrence=truncated_recurrence,
        )

    return jax.make_jaxpr(body)(tokens, params)


def _assert_outputs_close(left, right) -> None:
    np.testing.assert_allclose(
        left.logits, right.logits, rtol=FORWARD_RTOL, atol=FORWARD_ATOL
    )
    np.testing.assert_allclose(
        left.hidden, right.hidden, rtol=FORWARD_RTOL, atol=FORWARD_ATOL
    )
    assert len(left.mtp_logits) == len(right.mtp_logits)
    for lhs, rhs in zip(left.mtp_logits, right.mtp_logits, strict=True):
        np.testing.assert_allclose(lhs, rhs, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    np.testing.assert_allclose(
        left.router_probs, right.router_probs, rtol=FORWARD_RTOL, atol=FORWARD_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(left.expert_ids),
        np.asarray(right.expert_ids),
        rtol=0.0,
        atol=FORWARD_ATOL,
    )
    np.testing.assert_allclose(
        left.z_loss, right.z_loss, rtol=FORWARD_RTOL, atol=FORWARD_ATOL
    )
    assert left.r_used == right.r_used


def _assert_output_contract(out, config, seq: int) -> None:
    assert out.logits.shape == (BATCH, seq, config.vocab_size)
    assert out.logits.dtype == jnp.float32
    assert out.hidden.shape == (BATCH, seq, config.d_model)
    assert out.hidden.dtype == jnp.float32
    assert len(out.mtp_logits) == config.mtp_heads
    for head in out.mtp_logits:
        assert head.shape == (BATCH, seq, config.vocab_size)
        assert head.dtype == jnp.float32
    assert out.router_probs.shape == (BATCH, seq, config.n_routed_experts)
    assert out.router_probs.dtype == jnp.float32
    ids = np.asarray(out.expert_ids)
    assert ids.shape == (BATCH, seq, config.top_k)
    assert ids.dtype == np.int32
    assert int(ids.min()) >= 0
    assert int(ids.max()) < config.n_routed_experts
    assert tuple(np.asarray(out.z_loss).shape) == ()
    assert np.asarray(out.z_loss).dtype == np.float32
    assert type(out.r_used) is int
    assert np.isfinite(np.asarray(out.logits)).all()
    assert np.isfinite(np.asarray(out.z_loss)).all()


def _mismatch_paths(left, right, rtol: float, atol: float) -> list:
    bad = []

    def check(path, lhs, rhs):
        lhs_np = np.asarray(lhs)
        rhs_np = np.asarray(rhs)
        if lhs_np.shape != rhs_np.shape or not np.allclose(
            lhs_np, rhs_np, rtol=rtol, atol=atol, equal_nan=False
        ):
            bad.append(path)

    jax.tree_util.tree_map_with_path(check, left, right)
    return bad


def _assert_finite_tree(tree, label: str) -> None:
    bad = []

    def check(path, leaf):
        if not np.isfinite(np.asarray(leaf)).all():
            bad.append(path)

    jax.tree_util.tree_map_with_path(check, tree)
    assert not bad, f"{label} has non-finite leaves"


def _assert_some_nonzero(tree, label: str) -> None:
    found = False

    def check(leaf):
        nonlocal found
        if np.any(np.asarray(leaf) != 0):
            found = True

    jax.tree_util.tree_map(check, tree)
    assert found, f"{label} is all zeros"


def _assert_grads_differ(short, full) -> None:
    _assert_finite_tree(short, "short-window grad")
    _assert_finite_tree(full, "full-window grad")
    _assert_some_nonzero(full, "full-window grad")
    n_leaves = len(jax.tree_util.tree_leaves(short))
    assert n_leaves > 0
    bad = _mismatch_paths(short, full, GRAD_RTOL, GRAD_ATOL)
    assert bad, (
        "truncated-window grad matches the full unroll on every param leaf "
        f"(rtol={GRAD_RTOL}, atol={GRAD_ATOL}, leaves={n_leaves})"
    )


def _assert_grads_close(
    left,
    right,
    label: str,
    rtol: float = GRAD_RTOL,
    atol: float = GRAD_ATOL,
) -> None:
    _assert_finite_tree(left, label)
    _assert_finite_tree(right, label)
    bad = _mismatch_paths(left, right, rtol, atol)
    assert not bad, f"{label} mismatch on {bad[:3]}"


def _both_grads(r: int, truncated_recurrence: int):
    key = (r, truncated_recurrence)
    if key not in _GRAD_CACHE:
        config, params, tokens = _bundle()

        def pair(pars):
            out = forward(
                tokens,
                pars,
                config,
                r=r,
                truncated_recurrence=truncated_recurrence,
            )
            return jnp.sum(out.logits), out.z_loss

        _primals, vjp_fun = jax.vjp(pair, params)
        one = jnp.ones((), dtype=jnp.float32)
        zero = jnp.zeros((), dtype=jnp.float32)
        g_logits = vjp_fun((one, zero))[0]
        g_z = vjp_fun((zero, one))[0]
        jax.tree_util.tree_leaves(g_logits)[0].block_until_ready()
        jax.tree_util.tree_leaves(g_z)[0].block_until_ready()
        _GRAD_CACHE[key] = (g_logits, g_z)
    return _GRAD_CACHE[key]


def test_reference_does_not_import_model() -> None:
    import ast

    import tests.reference.recurrence_cut as toy

    tree = ast.parse(inspect.getsource(toy))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] != "model"
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module.split(".")[0] != "model"


def test_toy_grad_truncated1_differs_from_full_and_matches_hand_stop() -> None:
    param = jnp.array([0.3, -0.4, 0.8], dtype=jnp.float32)
    init = jnp.array([0.2, 0.5, -0.1], dtype=jnp.float32)

    def update(carry, step, p):
        del step
        return 0.5 * carry + p

    def run(scan, p, r_used, trunc):
        return scan(lambda carry, step: update(carry, step, p), init, r_used, trunc)

    for r_used, trunc in ((4, 1), (5, 4), (3, 1)):
        value_short = run(truncated_scan, param, r_used, trunc)
        value_full = run(truncated_scan, param, r_used, r_used)
        value_hand = run(_hand_stopped, param, r_used, trunc)
        np.testing.assert_allclose(value_short, value_full, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(value_short, value_hand, rtol=1e-5, atol=1e-5)

        g_short = jax.grad(
            lambda p, ru=r_used, tr=trunc: jnp.sum(run(truncated_scan, p, ru, tr))
        )(param)
        g_full = jax.grad(
            lambda p, ru=r_used: jnp.sum(run(truncated_scan, p, ru, ru))
        )(param)
        g_hand = jax.grad(
            lambda p, ru=r_used, tr=trunc: jnp.sum(run(_hand_stopped, p, ru, tr))
        )(param)
        assert not np.allclose(
            np.asarray(g_short), np.asarray(g_full), rtol=GRAD_RTOL, atol=GRAD_ATOL
        )
        np.testing.assert_allclose(g_short, g_hand, rtol=1e-5, atol=1e-5)
        g_hand_full = jax.grad(lambda p, ru=r_used: jnp.sum(run(_hand_stopped, p, ru, ru)))(param)
        np.testing.assert_allclose(g_full, g_hand_full, rtol=1e-5, atol=1e-5)


def test_toy_pytree_carry_stop_matches_hand_stop() -> None:
    init = {"h": jnp.array([0.2, -0.3], dtype=jnp.float32), "z": jnp.array(0.4)}

    def update(carry, step):
        del step
        return {"h": 0.5 * carry["h"] + 0.25, "z": carry["z"] + 1.0}

    def loss(h0, scan, trunc):
        carry = {"h": h0, "z": init["z"]}
        out = scan(update, carry, 3, trunc)
        return jnp.sum(out["h"])

    g_short = jax.grad(lambda h: loss(h, truncated_scan, 1))(init["h"])
    g_full = jax.grad(lambda h: loss(h, truncated_scan, 3))(init["h"])
    g_hand = jax.grad(lambda h: loss(h, _hand_stopped, 1))(init["h"])
    assert not np.allclose(g_short, g_full, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    np.testing.assert_allclose(g_short, g_hand, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g_short, jnp.zeros_like(g_short), atol=1e-6)


def test_toy_finite_difference_matches_stopped_window() -> None:
    """FD the live window only. A primal FD does not see stop_gradient."""
    p0 = jnp.float32(0.35)
    init = jnp.float32(0.2)
    eps = jnp.float32(1e-3)

    def update(carry, step, p):
        del step
        return jnp.float32(0.5) * carry + p

    def analytic(r_used, trunc):
        def loss(p):
            return truncated_scan(lambda c, s: update(c, s, p), init, r_used, trunc)

        return jax.grad(loss)(p0)

    def numeric(r_used, trunc):
        cut = r_used - min(trunc, r_used)
        carry = init
        for step in range(cut):
            carry = update(carry, step, p0)
        frozen = jnp.asarray(np.array(carry, dtype=np.float32))

        def live(p):
            out = frozen
            for step in range(cut, r_used):
                out = update(out, step, p)
            return out

        return (live(p0 + eps) - live(p0 - eps)) / (jnp.float32(2.0) * eps)

    for r_used, trunc in ((3, 1), (5, 4)):
        np.testing.assert_allclose(
            analytic(r_used, trunc), numeric(r_used, trunc), rtol=1e-3, atol=1e-4
        )


def test_toy_jaxpr_stop_gradient_only_when_cut_positive() -> None:
    init = jnp.ones((4,), dtype=jnp.float32)

    def update(carry, step):
        del step
        return 0.5 * carry + 0.1

    def names_for(r_used, trunc):
        closed = jax.make_jaxpr(lambda carry: truncated_scan(update, carry, r_used, trunc))(init)
        names, _stops = _jaxpr_facts(closed)
        return names

    short = names_for(5, 1)
    full = names_for(5, 5)
    longer = names_for(5, 15)
    assert "stop_gradient" in short
    assert "stop_gradient" not in full
    assert "stop_gradient" not in longer


def test_toy_rejects_bool_float_and_numpy_int() -> None:
    init = jnp.zeros((2,), dtype=jnp.float32)

    def update(carry, step):
        del step
        return carry

    for bad in (True, 1.5, np.int64(4), 0, -1):
        with pytest.raises(ValueError):
            truncated_scan(update, init, 2, bad)


def test_walker_finds_stop_gradient_nested_in_scan_cond_while() -> None:
    def fn(x):
        def scan_body(carry, _unused):
            carry = jax.lax.cond(
                carry[0] > 0,
                jax.lax.stop_gradient,
                lambda z: z,
                carry,
            )
            return carry, None

        carry, _ys = jax.lax.scan(scan_body, x, xs=None, length=2)

        def cond_fun(state):
            return state[0] < 4

        def body_fun(state):
            return jax.lax.stop_gradient(state) + 1

        return jax.lax.while_loop(cond_fun, body_fun, carry)

    closed = jax.make_jaxpr(fn)(jnp.ones((4,), dtype=jnp.float32))
    names, stops = _jaxpr_facts(closed)
    assert names.count("stop_gradient") >= 2
    assert stops
    assert "scan" in names
    assert "while" in names


def test_truncated_recurrence_default_is_four() -> None:
    assert type(TRUNCATED_RECURRENCE) is int
    assert TRUNCATED_RECURRENCE == 4
    default = inspect.signature(forward).parameters["truncated_recurrence"].default
    assert type(default) is int
    assert default == TRUNCATED_RECURRENCE


def test_forward_independent_of_truncated_recurrence() -> None:
    config, params, tokens = _bundle()
    for r in (3, 5):
        windows = (1, 4, r, r + 10)
        outputs = [
            forward(tokens, params, config, r=r, truncated_recurrence=trunc)
            for trunc in windows
        ]
        for out in outputs:
            _assert_output_contract(out, config, SEQ)
            assert out.r_used == r
        for other in outputs[1:]:
            _assert_outputs_close(outputs[0], other)
        default_out = forward(tokens, params, config, r=r)
        _assert_outputs_close(outputs[0], default_out)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="bool"),
        pytest.param(1.5, id="float"),
        pytest.param(np.int64(4), id="np-int64"),
    ],
)
def test_truncated_recurrence_rejects_bad_values(bad) -> None:
    config, params, tokens = _bundle()
    with pytest.raises(ConfigError):
        forward(tokens, params, config, r=3, truncated_recurrence=bad)


@pytest.mark.parametrize(
    "r,short",
    [pytest.param(3, 1, id="r3-trunc1"), pytest.param(5, 4, id="r5-trunc4")],
)
def test_grad_logits_short_window_differs_from_full(r, short) -> None:
    g_short, _z_short = _both_grads(r, short)
    g_full, _z_full = _both_grads(r, r)
    _assert_grads_differ(g_short, g_full)


@pytest.mark.parametrize(
    "r,short",
    [pytest.param(3, 1, id="r3-trunc1"), pytest.param(5, 4, id="r5-trunc4")],
)
def test_grad_z_loss_short_window_differs_from_full(r, short) -> None:
    _g_short, z_short = _both_grads(r, short)
    _g_full, z_full = _both_grads(r, r)
    _assert_grads_differ(z_short, z_full)


@pytest.mark.parametrize("r", [3, 5])
def test_grad_full_window_matches_larger_window(r) -> None:
    g_full, z_full = _both_grads(r, r)
    g_long, z_long = _both_grads(r, r + 10)
    _assert_grads_close(g_full, g_long, f"logits r={r}")
    _assert_grads_close(z_full, z_long, f"z_loss r={r}")


@pytest.mark.parametrize(
    "r,trunc,expect_carry_stop",
    [
        pytest.param(3, 1, True, id="r3-trunc1"),
        pytest.param(5, 4, True, id="r5-trunc4"),
        pytest.param(3, 3, False, id="r3-full"),
        pytest.param(3, 13, False, id="r3-plus10"),
        pytest.param(5, 5, False, id="r5-full"),
        pytest.param(5, 15, False, id="r5-plus10"),
    ],
)
def test_jaxpr_carry_stop_gradient_follows_effective_window(r, trunc, expect_carry_stop) -> None:
    config, params, tokens = _bundle()
    out = forward(tokens, params, config, r=r, truncated_recurrence=trunc)
    assert out.r_used == r
    effective = min(trunc, out.r_used)
    assert (effective < out.r_used) is expect_carry_stop

    closed = _forward_jaxpr(tokens, params, config, r, trunc)
    names, stops = _jaxpr_facts(closed)
    # Walker must see the real jaxpr. logsumexp inserts stop_gradient on a
    # length-seq shift in every window, so a raw name check cannot prove the cut.
    assert "stop_gradient" in names
    assert "scan" in names or r == 1
    carry = _carry_stops(stops, config.d_model)
    if expect_carry_stop:
        assert carry, (
            "missing carry stop_gradient when effective < r_used "
            f"(r={r}, truncated_recurrence={trunc}, n_stop={names.count('stop_gradient')})"
        )
        assert names.count("stop_gradient") > 0
    else:
        assert not carry, (
            "full-window jaxpr must not stop the carry "
            f"(r={r}, truncated_recurrence={trunc}, carry_stops={carry[:3]})"
        )


def test_jit_forward_matches_eager() -> None:
    config, params, tokens = _bundle()
    jitted = jax.jit(forward, static_argnames=("config", "r", "truncated_recurrence"))
    for trunc in (1, 4):
        eager = forward(tokens, params, config, r=2, truncated_recurrence=trunc)
        compiled = jitted(tokens, params, config, r=2, truncated_recurrence=trunc)
        _assert_output_contract(eager, config, SEQ)
        _assert_outputs_close(eager, compiled)
        assert eager.r_used == 2
        assert compiled.r_used == 2


def test_jit_grad_matches_eager_grad() -> None:
    config, params, tokens = _bundle()
    jitted = jax.jit(forward, static_argnames=("config", "r", "truncated_recurrence"))
    for trunc in (1, 4):

        def eager_loss(pars, truncated=trunc):
            return jnp.sum(
                forward(
                    tokens,
                    pars,
                    config,
                    r=2,
                    truncated_recurrence=truncated,
                ).logits
            )

        def jit_forward_loss(pars, truncated=trunc):
            return jnp.sum(
                jitted(
                    tokens,
                    pars,
                    config,
                    r=2,
                    truncated_recurrence=truncated,
                ).logits
            )

        g_eager = jax.grad(eager_loss)(params)
        g_of_jit = jax.grad(jit_forward_loss)(params)
        g_jit_grad = jax.jit(jax.grad(eager_loss))(params)
        jax.tree_util.tree_leaves(g_eager)[0].block_until_ready()
        _assert_grads_close(
            g_eager,
            g_of_jit,
            f"grad of jitted forward trunc={trunc}",
            rtol=JIT_GRAD_RTOL,
            atol=JIT_GRAD_ATOL,
        )
        _assert_grads_close(
            g_eager,
            g_jit_grad,
            f"jit jax.grad trunc={trunc}",
            rtol=JIT_GRAD_RTOL,
            atol=JIT_GRAD_ATOL,
        )


def test_r_none_returns_positive_r_used() -> None:
    config, params, tokens = _bundle()
    out = forward(tokens, params, config, r=None)
    assert type(out.r_used) is int
    assert out.r_used >= 1
    _assert_output_contract(out, config, SEQ)
    out_short = forward(tokens, params, config, r=None, truncated_recurrence=1)
    assert out_short.r_used >= 1
    _assert_outputs_close(out, out_short)
