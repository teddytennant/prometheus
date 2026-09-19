"""A2-train-step-jit oracle: ``jax.jit`` of ``train_step`` must match eager at 1e-5.

These tests call the public ``train.train_step`` signature with ``model.tiny_config()``,
``train.tiny_train_config()``, and a tiny ``Batch`` (same construction as
``tests/test_train.py::_tiny_batch``). Python scalars ``step`` (1-based int),
``TrainConfig``, and ``ModelConfig`` stay host-side (closed over or
``static_argnums``). ``jax.grad`` through the whole step is not required —
``train_step`` already contains ``value_and_grad``.

They must fail on the current post-``value_and_grad`` host conversions
(``float(np.sqrt(_tree_sum_sq(grads)))``, Python ``if clip > 0 and gnorm > clip``,
``numpy.asarray`` inside QK-clip / router-bias / ``classify_param``,
``float(ce)`` into ``LossBreakdown``) with ``TracerArrayConversionError`` or
``ConcretizationTypeError``, and pass once ``train_step`` stays in JAX after
``value_and_grad``.

A full independent NumPy ``train_step`` is not added under ``tests/reference/``.
It would reimplement ``model.forward`` + autodiff + the MuonClip / AdamW tree
walk + QK-clip + router-bias + WSD, which is larger than this step and would
copy production structure. Existing ``tests/reference/train.py`` pieces already
used by ``tests/test_train.py`` (``wsd_lr``, ``total_loss``, ``qk_clip``) are
compared against jitted outputs instead, plus jit vs eager on ``loss.total``
and param / opt_state leaves.

Dataclass returns (``StepOutput``, ``LossBreakdown``) are unpacked to tuples
inside a jitted wrapper so a JAX math fix does not also require pytree
registration (same pattern as ``tests/test_latent_jit.py``). One test still
calls ``jax.jit(train.train_step, ...)`` itself; tests never register pytrees.
Production must not import ``tests``. Do not mark gpu.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

import model
import train
from tests.reference import train as ref

TOL = dict(rtol=1e-5, atol=1e-5)
STEP = 1


def _np(x: object) -> np.ndarray:
    return np.asarray(x)


def _close(got: object, exp: object) -> None:
    np.testing.assert_allclose(_np(got), _np(exp), **TOL)


def _tiny_batch(mcfg: model.ModelConfig, batch: int = 2, seq: int = 8) -> train.Batch:
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


def _unpack(out: train.StepOutput) -> tuple:
    return (
        out.loss.ce,
        out.loss.mtp,
        out.loss.z,
        out.loss.total,
        out.grad_norm,
        out.params,
        out.opt_state,
        out.lr,
        out.step,
    )


def _array_leaves(tree: object) -> list[np.ndarray]:
    leaves: list[np.ndarray] = []

    def rec(node: object) -> None:
        if isinstance(node, dict):
            for key in sorted(node):
                rec(node[key])
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                rec(child)
            return
        if hasattr(node, "shape"):
            leaves.append(np.asarray(node))

    rec(tree)
    return leaves


def _close_tree(got: object, exp: object) -> None:
    got_leaves = _array_leaves(got)
    exp_leaves = _array_leaves(exp)
    assert len(got_leaves) == len(exp_leaves)
    for a, b in zip(got_leaves, exp_leaves, strict=True):
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, **TOL)


def _qk_max_abs(params: object) -> float:
    """Max |q k^T| over linear-attn leaves, matching production's 2D reshape."""
    peak = 0.0

    def rec(node: object) -> None:
        nonlocal peak
        if isinstance(node, dict):
            if "W_q" in node and "W_k" in node:
                q = np.asarray(node["W_q"], dtype=np.float32)
                k = np.asarray(node["W_k"], dtype=np.float32)
                if q.shape == k.shape and q.ndim >= 2:
                    qn = q.reshape(-1, int(q.shape[-1]))
                    kn = k.reshape(-1, int(k.shape[-1]))
                    scores = np.einsum("id,jd->ij", qn, kn)
                    peak = max(peak, float(np.max(np.abs(scores))))
            for child in node.values():
                rec(child)
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                rec(child)

    rec(params)
    return peak


def _inject_router_bias(params: dict, n_experts: int) -> dict:
    out = dict(params)
    layers = []
    injected = False
    for layer in out["layers"]:
        layer = dict(layer)
        if (not injected) and "router" in layer:
            layer["router_bias"] = jnp.zeros((int(n_experts),), dtype=jnp.float32)
            injected = True
        layers.append(layer)
    out["layers"] = layers
    return out


def _scale_qk(params: dict, factor: float) -> dict:
    out = dict(params)
    layers = []
    scale = jnp.float32(factor)
    for layer in out["layers"]:
        layer = dict(layer)
        if "W_q" in layer and "W_k" in layer:
            layer["W_q"] = layer["W_q"] * scale
            layer["W_k"] = layer["W_k"] * scale
        layers.append(layer)
    out["layers"] = layers
    return out


def _setup() -> tuple:
    mcfg = model.tiny_config()
    tcfg = train.tiny_train_config()
    params = model.init_params(mcfg, 0)
    batch = _as_batch(_tiny_batch(mcfg))
    return mcfg, tcfg, params, batch


def _jitted_closed(batch: train.Batch, mcfg, tcfg, step: int):
    """Unpack dataclass returns so tests need not register pytrees."""

    def fields(params, opt_state):
        out = train.train_step(params, opt_state, batch, mcfg, tcfg, step=step)
        return _unpack(out)

    return jax.jit(fields)


def _assert_step_outputs(got: tuple, eager: train.StepOutput, tcfg: train.TrainConfig) -> None:
    ce, mtp, z, total, gnorm, params, opt_state, lr, step = got
    _close(ce, eager.loss.ce)
    _close(mtp, eager.loss.mtp)
    _close(z, eager.loss.z)
    _close(total, eager.loss.total)
    _close(gnorm, eager.grad_norm)
    _close_tree(params, eager.params)
    _close_tree(opt_state, eager.opt_state)
    _close(lr, eager.lr)
    assert int(step) == int(eager.step) == STEP
    _close(lr, ref.wsd_lr(STEP, tcfg))
    _close(total, ref.total_loss(ce, mtp, z, tcfg))


# ---------------------------------------------------------------------------
# jit vs eager (1e-5) on public train_step outputs
# ---------------------------------------------------------------------------
def test_train_step_jit_matches_eager() -> None:
    """Closed-over configs/step/batch: jitted train_step matches eager at 1e-5."""
    mcfg, tcfg, params, batch = _setup()
    opt_jit = train.init_opt_state(params, tcfg)
    jitted = _jitted_closed(batch, mcfg, tcfg, STEP)
    got = jitted(params, opt_jit)

    opt_e = train.init_opt_state(params, tcfg)
    eager = train.train_step(params, opt_e, batch, mcfg, tcfg, step=STEP)
    _assert_step_outputs(got, eager, tcfg)

    embed_got = got[5]["embed"]
    embed_exp = eager.params["embed"]
    _close(embed_got, embed_exp)
    assert not np.allclose(_np(embed_got), _np(params["embed"]), atol=0.0)


def test_train_step_jit_vs_reference_pieces() -> None:
    """Jitted loss.total / one param leaf vs eager; lr vs ref.wsd_lr at 1e-5.

    A full independent train_step NumPy reference is omitted (would need the
    whole recurrent MoE/MLA forward, autodiff, and optimizer tree walk).
    ``ref.total_loss`` / ``ref.wsd_lr`` are the existing oracle pieces.
    """
    mcfg, tcfg, params, batch = _setup()
    opt_jit = train.init_opt_state(params, tcfg)
    ce, mtp, z, total, _gnorm, new_params, _opt, lr, step = _jitted_closed(
        batch, mcfg, tcfg, STEP
    )(params, opt_jit)

    opt_e = train.init_opt_state(params, tcfg)
    eager = train.train_step(params, opt_e, batch, mcfg, tcfg, step=STEP)
    _close(total, eager.loss.total)
    _close(new_params["embed"], eager.params["embed"])
    _close(lr, ref.wsd_lr(STEP, tcfg))
    _close(total, ref.total_loss(ce, mtp, z, tcfg))
    assert int(step) == STEP


# ---------------------------------------------------------------------------
# Grad clip must not Python-branch on traced grad_norm
# ---------------------------------------------------------------------------
def test_train_step_jit_clip_and_no_clip_no_python_branch() -> None:
    """Both a clipping and a non-clipping threshold must run under jax.jit.

    ``tiny_train_config.grad_clip`` is 1.0; tiny grads typically exceed it, so
    both paths use a reconstructed frozen ``TrainConfig`` with clip > 0
    (tiny vs huge). A Python ``if gnorm > clip`` on a tracer must fail.
    """
    mcfg, tcfg, params, batch = _setup()
    clip_cfg = replace(tcfg, grad_clip=1e-12)
    no_clip_cfg = replace(tcfg, grad_clip=1e9)

    opt_c = train.init_opt_state(params, clip_cfg)
    got_c = _jitted_closed(batch, mcfg, clip_cfg, STEP)(params, opt_c)
    opt_n = train.init_opt_state(params, no_clip_cfg)
    got_n = _jitted_closed(batch, mcfg, no_clip_cfg, STEP)(params, opt_n)

    assert _np(got_c[4]).dtype == np.float32
    assert _np(got_n[4]).dtype == np.float32
    assert float(_np(got_c[4])) > 0.0
    assert float(_np(got_n[4])) > 0.0
    # Clipped vs unclipped updates must differ on at least one param leaf.
    assert not np.allclose(_np(got_c[5]["embed"]), _np(got_n[5]["embed"]), **TOL)


# ---------------------------------------------------------------------------
# QK-clip path stays inside the jitted step
# ---------------------------------------------------------------------------
def test_train_step_jit_qk_clip_path() -> None:
    """tiny_config.qk_clip > 0: jitted step runs; identity-ish when under cap."""
    mcfg, tcfg, params, batch = _setup()
    assert float(tcfg.qk_clip) > 0.0
    assert _qk_max_abs(params) <= float(tcfg.qk_clip)

    opt = train.init_opt_state(params, tcfg)
    got = _jitted_closed(batch, mcfg, tcfg, STEP)(params, opt)
    new_params = got[5]
    cap = float(tcfg.qk_clip)
    assert _qk_max_abs(new_params) <= cap + 1e-5

    # Under-cap weights: independent qk_clip is identity on a linear leaf.
    for layer in new_params["layers"]:
        if "W_q" in layer and "W_k" in layer:
            q = np.asarray(layer["W_q"], dtype=np.float32)
            k = np.asarray(layer["W_k"], dtype=np.float32)
            qn = q.reshape(-1, int(q.shape[-1]))
            kn = k.reshape(-1, int(k.shape[-1]))
            rq, rk = ref.qk_clip(qn, kn, cap)
            _close(rq, qn)
            _close(rk, kn)
            break
    else:
        raise AssertionError("tiny model has no W_q/W_k pair")


def test_train_step_jit_qk_clip_caps_when_over() -> None:
    """Jitted step with qk_clip > 0 still caps linear W_q/W_k that start over τ."""
    mcfg, tcfg, params, batch = _setup()
    cap = float(tcfg.qk_clip)
    params = _scale_qk(params, 1.0e3)
    assert _qk_max_abs(params) > cap

    opt = train.init_opt_state(params, tcfg)
    got = _jitted_closed(batch, mcfg, tcfg, STEP)(params, opt)
    assert _qk_max_abs(got[5]) <= cap + 1e-4


# ---------------------------------------------------------------------------
# Router-bias update stays inside the jitted step (no numpy walk on tracers)
# ---------------------------------------------------------------------------
def test_train_step_jit_router_bias_update_stays_in_jax() -> None:
    """Injected router_bias is updated under jit (no numpy.mean on traced load)."""
    mcfg, tcfg, params, batch = _setup()
    params = _inject_router_bias(params, mcfg.n_routed_experts)
    opt = train.init_opt_state(params, tcfg)
    got = _jitted_closed(batch, mcfg, tcfg, STEP)(params, opt)

    bias = None
    for layer in got[5]["layers"]:
        if "router_bias" in layer:
            bias = np.asarray(layer["router_bias"], dtype=np.float32)
            break
    assert bias is not None
    assert bias.shape == (int(mcfg.n_routed_experts),)
    assert bias.dtype == np.float32
    assert np.isfinite(bias).all()
    assert float(np.linalg.norm(bias)) > 0.0


# ---------------------------------------------------------------------------
# Host-side Python scalars: static_argnums and closed-over
# ---------------------------------------------------------------------------
def test_train_step_jit_static_argnums_configs_and_step() -> None:
    """jax.jit(..., static_argnums) for ModelConfig, TrainConfig, and step."""
    mcfg, tcfg, params, batch = _setup()

    def fields(params, opt_state, model_config, train_config, step):
        out = train.train_step(
            params, opt_state, batch, model_config, train_config, step=step
        )
        return _unpack(out)

    jitted = jax.jit(fields, static_argnums=(2, 3, 4))
    opt_jit = train.init_opt_state(params, tcfg)
    got = jitted(params, opt_jit, mcfg, tcfg, STEP)

    opt_e = train.init_opt_state(params, tcfg)
    eager = train.train_step(params, opt_e, batch, mcfg, tcfg, step=STEP)
    _assert_step_outputs(got, eager, tcfg)


def test_train_step_jits_itself_without_pytree_registration() -> None:
    """``jax.jit`` of a call to ``train.train_step`` itself; tests register nothing.

    ``Batch`` / ``StepOutput`` / ``LossBreakdown`` are not registered here.
    Configs and step are closed over; dataclass returns are unpacked to tuples
    so a JAX math fix does not also require pytree registration (same pattern
    as ``tests/test_latent_jit.py``).
    """
    mcfg, tcfg, params, batch = _setup()
    opt_jit = train.init_opt_state(params, tcfg)

    def call_train_step(params, opt_state):
        out = train.train_step(params, opt_state, batch, mcfg, tcfg, step=STEP)
        return _unpack(out)

    got = jax.jit(call_train_step)(params, opt_jit)
    opt_e = train.init_opt_state(params, tcfg)
    eager = train.train_step(params, opt_e, batch, mcfg, tcfg, step=STEP)
    _assert_step_outputs(got, eager, tcfg)


# ---------------------------------------------------------------------------
# Shapes / dtypes of jitted outputs
# ---------------------------------------------------------------------------
def test_train_step_jit_shapes_and_dtypes() -> None:
    """Jitted scalars are float32; param / opt_state leaves keep FP32 masters."""
    mcfg, tcfg, params, batch = _setup()
    opt = train.init_opt_state(params, tcfg)
    ce, mtp, z, total, gnorm, new_params, new_opt, lr, step = _jitted_closed(
        batch, mcfg, tcfg, STEP
    )(params, opt)

    for name, val in (("ce", ce), ("mtp", mtp), ("z", z), ("total", total), ("gnorm", gnorm)):
        arr = _np(val)
        assert arr.dtype == np.float32, name
        assert arr.shape == () or arr.ndim == 0, name
        assert np.isfinite(arr).all(), name

    init_leaves = _array_leaves(params)
    new_leaves = _array_leaves(new_params)
    assert len(init_leaves) == len(new_leaves)
    for a, b in zip(init_leaves, new_leaves, strict=True):
        assert a.shape == b.shape
        assert b.dtype == np.float32

    for leaf in _array_leaves(new_opt):
        assert leaf.dtype == np.float32

    assert float(_np(lr)) == float(ref.wsd_lr(STEP, tcfg))
    assert int(step) == STEP
