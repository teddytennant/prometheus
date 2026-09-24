"""Failing tests for spec 4.2 layout and spec 4.3 segment decode / CODI.

Expected values come from ``tests.reference.latent_segment``. Production
stubs raise ``NotImplementedError``, so every test that calls them fails.
"""

from __future__ import annotations

import numpy as np
import pytest

import model.latent as latent
from tests.reference import latent_segment as ref


def _cfg(**kw) -> latent.LatentConfig:
    return latent.LatentConfig(**kw)


def test_plan_golden_two_chunks() -> None:
    got = latent.plan_think_layout((4, 8), (7, 8), _cfg())
    exp = ref.plan_think_layout((4, 8), (7, 8), _cfg())
    assert got == exp
    assert got[0].kind is latent.SpanKind.LATENT
    assert got[1].kind is latent.SpanKind.ANCHOR
    assert got[1].token_ids == (7, 8)
    assert got[2].n_thoughts == 8


def test_plan_error_order() -> None:
    with pytest.raises(latent.LatentError, match="chunk_lengths empty"):
        latent.plan_think_layout((), (1,), _cfg())
    with pytest.raises(latent.LatentError, match="chunk length out of range"):
        latent.plan_think_layout((3,), (1,), _cfg())
    with pytest.raises(latent.LatentError, match="chunk length out of range"):
        latent.plan_think_layout((True,), (1,), _cfg())  # type: ignore[arg-type]
    with pytest.raises(latent.LatentError, match="anchor empty"):
        latent.plan_think_layout((4,), (), _cfg())
    with pytest.raises(latent.LatentError, match="anchor token id"):
        latent.plan_think_layout((4,), (-1,), _cfg())
    with pytest.raises(latent.LatentError, match="anchor token id"):
        latent.plan_think_layout((4,), (True,), _cfg())  # type: ignore[arg-type]


def test_insert_tool_between_anchor_and_next_latent() -> None:
    layout = latent.plan_think_layout((4, 8), (7, 8), _cfg())
    got = latent.insert_discrete_tool(layout, 0, (1, 2))
    exp = ref.insert_discrete_tool(ref.plan_think_layout((4, 8), (7, 8), _cfg()), 0, (1, 2))
    assert got == exp
    assert got[2].kind is latent.SpanKind.TOOL
    assert got[2].token_ids == (1, 2)
    assert got[3].kind is latent.SpanKind.LATENT


def test_insert_tool_after_last_chunk_appends() -> None:
    layout = latent.plan_think_layout((4, 8), (7, 8), _cfg())
    got = latent.insert_discrete_tool(layout, 1, (3,))
    assert got[-1].kind is latent.SpanKind.TOOL
    assert got[-1].token_ids == (3,)
    assert got[-2].kind is latent.SpanKind.LATENT


def test_insert_repeated_appends_in_order() -> None:
    layout = latent.plan_think_layout((4, 8), (7,), _cfg())
    once = latent.insert_discrete_tool(layout, 0, (1,))
    twice = latent.insert_discrete_tool(once, 0, (2,))
    tools = [s for s in twice if s.kind is latent.SpanKind.TOOL]
    assert [s.token_ids for s in tools] == [(1,), (2,)]


def test_insert_error_order() -> None:
    layout = latent.plan_think_layout((4, 8), (1,), _cfg())
    with pytest.raises(latent.LatentError, match="layout"):
        latent.insert_discrete_tool((), 0, (1,))
    with pytest.raises(latent.LatentError, match="after_chunk"):
        latent.insert_discrete_tool(layout, 2, (1,))
    with pytest.raises(latent.LatentError, match="after_chunk"):
        latent.insert_discrete_tool(layout, True, (1,))  # type: ignore[arg-type]
    with pytest.raises(latent.LatentError, match="tool empty"):
        latent.insert_discrete_tool(layout, 0, ())
    with pytest.raises(latent.LatentError, match="tool token id"):
        latent.insert_discrete_tool(layout, 0, (-1,))


def test_decode_one_token_is_tied_head_and_ignores_decoder() -> None:
    rng = np.random.default_rng(0)
    h = rng.normal(size=(4,)).astype(np.float64)
    u = rng.normal(size=(3, 4)).astype(np.float64)
    fn = rng.normal(size=(4,)).astype(np.float64)
    bad = latent.SegmentDecoder(
        w_in=np.full((4, 4), np.nan),
        w_h=np.full((4, 4), np.nan),
        bias=np.full((4,), np.nan),
    )
    got = latent.decode_thought_segment(h, 1, u, decoder=bad, final_norm=fn)
    exp = ref.decode_thought_segment(h, 1, u, decoder=None, final_norm=fn)
    assert np.asarray(got).shape == (3,)
    assert np.allclose(np.asarray(got), exp, atol=1e-5)


def test_decode_multi_token_matches_reference() -> None:
    rng = np.random.default_rng(1)
    d, vocab, n = 4, 3, 5
    h = rng.normal(size=(d,)).astype(np.float64)
    u = rng.normal(size=(vocab, d)).astype(np.float64)
    dec = latent.SegmentDecoder(
        w_in=rng.normal(size=(d, d)).astype(np.float64) * 0.1,
        w_h=rng.normal(size=(d, d)).astype(np.float64) * 0.1,
        bias=rng.normal(size=(d,)).astype(np.float64) * 0.1,
    )
    got = latent.decode_thought_segment(h, n, u, decoder=dec)
    exp = ref.decode_thought_segment(h, n, u, decoder=dec)
    assert np.asarray(got).shape == (n, vocab)
    assert np.allclose(np.asarray(got), exp, atol=1e-5)


def test_decode_error_order() -> None:
    h = np.ones(4)
    u = np.ones((2, 4))
    with pytest.raises(latent.LatentError, match="teacher_len"):
        latent.decode_thought_segment(h, True, u)  # type: ignore[arg-type]
    with pytest.raises(latent.LatentError, match="hidden"):
        latent.decode_thought_segment(np.array([1.0, np.nan, 0.0, 0.0]), 1, u)
    with pytest.raises(latent.LatentError, match="unembed"):
        latent.decode_thought_segment(h, 1, np.ones((2, 3)))
    with pytest.raises(latent.LatentError, match="final_norm"):
        latent.decode_thought_segment(h, 1, u, final_norm=np.ones(3))
    with pytest.raises(latent.LatentError, match="decoder"):
        latent.decode_thought_segment(h, 2, u, decoder=None)


def test_decode_grad_hidden_and_w_in() -> None:
    import jax
    import jax.numpy as jnp

    d, vocab, n = 3, 2, 4
    h = jnp.array([0.2, -0.1, 0.4])
    u = jnp.arange(vocab * d, dtype=jnp.float32).reshape(vocab, d) * 0.01
    w_in = jnp.eye(d) * 0.1
    w_h = jnp.zeros((d, d))
    bias = jnp.zeros((d,))

    def loss(h, w_in):
        dec = latent.SegmentDecoder(w_in=w_in, w_h=w_h, bias=bias)
        logits = latent.decode_thought_segment(h, n, u, decoder=dec)
        return jnp.sum(logits)

    g_h, g_w = jax.grad(loss, argnums=(0, 1))(h, w_in)
    assert np.all(np.isfinite(np.asarray(g_h)))
    assert np.all(np.isfinite(np.asarray(g_w)))
    # Finite difference on one hidden coordinate.
    eps = 1e-3
    hp = h.at[0].add(eps)
    hm = h.at[0].add(-eps)
    num = (loss(hp, w_in) - loss(hm, w_in)) / (2 * eps)
    assert np.allclose(float(g_h[0]), float(num), atol=1e-2)


def test_codi_value_and_stop_grad() -> None:
    import jax
    import jax.numpy as jnp

    s = np.array([[1.0, 2.0], [0.0, -1.0]], dtype=np.float64)
    t = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    got = latent.codi_alignment(s, t)
    exp = ref.codi_alignment(s, t)
    assert np.allclose(float(np.asarray(got)), exp, atol=1e-6)
    sj = jnp.asarray(s)
    tj = jnp.asarray(t)

    def f(student, teacher):
        return latent.codi_alignment(student, teacher)

    gs, gt = jax.grad(f, argnums=(0, 1))(sj, tj)
    assert np.allclose(np.asarray(gt), 0.0)
    expect = 2.0 * (s - t) / s.size
    assert np.allclose(np.asarray(gs), expect, atol=1e-5)


def test_codi_error_order() -> None:
    good = np.ones((2, 2))
    with pytest.raises(latent.LatentError, match="student_h"):
        latent.codi_alignment(np.array([1.0, 2.0]), good)
    with pytest.raises(latent.LatentError, match="teacher_h"):
        latent.codi_alignment(good, np.ones((0, 2)))
    with pytest.raises(latent.LatentError, match="shape"):
        latent.codi_alignment(good, np.ones((2, 3)))


def test_stage_b_objective_adds_alignment() -> None:
    halt = np.array([0.0, 0.0], dtype=np.float64)
    answer = np.array([[8.0, 0.0], [0.0, 8.0]], dtype=np.float64)
    thought = np.array([[8.0, 0.0]], dtype=np.float64)
    ids = np.array([0], dtype=np.int64)
    mask = np.array([1], dtype=np.int64)
    s = np.ones((1, 2))
    t = np.zeros((1, 2))
    cfg = _cfg()
    got = latent.stage_b_objective(
        halt, answer, thought, ids, mask, 0, 0, cfg, s, t, align_weight=2.0
    )
    exp = ref.stage_b_objective(
        halt, answer, thought, ids, mask, 0, 0, cfg, s, t, align_weight=2.0
    )
    assert np.allclose(float(np.asarray(got.l_align)), exp.l_align, atol=1e-5)
    assert np.allclose(float(np.asarray(got.total)), exp.total, atol=1e-5)
    assert np.allclose(float(np.asarray(got.stage_b.total)), float(exp.stage_b.total), atol=1e-5)


def test_stage_b_objective_align_weight_first() -> None:
    with pytest.raises(latent.LatentError, match="align_weight"):
        latent.stage_b_objective(
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            0,
            0,
            _cfg(max_thoughts=-1),
            np.ones((1, 1)),
            np.ones((1, 1)),
            align_weight=True,  # type: ignore[arg-type]
        )
