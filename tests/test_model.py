"""A1 oracle: production ``model`` API vs the NumPy reference.

Groups
------
constants: flagship numbers, LINEAR_TO_MLA, tiny_config fields, enums.
  These may pass against the stub (values live on the interface).
validate_config: tiny/flagship accepted; broken hybrid, layer split,
  prelude/coda, top_k rejected with ConfigError.
attention_kind / ffn_kind: 3:1 hybrid starting at linear; first n_dense
  dense then MoE; OOB raises.
rms_norm: vs reference (1e-5), dtype/shape, zeros, scale invariance,
  directional finite-difference vs reference.
rope / rope_2d: vs reference, partial passthrough, 2-D split, isometry
  on rotated pairs, dtype.
linear_attention: vs reference, causality, chunk/state carry, shapes.
mla_attention: vs reference, causal, QK-norm, output last-dim = d_nope.
moe: vs reference, top-k cardinality, node-limited racks, shared path.
latent_adapter: production call (shape/dtype); numerical check via forward.
init_params / param_count: tree shapes, float32, ~10M tiny, no flagship init.
forward: vs reference on tiny (r=1, r=2, thoughts=None vs thoughts),
  shapes, MTP length, discrete-only.
jax.jit forward: jit(config static, r a Python int) matches eager
  ForwardOutput at 1e-5 (thoughts=None and thoughts provided); jax.grad
  of mean(logits) through the jitted call matches eager at 1e-4; eager
  OOV tokens still raise ConfigError. Not marked gpu.
gpu (V1): JAX GPU vs reference logits 1e-5; skipped without a GPU.

Every test that calls a stubbed function must fail until production is filled in.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import model
from tests.reference import model as ref

TOL = dict(rtol=1e-5, atol=1e-5)


def _np(x: object) -> np.ndarray:
    return np.asarray(x)


def _tree_shapes(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _tree_shapes(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tree_shapes(v) for v in obj]
    if hasattr(obj, "shape"):
        return tuple(int(s) for s in obj.shape)
    return None


def _require_gpu():
    jax = pytest.importorskip("jax")
    gpus = [d for d in jax.devices() if d.platform in ("gpu", "cuda", "tpu")]
    if not gpus:
        pytest.skip("V1 GPU test requires a GPU device")
    return jax


# ---------------------------------------------------------------------------
# constants / tiny_config / enums (may pass on the stub)
# ---------------------------------------------------------------------------


def test_linear_to_mla_constant():
    assert model.LINEAR_TO_MLA == 3


def test_flagship_config_spec_31_numbers():
    cfg = model.flagship_config()
    assert cfg.d_model == 12_288
    assert cfg.n_layers == 96
    assert cfg.n_dense == 3
    assert cfg.n_moe == 93
    assert cfg.n_linear_attn == 72
    assert cfg.n_mla == 24
    assert cfg.n_routed_experts == 512
    assert cfg.n_shared_experts == 2
    assert cfg.top_k == 20
    assert cfg.expert_hidden == 4096
    assert cfg.mtp_heads == 2
    assert cfg.core_block_layers == 8
    assert cfg.recurrence_train_mean == 3.0
    assert cfg.recurrence_max == 16
    assert cfg.vocab_size == 256_000
    assert cfg.max_context == 16_384
    assert cfg.max_racks == 4


def test_tiny_config_v1_field_values():
    cfg = model.tiny_config()
    assert cfg.d_model == 128
    assert cfg.n_layers == 8
    assert cfg.n_dense == 1
    assert cfg.n_moe == 7
    assert cfg.n_linear_attn == 6
    assert cfg.n_mla == 2
    assert cfg.n_routed_experts == 8
    assert cfg.n_shared_experts == 2
    assert cfg.top_k == 2
    assert cfg.expert_hidden == 256
    assert cfg.mtp_heads == 2
    assert cfg.core_block_layers == 2
    assert cfg.vocab_size == 256
    assert cfg.max_context == 128
    assert cfg.prelude_layers == 3
    assert cfg.coda_layers == 3
    assert cfg.adapter_hidden == 128


def test_attention_and_ffn_enums():
    assert model.AttentionKind.LINEAR == "linear"
    assert model.AttentionKind.MLA == "mla"
    assert model.FfnKind.DENSE == "dense"
    assert model.FfnKind.MOE == "moe"
    assert set(model.AttentionKind) == {
        model.AttentionKind.LINEAR,
        model.AttentionKind.MLA,
    }
    assert set(model.FfnKind) == {model.FfnKind.DENSE, model.FfnKind.MOE}


def test_tiny_hybrid_and_ffn_counts_consistent():
    cfg = model.tiny_config()
    assert cfg.n_dense + cfg.n_moe == cfg.n_layers
    assert cfg.n_linear_attn + cfg.n_mla == cfg.n_layers
    assert cfg.n_linear_attn == model.LINEAR_TO_MLA * cfg.n_mla
    assert cfg.prelude_layers + cfg.core_block_layers + cfg.coda_layers == cfg.n_layers


def test_flagship_hybrid_counts_consistent():
    cfg = model.flagship_config()
    assert cfg.n_dense + cfg.n_moe == cfg.n_layers
    assert cfg.n_linear_attn + cfg.n_mla == cfg.n_layers
    assert cfg.n_linear_attn == model.LINEAR_TO_MLA * cfg.n_mla
    assert cfg.n_layers % (model.LINEAR_TO_MLA + 1) == 0


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


def test_validate_config_accepts_tiny_and_flagship():
    model.validate_config(model.tiny_config())
    model.validate_config(model.flagship_config())


def test_validate_config_rejects_broken_hybrid_ratio():
    cfg = dataclasses.replace(model.tiny_config(), n_linear_attn=5, n_mla=3)
    with pytest.raises(model.ConfigError):
        model.validate_config(cfg)


def test_validate_config_rejects_layer_split_mismatch():
    cfg = dataclasses.replace(model.tiny_config(), n_dense=2, n_moe=5)
    with pytest.raises(model.ConfigError):
        model.validate_config(cfg)


def test_validate_config_rejects_prelude_coda_mismatch():
    cfg = dataclasses.replace(model.tiny_config(), prelude_layers=1, coda_layers=1)
    with pytest.raises(model.ConfigError):
        model.validate_config(cfg)


def test_validate_config_rejects_top_k_too_large():
    cfg = dataclasses.replace(model.tiny_config(), top_k=64)
    with pytest.raises(model.ConfigError):
        model.validate_config(cfg)


# ---------------------------------------------------------------------------
# attention_kind / ffn_kind
# ---------------------------------------------------------------------------


def test_attention_kind_hybrid_pattern_tiny():
    cfg = model.tiny_config()
    kinds = [model.attention_kind(i, cfg) for i in range(cfg.n_layers)]
    expected = [
        model.AttentionKind.LINEAR,
        model.AttentionKind.LINEAR,
        model.AttentionKind.LINEAR,
        model.AttentionKind.MLA,
        model.AttentionKind.LINEAR,
        model.AttentionKind.LINEAR,
        model.AttentionKind.LINEAR,
        model.AttentionKind.MLA,
    ]
    assert kinds == expected
    assert kinds.count(model.AttentionKind.LINEAR) == cfg.n_linear_attn
    assert kinds.count(model.AttentionKind.MLA) == cfg.n_mla
    assert kinds[0] is model.AttentionKind.LINEAR


def test_attention_kind_index_zero_is_linear_flagship_period():
    cfg = model.flagship_config()
    assert model.attention_kind(0, cfg) is model.AttentionKind.LINEAR
    assert model.attention_kind(3, cfg) is model.AttentionKind.MLA
    assert model.attention_kind(4, cfg) is model.AttentionKind.LINEAR
    assert model.attention_kind(95, cfg) is model.AttentionKind.MLA


def test_attention_kind_out_of_range():
    cfg = model.tiny_config()
    with pytest.raises(model.AttentionKindError):
        model.attention_kind(-1, cfg)
    with pytest.raises(model.AttentionKindError):
        model.attention_kind(cfg.n_layers, cfg)


def test_ffn_kind_dense_then_moe():
    cfg = model.tiny_config()
    kinds = [model.ffn_kind(i, cfg) for i in range(cfg.n_layers)]
    assert kinds[0] is model.FfnKind.DENSE
    assert all(k is model.FfnKind.MOE for k in kinds[1:])
    assert kinds.count(model.FfnKind.DENSE) == cfg.n_dense
    assert kinds.count(model.FfnKind.MOE) == cfg.n_moe


def test_ffn_kind_out_of_range():
    cfg = model.tiny_config()
    with pytest.raises(model.ConfigError):
        model.ffn_kind(-1, cfg)
    with pytest.raises(model.ConfigError):
        model.ffn_kind(cfg.n_layers, cfg)


# ---------------------------------------------------------------------------
# rms_norm
# ---------------------------------------------------------------------------


def test_rms_norm_matches_reference():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 5, 16)).astype(np.float32)
    w = rng.standard_normal((16,)).astype(np.float32)
    got = _np(model.rms_norm(x, w))
    exp = ref.rms_norm(x, w)
    np.testing.assert_allclose(got, exp, **TOL)
    assert got.dtype == np.float32
    assert got.shape == x.shape


def test_rms_norm_zeros_and_unit_weight():
    x = np.zeros((3, 8), dtype=np.float32)
    w = np.ones((8,), dtype=np.float32)
    y = _np(model.rms_norm(x, w))
    np.testing.assert_allclose(y, 0.0, **TOL)


def test_rms_norm_positive_scale_invariance():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 7)).astype(np.float32)
    w = rng.standard_normal((7,)).astype(np.float32)
    y1 = _np(model.rms_norm(x, w))
    y2 = _np(model.rms_norm(3.0 * x, w))
    np.testing.assert_allclose(y1, y2, **TOL)


def test_rms_norm_directional_fd_matches_reference():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    w = rng.standard_normal((8,)).astype(np.float32)
    v = rng.standard_normal(x.shape).astype(np.float32)
    eps = 1e-3

    def prod_sum(z: np.ndarray) -> float:
        return float(np.sum(_np(model.rms_norm(z, w))))

    def ref_sum(z: np.ndarray) -> float:
        return float(np.sum(ref.rms_norm(z, w)))

    d_prod = (prod_sum(x + eps * v) - prod_sum(x - eps * v)) / (2 * eps)
    d_ref = (ref_sum(x + eps * v) - ref_sum(x - eps * v)) / (2 * eps)
    assert d_prod == pytest.approx(d_ref, rel=1e-3, abs=1e-3)


# ---------------------------------------------------------------------------
# rope / rope_2d
# ---------------------------------------------------------------------------


def test_rope_matches_reference_partial_and_full():
    rng = np.random.default_rng(3)
    q = rng.standard_normal((2, 6, 4, 8)).astype(np.float32)
    k = rng.standard_normal((2, 6, 4, 8)).astype(np.float32)
    pos = np.arange(6, dtype=np.float32)
    for partial in (True, False):
        q_m, k_m = model.rope(q, k, pos, partial=partial)
        q_r, k_r = ref.rope(q, k, pos, partial=partial)
        np.testing.assert_allclose(_np(q_m), q_r, **TOL)
        np.testing.assert_allclose(_np(k_m), k_r, **TOL)
        assert _np(q_m).shape == q.shape
        assert _np(q_m).dtype == np.float32


def test_rope_partial_leaves_unrotated_tail():
    rng = np.random.default_rng(4)
    q = rng.standard_normal((1, 3, 8)).astype(np.float32)
    k = rng.standard_normal((1, 3, 8)).astype(np.float32)
    pos = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    q_out, k_out = model.rope(q, k, pos, partial=True)
    # partial rotates half the pairs → first 4 dims; last 4 passthrough
    np.testing.assert_allclose(_np(q_out)[..., 4:], q[..., 4:], **TOL)
    np.testing.assert_allclose(_np(k_out)[..., 4:], k[..., 4:], **TOL)


def test_rope_preserves_pair_norm():
    rng = np.random.default_rng(5)
    q = rng.standard_normal((2, 5, 8)).astype(np.float32)
    k = rng.standard_normal((2, 5, 8)).astype(np.float32)
    pos = np.arange(5, dtype=np.float32)
    q_out, k_out = model.rope(q, k, pos, partial=False)
    np.testing.assert_allclose(
        np.linalg.norm(_np(q_out), axis=-1),
        np.linalg.norm(q, axis=-1),
        **TOL,
    )
    np.testing.assert_allclose(
        np.linalg.norm(_np(k_out), axis=-1),
        np.linalg.norm(k, axis=-1),
        **TOL,
    )


def test_rope_2d_matches_reference():
    rng = np.random.default_rng(6)
    q = rng.standard_normal((2, 9, 8)).astype(np.float32)
    k = rng.standard_normal((2, 9, 8)).astype(np.float32)
    row = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.float32)
    col = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.float32)
    q_m, k_m = model.rope_2d(q, k, row, col)
    q_r, k_r = ref.rope_2d(q, k, row, col)
    np.testing.assert_allclose(_np(q_m), q_r, **TOL)
    np.testing.assert_allclose(_np(k_m), k_r, **TOL)


# ---------------------------------------------------------------------------
# linear_attention
# ---------------------------------------------------------------------------


def test_linear_attention_matches_reference():
    rng = np.random.default_rng(7)
    q = rng.standard_normal((2, 5, 3, 8)).astype(np.float32)
    k = rng.standard_normal((2, 5, 3, 8)).astype(np.float32)
    v = rng.standard_normal((2, 5, 3, 8)).astype(np.float32)
    out_m, st_m = model.linear_attention(q, k, v)
    out_r, st_r = ref.linear_attention(q, k, v)
    np.testing.assert_allclose(_np(out_m), out_r, **TOL)
    np.testing.assert_allclose(_np(st_m), st_r, **TOL)
    assert _np(out_m).shape == q.shape
    assert _np(st_m).shape == (2, 3, 8, 8)
    assert _np(out_m).dtype == np.float32


def test_linear_attention_is_causal():
    rng = np.random.default_rng(8)
    q = rng.standard_normal((1, 6, 2, 4)).astype(np.float32)
    k = rng.standard_normal((1, 6, 2, 4)).astype(np.float32)
    v = rng.standard_normal((1, 6, 2, 4)).astype(np.float32)
    out, _ = model.linear_attention(q, k, v)
    q2 = q.copy()
    k2 = k.copy()
    v2 = v.copy()
    q2[:, -1] += 1.5
    k2[:, -1] += 1.5
    v2[:, -1] += 1.5
    out2, _ = model.linear_attention(q2, k2, v2)
    np.testing.assert_allclose(_np(out)[:, :-1], _np(out2)[:, :-1], **TOL)


def test_linear_attention_state_chunks_match_full():
    rng = np.random.default_rng(9)
    q = rng.standard_normal((2, 6, 2, 4)).astype(np.float32)
    k = rng.standard_normal((2, 6, 2, 4)).astype(np.float32)
    v = rng.standard_normal((2, 6, 2, 4)).astype(np.float32)
    full, st_full = model.linear_attention(q, k, v)
    a, st = model.linear_attention(q[:, :2], k[:, :2], v[:, :2])
    b, st2 = model.linear_attention(q[:, 2:], k[:, 2:], v[:, 2:], state=st)
    cat = np.concatenate([_np(a), _np(b)], axis=1)
    np.testing.assert_allclose(cat, _np(full), **TOL)
    np.testing.assert_allclose(_np(st2), _np(st_full), **TOL)


def test_linear_attention_3d_squeezes_heads():
    rng = np.random.default_rng(10)
    q = rng.standard_normal((2, 4, 8)).astype(np.float32)
    k = rng.standard_normal((2, 4, 8)).astype(np.float32)
    v = rng.standard_normal((2, 4, 8)).astype(np.float32)
    out, st = model.linear_attention(q, k, v)
    assert _np(out).shape == q.shape
    assert _np(st).shape == (2, 8, 8)
    out_r, st_r = ref.linear_attention(q, k, v)
    np.testing.assert_allclose(_np(out), out_r, **TOL)
    np.testing.assert_allclose(_np(st), st_r, **TOL)


# ---------------------------------------------------------------------------
# mla_attention
# ---------------------------------------------------------------------------


def test_mla_attention_matches_reference():
    rng = np.random.default_rng(11)
    b, s, h, dc, dr = 2, 4, 3, 8, 4
    q = rng.standard_normal((b, s, h, dc + dr)).astype(np.float32)
    ckv = rng.standard_normal((b, s, 1, dc)).astype(np.float32)
    rk = rng.standard_normal((b, s, 1, dr)).astype(np.float32)
    got = _np(model.mla_attention(q, ckv, rk, qk_norm=True))
    exp = ref.mla_attention(q, ckv, rk, qk_norm=True)
    np.testing.assert_allclose(got, exp, **TOL)
    assert got.shape == (b, s, h, dc)
    assert got.dtype == np.float32


def test_mla_attention_qk_norm_false_matches_reference():
    rng = np.random.default_rng(12)
    q = rng.standard_normal((1, 3, 2, 6)).astype(np.float32)
    ckv = rng.standard_normal((1, 3, 2, 4)).astype(np.float32)
    rk = rng.standard_normal((1, 3, 2, 2)).astype(np.float32)
    got = _np(model.mla_attention(q, ckv, rk, qk_norm=False))
    exp = ref.mla_attention(q, ckv, rk, qk_norm=False)
    np.testing.assert_allclose(got, exp, **TOL)


def test_mla_attention_is_causal():
    rng = np.random.default_rng(13)
    q = rng.standard_normal((1, 5, 2, 6)).astype(np.float32)
    ckv = rng.standard_normal((1, 5, 1, 4)).astype(np.float32)
    rk = rng.standard_normal((1, 5, 1, 2)).astype(np.float32)
    out = _np(model.mla_attention(q, ckv, rk))
    q2 = q.copy()
    ckv2 = ckv.copy()
    rk2 = rk.copy()
    q2[:, -1] += 0.75
    ckv2[:, -1] += 0.75
    rk2[:, -1] += 0.75
    out2 = _np(model.mla_attention(q2, ckv2, rk2))
    np.testing.assert_allclose(out[:, :-1], out2[:, :-1], **TOL)


# ---------------------------------------------------------------------------
# moe
# ---------------------------------------------------------------------------


def _toy_moe_weights(rng: np.random.Generator, d: int, h: int, n_r: int, n_s: int):
    router = rng.standard_normal((d, n_r)).astype(np.float32) * 0.02
    routed = (
        rng.standard_normal((n_r, d, h)).astype(np.float32) * 0.02,
        rng.standard_normal((n_r, d, h)).astype(np.float32) * 0.02,
        rng.standard_normal((n_r, h, d)).astype(np.float32) * 0.02,
    )
    shared = (
        rng.standard_normal((n_s, d, h)).astype(np.float32) * 0.02,
        rng.standard_normal((n_s, d, h)).astype(np.float32) * 0.02,
        rng.standard_normal((n_s, h, d)).astype(np.float32) * 0.02,
    )
    return router, routed, shared


def test_moe_matches_reference():
    rng = np.random.default_rng(14)
    x = rng.standard_normal((2, 3, 16)).astype(np.float32)
    router, routed, shared = _toy_moe_weights(rng, 16, 32, 8, 2)
    out_m, probs_m, ids_m = model.moe(
        x,
        router_weight=router,
        routed_weights=routed,
        shared_weights=shared,
        top_k=2,
        max_racks=4,
    )
    out_r, probs_r, ids_r = ref.moe(
        x,
        router_weight=router,
        routed_weights=routed,
        shared_weights=shared,
        top_k=2,
        max_racks=4,
    )
    np.testing.assert_allclose(_np(out_m), out_r, **TOL)
    np.testing.assert_allclose(_np(probs_m), probs_r, **TOL)
    np.testing.assert_array_equal(_np(ids_m), ids_r)
    assert _np(out_m).shape == x.shape
    assert _np(probs_m).shape == (2, 3, 8)
    assert _np(ids_m).shape == (2, 3, 2)
    assert _np(out_m).dtype == np.float32


def test_moe_top_k_unique_and_in_range():
    rng = np.random.default_rng(15)
    x = rng.standard_normal((4, 16)).astype(np.float32)
    router, routed, shared = _toy_moe_weights(rng, 16, 8, 6, 1)
    _, _, ids = model.moe(
        x,
        router_weight=router,
        routed_weights=routed,
        shared_weights=shared,
        top_k=3,
        max_racks=4,
    )
    ids_np = _np(ids)
    assert ids_np.shape == (4, 3)
    assert ids_np.min() >= 0
    assert ids_np.max() < 6
    for row in ids_np:
        assert len(set(row.tolist())) == 3


def test_moe_node_limited_respects_max_racks():
    rng = np.random.default_rng(16)
    x = rng.standard_normal((5, 8)).astype(np.float32)
    router, routed, shared = _toy_moe_weights(rng, 8, 8, 8, 1)
    _, _, ids = model.moe(
        x,
        router_weight=router,
        routed_weights=routed,
        shared_weights=shared,
        top_k=4,
        max_racks=2,
    )
    racks = ref._expert_to_rack(8, 2)
    for row in _np(ids):
        used = {int(racks[int(e)]) for e in row}
        assert len(used) <= 2


# ---------------------------------------------------------------------------
# latent_adapter
# ---------------------------------------------------------------------------


def test_latent_adapter_shape_dtype():
    rng = np.random.default_rng(17)
    x = rng.standard_normal((2, 3, 128)).astype(np.float32)
    y = _np(model.latent_adapter(x, hidden=128))
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.isfinite(y).all()


# ---------------------------------------------------------------------------
# init_params / param_count
# ---------------------------------------------------------------------------


def test_init_params_tree_shapes_match_reference():
    cfg = model.tiny_config()
    got = model.init_params(cfg, rng=0)
    exp = ref.init_params(cfg, rng=0)
    assert _tree_shapes(got) == _tree_shapes(exp)
    assert _np(got["embed"]).dtype == np.float32
    assert len(got["layers"]) == cfg.n_layers
    assert len(got["mtp"]) == cfg.mtp_heads


def test_param_count_matches_reference_tree():
    cfg = model.tiny_config()
    params = model.init_params(cfg, rng=1)
    assert model.param_count(params) == ref.param_count(params)
    ref_params = ref.init_params(cfg, rng=1)
    assert model.param_count(params) == ref.param_count(ref_params)


def test_param_count_on_reference_tree():
    cfg = model.tiny_config()
    params = ref.init_params(cfg, rng=1)
    assert model.param_count(params) == ref.param_count(params)
    assert model.param_count(params) == 7_702_016


def test_tiny_param_count_is_v1_scale():
    cfg = model.tiny_config()
    n = model.param_count(model.init_params(cfg, rng=2))
    assert 1_000_000 <= n <= 20_000_000


def test_init_params_linear_vs_mla_keys():
    cfg = model.tiny_config()
    params = model.init_params(cfg, rng=3)
    linear = params["layers"][0]
    mla = params["layers"][3]
    for key in ("W_q", "W_k", "W_v", "W_o"):
        assert key in linear
    assert "W_kv_compress" in mla
    assert "W_kv_up" in mla
    assert "W_rope_k" in mla
    assert "router" not in params["layers"][0]
    assert "router" in params["layers"][1]


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


def _tiny_batch(cfg: model.ModelConfig, rng: np.random.Generator, seq: int = 4):
    tokens = rng.integers(0, cfg.vocab_size, size=(2, seq), endpoint=False)
    return tokens.astype(np.int32)


def test_forward_matches_reference_discrete():
    cfg = model.tiny_config()
    rng = np.random.default_rng(20)
    params = ref.init_params(cfg, rng=20)
    tokens = _tiny_batch(cfg, rng)
    got = model.forward(tokens, params, cfg, r=1, thoughts=None)
    exp = ref.forward(tokens, params, cfg, r=1, thoughts=None)
    np.testing.assert_allclose(_np(got.logits), exp.logits, **TOL)
    np.testing.assert_allclose(_np(got.hidden), exp.hidden, **TOL)
    assert len(got.mtp_logits) == cfg.mtp_heads
    for a, b in zip(got.mtp_logits, exp.mtp_logits, strict=True):
        np.testing.assert_allclose(_np(a), b, **TOL)
    np.testing.assert_allclose(_np(got.z_loss), exp.z_loss, **TOL)
    assert int(got.r_used) == 1
    assert _np(got.logits).shape == (tokens.shape[0], tokens.shape[1], cfg.vocab_size)
    assert _np(got.logits).dtype == np.float32


def test_forward_thoughts_match_reference_and_differ_from_discrete():
    cfg = model.tiny_config()
    rng = np.random.default_rng(21)
    params = ref.init_params(cfg, rng=21)
    tokens = _tiny_batch(cfg, rng, seq=3)
    thoughts = rng.standard_normal((2, 2, cfg.d_model)).astype(np.float32)
    with_t = model.forward(tokens, params, cfg, r=1, thoughts=thoughts)
    no_t = model.forward(tokens, params, cfg, r=1, thoughts=None)
    exp = ref.forward(tokens, params, cfg, r=1, thoughts=thoughts)
    np.testing.assert_allclose(_np(with_t.logits), exp.logits, **TOL)
    assert not np.allclose(_np(with_t.logits), _np(no_t.logits), atol=1e-5)
    assert _np(with_t.logits).shape == (2, 3, cfg.vocab_size)
    assert _np(with_t.hidden).shape == (2, 3, cfg.d_model)


def test_forward_recurrence_r2_differs_from_r1_and_matches_reference():
    cfg = model.tiny_config()
    rng = np.random.default_rng(22)
    params = ref.init_params(cfg, rng=22)
    tokens = _tiny_batch(cfg, rng, seq=3)
    y1 = model.forward(tokens, params, cfg, r=1)
    y2 = model.forward(tokens, params, cfg, r=2)
    e2 = ref.forward(tokens, params, cfg, r=2)
    np.testing.assert_allclose(_np(y2.logits), e2.logits, **TOL)
    assert int(y2.r_used) == 2
    assert not np.allclose(_np(y1.logits), _np(y2.logits), atol=1e-5)


def test_forward_seq_one_edge():
    cfg = model.tiny_config()
    rng = np.random.default_rng(23)
    params = ref.init_params(cfg, rng=23)
    tokens = rng.integers(0, cfg.vocab_size, size=(1, 1)).astype(np.int32)
    got = model.forward(tokens, params, cfg, r=1)
    exp = ref.forward(tokens, params, cfg, r=1)
    np.testing.assert_allclose(_np(got.logits), exp.logits, **TOL)
    assert _np(got.logits).shape == (1, 1, cfg.vocab_size)


def test_forward_empty_thoughts_is_discrete():
    cfg = model.tiny_config()
    rng = np.random.default_rng(24)
    params = ref.init_params(cfg, rng=24)
    tokens = _tiny_batch(cfg, rng, seq=3)
    empty = np.zeros((2, 0, cfg.d_model), dtype=np.float32)
    a = model.forward(tokens, params, cfg, r=1, thoughts=empty)
    b = model.forward(tokens, params, cfg, r=1, thoughts=None)
    np.testing.assert_allclose(_np(a.logits), _np(b.logits), **TOL)


# ---------------------------------------------------------------------------
# jax.jit(forward) — config static, r a Python int (not GPU-marked)
# ---------------------------------------------------------------------------


def _assert_forward_outputs_close(
    got: model.ForwardOutput,
    exp: model.ForwardOutput,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Compare ForwardOutput fields at FP32 parity; expert_ids and r_used exact."""
    tol = dict(rtol=rtol, atol=atol)
    np.testing.assert_allclose(_np(got.logits), _np(exp.logits), **tol)
    np.testing.assert_allclose(_np(got.hidden), _np(exp.hidden), **tol)
    assert len(got.mtp_logits) == len(exp.mtp_logits)
    for a, b in zip(got.mtp_logits, exp.mtp_logits, strict=True):
        np.testing.assert_allclose(_np(a), _np(b), **tol)
    np.testing.assert_allclose(_np(got.router_probs), _np(exp.router_probs), **tol)
    np.testing.assert_array_equal(_np(got.expert_ids), _np(exp.expert_ids))
    np.testing.assert_allclose(_np(got.z_loss), _np(exp.z_loss), **tol)
    assert int(got.r_used) == int(exp.r_used)


def test_forward_jax_jit_matches_eager_discrete():
    """jit(forward) with config closed over, r=1, thoughts=None matches eager at 1e-5.

    Token-id range checks must not call Python int() on traced token values.
    r=None sampling stays host-side and is not required to jit.
    """
    cfg = model.tiny_config()
    rng = np.random.default_rng(40)
    params = model.init_params(cfg, rng=40)
    tokens = _tiny_batch(cfg, rng, seq=4)
    eager = model.forward(tokens, params, cfg, r=1, thoughts=None)
    jitted = jax.jit(lambda t, p: model.forward(t, p, cfg, r=1, thoughts=None))
    got = jitted(tokens, params)
    _assert_forward_outputs_close(got, eager)
    assert _np(got.logits).shape == (tokens.shape[0], tokens.shape[1], cfg.vocab_size)
    assert _np(got.logits).dtype == np.float32
    assert _np(got.hidden).dtype == np.float32
    assert int(got.r_used) == 1
    exp_ref = ref.forward(tokens, params, cfg, r=1, thoughts=None)
    np.testing.assert_allclose(_np(got.logits), exp_ref.logits, **TOL)


def test_forward_jax_jit_matches_eager_with_thoughts():
    """Same jit contract with thoughts of shape (batch, n_thoughts, d_model)."""
    cfg = model.tiny_config()
    rng = np.random.default_rng(41)
    params = model.init_params(cfg, rng=41)
    tokens = _tiny_batch(cfg, rng, seq=3)
    thoughts = rng.standard_normal((tokens.shape[0], 2, cfg.d_model)).astype(np.float32)
    eager = model.forward(tokens, params, cfg, r=1, thoughts=thoughts)
    jitted = jax.jit(lambda t, p, th: model.forward(t, p, cfg, r=1, thoughts=th))
    got = jitted(tokens, params, thoughts)
    _assert_forward_outputs_close(got, eager)
    assert _np(got.logits).shape == (tokens.shape[0], tokens.shape[1], cfg.vocab_size)
    assert _np(got.hidden).shape == (tokens.shape[0], tokens.shape[1], cfg.d_model)
    assert int(got.r_used) == 1


def test_forward_jax_grad_through_jit_matches_eager():
    """jax.grad of mean(logits) through jitted forward matches eager at 1e-4."""
    cfg = model.tiny_config()
    rng = np.random.default_rng(42)
    params = model.init_params(cfg, rng=42)
    tokens = _tiny_batch(cfg, rng, seq=2)

    def mean_logits(p, t):
        return jnp.mean(model.forward(t, p, cfg, r=1, thoughts=None).logits)

    eager_g = jax.grad(mean_logits, argnums=0)(params, tokens)
    jitted_mean = jax.jit(mean_logits)
    jit_g = jax.grad(jitted_mean, argnums=0)(params, tokens)
    np.testing.assert_allclose(
        _np(jit_g["unembed"]), _np(eager_g["unembed"]), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        _np(jit_g["embed"]), _np(eager_g["embed"]), rtol=1e-4, atol=1e-4
    )


def test_forward_out_of_vocab_raises_eager():
    """Eager path still raises ConfigError on OOV / negative token ids.

    Under jit the raise is not required (XLA cannot raise Python exceptions
    the same way). jit-vs-eager match tests use in-vocab tokens.
    """
    cfg = model.tiny_config()
    params = model.init_params(cfg, rng=43)
    high = np.array([[0, cfg.vocab_size]], dtype=np.int32)
    with pytest.raises(model.ConfigError, match="token id out of vocab"):
        model.forward(high, params, cfg, r=1)
    neg = np.array([[-1, 0]], dtype=np.int32)
    with pytest.raises(model.ConfigError, match="token id out of vocab"):
        model.forward(neg, params, cfg, r=1)


# ---------------------------------------------------------------------------
# V1 GPU (skipped on CPU / without JAX GPU)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_v1_gpu_logits_match_reference_fp32():
    """V1: tiny flagship-shape model, production vs numpy reference to 1e-5."""
    _require_gpu()
    cfg = model.tiny_config()
    rng = np.random.default_rng(30)
    params = ref.init_params(cfg, rng=30)
    tokens = _tiny_batch(cfg, rng, seq=8)
    got = model.forward(tokens, params, cfg, r=1)
    exp = ref.forward(tokens, params, cfg, r=1)
    np.testing.assert_allclose(_np(got.logits), exp.logits, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(_np(got.hidden), exp.hidden, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
def test_v1_gpu_param_count_and_forward_finite():
    """V1: ~10M-param tiny model runs on GPU and stays finite."""
    _require_gpu()
    cfg = model.tiny_config()
    params = model.init_params(cfg, rng=31)
    n = model.param_count(params)
    assert 1_000_000 <= n <= 20_000_000
    rng = np.random.default_rng(31)
    tokens = _tiny_batch(cfg, rng, seq=8)
    out = model.forward(tokens, params, cfg, r=2)
    assert np.isfinite(_np(out.logits)).all()
    assert int(out.r_used) == 2
