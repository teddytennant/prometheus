"""Oracle tests for A2 ``train/`` (spec 5.3 MuonClip / precision, 5.4 WSD, 3.1 losses).

Constants/enums/dataclasses may pass against the committed stubs. Every test that
calls a stubbed function must fail on the stub with NotImplementedError.

Coverage
--------
constants: flagship/tiny match the interface; tiny WSD 2/8/4; n_layers 8.
validate_train_config: accepts tiny and flagship; rejects zero/negative lr,
    warmup+stable+decay == 0, bf16_tail_layers > n_layers.
classify_param: 2D hidden weights Muon; embeddings, norms, biases, sigma AdamW.
precision_for: hidden linears FP8; router/norm/embed/softmax/last two layers/
    latent adapter BF16; never FP32_MASTER (masters are conceptual).
newton_schulz: odd steps >= 1; roughly orthogonal (compact Gram ~ I up to
    scale) on a random matrix; vs reference rtol 1e-5; scale-odd properties.
muon_update: vs reference; momentum carry.
qk_clip: max |q k^T| <= max_logit after clip; vs reference.
adamw_update: decoupled wd, 1-based bias correction, vs reference.
wsd_lr: step 1-based; linear warmup to peak_lr; stable = peak; linear decay
    to 0 at warmup+stable+decay.
soft_cap, cross_entropy (masked), mtp_loss (head i predicts token t+i+1),
    z_loss, total_loss vs reference.
init_opt_state: tree matches classify_param (Muon momentum / AdamW m,v).
apply_precision: does not mutate master dtypes.
train_step: tiny_config + model.tiny_config() + tiny Batch overfits
    directionally (loss decreases); masters stay FP32.
finite-difference: directional derivative of newton_schulz vs reference.
gpu (V1): JAX GPU vs reference Newton–Schulz 1e-5; skipped without a GPU.

Frozen golden scalars live in this file and were computed from the NumPy
reference, not from production.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import model
import train
from tests.reference import train as ref

TOL = dict(rtol=1e-5, atol=1e-5)

# ---------------------------------------------------------------------------
# Frozen goldens (NumPy reference, not production).
# newton_schulz of [[1,2],[3,4],[5,6]] with 5 quintic steps.
# ---------------------------------------------------------------------------
GOLDEN_NS_G = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
GOLDEN_NS_00 = -0.46751821
GOLDEN_NS_21 = 0.24543214
GOLDEN_NS_SUM = 1.4456556

GOLDEN_MUON_G = np.array(
    [[0.5, -0.25], [0.1, 0.8], [-0.3, 0.4]], dtype=np.float32
)
GOLDEN_MUON_D1_00 = 0.01596472
GOLDEN_MUON_D1_SUM = 0.036446046
GOLDEN_MUON_M1_SUM = 1.25

GOLDEN_ADAMW_P = 0.99978
GOLDEN_ADAMW_M = 0.05
GOLDEN_ADAMW_V = 0.0125

GOLDEN_SOFTCAP = np.array(
    [0.0, 22.847826, 28.920828, -13.863516], dtype=np.float32
)
GOLDEN_CE = 0.13238446
GOLDEN_CE_MASKED = 0.16984603
GOLDEN_MTP = 0.035976347
GOLDEN_Z = 2.1461992
GOLDEN_TOTAL = 3.003

GOLDEN_QK_MAX_LOGIT = 1.5
GOLDEN_QK_Q00 = 0.70710677  # sqrt(1.5/3) = sqrt(1/2)


def _np(x: object) -> np.ndarray:
    return np.asarray(x)


def _flatten(obj: object, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(obj, dict):
        out: dict[str, np.ndarray] = {}
        for key, val in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(val, path))
        return out
    if isinstance(obj, (list, tuple)):
        out = {}
        for i, val in enumerate(obj):
            path = f"{prefix}.{i}" if prefix else str(i)
            out.update(_flatten(val, path))
        return out
    if hasattr(obj, "shape"):
        return {prefix: _np(obj)}
    return {}


def expected_kind(name: str, value: object) -> train.ParamKind:
    """Spec 5.3: 2D hidden weights Muon; embeddings, norms, biases, sigma AdamW."""
    n = name.lower()
    if any(tok in n for tok in ("embed", "norm", "bias", "sigma")):
        return train.ParamKind.ADAMW
    if _np(value).ndim == 2:
        return train.ParamKind.MUON_2D
    return train.ParamKind.ADAMW


def expected_precision(name: str, config: train.TrainConfig) -> train.PrecisionKind:
    """Spec 5.3: FP8 hidden linears; BF16 router/norm/embed/softmax/tail/adapter."""
    n = name.lower()
    match = re.match(r"layers\.(\d+)\.", n)
    if match is not None:
        idx = int(match.group(1))
        if idx >= config.n_layers - config.bf16_tail_layers:
            return train.PrecisionKind.BF16_STABLE
    if any(
        tok in n
        for tok in ("router", "norm", "embed", "adapter", "softmax", "sigma")
    ):
        return train.PrecisionKind.BF16_STABLE
    return train.PrecisionKind.FP8_LINEAR


def _tiny_batch(mcfg: model.ModelConfig, batch: int = 2, seq: int = 8) -> train.Batch:
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, mcfg.vocab_size, size=(batch, seq), dtype=np.int32)
    loss_mask = np.ones((batch, seq), dtype=np.float32)
    loss_mask[:, 0] = 0.0
    positions = np.broadcast_to(np.arange(seq, dtype=np.int32), (batch, seq)).copy()
    return train.Batch(tokens=tokens, loss_mask=loss_mask, positions=positions)


def _require_gpu():  # pragma: no cover - imported only on GPU hosts
    jax = pytest.importorskip("jax")
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []
    if not gpus:
        pytest.skip("no GPU")
    return jax


# ---------------------------------------------------------------------------
# Constants (may pass on the stub)
# ---------------------------------------------------------------------------
def test_flagship_constants_match_interface() -> None:
    assert train.FLAGSHIP_PEAK_LR == 2e-4
    assert train.FLAGSHIP_MUON_LR == 2e-2
    assert train.FLAGSHIP_ADAMW_LR == 2e-4
    assert train.FLAGSHIP_MUON_MOMENTUM == 0.95
    assert train.FLAGSHIP_MUON_NS_STEPS == 5
    assert train.FLAGSHIP_ADAMW_BETA1 == 0.9
    assert train.FLAGSHIP_ADAMW_BETA2 == 0.95
    assert train.FLAGSHIP_ADAMW_EPS == 1e-8
    assert train.FLAGSHIP_ADAMW_WD == 0.1
    assert train.FLAGSHIP_WARMUP_STEPS == 2_000
    assert train.FLAGSHIP_STABLE_STEPS == 500_000
    assert train.FLAGSHIP_DECAY_STEPS == 100_000
    assert train.FLAGSHIP_Z_LOSS_WEIGHT == 1e-3
    assert train.FLAGSHIP_SOFTCAP == 30.0
    assert train.FLAGSHIP_QK_CLIP == 100.0
    assert train.FLAGSHIP_GRAD_CLIP == 1.0
    assert train.FLAGSHIP_FP8_BLOCK == 128
    assert train.FLAGSHIP_BF16_TAIL_LAYERS == 2
    assert train.FLAGSHIP_N_LAYERS == 96
    assert train.FLAGSHIP_N_LAYERS == model.FLAGSHIP_N_LAYERS


def test_tiny_constants_wsd_lengths_and_n_layers() -> None:
    cfg = train.tiny_train_config()
    assert cfg.warmup_steps == 2
    assert cfg.stable_steps == 8
    assert cfg.decay_steps == 4
    assert cfg.n_layers == 8
    assert cfg.fp8_block == 16
    assert cfg.peak_lr == train.FLAGSHIP_PEAK_LR
    assert cfg.muon_lr == train.FLAGSHIP_MUON_LR
    assert cfg.adamw_lr == train.FLAGSHIP_ADAMW_LR
    assert cfg.muon_ns_steps == 5
    assert cfg.bf16_tail_layers == 2
    assert cfg.z_loss_weight == train.FLAGSHIP_Z_LOSS_WEIGHT
    assert cfg.softcap == train.FLAGSHIP_SOFTCAP
    assert cfg.qk_clip == train.FLAGSHIP_QK_CLIP


def test_param_and_precision_enums() -> None:
    assert train.ParamKind.MUON_2D.value == "muon_2d"
    assert train.ParamKind.ADAMW.value == "adamw"
    assert train.PrecisionKind.FP8_LINEAR.value == "fp8_linear"
    assert train.PrecisionKind.BF16_STABLE.value == "bf16_stable"
    assert train.PrecisionKind.FP32_MASTER.value == "fp32_master"


# ---------------------------------------------------------------------------
# validate_train_config
# ---------------------------------------------------------------------------
def test_validate_train_config_accepts_tiny_and_flagship() -> None:
    assert train.validate_train_config(train.tiny_train_config()) is None
    assert train.validate_train_config(train.flagship_train_config()) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("peak_lr", 0.0),
        ("peak_lr", -1e-4),
        ("muon_lr", 0.0),
        ("muon_lr", -0.02),
        ("adamw_lr", 0.0),
        ("adamw_lr", -2e-4),
    ],
)
def test_validate_train_config_rejects_zero_or_negative_lr(
    field: str, value: float
) -> None:
    cfg = replace(train.tiny_train_config(), **{field: value})
    with pytest.raises(train.TrainConfigError):
        train.validate_train_config(cfg)


def test_validate_train_config_rejects_zero_total_schedule() -> None:
    cfg = replace(
        train.tiny_train_config(),
        warmup_steps=0,
        stable_steps=0,
        decay_steps=0,
    )
    with pytest.raises(train.TrainConfigError):
        train.validate_train_config(cfg)


def test_validate_train_config_rejects_tail_gt_n_layers() -> None:
    cfg = replace(train.tiny_train_config(), bf16_tail_layers=9)
    assert cfg.n_layers == 8
    with pytest.raises(train.TrainConfigError):
        train.validate_train_config(cfg)


def test_validate_train_config_accepts_tail_equal_n_layers() -> None:
    cfg = replace(train.tiny_train_config(), bf16_tail_layers=8)
    assert train.validate_train_config(cfg) is None


# ---------------------------------------------------------------------------
# classify_param
# ---------------------------------------------------------------------------
def test_classify_param_2d_hidden_is_muon() -> None:
    w = np.zeros((8, 16), dtype=np.float32)
    for name in (
        "layers.0.ffn_gate",
        "layers.0.ffn_up",
        "layers.0.ffn_down",
        "layers.0.W_q",
        "layers.1.shared_gate",
        "mtp.0.proj",
        "adapter_w1",
        "layers.0.router",
    ):
        assert train.classify_param(name, w) == train.ParamKind.MUON_2D, name
        assert train.classify_param(name, w) == expected_kind(name, w)


def test_classify_param_embed_norm_bias_sigma_are_adamw() -> None:
    e = np.zeros((32, 8), dtype=np.float32)
    v = np.ones((8,), dtype=np.float32)
    s = np.array(1.0, dtype=np.float32)
    cases = [
        ("embed", e),
        ("unembed", e),
        ("mtp.0.unembed", e),
        ("final_norm", v),
        ("layers.0.pre_attn_norm", v),
        ("layers.0.pre_ffn_norm", v),
        ("adapter_norm", v),
        ("layers.0.router_bias", v),
        ("bias", v),
        ("latent_sigma", s),
        ("sigma", s),
    ]
    for name, value in cases:
        assert train.classify_param(name, value) == train.ParamKind.ADAMW, name
        assert train.classify_param(name, value) == expected_kind(name, value)


def test_classify_param_3d_is_not_muon_2d() -> None:
    w3 = np.zeros((8, 4, 4), dtype=np.float32)
    assert train.classify_param("layers.0.W_q", w3) == train.ParamKind.ADAMW
    assert train.classify_param("layers.0.routed_gate", w3) == train.ParamKind.ADAMW


def test_classify_param_on_tiny_model_tree() -> None:
    params = model.init_params(model.tiny_config(), rng=0)
    for name, value in _flatten(params).items():
        got = train.classify_param(name, value)
        assert got == expected_kind(name, value), name


# ---------------------------------------------------------------------------
# precision_for
# ---------------------------------------------------------------------------
def test_precision_for_hidden_linears_fp8() -> None:
    cfg = train.tiny_train_config()
    for name in (
        "layers.0.ffn_gate",
        "layers.0.ffn_up",
        "layers.0.W_q",
        "layers.5.W_o",
        "mtp.0.proj",
    ):
        assert train.precision_for(name, cfg) == train.PrecisionKind.FP8_LINEAR
        assert train.precision_for(name, cfg) == expected_precision(name, cfg)


def test_precision_for_router_norm_embed_adapter_softmax_bf16() -> None:
    cfg = train.tiny_train_config()
    for name in (
        "embed",
        "unembed",
        "final_norm",
        "layers.0.pre_attn_norm",
        "layers.0.router",
        "adapter_w1",
        "adapter_w2",
        "adapter_norm",
        "mtp.0.norm",
        "mtp.0.unembed",
        "latent_sigma",
        "layers.0.softmax_scale",
    ):
        assert train.precision_for(name, cfg) == train.PrecisionKind.BF16_STABLE, name
        assert train.precision_for(name, cfg) == expected_precision(name, cfg)


def test_precision_for_last_two_layers_bf16() -> None:
    cfg = train.tiny_train_config()
    assert cfg.n_layers == 8 and cfg.bf16_tail_layers == 2
    assert train.precision_for("layers.6.ffn_gate", cfg) == train.PrecisionKind.BF16_STABLE
    assert train.precision_for("layers.7.W_q", cfg) == train.PrecisionKind.BF16_STABLE
    assert train.precision_for("layers.5.ffn_gate", cfg) == train.PrecisionKind.FP8_LINEAR
    flag = train.flagship_train_config()
    assert train.precision_for("layers.93.W_q", flag) == train.PrecisionKind.FP8_LINEAR
    assert train.precision_for("layers.94.W_q", flag) == train.PrecisionKind.BF16_STABLE
    assert train.precision_for("layers.95.ffn_down", flag) == train.PrecisionKind.BF16_STABLE


def test_precision_for_never_returns_fp32_master() -> None:
    cfg = train.tiny_train_config()
    names = [
        "embed",
        "layers.0.ffn_gate",
        "layers.7.W_q",
        "adapter_w1",
        "latent_sigma",
        "mtp.0.proj",
    ]
    for name in names:
        assert train.precision_for(name, cfg) != train.PrecisionKind.FP32_MASTER


# ---------------------------------------------------------------------------
# newton_schulz
# ---------------------------------------------------------------------------
def test_newton_schulz_rejects_even_and_nonpositive_steps() -> None:
    g = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError):
        train.newton_schulz(g, steps=2)
    with pytest.raises(ValueError):
        train.newton_schulz(g, steps=0)
    with pytest.raises(ValueError):
        train.newton_schulz(g, steps=-1)


def test_newton_schulz_shape_dtype_and_reference() -> None:
    rng = np.random.default_rng(0)
    for shape in ((8, 4), (4, 8), (5, 5)):
        g = rng.standard_normal(shape).astype(np.float32)
        got = train.newton_schulz(g, steps=5)
        exp = ref.newton_schulz(g, steps=5)
        assert _np(got).shape == shape
        assert _np(got).dtype == np.float32
        np.testing.assert_allclose(_np(got), exp, **TOL)


def test_newton_schulz_roughly_orthogonal_tall_matrix() -> None:
    """Compact Gram ~ I up to scale. Tall (8, 4): G.T @ G; also check a wide G @ G.T."""
    rng = np.random.default_rng(0)
    g_tall = rng.standard_normal((8, 4)).astype(np.float32)
    out_tall = _np(train.newton_schulz(g_tall, steps=5))
    np.testing.assert_allclose(out_tall, ref.newton_schulz(g_tall, steps=5), **TOL)
    gram_tall = out_tall.T @ out_tall
    scale = float(np.trace(gram_tall) / gram_tall.shape[0])
    np.testing.assert_allclose(
        gram_tall / scale, np.eye(4, dtype=np.float32), rtol=0.5, atol=0.5
    )
    s = np.linalg.svd(out_tall, compute_uv=False)
    assert 0.3 < float(s.min()) <= float(s.max()) < 2.0

    g_wide = rng.standard_normal((4, 8)).astype(np.float32)
    out_wide = _np(train.newton_schulz(g_wide, steps=5))
    gram_wide = out_wide @ out_wide.T
    scale_w = float(np.trace(gram_wide) / gram_wide.shape[0])
    np.testing.assert_allclose(
        gram_wide / scale_w, np.eye(4, dtype=np.float32), rtol=0.5, atol=0.5
    )


def test_newton_schulz_golden_and_odd_scale_invariance() -> None:
    got = _np(train.newton_schulz(GOLDEN_NS_G, steps=5))
    exp = ref.newton_schulz(GOLDEN_NS_G, steps=5)
    np.testing.assert_allclose(got, exp, **TOL)
    assert got[0, 0] == pytest.approx(GOLDEN_NS_00, rel=1e-5, abs=1e-5)
    assert got[2, 1] == pytest.approx(GOLDEN_NS_21, rel=1e-5, abs=1e-5)
    assert float(got.sum()) == pytest.approx(GOLDEN_NS_SUM, rel=1e-5, abs=1e-5)
    rng = np.random.default_rng(4)
    g = rng.standard_normal((6, 3)).astype(np.float32)
    a = _np(train.newton_schulz(g, steps=5))
    b = _np(train.newton_schulz(np.float32(2.5) * g, steps=5))
    np.testing.assert_allclose(a, b, **TOL)
    c = _np(train.newton_schulz(-g, steps=5))
    np.testing.assert_allclose(c, -a, **TOL)


def test_newton_schulz_zero_matrix() -> None:
    z = np.zeros((5, 3), dtype=np.float32)
    got = _np(train.newton_schulz(z, steps=5))
    np.testing.assert_allclose(got, ref.newton_schulz(z, steps=5), **TOL)
    np.testing.assert_allclose(got, 0.0, atol=1e-7)


def test_newton_schulz_directional_fd_matches_reference() -> None:
    rng = np.random.default_rng(7)
    g = rng.standard_normal((5, 3)).astype(np.float32)
    v = rng.standard_normal(g.shape).astype(np.float32)
    eps = 1e-3

    def prod_sum(z: np.ndarray) -> float:
        return float(np.sum(_np(train.newton_schulz(z, steps=1))))

    def ref_sum(z: np.ndarray) -> float:
        return float(np.sum(ref.newton_schulz(z, steps=1)))

    d_prod = (prod_sum(g + eps * v) - prod_sum(g - eps * v)) / (2.0 * eps)
    d_ref = (ref_sum(g + eps * v) - ref_sum(g - eps * v)) / (2.0 * eps)
    assert d_prod == pytest.approx(d_ref, rel=1e-3, abs=1e-3)


# ---------------------------------------------------------------------------
# muon_update
# ---------------------------------------------------------------------------
def test_muon_update_vs_reference_and_momentum_carry() -> None:
    g = GOLDEN_MUON_G
    m0 = np.zeros_like(g)
    d1, m1 = train.muon_update(
        g, m0, lr=0.02, momentum_coeff=0.95, ns_steps=5
    )
    e1, em1 = ref.muon_update(
        g, m0, lr=0.02, momentum_coeff=0.95, ns_steps=5
    )
    np.testing.assert_allclose(_np(d1), e1, **TOL)
    np.testing.assert_allclose(_np(m1), em1, **TOL)
    assert _np(d1).shape == g.shape and _np(m1).shape == g.shape
    assert _np(d1).dtype == np.float32 and _np(m1).dtype == np.float32
    assert float(_np(d1)[0, 0]) == pytest.approx(GOLDEN_MUON_D1_00, rel=1e-5, abs=1e-5)
    assert float(_np(d1).sum()) == pytest.approx(GOLDEN_MUON_D1_SUM, rel=1e-5, abs=1e-5)
    assert float(_np(m1).sum()) == pytest.approx(GOLDEN_MUON_M1_SUM, rel=1e-5, abs=1e-5)

    d2, m2 = train.muon_update(
        g, m1, lr=0.02, momentum_coeff=0.95, ns_steps=5
    )
    e2, em2 = ref.muon_update(
        g, em1, lr=0.02, momentum_coeff=0.95, ns_steps=5
    )
    np.testing.assert_allclose(_np(d2), e2, **TOL)
    np.testing.assert_allclose(_np(m2), em2, **TOL)
    assert not np.allclose(_np(m2), _np(m1), atol=1e-6)
    # Nesterov carry: m_t = β m + g. With m0 = 0, m1 = g.
    np.testing.assert_allclose(_np(m1), g, **TOL)


def test_muon_update_zero_grad_keeps_decayed_momentum() -> None:
    g = np.zeros((3, 2), dtype=np.float32)
    m = np.ones((3, 2), dtype=np.float32)
    d, m_new = train.muon_update(
        g, m, lr=0.02, momentum_coeff=0.95, ns_steps=5
    )
    ed, em = ref.muon_update(g, m, lr=0.02, momentum_coeff=0.95, ns_steps=5)
    np.testing.assert_allclose(_np(d), ed, **TOL)
    np.testing.assert_allclose(_np(m_new), em, **TOL)


# ---------------------------------------------------------------------------
# qk_clip
# ---------------------------------------------------------------------------
def test_qk_clip_caps_max_logit_vs_reference() -> None:
    q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    k = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]], dtype=np.float32)
    assert float(np.max(np.abs(q @ k.T))) > GOLDEN_QK_MAX_LOGIT
    q2, k2 = train.qk_clip(q, k, GOLDEN_QK_MAX_LOGIT)
    eq, ek = ref.qk_clip(q, k, GOLDEN_QK_MAX_LOGIT)
    np.testing.assert_allclose(_np(q2), eq, **TOL)
    np.testing.assert_allclose(_np(k2), ek, **TOL)
    scores = _np(q2) @ _np(k2).T
    assert float(np.max(np.abs(scores))) <= GOLDEN_QK_MAX_LOGIT + 1e-5
    assert float(_np(q2)[0, 0]) == pytest.approx(GOLDEN_QK_Q00, rel=1e-5, abs=1e-5)
    # Identity when already under the cap.
    q3, k3 = train.qk_clip(q2, k2, 100.0)
    np.testing.assert_allclose(_np(q3), _np(q2), **TOL)
    np.testing.assert_allclose(_np(k3), _np(k2), **TOL)


def test_qk_clip_batched_and_rejects_nonpositive_cap() -> None:
    rng = np.random.default_rng(2)
    q = rng.standard_normal((2, 6, 4)).astype(np.float32)
    k = rng.standard_normal((2, 6, 4)).astype(np.float32)
    q2, k2 = train.qk_clip(q, k, 2.0)
    eq, ek = ref.qk_clip(q, k, 2.0)
    np.testing.assert_allclose(_np(q2), eq, **TOL)
    np.testing.assert_allclose(_np(k2), ek, **TOL)
    scores = np.einsum("...id,...jd->...ij", _np(q2), _np(k2))
    assert float(np.max(np.abs(scores))) <= 2.0 + 1e-5
    with pytest.raises(ValueError):
        train.qk_clip(q, k, 0.0)
    with pytest.raises(ValueError):
        train.qk_clip(q, k, -1.0)


# ---------------------------------------------------------------------------
# adamw_update
# ---------------------------------------------------------------------------
def test_adamw_update_decoupled_and_one_based_vs_reference() -> None:
    p = np.array([1.0], dtype=np.float32)
    g = np.array([0.5], dtype=np.float32)
    m = np.zeros(1, dtype=np.float32)
    v = np.zeros(1, dtype=np.float32)
    p1, m1, v1 = train.adamw_update(
        p,
        g,
        m,
        v,
        lr=2e-4,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        wd=0.1,
        step=1,
    )
    ep, em, ev = ref.adamw_update(
        p,
        g,
        m,
        v,
        lr=2e-4,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        wd=0.1,
        step=1,
    )
    np.testing.assert_allclose(_np(p1), ep, **TOL)
    np.testing.assert_allclose(_np(m1), em, **TOL)
    np.testing.assert_allclose(_np(v1), ev, **TOL)
    assert float(_np(p1)[0]) == pytest.approx(GOLDEN_ADAMW_P, rel=1e-5, abs=1e-5)
    assert float(_np(m1)[0]) == pytest.approx(GOLDEN_ADAMW_M, rel=1e-5, abs=1e-5)
    assert float(_np(v1)[0]) == pytest.approx(GOLDEN_ADAMW_V, rel=1e-5, abs=1e-5)
    # Decoupled: even with g=0 and m=v=0, wd shrinks p.
    p0 = np.array([1.0, -2.0], dtype=np.float32)
    z = np.zeros_like(p0)
    pwd, _, _ = train.adamw_update(
        p0, z, z, z, lr=0.1, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.1, step=1
    )
    np.testing.assert_allclose(
        _np(pwd), p0 * np.float32(1.0 - 0.1 * 0.1), **TOL
    )


def test_adamw_update_step_two_differs_from_step_one() -> None:
    rng = np.random.default_rng(3)
    p = rng.standard_normal((4, 4)).astype(np.float32)
    g = rng.standard_normal((4, 4)).astype(np.float32)
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    kwargs = dict(lr=2e-4, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.1)
    p1, m1, v1 = train.adamw_update(p, g, m, v, step=1, **kwargs)
    p2, m2, v2 = train.adamw_update(p, g, m, v, step=2, **kwargs)
    e1 = ref.adamw_update(p, g, m, v, step=1, **kwargs)
    e2 = ref.adamw_update(p, g, m, v, step=2, **kwargs)
    np.testing.assert_allclose(_np(p1), e1[0], **TOL)
    np.testing.assert_allclose(_np(p2), e2[0], **TOL)
    assert not np.allclose(_np(p1), _np(p2), atol=1e-8)
    with pytest.raises(ValueError):
        train.adamw_update(p, g, m, v, step=0, **kwargs)


# ---------------------------------------------------------------------------
# wsd_lr
# ---------------------------------------------------------------------------
def test_wsd_lr_tiny_warmup_stable_decay() -> None:
    cfg = train.tiny_train_config()
    assert cfg.warmup_steps == 2
    assert cfg.stable_steps == 8
    assert cfg.decay_steps == 4
    peak = cfg.peak_lr
    # 1-based linear warmup to peak.
    assert train.wsd_lr(1, cfg) == pytest.approx(peak * 0.5)
    assert train.wsd_lr(2, cfg) == pytest.approx(peak)
    # Stable = peak (steps 3..10).
    for step in range(3, 11):
        assert train.wsd_lr(step, cfg) == pytest.approx(peak)
    # Linear decay to 0 at warmup+stable+decay = 14.
    assert train.wsd_lr(11, cfg) == pytest.approx(peak * 0.75)
    assert train.wsd_lr(12, cfg) == pytest.approx(peak * 0.5)
    assert train.wsd_lr(13, cfg) == pytest.approx(peak * 0.25)
    assert train.wsd_lr(14, cfg) == pytest.approx(0.0)
    assert train.wsd_lr(15, cfg) == pytest.approx(0.0)
    for step in range(1, 16):
        assert train.wsd_lr(step, cfg) == pytest.approx(ref.wsd_lr(step, cfg))


def test_wsd_lr_flagship_points_and_rejects_step_lt_1() -> None:
    cfg = train.flagship_train_config()
    peak = cfg.peak_lr
    assert train.wsd_lr(1, cfg) == pytest.approx(peak / 2000)
    assert train.wsd_lr(2000, cfg) == pytest.approx(peak)
    assert train.wsd_lr(2001, cfg) == pytest.approx(peak)
    assert train.wsd_lr(2_000 + 500_000, cfg) == pytest.approx(peak)
    assert train.wsd_lr(2_000 + 500_000 + 100_000, cfg) == pytest.approx(0.0)
    assert train.wsd_lr(1, cfg) == pytest.approx(ref.wsd_lr(1, cfg))
    with pytest.raises(ValueError):
        train.wsd_lr(0, cfg)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def test_soft_cap_vs_reference_and_golden() -> None:
    x = np.array([0.0, 30.0, 60.0, -15.0], dtype=np.float32)
    got = _np(train.soft_cap(x, 30.0))
    np.testing.assert_allclose(got, ref.soft_cap(x, 30.0), **TOL)
    np.testing.assert_allclose(got, GOLDEN_SOFTCAP, **TOL)
    np.testing.assert_allclose(got, 30.0 * np.tanh(x / 30.0), **TOL)
    assert got.dtype == np.float32
    with pytest.raises(ValueError):
        train.soft_cap(x, 0.0)


def test_cross_entropy_masked_vs_reference() -> None:
    logits = np.array([[[2.0, 0.0, -1.0], [0.0, 3.0, 0.0]]], dtype=np.float32)
    targets = np.array([[0, 1]], dtype=np.int32)
    mask = np.array([[1.0, 1.0]], dtype=np.float32)
    got = _np(train.cross_entropy(logits, targets, mask))
    assert got.shape == () or got.ndim == 0
    assert float(got) == pytest.approx(GOLDEN_CE, rel=1e-5, abs=1e-5)
    np.testing.assert_allclose(got, ref.cross_entropy(logits, targets, mask), **TOL)
    mask2 = np.array([[1.0, 0.0]], dtype=np.float32)
    got2 = float(_np(train.cross_entropy(logits, targets, mask2)))
    assert got2 == pytest.approx(GOLDEN_CE_MASKED, rel=1e-5, abs=1e-5)
    # All-zero mask → 0.
    z = float(_np(train.cross_entropy(logits, targets, np.zeros_like(mask))))
    assert z == pytest.approx(0.0, abs=1e-7)
    # Masked position does not affect the loss.
    logits_b = logits.copy()
    logits_b[0, 1] = np.array([9.0, -9.0, 4.0], dtype=np.float32)
    a = float(_np(train.cross_entropy(logits, targets, mask2)))
    b = float(_np(train.cross_entropy(logits_b, targets, mask2)))
    assert a == pytest.approx(b, rel=1e-5, abs=1e-5)


def test_mtp_loss_head_i_predicts_token_t_plus_i_plus_1() -> None:
    tokens = np.array([[1, 2, 0, 1]], dtype=np.int32)
    mask = np.ones((1, 4), dtype=np.float32)
    h0 = np.zeros((1, 4, 3), dtype=np.float32)
    h0[0, 0, 2] = 4.0
    h0[0, 1, 0] = 4.0
    h0[0, 2, 1] = 4.0
    h1 = np.zeros((1, 4, 3), dtype=np.float32)
    h1[0, 0, 0] = 4.0
    h1[0, 1, 1] = 4.0
    got = _np(train.mtp_loss((h0, h1), tokens, mask))
    np.testing.assert_allclose(got, ref.mtp_loss((h0, h1), tokens, mask), **TOL)
    assert float(got) == pytest.approx(GOLDEN_MTP, rel=1e-5, abs=1e-5)
    # Sequence shorter than any extra-token offset → 0.
    short = np.array([[1]], dtype=np.int32)
    short_m = np.ones((1, 1), dtype=np.float32)
    h = np.zeros((1, 1, 3), dtype=np.float32)
    z = float(_np(train.mtp_loss((h,), short, short_m)))
    assert z == pytest.approx(0.0, abs=1e-7)


def test_z_loss_vs_reference() -> None:
    probs = np.array([[[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]], dtype=np.float32)
    z = _np(train.z_loss(probs))
    np.testing.assert_allclose(z, ref.z_loss(probs), **TOL)
    assert float(z) == pytest.approx(GOLDEN_Z, rel=1e-5, abs=1e-5)


def test_total_loss_vs_reference() -> None:
    cfg = SimpleNamespace(z_loss_weight=1e-3)
    tot = _np(train.total_loss(2.0, 1.0, 3.0, cfg))
    np.testing.assert_allclose(tot, ref.total_loss(2.0, 1.0, 3.0, cfg), **TOL)
    assert float(tot) == pytest.approx(GOLDEN_TOTAL, rel=1e-5, abs=1e-5)
    tcfg = train.tiny_train_config()
    tot2 = _np(train.total_loss(2.0, 1.0, 3.0, tcfg))
    assert float(tot2) == pytest.approx(
        2.0 + 1.0 + tcfg.z_loss_weight * 3.0, rel=1e-5, abs=1e-5
    )


# ---------------------------------------------------------------------------
# init_opt_state
# ---------------------------------------------------------------------------
def _assert_opt_state_matches(params: object, state: object, prefix: str = "") -> None:
    if isinstance(params, dict):
        assert isinstance(state, dict), prefix
        for key, val in params.items():
            assert key in state, f"missing opt state for {prefix}.{key}"
            path = f"{prefix}.{key}" if prefix else str(key)
            _assert_opt_state_matches(val, state[key], path)
        return
    if isinstance(params, (list, tuple)):
        assert len(state) == len(params), prefix
        for i, (pval, sval) in enumerate(zip(params, state, strict=True)):
            path = f"{prefix}.{i}" if prefix else str(i)
            _assert_opt_state_matches(pval, sval, path)
        return
    kind = train.classify_param(prefix, params)
    assert isinstance(state, dict), prefix
    shape = tuple(_np(params).shape)
    if kind == train.ParamKind.MUON_2D:
        mom = _np(state["momentum"])
        assert mom.shape == shape
        assert mom.dtype == np.float32
        np.testing.assert_allclose(mom, 0.0, atol=0.0)
    else:
        m = _np(state["m"])
        v = _np(state["v"])
        assert m.shape == v.shape == shape
        assert m.dtype == np.float32 and v.dtype == np.float32
        np.testing.assert_allclose(m, 0.0, atol=0.0)
        np.testing.assert_allclose(v, 0.0, atol=0.0)


def test_init_opt_state_tree_matches_classify_param() -> None:
    params = {
        "embed": np.zeros((4, 8), dtype=np.float32),
        "final_norm": np.ones((8,), dtype=np.float32),
        "layers": [
            {
                "ffn_gate": np.zeros((8, 16), dtype=np.float32),
                "pre_attn_norm": np.ones((8,), dtype=np.float32),
                "router_bias": np.zeros((4,), dtype=np.float32),
            }
        ],
        "latent_sigma": np.array(1.0, dtype=np.float32),
        "adapter_w1": np.zeros((8, 8), dtype=np.float32),
    }
    state = train.init_opt_state(params, train.tiny_train_config())
    _assert_opt_state_matches(params, state)
    assert "momentum" in state["layers"][0]["ffn_gate"]
    assert "m" in state["embed"] and "v" in state["embed"]
    assert "momentum" in state["adapter_w1"]
    assert "m" in state["latent_sigma"]


def test_init_opt_state_on_tiny_model_params() -> None:
    params = model.init_params(model.tiny_config(), rng=0)
    state = train.init_opt_state(params, train.tiny_train_config())
    _assert_opt_state_matches(params, state)


# ---------------------------------------------------------------------------
# apply_precision / train_step
# ---------------------------------------------------------------------------
def test_apply_precision_does_not_mutate_master_dtypes() -> None:
    cfg = train.tiny_train_config()
    params = {
        "embed": np.ones((4, 8), dtype=np.float32),
        "layers.0.ffn_gate": np.linspace(0.0, 1.0, 32, dtype=np.float32).reshape(8, 4),
        "layers.7.ffn_gate": np.ones((8, 4), dtype=np.float32),
        "final_norm": np.ones((8,), dtype=np.float32),
        "adapter_w1": np.ones((8, 8), dtype=np.float32),
    }
    snapshot = {k: v.copy() for k, v in params.items()}
    out = train.apply_precision(params, cfg)
    for key, val in params.items():
        assert val.dtype == np.float32, key
        np.testing.assert_array_equal(val, snapshot[key])
    out_leaves = _flatten(out)
    in_leaves = _flatten(params)
    assert set(out_leaves) == set(in_leaves)
    for key, val in in_leaves.items():
        assert out_leaves[key].shape == val.shape, key
    # Poke the returned tree; masters must stay FP32 copies.
    first_key = next(iter(out_leaves))
    view = np.asarray(out_leaves[first_key])
    if view.flags.writeable:
        view.reshape(-1)[0] = np.float32(123.0)
    for key, val in params.items():
        assert val.dtype == np.float32
        np.testing.assert_array_equal(val, snapshot[key])


def test_train_step_overfits_tiny_and_keeps_fp32_masters() -> None:
    mcfg = model.tiny_config()
    tcfg = train.tiny_train_config()
    params = model.init_params(mcfg, rng=0)
    opt = train.init_opt_state(params, tcfg)
    batch = _tiny_batch(mcfg)
    losses: list[float] = []
    p, o = params, opt
    last = None
    for step in range(1, 4):
        last = train.train_step(p, o, batch, mcfg, tcfg, step=step)
        losses.append(float(_np(last.loss.total)))
        p, o = last.params, last.opt_state
        assert last.step == step
        assert float(last.lr) == pytest.approx(ref.wsd_lr(step, tcfg))
        for name, val in _flatten(last.params).items():
            if _np(val).dtype.kind == "f":
                assert _np(val).dtype == np.float32, name
    assert last is not None
    assert np.isfinite(losses[-1])
    assert np.isfinite(float(_np(last.grad_norm)))
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# GPU (V1)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_v1_gpu_newton_schulz_matches_reference() -> None:
    """V1: Muon Newton–Schulz on GPU matches the NumPy reference to 1e-5."""
    jax = _require_gpu()
    rng = np.random.default_rng(0)
    g = rng.standard_normal((8, 4)).astype(np.float32)
    got = train.newton_schulz(jax.device_put(g), steps=5)
    exp = ref.newton_schulz(g, steps=5)
    np.testing.assert_allclose(_np(got), exp, rtol=1e-5, atol=1e-5)
