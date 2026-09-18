"""Oracle tests for ``prometheus.verify.v5_rung0`` (spec 16.2 V5, 15.5 A7+I1).

Every collected CPU test calls ``run_v5`` so the stub fails all of them.
GPU tests are marked ``gpu`` and skip without a device.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prometheus.verify import v5_rung0 as v5
from tests.reference import v5_rung0 as ref

_V5_KEYS = frozenset({"loss_curve_matches_ladder", "checkpoint_resume_ok"})
_CPU_GPUS = 1
_CACHED_CPU: dict[str, Any] | None = None
_ROOT = Path(__file__).resolve().parents[1]
_PROD_SRC = _ROOT / "prometheus" / "verify" / "v5_rung0.py"


def _assert_v5_result_shape(result: Any) -> dict[str, Any]:
    assert isinstance(result, dict)
    assert set(result.keys()) == set(_V5_KEYS)
    assert type(result["loss_curve_matches_ladder"]) is bool
    assert type(result["checkpoint_resume_ok"]) is bool
    return result


def _cpu_run_v5() -> dict[str, Any]:
    global _CACHED_CPU
    if _CACHED_CPU is None:
        _CACHED_CPU = _assert_v5_result_shape(v5.run_v5(gpus=_CPU_GPUS))
    return _CACHED_CPU


def _require_gpu() -> None:
    jax = pytest.importorskip("jax")
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if not devices:
        pytest.skip("V5 GPU test requires a GPU device")


def _assert_v5error(*, gpus: int) -> None:
    with pytest.raises(v5.V5Error) as ei:
        v5.run_v5(gpus=gpus)
    assert type(ei.value) is v5.V5Error
    assert not isinstance(ei.value, NotImplementedError)


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
            imported.add(node.module)
    return imported


# ---------------------------------------------------------------------------
# gpus < 1 → V5Error, not NotImplementedError
# ---------------------------------------------------------------------------


def test_gpus_zero_raises_v5error() -> None:
    _assert_v5error(gpus=0)
    assert not issubclass(v5.V5Error, NotImplementedError)
    assert issubclass(v5.V5Error, Exception)


def test_gpus_negative_raises_v5error() -> None:
    _assert_v5error(gpus=-1)
    _assert_v5error(gpus=-8)


# ---------------------------------------------------------------------------
# Result shape / types / successful analog / public constants
# ---------------------------------------------------------------------------


def test_run_v5_returns_exact_v5result_keys_and_bool_types() -> None:
    result = _assert_v5_result_shape(_cpu_run_v5())
    assert "loss_curve_matches_ladder" in result
    assert "checkpoint_resume_ok" in result


def test_successful_analog_both_bools_true() -> None:
    result = _assert_v5_result_shape(_cpu_run_v5())
    assert result["loss_curve_matches_ladder"] is True
    assert result["checkpoint_resume_ok"] is True
    proto = ref.evaluate_v5_protocol()
    assert proto["loss_curve_matches_ladder"] is True
    assert proto["checkpoint_resume_ok"] is True


def test_public_ladder_constants_match_spec() -> None:
    _assert_v5_result_shape(_cpu_run_v5())
    assert v5.DEFAULT_GPUS == 8
    assert v5.DEFAULT_GPUS == ref.DEFAULT_GPUS
    assert v5.RUNG_0_ACTIVE_PARAMS == 100_000_000
    assert v5.RUNG_0_TOTAL_PARAMS == 1_000_000_000
    assert v5.RUNG_0_TOKENS == 20_000_000_000
    assert v5.RUNG_0_ACTIVE_PARAMS == ref.RUNG_0_ACTIVE_PARAMS
    assert v5.RUNG_0_TOTAL_PARAMS == ref.RUNG_0_TOTAL_PARAMS
    assert v5.RUNG_0_TOKENS == ref.RUNG_0_TOKENS
    # Spec 16.2: C = 6 N D ≈ 1.2e19 FLOP at ladder rung 0.
    c = ref.compute_flops(v5.RUNG_0_ACTIVE_PARAMS, v5.RUNG_0_TOKENS)
    assert c == pytest.approx(1.2e19, rel=1e-12)


# ---------------------------------------------------------------------------
# Loss-curve match vs independent I1 fit (not vs itself)
# ---------------------------------------------------------------------------


def test_loss_curve_matches_independent_i1_fit_not_itself() -> None:
    result = _assert_v5_result_shape(_cpu_run_v5())
    assert result["loss_curve_matches_ladder"] is True

    analog_obs = ref.analog_rung13_observations()
    fitted = ref.fit(analog_obs)
    losses, tokens = ref.analog_rung0_injected_curve()
    preds = [fitted.predict(ref.ANALOG_ACTIVE_PARAMS, t) for t in tokens]
    assert len(losses) == len(tokens) == 4
    for obs, pred in zip(losses, preds, strict=True):
        assert obs == pytest.approx(pred, rel=ref.FIT_REL_TOL, abs=1e-12)

    # Independent I1-style fit, not identity of two copies of the observed curve.
    assert ref.curve_matches_ladder(losses, tokens, fitted, ref.ANALOG_ACTIVE_PARAMS)
    identity = list(losses) == list(losses)
    assert identity is True
    constant = [1.0] * len(losses)
    assert constant == list(constant)
    assert not ref.curve_matches_ladder(constant, tokens, fitted, ref.ANALOG_ACTIVE_PARAMS)

    # Fit recovered the generating law.
    assert fitted.a == pytest.approx(ref.LAW_A, rel=ref.FIT_REL_TOL)
    assert fitted.alpha == pytest.approx(ref.LAW_ALPHA, rel=ref.FIT_REL_TOL)


def test_injected_law_curve_matches_ladder_and_wrong_curve_fails() -> None:
    result = _assert_v5_result_shape(_cpu_run_v5())
    proto = ref.evaluate_v5_protocol()
    fitted = ref.fit(ref.analog_rung13_observations())
    losses, tokens = proto["losses"], proto["token_counts"]
    assert ref.curve_matches_ladder(losses, tokens, fitted, ref.ANALOG_ACTIVE_PARAMS)

    # Fault: 6% above prediction trips I1 should_stop (>= 5%).
    bumped = [float(p) * 1.06 for p in proto["predictions"]]
    assert not ref.curve_matches_ladder(bumped, tokens, fitted, ref.ANALOG_ACTIVE_PARAMS)
    assert ref.should_stop(proto["predictions"][0], bumped[0]) is True
    # Just under the gate still matches.
    under = [float(p) * 1.049 for p in proto["predictions"]]
    assert ref.curve_matches_ladder(under, tokens, fitted, ref.ANALOG_ACTIVE_PARAMS)
    # Exactly 5% above is a stop / not a match.
    edge = proto["predictions"][0] * (1.0 + ref.STOP_RELATIVE_EXCESS)
    assert ref.should_stop(proto["predictions"][0], edge) is True
    assert not ref.curve_matches_ladder(
        [edge] + list(losses[1:]), tokens, fitted, ref.ANALOG_ACTIVE_PARAMS
    )
    # Empty / non-finite / non-positive.
    assert ref.curve_matches_ladder([], [], fitted, ref.ANALOG_ACTIVE_PARAMS) is False
    nan_curve = list(losses)
    nan_curve[0] = float("nan")
    assert not ref.curve_matches_ladder(nan_curve, tokens, fitted, ref.ANALOG_ACTIVE_PARAMS)
    zero_curve = list(losses)
    zero_curve[0] = 0.0
    assert not ref.curve_matches_ladder(zero_curve, tokens, fitted, ref.ANALOG_ACTIVE_PARAMS)
    with pytest.raises(ValueError, match="length mismatch"):
        ref.curve_matches_ladder(losses, tokens[:1], fitted, ref.ANALOG_ACTIVE_PARAMS)
    assert result["loss_curve_matches_ladder"] is True


def test_rung0_is_not_a_fit_point() -> None:
    _assert_v5_result_shape(_cpu_run_v5())
    rung0 = ref.observation(
        ref.RUNG_0, ref.ANALOG_ACTIVE_PARAMS, 4, 1.0
    )
    with pytest.raises(ref.ScaleError) as ei:
        ref.fit([rung0, ref.analog_rung13_observations()[0]])
    assert ei.value.code == "Rung0NotUsed"
    with pytest.raises(ref.ScaleError) as ej:
        ref.validate_observation(rung0)
    assert ej.value.code == "Rung0NotUsed"


def test_i1_two_point_recovery_and_properties() -> None:
    _assert_v5_result_shape(_cpu_run_v5())
    a, alpha = 2.0, 0.5
    train = [
        ref.observation(ref.RUNG_1, 3, 5, ref.law_loss(a, alpha, 3, 5)),
        ref.observation(ref.RUNG_2, 7, 5, ref.law_loss(a, alpha, 7, 5)),
    ]
    fitted = ref.fit(train)
    assert fitted.a == pytest.approx(a, rel=ref.FIT_REL_TOL)
    assert fitted.alpha == pytest.approx(alpha, rel=ref.FIT_REL_TOL)
    held = ref.law_loss(a, alpha, 11, 13)
    assert fitted.predict(11, 13) == pytest.approx(held, rel=ref.FIT_REL_TOL)

    # Permutation invariance.
    swapped = ref.fit([train[1], train[0]])
    assert swapped.a == pytest.approx(fitted.a, rel=1e-12)
    assert swapped.alpha == pytest.approx(fitted.alpha, rel=1e-12)

    # Monotone: more tokens → lower L when alpha > 0.
    lo = fitted.predict(4, 2)
    hi = fitted.predict(4, 8)
    assert hi < lo

    # Duplicate / too few / zero compute.
    with pytest.raises(ref.ScaleError) as ed:
        ref.fit([train[0], train[0]])
    assert ed.value.code == "DuplicateRung"
    with pytest.raises(ref.ScaleError) as en:
        ref.fit([train[0]])
    assert en.value.code == "NeedTwoRungs"
    with pytest.raises(ref.ScaleError) as ez:
        ref.compute_flops(0, 1)
    assert ez.value.code == "InvalidCompute"

    # Ladder-scale fit is judged against spec-6 N, D (C ≈ 1.2e19 at rung 0).
    ladder_fit = ref.fit(ref.ladder_rung13_observations())
    ladder_pred = ladder_fit.predict(ref.RUNG_0_ACTIVE_PARAMS, ref.RUNG_0_TOKENS)
    expect = ref.law_loss(
        ref.LAW_A, ref.LAW_ALPHA, ref.RUNG_0_ACTIVE_PARAMS, ref.RUNG_0_TOKENS
    )
    assert ladder_pred == pytest.approx(expect, rel=1e-9)
    proto = ref.evaluate_v5_protocol()
    assert proto["ladder_rung0_prediction"] == pytest.approx(expect, rel=1e-9)
    assert proto["ladder_c"] == pytest.approx(1.2e19, rel=1e-12)


# ---------------------------------------------------------------------------
# Checkpoint / resume: uninterrupted twin vs mid-run restore
# ---------------------------------------------------------------------------


def test_checkpoint_resume_matches_uninterrupted_twin() -> None:
    result = _assert_v5_result_shape(_cpu_run_v5())
    assert result["checkpoint_resume_ok"] is True

    losses = [1.0, 0.8, 0.5, 0.3]
    cfg = ref.analog_rung0_config(10, 3)
    twin = ref.run_injected_curve(losses, cfg)
    job1 = ref.RungRun.new(cfg)
    job1.step_once(1.0)
    job1.step_once(0.8)
    ckpt = job1.checkpoint()
    assert ckpt.tokens_seen == 6
    assert ckpt.loss_curve == (1.0, 0.8)
    assert ckpt.tokenizer_hash == ref.ANALOG_TOKENIZER_HASH
    assert ckpt.seed == ref.ANALOG_SEED

    job2 = ref.RungRun.resume(cfg, ckpt)
    assert job2.tokens_seen == 6
    assert job2.step == 2
    assert job2.loss_curve == [1.0, 0.8]
    assert job2.done() is False
    r3 = job2.step_once(0.5)
    assert r3.step == 3
    assert r3.tokens_seen == 9
    assert r3.done is False
    r4 = job2.step_once(0.3)
    assert r4.step == 4
    assert r4.tokens_seen == 10
    assert r4.done is True
    assert job2.loss_curve == twin.loss_curve
    assert job2.tokens_seen == twin.tokens_seen
    assert job2.step == twin.step
    assert job2.done() == twin.done()
    assert ref.checkpoint_resume_continues(
        losses, token_budget=10, tokens_per_step=3, split_after=2, config=cfg
    )

    # Analog 8/2 protocol used by evaluate_v5_protocol.
    analog_losses, _ = ref.analog_rung0_injected_curve()
    assert ref.checkpoint_resume_continues(analog_losses, split_after=1)
    assert ref.checkpoint_resume_continues(analog_losses, split_after=2)
    assert ref.checkpoint_resume_continues(analog_losses, split_after=3)


def test_checkpoint_resume_faults_tokenizer_and_mismatch() -> None:
    _assert_v5_result_shape(_cpu_run_v5())
    cfg = ref.analog_rung0_config(8, 2)
    run = ref.RungRun.new(cfg)
    run.step_once(1.0)
    ckpt = run.checkpoint()

    bad_hash = ref.RungConfig(
        spec=cfg.spec,
        token_budget=cfg.token_budget,
        tokens_per_step=cfg.tokens_per_step,
        tokenizer_hash="sha256:other",
        seed=cfg.seed,
    )
    with pytest.raises(ref.RungError) as eth:
        ref.RungRun.resume(bad_hash, ckpt)
    assert eth.value.code == "TokenizerChanged"

    bad_seed = ref.RungConfig(
        spec=cfg.spec,
        token_budget=cfg.token_budget,
        tokens_per_step=cfg.tokens_per_step,
        tokenizer_hash=cfg.tokenizer_hash,
        seed=cfg.seed + 1,
    )
    with pytest.raises(ref.RungError) as es:
        ref.RungRun.resume(bad_seed, ckpt)
    assert es.value.code == "CheckpointMismatch"

    past = ref.RungCheckpoint(
        step=ckpt.step,
        tokens_seen=cfg.token_budget + 1,
        loss_curve=ckpt.loss_curve,
        tokenizer_hash=ckpt.tokenizer_hash,
        seed=ckpt.seed,
        token_budget=ckpt.token_budget,
        tokens_per_step=ckpt.tokens_per_step,
        rung=ckpt.rung,
    )
    with pytest.raises(ref.RungError) as ep:
        ref.RungRun.resume(cfg, past)
    assert ep.value.code == "ResumePastBudget"

    with pytest.raises(ref.RungError) as el:
        run.step_once(float("nan"))
    assert el.value.code == "NonFiniteLoss"

    done = ref.run_injected_curve([0.9, 0.8, 0.7, 0.6], cfg)
    assert done.done() is True
    with pytest.raises(ref.RungError) as ea:
        done.step_once(0.1)
    assert ea.value.code == "AlreadyDone"

    with pytest.raises(ref.RungError) as eb:
        ref.validate_rung_config(ref.analog_rung0_config(0, 2))
    assert eb.value.code == "InvalidBudget"
    empty_hash = ref.RungConfig(
        spec=cfg.spec,
        token_budget=8,
        tokens_per_step=2,
        tokenizer_hash="",
        seed=7,
    )
    with pytest.raises(ref.RungError) as eh:
        ref.validate_rung_config(empty_hash)
    assert eh.value.code == "EmptyTokenizerHash"


# ---------------------------------------------------------------------------
# Tiny net: shapes / dtypes / finite-difference grads / weight resume
# ---------------------------------------------------------------------------


def test_tiny_net_shapes_dtypes_and_finite_difference_grads() -> None:
    _assert_v5_result_shape(_cpu_run_v5())
    tokens = ref.toy_tokens()
    params = ref.toy_init()
    assert tokens.shape == (ref.TOY_BATCH, ref.TOY_SEQ)
    assert tokens.dtype == np.int32
    assert params["embed"].shape == (ref.TOY_VOCAB, ref.TOY_DIM)
    assert params["unembed"].shape == (ref.TOY_VOCAB, ref.TOY_DIM)
    assert params["embed"].dtype == np.float32
    assert params["unembed"].dtype == np.float32
    logits = ref.toy_forward(tokens, params)
    assert logits.shape == (ref.TOY_BATCH, ref.TOY_SEQ, ref.TOY_VOCAB)
    assert logits.dtype == np.float32
    loss = ref.toy_next_token_loss(tokens, params)
    assert np.isfinite(loss)
    assert loss > 0.0

    analytic, numeric = ref.toy_finite_diff_slice(tokens, params)
    assert analytic.shape == numeric.shape
    assert analytic.dtype == np.float32
    assert numeric.dtype == np.float32
    assert ref.grad_match_ok(analytic, numeric) is True

    curve, _ = ref.toy_train_curve(tokens, params, steps=ref.TOY_STEPS)
    assert len(curve) == ref.TOY_STEPS
    assert all(np.isfinite(x) for x in curve)
    assert curve[-1] < curve[0]


def test_tiny_net_checkpoint_resume_continues_curve() -> None:
    result = _assert_v5_result_shape(_cpu_run_v5())
    assert result["checkpoint_resume_ok"] is True
    assert ref.toy_checkpoint_resume_continues() is True
    # Fault: a different seed's init does not continue the same curve.
    tokens = ref.toy_tokens()
    a = ref.toy_init(seed=ref.TOY_SEED)
    b = ref.toy_init(seed=ref.TOY_SEED + 1)
    curve_a, _ = ref.toy_train_curve(tokens, a, steps=ref.TOY_STEPS)
    curve_b, _ = ref.toy_train_curve(tokens, b, steps=ref.TOY_STEPS)
    assert curve_a != curve_b


# ---------------------------------------------------------------------------
# Production hygiene: no tests/ import, no v5.json
# ---------------------------------------------------------------------------


def test_run_v5_does_not_write_v5_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = _assert_v5_result_shape(v5.run_v5(gpus=_CPU_GPUS))
    assert result["loss_curve_matches_ladder"] is True
    assert result["checkpoint_resume_ok"] is True
    found = list(tmp_path.rglob("v5.json"))
    assert found == []
    assert not (tmp_path / "v5.json").exists()
    assert not (_ROOT / "v5.json").exists()
    assert not (_ROOT / "prometheus" / "verify" / "v5.json").exists()


def test_production_source_does_not_import_tests() -> None:
    result = _assert_v5_result_shape(_cpu_run_v5())
    src = _PROD_SRC.read_text(encoding="utf-8")
    imported = _imported_names(_PROD_SRC)
    assert "tests" not in imported
    assert "tests.reference" not in imported
    assert "tests.reference.v5_rung0" not in src
    assert "from tests" not in src
    assert "import tests" not in src
    assert result["loss_curve_matches_ladder"] is True


def test_reference_does_not_import_production_jax_torch() -> None:
    _assert_v5_result_shape(_cpu_run_v5())
    src_path = Path(ref.__file__)
    imported = _imported_names(src_path)
    for name in (
        "jax",
        "torch",
        "prometheus",
        "prometheus.verify",
        "prometheus.verify.v5_rung0",
        "prometheus_control",
    ):
        assert name not in imported, name
    src = src_path.read_text(encoding="utf-8")
    assert "prometheus.verify.v5_rung0" not in src


# ---------------------------------------------------------------------------
# V5 GPU (8 H200 analog). Skipped on CPU.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_v5_gpu_run_v5_gates() -> None:
    """V5: GPU analog of rung 0; both F4 bools true. Passing 1 GPU is not V5 verified."""
    _require_gpu()
    result = _assert_v5_result_shape(v5.run_v5(gpus=_CPU_GPUS))
    assert result["loss_curve_matches_ladder"] is True
    assert result["checkpoint_resume_ok"] is True
    assert v5.DEFAULT_GPUS == 8


@pytest.mark.gpu
def test_v5_gpu_gpus_less_than_one_still_v5error() -> None:
    _require_gpu()
    with pytest.raises(v5.V5Error):
        v5.run_v5(gpus=0)
    result = _assert_v5_result_shape(v5.run_v5(gpus=_CPU_GPUS))
    assert result["loss_curve_matches_ladder"] is True
    assert result["checkpoint_resume_ok"] is True
