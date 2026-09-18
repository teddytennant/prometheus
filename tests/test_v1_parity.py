"""Oracle tests for ``prometheus.verify.v1_parity.run_v1`` (spec 16.2 V1).

Every collected CPU test calls the public interface so the suite fails against
the ``NotImplementedError`` stub. GPU tests are marked ``gpu`` (V1 / 1 H200)
and skip cleanly without a device.

Production is imported only here. ``tests.reference.v1_parity`` is plain NumPy
and must not import ``model/``, ``train/``, ``prometheus.verify.v1_parity``, or
JAX.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from prometheus.verify import v1_parity as v1
from tests.reference import v1_parity as ref

_CPU_RESULT: dict[str, Any] | None = None
_CPU_ERROR: BaseException | None = None


def _cpu_run_v1() -> dict[str, Any]:
    """Call ``run_v1(gpus=1)`` once per process (CPU JAX analog of V1)."""
    global _CPU_RESULT, _CPU_ERROR
    if _CPU_RESULT is not None:
        return _CPU_RESULT
    if _CPU_ERROR is not None:
        raise _CPU_ERROR
    try:
        out = v1.run_v1(gpus=1)
    except BaseException as exc:
        _CPU_ERROR = exc
        raise
    _CPU_RESULT = out
    return out


def _require_gpu():
    """Skip unless JAX sees a GPU. Marker ``gpu`` is the V1 stage tag."""
    jax = pytest.importorskip("jax")
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []
    if not gpus:
        pytest.skip("V1 GPU test requires a GPU device")
    return jax


def _assert_v1_result_shape(result: object) -> dict[str, Any]:
    assert isinstance(result, dict), "run_v1 must return a V1Result mapping"
    for key in ("logits_max_diff", "grad_ok", "overfit_ok"):
        assert key in result, f"missing V1Result key {key!r}"
    diff = result["logits_max_diff"]
    assert not isinstance(diff, bool), "logits_max_diff must not be bool"
    assert isinstance(diff, (float, int, np.floating, np.integer))
    d = float(diff)
    assert np.isfinite(d), "logits_max_diff must be finite"
    assert d >= 0.0, "max-abs diff is non-negative"
    assert type(result["grad_ok"]) is bool
    assert type(result["overfit_ok"]) is bool
    return result


# ---------------------------------------------------------------------------
# gpus < 1 → V1Error (not a generic Exception-pass against the stub)
# ---------------------------------------------------------------------------


def test_gpus_zero_raises_v1error() -> None:
    with pytest.raises(v1.V1Error) as ei:
        v1.run_v1(gpus=0)
    assert type(ei.value) is v1.V1Error
    assert not isinstance(ei.value, NotImplementedError)


def test_gpus_negative_raises_v1error() -> None:
    for gpus in (-1, -3, -8):
        with pytest.raises(v1.V1Error) as ei:
            v1.run_v1(gpus=gpus)
        assert type(ei.value) is v1.V1Error
        assert not isinstance(ei.value, NotImplementedError)


# ---------------------------------------------------------------------------
# Successful CPU run_v1(gpus=1): keys, dtypes, spec gates
# ---------------------------------------------------------------------------


def test_run_v1_gpus_1_returns_v1result_keys() -> None:
    result = _assert_v1_result_shape(_cpu_run_v1())
    assert set(result) >= {"logits_max_diff", "grad_ok", "overfit_ok"}
    assert v1.LOGITS_MAX_ABS == 1e-5
    assert ref.LOGITS_MAX_ABS == v1.LOGITS_MAX_ABS


def test_logits_max_diff_finite_and_within_1e_5() -> None:
    result = _assert_v1_result_shape(_cpu_run_v1())
    diff = float(result["logits_max_diff"])
    assert ref.meets_logits_gate(diff, limit=v1.LOGITS_MAX_ABS)
    assert diff <= v1.LOGITS_MAX_ABS


def test_grad_ok_true_when_reverse_mode_matches_finite_differences() -> None:
    result = _assert_v1_result_shape(_cpu_run_v1())
    assert result["grad_ok"] is True
    # Independent protocol: reverse-mode vs central differences on a tiny slice.
    assert ref.toy_grad_ok() is True
    params = ref.toy_init()
    tokens = ref.toy_tokens()
    analytic, numeric = ref.toy_finite_diff_slice(tokens, params)
    assert analytic.shape == numeric.shape
    assert analytic.dtype == np.float32
    assert numeric.dtype == np.float32
    assert ref.grad_match_ok(analytic, numeric) is True


def test_overfit_ok_true_when_one_batch_loss_strictly_drops() -> None:
    result = _assert_v1_result_shape(_cpu_run_v1())
    assert result["overfit_ok"] is True
    params = ref.toy_init()
    tokens = ref.toy_tokens()
    before = ref.toy_next_token_loss(tokens, params)
    after_params = ref.toy_sgd_step(tokens, params)
    after = ref.toy_next_token_loss(tokens, after_params)
    assert ref.overfit_loss_dropped(before, after) is True
    assert ref.toy_overfit_ok(tokens, params) is True
    # Equal or rising loss is not overfit.
    assert ref.overfit_loss_dropped(1.0, 1.0) is False
    assert ref.overfit_loss_dropped(1.0, 1.1) is False
    assert ref.overfit_loss_dropped(float("nan"), 0.1) is False


# ---------------------------------------------------------------------------
# Production vs independent reference: property / shape / dtype / golden scale
# ---------------------------------------------------------------------------


def test_production_vs_reference_logits_max_diff_scale() -> None:
    result = _assert_v1_result_shape(_cpu_run_v1())
    proto = ref.evaluate_toy_protocol()
    logits = proto["logits"]
    assert logits.dtype == np.float32
    assert logits.shape == (ref.TOY_BATCH, ref.TOY_SEQ, ref.TOY_VOCAB)
    assert proto["tokens"].dtype == np.int32
    assert proto["tokens"].shape == (ref.TOY_BATCH, ref.TOY_SEQ)

    match_diff = ref.logits_max_abs_diff(logits, logits)
    assert match_diff == ref.GOLDEN_MATCH_DIFF
    assert ref.meets_logits_gate(match_diff)

    # Golden scale: a 2e-5 uniform shift is just above the 1e-5 FP32 gate.
    shifted = ref.shift_logits(logits, ref.GOLDEN_SHIFT_ABOVE_GATE)
    assert shifted.dtype == np.float32
    assert shifted.shape == logits.shape
    golden = ref.logits_max_abs_diff(logits, shifted)
    assert golden == pytest.approx(
        float(np.float32(ref.GOLDEN_SHIFT_ABOVE_GATE)), abs=1e-9
    )
    assert golden > v1.LOGITS_MAX_ABS
    assert not ref.meets_logits_gate(golden)

    # Production report must live on the matching-forward side of the gate.
    assert float(result["logits_max_diff"]) <= v1.LOGITS_MAX_ABS
    assert float(result["logits_max_diff"]) <= max(match_diff, v1.LOGITS_MAX_ABS)
    assert proto["grad_ok"] is True
    assert proto["overfit_ok"] is True


def test_logits_max_abs_diff_properties() -> None:
    _assert_v1_result_shape(_cpu_run_v1())
    rng = np.random.default_rng(4)
    a = rng.standard_normal((2, 3, 5)).astype(np.float32)
    b = rng.standard_normal((2, 3, 5)).astype(np.float32)
    d_ab = ref.logits_max_abs_diff(a, b)
    d_ba = ref.logits_max_abs_diff(b, a)
    assert d_ab == pytest.approx(d_ba, abs=0.0)
    assert d_ab >= 0.0
    assert ref.logits_max_abs_diff(a, a) == 0.0
    c = np.float32(0.25)
    assert ref.logits_max_abs_diff(a, a + c) == pytest.approx(float(c), abs=1e-6)
    with pytest.raises(ValueError, match="shape mismatch"):
        ref.logits_max_abs_diff(a, a[:, :2])


def test_deliberately_wrong_forward_exceeds_1e_5() -> None:
    """Fault injection on the independent path (no production implementation)."""
    result = _assert_v1_result_shape(_cpu_run_v1())
    assert float(result["logits_max_diff"]) <= v1.LOGITS_MAX_ABS

    proto = ref.evaluate_toy_protocol()
    logits = proto["logits"]
    assert ref.wrong_forward_exceeds_gate(logits) is True
    wrong = ref.shift_logits(logits, ref.WRONG_FORWARD_SHIFT)
    diff = ref.logits_max_abs_diff(logits, wrong)
    assert diff == pytest.approx(ref.WRONG_FORWARD_SHIFT, abs=1e-6)
    assert diff > v1.LOGITS_MAX_ABS
    assert not ref.meets_logits_gate(diff)

    # A dummy that always reports 0 would still pass the production gate; the
    # independent path must actually move when the forward is wrong.
    assert diff != pytest.approx(float(result["logits_max_diff"]), abs=1e-8)


def test_reference_does_not_import_production_or_jax() -> None:
    _assert_v1_result_shape(_cpu_run_v1())
    import ast

    src = ref.__file__
    assert src is not None
    tree = ast.parse(open(src, encoding="utf-8").read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
            imported.add(node.module)
    for name in ("jax", "torch", "model", "train", "prometheus"):
        assert name not in imported, name


# ---------------------------------------------------------------------------
# V1 GPU scale (tiny_config, 1 GPU). Skipped on CPU.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_v1_gpu_run_v1_tiny_config_gates() -> None:
    """V1: 1 GPU, tiny_config JAX vs independent, FP32 1e-5, grad, overfit."""
    _require_gpu()
    result = _assert_v1_result_shape(v1.run_v1(gpus=1))
    assert float(result["logits_max_diff"]) <= v1.LOGITS_MAX_ABS
    assert result["grad_ok"] is True
    assert result["overfit_ok"] is True


@pytest.mark.gpu
def test_v1_gpu_gpus_less_than_one_still_v1error() -> None:
    _require_gpu()
    with pytest.raises(v1.V1Error):
        v1.run_v1(gpus=0)
    result = _assert_v1_result_shape(v1.run_v1(gpus=1))
    assert result["grad_ok"] is True
    assert result["overfit_ok"] is True
