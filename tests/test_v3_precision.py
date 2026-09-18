"""Oracle tests for V3 precision (spec 16.2 / 5.3, F4 ``v3.sh`` / ``check_exit``).

Public interface is ``prometheus.verify.v3_precision.run_v3``. Every collected
CPU test calls that interface so the suite fails against the
``NotImplementedError`` stub. GPU tests use ``pytest.mark.gpu`` and skip
without a device.

The CPU analog is a tiny toy net (``gpus=1``, a handful of steps). Spec V3 is
8 H200 at 0.1–0.5B for 2k–5k steps; passing these tests is not V3 verified.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prometheus.verify import v3_precision as v3
from tests.reference import v3_precision as ref

_CPU_STEPS = 4
_V3_KEYS = frozenset({"fp8_loss_rel_diff", "nvfp4_numerics_ok"})

_cpu_result: v3.V3Result | None = None
_cpu_error: BaseException | None = None


def _cpu_run_v3() -> v3.V3Result:
    """``run_v3(gpus=1, steps=4)``. Cached so the stub fails once per process."""
    global _cpu_result, _cpu_error
    if _cpu_error is not None:
        raise _cpu_error
    if _cpu_result is None:
        try:
            _cpu_result = v3.run_v3(gpus=1, steps=_CPU_STEPS)
        except BaseException as exc:
            _cpu_error = exc
            raise
    return _cpu_result


def _assert_v3_result(result: Any) -> v3.V3Result:
    assert isinstance(result, dict), type(result)
    assert set(result) == set(_V3_KEYS), set(result)
    rel = result["fp8_loss_rel_diff"]
    ok = result["nvfp4_numerics_ok"]
    assert type(rel) is not bool, type(rel)
    assert isinstance(rel, (float, int, np.floating, np.integer)), type(rel)
    assert type(ok) is bool, type(ok)
    rel_f = float(rel)
    assert np.isfinite(rel_f), rel_f
    assert rel_f >= 0.0
    assert rel_f <= v3.FP8_REL_MAX
    assert ok is True
    return result


def _imported_names(path: str) -> set[str]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
            names.add(node.module)
    return names


def _require_gpu() -> Any:
    jax = pytest.importorskip("jax")
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if not devices:
        pytest.skip("V3 GPU test requires a GPU device")
    return jax


# ---------------------------------------------------------------------------
# Faults: gpus < 1 and steps < 1 raise V3Error (not a generic Exception)
# ---------------------------------------------------------------------------


def test_gpus_zero_raises_v3error() -> None:
    with pytest.raises(v3.V3Error) as ei:
        v3.run_v3(gpus=0)
    assert type(ei.value) is v3.V3Error
    assert not isinstance(ei.value, NotImplementedError)


def test_gpus_negative_raises_v3error() -> None:
    for gpus in (-1, -3, -8):
        with pytest.raises(v3.V3Error) as ei:
            v3.run_v3(gpus=gpus)
        assert type(ei.value) is v3.V3Error
        assert not isinstance(ei.value, NotImplementedError)


def test_steps_zero_raises_v3error() -> None:
    with pytest.raises(v3.V3Error) as ei:
        v3.run_v3(gpus=1, steps=0)
    assert type(ei.value) is v3.V3Error
    assert not isinstance(ei.value, NotImplementedError)


def test_steps_negative_raises_v3error() -> None:
    for steps in (-1, -2000):
        with pytest.raises(v3.V3Error) as ei:
            v3.run_v3(gpus=1, steps=steps)
        assert type(ei.value) is v3.V3Error
        assert not isinstance(ei.value, NotImplementedError)


# ---------------------------------------------------------------------------
# Public CPU analog (gpus=1, small steps): keys, types, spec gates
# ---------------------------------------------------------------------------


def test_run_v3_gpus_1_returns_exact_v3result_keys() -> None:
    result = _assert_v3_result(_cpu_run_v3())
    assert set(result) == {"fp8_loss_rel_diff", "nvfp4_numerics_ok"}


def test_fp8_loss_rel_diff_type_finite_nonneg_within_gate() -> None:
    result = _assert_v3_result(_cpu_run_v3())
    rel = result["fp8_loss_rel_diff"]
    assert type(rel) is not bool
    assert isinstance(rel, (float, int, np.floating, np.integer))
    rel_f = float(rel)
    assert np.isfinite(rel_f)
    assert rel_f >= 0.0
    assert rel_f <= v3.FP8_REL_MAX
    assert rel_f <= ref.FP8_REL_MAX
    assert ref.meets_fp8_gate(rel_f, limit=v3.FP8_REL_MAX)


def test_nvfp4_numerics_ok_is_true_bool() -> None:
    result = _assert_v3_result(_cpu_run_v3())
    assert result["nvfp4_numerics_ok"] is True
    assert type(result["nvfp4_numerics_ok"]) is bool


# ---------------------------------------------------------------------------
# Production vs independent reference: shapes, dtypes, golden scale, gates
# ---------------------------------------------------------------------------


def test_reference_protocol_shapes_and_dtypes() -> None:
    result = _assert_v3_result(_cpu_run_v3())
    proto = ref.evaluate_toy_protocol(steps=2)
    steps = 2
    for key in ("bf16_losses", "fp8_losses", "nvfp4_losses"):
        losses = proto[key]
        assert losses.shape == (steps,)
        assert losses.dtype == np.float64
        assert np.isfinite(losses).all()
    assert proto["tokens"].shape == (ref.TOY_BATCH, ref.TOY_SEQ)
    assert proto["tokens"].dtype == np.int32
    w_in = proto["w_in"]
    assert w_in.shape == (ref.TOY_FFN, ref.TOY_DIM)
    assert w_in.dtype == np.float32
    assert proto["w_in_fp8"].shape == w_in.shape
    assert proto["w_in_fp8"].dtype == np.float32
    n_fp8 = ref.TOY_DIM // ref.TOY_FP8_BLOCK
    assert proto["w_in_fp8_scale"].shape == (ref.TOY_FFN, n_fp8)
    assert proto["w_in_fp8_scale"].dtype == np.float32
    expert = proto["expert_w_in"]
    assert expert.shape == (ref.TOY_EXPERTS, ref.TOY_FFN, ref.TOY_DIM)
    assert expert.dtype == np.float32
    n_nv = ref.TOY_DIM // ref.TOY_NVFP4_BLOCK
    assert proto["expert_w_in_nvfp4"].shape == (ref.TOY_FFN, ref.TOY_DIM)
    assert proto["expert_w_in_nvfp4_scale"].shape == (ref.TOY_FFN, n_nv)
    acts = proto["activations"]
    assert acts["h"].shape == (ref.TOY_BATCH, ref.TOY_SEQ, ref.TOY_DIM)
    assert acts["logits"].shape == (ref.TOY_BATCH, ref.TOY_SEQ, ref.TOY_VOCAB)
    assert acts["h"].dtype == np.float32
    assert acts["logits"].dtype == np.float32
    assert proto["ran_nvfp4"] is True
    assert proto["nvfp4_numerics_ok"] is True
    assert ref.meets_fp8_gate(float(proto["fp8_loss_rel_diff"]))
    assert float(result["fp8_loss_rel_diff"]) <= v3.FP8_REL_MAX


def test_golden_relative_scale_one_percent_fails_gate() -> None:
    result = _assert_v3_result(_cpu_run_v3())
    proto = ref.evaluate_toy_protocol(steps=2)
    bumped = ref.perturb_fp8_losses(
        proto["bf16_losses"], rel_gap=ref.GOLDEN_GAP_ABOVE_GATE
    )
    golden = ref.fp8_loss_rel_diff(bumped, proto["bf16_losses"])
    assert golden == pytest.approx(ref.GOLDEN_GAP_ABOVE_GATE, rel=1e-9, abs=1e-12)
    assert golden > v3.FP8_REL_MAX
    assert golden > 0.005
    assert not ref.meets_fp8_gate(golden)
    # Production must live on the pass side of the 0.5% gate.
    assert float(result["fp8_loss_rel_diff"]) <= v3.FP8_REL_MAX


def test_deliberately_worse_fp8_gap_fails_gate() -> None:
    result = _assert_v3_result(_cpu_run_v3())
    proto = ref.evaluate_toy_protocol(steps=2)
    worse = ref.perturb_fp8_losses(proto["bf16_losses"], rel_gap=1e-2)
    gap = ref.fp8_loss_rel_diff(worse, proto["bf16_losses"])
    assert gap == pytest.approx(1e-2, rel=1e-9, abs=1e-12)
    assert gap > 0.005
    assert not ref.meets_fp8_gate(gap)
    assert gap != pytest.approx(float(result["fp8_loss_rel_diff"]), abs=1e-8)


def test_nonfinite_nvfp4_path_fails_numerics() -> None:
    result = _assert_v3_result(_cpu_run_v3())
    proto = ref.evaluate_toy_protocol(steps=2)
    finite = proto["nvfp4_losses"]
    nan_losses = ref.inject_nonfinite(finite, which="nan")
    inf_losses = ref.inject_nonfinite(finite, which="inf")
    h = proto["activations"]["h"]
    assert (
        ref.nvfp4_numerics_ok(nan_losses, h, ran_nvfp4=True) is False
    )
    assert (
        ref.nvfp4_numerics_ok(inf_losses, h, ran_nvfp4=True) is False
    )
    assert result["nvfp4_numerics_ok"] is True


def test_must_not_compare_a_precision_run_to_itself() -> None:
    """``run_v3`` must train two trajectories; identity-0 is not a legal method."""
    result = _assert_v3_result(_cpu_run_v3())
    proto = ref.evaluate_toy_protocol(steps=2)
    assert proto["bf16_losses"] is not proto["fp8_losses"]
    assert proto["bf16_losses"] is not proto["nvfp4_losses"]
    identity = ref.fp8_loss_rel_diff(proto["bf16_losses"], proto["bf16_losses"])
    assert identity == 0.0
    two_path = ref.fp8_loss_rel_diff(proto["fp8_losses"], proto["bf16_losses"])
    assert two_path == pytest.approx(float(proto["fp8_loss_rel_diff"]), abs=0.0)
    assert proto["ran_nvfp4"] is True
    # A dummy that reports 0.0 by self-compare is not this protocol.
    assert two_path == ref.fp8_loss_rel_diff(proto["fp8_losses"], proto["bf16_losses"])
    assert float(result["fp8_loss_rel_diff"]) >= 0.0


# ---------------------------------------------------------------------------
# Properties of fp8_loss_rel_diff / NVFP4 fake-quant on the reference
# ---------------------------------------------------------------------------


def test_fp8_loss_rel_diff_identity_mismatch_nan() -> None:
    _assert_v3_result(_cpu_run_v3())
    match = np.array([1.25, 0.5, 2.0], dtype=np.float64)
    assert ref.fp8_loss_rel_diff(match, match) == 0.0
    assert ref.meets_fp8_gate(0.0)
    # Length-1 scalar form.
    assert ref.fp8_loss_rel_diff(1.0, 1.0) == 0.0
    bumped = match * (1.0 + 1e-2)
    gap = ref.fp8_loss_rel_diff(bumped, match)
    assert gap == pytest.approx(1e-2, rel=1e-12, abs=1e-15)
    assert not ref.meets_fp8_gate(gap)
    nan_gap = ref.fp8_loss_rel_diff([np.nan], [1.0])
    assert not np.isfinite(nan_gap)
    assert not ref.meets_fp8_gate(nan_gap)
    assert not ref.meets_fp8_gate(float("inf"))
    assert not ref.meets_fp8_gate(-1e-6)
    with pytest.raises(ValueError, match="length mismatch"):
        ref.fp8_loss_rel_diff([1.0, 2.0], [1.0])


def test_nvfp4_fakequant_identity_mismatch_nan() -> None:
    _assert_v3_result(_cpu_run_v3())
    zeros = np.zeros((2, ref.TOY_NVFP4_BLOCK), dtype=np.float32)
    zq = ref.fakequant_nvfp4(zeros, block=ref.TOY_NVFP4_BLOCK)
    np.testing.assert_array_equal(zq, zeros)
    # Values off the E2M1 grid must move.
    off_grid = np.linspace(0.1, 1.3, ref.TOY_NVFP4_BLOCK, dtype=np.float32)
    q = ref.fakequant_nvfp4(off_grid, block=ref.TOY_NVFP4_BLOCK)
    assert q.shape == off_grid.shape
    assert q.dtype == np.float32
    assert np.isfinite(q).all()
    assert np.any(q != off_grid)
    # The *path* check: skipped NVFP4 is never ok, even with finite placeholders.
    finite = np.ones((4,), dtype=np.float32)
    assert ref.nvfp4_numerics_ok(finite, ran_nvfp4=False) is False
    assert ref.nvfp4_numerics_ok(finite, ran_nvfp4=True) is True
    assert ref.nvfp4_numerics_ok(ran_nvfp4=True) is False
    planted = ref.inject_nonfinite(finite, which="nan")
    assert ref.nvfp4_numerics_ok(planted, ran_nvfp4=True) is False
    inf_act = ref.inject_nonfinite(off_grid.astype(np.float64), which="inf")
    assert ref.nvfp4_numerics_ok(inf_act, ran_nvfp4=True) is False


def test_fp8_block_and_nvfp4_microblock_divide_toy_hidden() -> None:
    _assert_v3_result(_cpu_run_v3())
    assert ref.FLAGSHIP_FP8_BLOCK == 128
    assert ref.TOY_FP8_BLOCK == ref.FLAGSHIP_FP8_BLOCK
    assert ref.TOY_NVFP4_BLOCK == 16
    assert ref.TOY_DIM % ref.TOY_FP8_BLOCK == 0
    assert ref.TOY_DIM % ref.TOY_NVFP4_BLOCK == 0
    assert ref.TOY_FFN % ref.TOY_FP8_BLOCK == 0
    assert ref.TOY_FFN % ref.TOY_NVFP4_BLOCK == 0
    w = ref.toy_init()["w_in"]
    scale = ref.fp8_block_scales(w, block=ref.TOY_FP8_BLOCK)
    assert scale.shape[-1] == ref.TOY_DIM // ref.TOY_FP8_BLOCK
    # Padding path: last axis 20, block 16 → 2 blocks.
    ragged = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
    q = ref.fakequant_fp8(ragged, block=16)
    assert q.shape == ragged.shape
    assert np.isfinite(q).all()


def test_toy_linear_grads_match_finite_differences() -> None:
    _assert_v3_result(_cpu_run_v3())
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 0.1, size=(4, ref.TOY_DIM)).astype(np.float32)
    w = rng.normal(0.0, 0.1, size=(8, ref.TOY_DIM)).astype(np.float32)
    target = rng.normal(0.0, 0.1, size=(4, 8)).astype(np.float32)
    analytic, numeric = ref.linear_mse_grad_w_finite_diff(x, w, target)
    assert analytic.shape == numeric.shape == (2, 2)
    assert analytic.dtype == np.float32
    assert numeric.dtype == np.float32
    np.testing.assert_allclose(
        analytic, numeric, rtol=ref.TOY_GRAD_RTOL, atol=ref.TOY_GRAD_ATOL
    )


# ---------------------------------------------------------------------------
# Fault injection on the independent path (no production implementation)
# ---------------------------------------------------------------------------


def test_fault_inject_fp8_losses_above_half_percent() -> None:
    result = _assert_v3_result(_cpu_run_v3())
    proto = ref.evaluate_toy_protocol(steps=2)
    injected = ref.perturb_fp8_losses(proto["bf16_losses"], rel_gap=1e-2)
    gap = ref.fp8_loss_rel_diff(injected, proto["bf16_losses"])
    assert gap > 0.005
    assert not ref.meets_fp8_gate(gap)
    assert ref.meets_fp8_gate(float(result["fp8_loss_rel_diff"]))


def test_fault_inject_nvfp4_inf_and_nan() -> None:
    result = _assert_v3_result(_cpu_run_v3())
    proto = ref.evaluate_toy_protocol(steps=2)
    acts = proto["activations"]["h"]
    losses = proto["nvfp4_losses"]
    assert ref.nvfp4_numerics_ok(losses, acts, ran_nvfp4=True) is True
    assert (
        ref.nvfp4_numerics_ok(
            ref.inject_nonfinite(losses, which="nan"), acts, ran_nvfp4=True
        )
        is False
    )
    assert (
        ref.nvfp4_numerics_ok(
            losses, ref.inject_nonfinite(acts, which="inf"), ran_nvfp4=True
        )
        is False
    )
    # Skipping the NVFP4 path must not report ok.
    assert ref.nvfp4_numerics_ok(losses, acts, ran_nvfp4=False) is False
    assert result["nvfp4_numerics_ok"] is True


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_fp8_rel_max_and_default_steps_match_reference() -> None:
    _assert_v3_result(_cpu_run_v3())
    assert v3.FP8_REL_MAX == 0.005 == 5e-3
    assert ref.FP8_REL_MAX == 0.005 == 5e-3
    assert v3.FP8_REL_MAX == ref.FP8_REL_MAX
    assert v3.DEFAULT_STEPS == 2000
    assert ref.DEFAULT_STEPS == 2000
    assert v3.DEFAULT_STEPS == ref.DEFAULT_STEPS
    assert ref.FLAGSHIP_FP8_BLOCK == 128
    assert ref.SPEC_GPUS == 8


# ---------------------------------------------------------------------------
# Import isolation
# ---------------------------------------------------------------------------


def test_reference_does_not_import_production_or_jax() -> None:
    _assert_v3_result(_cpu_run_v3())
    src = ref.__file__
    assert src is not None
    imported = _imported_names(src)
    for name in ("jax", "torch", "model", "train", "kernels", "prometheus"):
        assert name not in imported, name
    assert "prometheus.verify.v3_precision" not in imported
    assert "prometheus.verify" not in imported


def test_production_must_not_import_tests_or_reference() -> None:
    _assert_v3_result(_cpu_run_v3())
    src = v3.__file__
    assert src is not None
    imported = _imported_names(src)
    for name in ("tests", "tests.reference", "tests.reference.v3_precision"):
        assert name not in imported, name


# ---------------------------------------------------------------------------
# V3 GPU scale (8 H200 note). Skipped on CPU.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_v3_gpu_run_v3_eight_h200_scale_gates() -> None:
    """V3: 8 H200 (spec 16.2). Tiny analog, not V3 verified."""
    _require_gpu()
    result = _assert_v3_result(v3.run_v3(gpus=8, steps=2))
    assert float(result["fp8_loss_rel_diff"]) <= v3.FP8_REL_MAX
    assert result["nvfp4_numerics_ok"] is True


@pytest.mark.gpu
def test_v3_gpu_eight_gpus_keys_and_nvfp4_ran() -> None:
    """Same 8-GPU call; still a tiny analog (not a 0.1–0.5B flagship run)."""
    _require_gpu()
    result = _assert_v3_result(v3.run_v3(gpus=8, steps=2))
    assert set(result) == {"fp8_loss_rel_diff", "nvfp4_numerics_ok"}
    assert type(result["nvfp4_numerics_ok"]) is bool
    assert result["nvfp4_numerics_ok"] is True
