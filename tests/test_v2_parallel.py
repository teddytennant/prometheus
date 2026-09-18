"""V2 parallel equivalence (spec 16.2): single-device vs SPMD mesh.

CPU tests call ``run_v2(gpus=1, steps=CPU_STEPS)`` so the suite fails against
the ``NotImplementedError`` stub (0 passed). GPU tests are marked ``gpu`` and
skip without a device; they use the spec default of 200 steps.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prometheus.verify import v2_parallel as v2
from tests.reference import v2_parallel as ref

CPU_STEPS = 2

_CPU_RESULT: v2.V2Result | None = None
_CPU_ERROR: BaseException | None = None


def _cpu_run_v2() -> v2.V2Result:
    """One ``run_v2(gpus=1, steps=CPU_STEPS)`` per process; re-raise cached errors."""
    global _CPU_RESULT, _CPU_ERROR
    if _CPU_RESULT is not None:
        return _CPU_RESULT
    if _CPU_ERROR is not None:
        raise _CPU_ERROR
    try:
        out = v2.run_v2(gpus=1, steps=CPU_STEPS)
    except BaseException as exc:
        _CPU_ERROR = exc
        raise
    _CPU_RESULT = out
    return out


def _assert_v2_result_shape(result: Mapping[str, Any]) -> v2.V2Result:
    assert set(result.keys()) == {"loss_rel_diff", "routing_identical"}
    diff = result["loss_rel_diff"]
    identical = result["routing_identical"]
    assert not isinstance(diff, bool)
    assert type(diff) in (float, np.floating, int, np.integer)
    assert type(identical) is bool
    return result  # type: ignore[return-value]


def _require_gpu():
    jax = pytest.importorskip("jax")
    gpus = [d for d in jax.devices() if getattr(d, "platform", None) == "gpu"]
    if not gpus:
        pytest.skip("no GPU")
    return jax


def _assert_v2error(gpus: int, steps: int) -> None:
    with pytest.raises(v2.V2Error) as ei:
        v2.run_v2(gpus=gpus, steps=steps)
    assert type(ei.value) is v2.V2Error
    assert not isinstance(ei.value, NotImplementedError)


# ---------------------------------------------------------------------------
# Faults: gpus < 1 / steps < 1 -> V2Error (not a generic Exception)
# ---------------------------------------------------------------------------


def test_gpus_zero_raises_v2error() -> None:
    _assert_v2error(gpus=0, steps=CPU_STEPS)


def test_gpus_negative_raises_v2error() -> None:
    _assert_v2error(gpus=-1, steps=CPU_STEPS)
    _assert_v2error(gpus=-8, steps=CPU_STEPS)


def test_steps_zero_raises_v2error() -> None:
    _assert_v2error(gpus=1, steps=0)


def test_steps_negative_raises_v2error() -> None:
    _assert_v2error(gpus=1, steps=-1)
    _assert_v2error(gpus=1, steps=-200)


# ---------------------------------------------------------------------------
# Public interface on CPU (gpus=1 analog, small steps)
# ---------------------------------------------------------------------------


def test_run_v2_gpus_1_returns_v2result_keys() -> None:
    assert v2.LOSS_REL_MAX == 1e-6
    assert v2.LOSS_REL_MAX == ref.LOSS_REL_MAX
    assert v2.DEFAULT_STEPS == 200
    assert v2.DEFAULT_STEPS == ref.DEFAULT_STEPS
    result = _assert_v2_result_shape(_cpu_run_v2())
    assert "loss_rel_diff" in result
    assert "routing_identical" in result


def test_loss_rel_diff_finite_nonneg_within_1e_6() -> None:
    result = _assert_v2_result_shape(_cpu_run_v2())
    diff = float(result["loss_rel_diff"])
    assert np.isfinite(diff)
    assert diff >= 0.0
    assert diff <= v2.LOSS_REL_MAX
    assert diff <= ref.LOSS_REL_MAX


def test_routing_identical_true_when_routing_matches() -> None:
    result = _assert_v2_result_shape(_cpu_run_v2())
    assert result["routing_identical"] is True


def test_production_vs_reference_v2_gates() -> None:
    """Production report must sit on the matching-mesh side of the V2 gates."""
    result = _assert_v2_result_shape(_cpu_run_v2())
    proto = ref.evaluate_v2_protocol(steps=CPU_STEPS)
    assert proto["single_losses"].shape == (CPU_STEPS,)
    assert proto["mesh_losses"].shape == (CPU_STEPS,)
    assert proto["single_losses"].dtype == np.float32
    assert proto["mesh_losses"].dtype == np.float32
    assert proto["single_ids"].dtype == np.int32
    assert proto["mesh_ids"].dtype == np.int32
    assert proto["tokens"].shape == (ref.TOY_BATCH, ref.TOY_SEQ)
    assert proto["tokens"].dtype == np.int32

    match_rel = ref.loss_rel_diff(proto["single_losses"], proto["mesh_losses"])
    assert match_rel == pytest.approx(ref.GOLDEN_MATCH_REL, abs=1e-12)
    assert match_rel <= v2.LOSS_REL_MAX
    assert proto["routing_identical"] is True
    assert proto["meets_gates"] is True
    assert proto["grad_ok"] is True

    # Golden scale: a 2e-6 relative bump sits just above the 1e-6 FP32 gate.
    # float32 cannot realize 2e-6 exactly on losses ~O(1), so allow 5% rel.
    bumped = proto["single_losses"] * np.float32(1.0 + ref.GOLDEN_REL_ABOVE_GATE)
    golden = ref.loss_rel_diff(proto["single_losses"], bumped)
    assert golden == pytest.approx(ref.GOLDEN_REL_ABOVE_GATE, rel=0.05, abs=1e-7)
    assert golden > v2.LOSS_REL_MAX
    assert not ref.meets_v2_gates(golden, True)

    assert float(result["loss_rel_diff"]) <= v2.LOSS_REL_MAX
    assert result["routing_identical"] is True
    assert ref.meets_v2_gates(float(result["loss_rel_diff"]), result["routing_identical"])


def test_loss_rel_diff_properties() -> None:
    _assert_v2_result_shape(_cpu_run_v2())
    rng = np.random.default_rng(4)
    a = np.abs(rng.standard_normal(8)).astype(np.float32) + np.float32(0.5)
    b = np.abs(rng.standard_normal(8)).astype(np.float32) + np.float32(0.5)
    d_ab = ref.loss_rel_diff(a, b)
    assert d_ab >= 0.0
    assert np.isfinite(d_ab)
    assert ref.loss_rel_diff(a, a) == 0.0
    assert ref.loss_rel_diff(1.25, 1.25) == 0.0
    rel = np.float32(1e-4)
    shifted = (a * (np.float32(1.0) + rel)).astype(np.float32)
    got = ref.loss_rel_diff(a, shifted)
    assert got == pytest.approx(float(rel), rel=1e-3, abs=1e-12)
    with pytest.raises(ValueError, match="shape mismatch"):
        ref.loss_rel_diff(a, a[:2])
    nan_a = np.array([np.nan], dtype=np.float32)
    nan_b = np.array([1.0], dtype=np.float32)
    nan_rel = ref.loss_rel_diff(nan_a, nan_b)
    assert not np.isfinite(nan_rel)
    assert not ref.meets_v2_gates(nan_rel, True)
    empty_f = np.zeros((0,), dtype=np.float32)
    empty_i = np.zeros((0,), dtype=np.int32)
    assert ref.loss_rel_diff(empty_f, empty_f) == 0.0
    assert ref.routing_identical(empty_i, empty_i) is True
    assert ref.routing_identical(np.array([0, 1]), np.array([0, 1])) is True
    assert ref.routing_identical(np.array([0, 1]), np.array([0, 2])) is False
    assert ref.routing_identical(np.array([0, 1]), np.array([[0, 1]])) is False


def test_deliberately_mismatched_routing_and_loss_fail_gates() -> None:
    """Fault injection on the independent path (no production implementation)."""
    result = _assert_v2_result_shape(_cpu_run_v2())
    assert float(result["loss_rel_diff"]) <= v2.LOSS_REL_MAX
    assert result["routing_identical"] is True

    proto = ref.evaluate_v2_protocol(steps=CPU_STEPS)
    wrong_loss = ref.perturb_losses(proto["mesh_losses"], rel=ref.WRONG_REL_SHIFT)
    wrong_rel = ref.loss_rel_diff(proto["single_losses"], wrong_loss)
    assert wrong_rel == pytest.approx(ref.WRONG_REL_SHIFT, rel=1e-3, abs=1e-12)
    assert wrong_rel > v2.LOSS_REL_MAX
    assert not ref.meets_v2_gates(wrong_rel, True)

    flipped = ref.flip_one_expert_id(proto["mesh_ids"], ref.TOY_EXPERTS)
    assert ref.routing_identical(proto["single_ids"], flipped) is False
    assert not ref.meets_v2_gates(proto["loss_rel_diff"], False)

    # A dummy that always reports 0 / True would still pass the production gate;
    # the independent path must actually move when the mesh is wrong.
    assert wrong_rel != pytest.approx(float(result["loss_rel_diff"]), abs=1e-8)


def test_toy_mesh_matches_single_device_shapes_and_spec_meshes() -> None:
    _assert_v2_result_shape(_cpu_run_v2())
    assert ref.SPEC_MESH_8GPU_EP.n_devices == 8
    assert ref.SPEC_MESH_8GPU_PP_CP.n_devices == 8
    assert ref.SPEC_MESH_8GPU_PP_CP.gpus_per_pp_stage == 4
    assert ref.SPEC_MESH_2X4_DP.n_devices == 8
    assert ref.TINY_MESH.n_devices == 8
    assert ref.TINY_MESH.gpus_per_pp_stage == 4
    assert ref.pp_layer_ranges(ref.TOY_LAYERS, 2) == ((0, 1), (1, 2))

    proto = ref.evaluate_v2_protocol(steps=CPU_STEPS)
    ids = proto["single_ids"]
    assert ids.ndim == 5
    assert ids.shape == (
        CPU_STEPS,
        ref.TOY_LAYERS,
        ref.TOY_BATCH,
        ref.TOY_SEQ,
        ref.TOY_TOP_K,
    )
    assert proto["mesh_ids"].shape == ids.shape
    assert int(ids.min()) >= 0
    assert int(ids.max()) < ref.TOY_EXPERTS

    pair_dp = ref.toy_train_pair(CPU_STEPS, mesh=ref.MeshSpec(dp=2, fsdp=1, ep=1, pp=1, cp=1))
    assert ref.loss_rel_diff(pair_dp["single_losses"], pair_dp["mesh_losses"]) <= ref.LOSS_REL_MAX
    assert ref.routing_identical(pair_dp["single_ids"], pair_dp["mesh_ids"]) is True


def test_toy_gradient_finite_differences() -> None:
    _assert_v2_result_shape(_cpu_run_v2())
    params = ref.toy_init()
    tokens = ref.toy_tokens()
    analytic, numeric = ref.toy_finite_diff_unembed_slice(tokens, params)
    assert analytic.shape == numeric.shape
    assert analytic.dtype == np.float32
    assert numeric.dtype == np.float32
    assert ref.grad_match_ok(analytic, numeric)
    assert ref.toy_grad_ok() is True

    logits = np.zeros((2, 3, ref.TOY_VOCAB), dtype=np.float32)
    targets = np.zeros((2, 3), dtype=np.int32)
    g = ref.ce_logits_grad(logits, targets)
    assert g.shape == logits.shape
    # Central FD on logits[0,0,0] for mean CE.
    eps = 1e-3
    plus = logits.copy()
    minus = logits.copy()
    plus[0, 0, 0] += np.float32(eps)
    minus[0, 0, 0] -= np.float32(eps)
    fd = (ref.mean_cross_entropy(plus, targets) - ref.mean_cross_entropy(minus, targets)) / (
        2.0 * eps
    )
    assert g[0, 0, 0] == pytest.approx(fd, rel=5e-2, abs=5e-3)


def test_reference_does_not_import_production() -> None:
    _assert_v2_result_shape(_cpu_run_v2())
    src = ref.__file__
    assert src is not None
    tree = ast.parse(Path(src).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
            imported.add(node.module)
    for name in (
        "jax",
        "torch",
        "model",
        "train",
        "parallel",
        "prometheus",
        "prometheus.verify",
        "prometheus.verify.v2_parallel",
    ):
        assert name not in imported, name


def test_production_source_does_not_import_tests() -> None:
    _assert_v2_result_shape(_cpu_run_v2())
    src = v2.__file__
    assert src is not None
    tree = ast.parse(Path(src).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
            imported.add(node.module)
    for name in ("tests", "tests.reference", "tests.reference.v2_parallel"):
        assert name not in imported, name


# ---------------------------------------------------------------------------
# V2 GPU scale. Skipped on CPU.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_v2_gpu_run_v2_gates() -> None:
    """V2: 1 GPU (or spec mesh if present), default 200 steps, FP32 1e-6, routing."""
    _require_gpu()
    result = _assert_v2_result_shape(v2.run_v2(gpus=1))
    assert float(result["loss_rel_diff"]) <= v2.LOSS_REL_MAX
    assert np.isfinite(float(result["loss_rel_diff"]))
    assert float(result["loss_rel_diff"]) >= 0.0
    assert result["routing_identical"] is True


@pytest.mark.gpu
def test_v2_gpu_gpus_less_than_one_still_v2error() -> None:
    _require_gpu()
    with pytest.raises(v2.V2Error) as ei:
        v2.run_v2(gpus=0)
    assert type(ei.value) is v2.V2Error
    result = _assert_v2_result_shape(v2.run_v2(gpus=1))
    assert result["routing_identical"] is True
    assert float(result["loss_rel_diff"]) <= v2.LOSS_REL_MAX
