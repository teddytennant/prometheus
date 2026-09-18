"""Oracle tests for ``prometheus.verify.v6_latent.run_v6`` (spec 16.2 V6).

Every collected CPU test calls the public interface so the suite fails against
the ``NotImplementedError`` stub. GPU tests are marked ``gpu`` (V6 / 4–8 H200)
and skip cleanly without a device.

The CPU analog is Stage A then Stage B plus a 1x-to-8x latent-budget sweep
(host stand-in for I5). Spec V6 is 4 to 8 H200; passing these tests is not V6
verified.

Production is imported only here. ``tests.reference.v6_latent`` is plain NumPy
and must not import ``model/``, ``train/``, ``kernels/``, JAX, torch, or the
Rust crates.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prometheus.verify import v6_latent as v6
from tests.reference import v6_latent as ref

_CPU_RESULT: v6.V6Result | None = None
_CPU_ERROR: BaseException | None = None

V6_KEYS = (
    "curriculum_no_collapse",
    "accuracy_rises_with_latent_budget",
    "thoughts_decode",
)


def _cpu_run_v6() -> v6.V6Result:
    """One ``run_v6(gpus=1)`` per process (CPU analog of V6); re-raise cached errors."""
    global _CPU_RESULT, _CPU_ERROR
    if _CPU_RESULT is not None:
        return _CPU_RESULT
    if _CPU_ERROR is not None:
        raise _CPU_ERROR
    try:
        out = v6.run_v6(gpus=1)
    except BaseException as exc:
        _CPU_ERROR = exc
        raise
    _CPU_RESULT = out
    return out


def _assert_v6_result_shape(result: Any) -> v6.V6Result:
    assert isinstance(result, dict), type(result)
    assert set(result) == set(V6_KEYS), set(result)
    for key in V6_KEYS:
        val = result[key]
        assert type(val) is bool, (key, type(val))
    return result  # type: ignore[return-value]


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


def _require_gpu():
    """Skip unless JAX sees a GPU. Marker ``gpu`` is the V6 stage tag."""
    jax = pytest.importorskip("jax")
    try:
        gpus = [d for d in jax.devices() if getattr(d, "platform", None) == "gpu"]
    except RuntimeError:
        gpus = []
    if not gpus:
        pytest.skip("no GPU")
    return jax


def _assert_v6error(gpus: int) -> None:
    with pytest.raises(v6.V6Error) as ei:
        v6.run_v6(gpus=gpus)
    assert type(ei.value) is v6.V6Error
    assert not isinstance(ei.value, NotImplementedError)


# ---------------------------------------------------------------------------
# Faults: gpus < 1 -> V6Error (not NotImplementedError)
# ---------------------------------------------------------------------------


def test_gpus_zero_raises_v6error() -> None:
    _assert_v6error(gpus=0)


def test_gpus_negative_raises_v6error() -> None:
    _assert_v6error(gpus=-1)
    _assert_v6error(gpus=-8)


# ---------------------------------------------------------------------------
# Result shape / types; all three bools true on a successful analog
# ---------------------------------------------------------------------------


def test_run_v6_result_shape_and_types() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    assert result["curriculum_no_collapse"] is True
    assert result["accuracy_rises_with_latent_budget"] is True
    assert result["thoughts_decode"] is True


def test_successful_analog_all_three_bools_true() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    assert result["curriculum_no_collapse"] is True
    assert result["accuracy_rises_with_latent_budget"] is True
    assert result["thoughts_decode"] is True
    assert ref.meets_v6_gates(result) is True
    proto = ref.evaluate_v6_protocol()
    assert proto["meets_gates"] is True
    assert proto["curriculum_no_collapse"] is True
    assert proto["accuracy_rises_with_latent_budget"] is True
    assert proto["thoughts_decode"] is True


def test_run_v6_result_json_serializable() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    payload = json.dumps(dict(result))
    loaded = json.loads(payload)
    assert set(loaded.keys()) == set(V6_KEYS)
    for key in V6_KEYS:
        assert loaded[key] is True
        assert type(loaded[key]) is bool


def test_default_gpus_is_four() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    assert result["curriculum_no_collapse"] is True
    assert v6.DEFAULT_GPUS == 4
    assert ref.DEFAULT_GPUS == 4
    assert v6.DEFAULT_GPUS == ref.DEFAULT_GPUS
    assert ref.DEFAULT_GPUS_MIN == 1
    assert ref.DEFAULT_GPUS_MAX == 8
    assert ref.SPEC_GPUS_MIN == 4
    assert ref.SPEC_GPUS_MAX == 8
    assert ref.LATENT_BUDGETS == (1, 2, 4, 8)
    assert ref.CURRICULUM_K == (1, 2, 4, 8)


def test_cpu_analog_accepts_gpus_ge_1() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    assert result["curriculum_no_collapse"] is True
    assert result["accuracy_rises_with_latent_budget"] is True
    assert result["thoughts_decode"] is True


def test_v6error_is_exception_not_notimplemented() -> None:
    _assert_v6_result_shape(_cpu_run_v6())
    assert issubclass(v6.V6Error, Exception)
    assert not issubclass(v6.V6Error, NotImplementedError)


# ---------------------------------------------------------------------------
# Curriculum actually runs Stage A then Stage B; throwaways collapse
# ---------------------------------------------------------------------------


def test_curriculum_stage_a_then_stage_b() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    proto = ref.evaluate_v6_protocol()
    curr = proto["curriculum"]
    assert curr["did_stage_a"] is True
    assert curr["did_stage_b"] is True
    assert curr["k_schedule"] == ref.CURRICULUM_K
    assert curr["k_raised"] is True
    assert curr["stage_a_loss_fell"] is True
    assert curr["depth_rose"] is True
    assert curr["halt_collapsed"] is False
    assert curr["losses_finite"] is True
    assert curr["curriculum_no_collapse"] is True
    assert len(curr["stage_a_losses"]) == ref.TOY_STEPS_A
    assert all(math.isfinite(x) for x in curr["stage_a_losses"])
    assert curr["stage_a_losses"][-1] < curr["stage_a_losses"][0]
    depths = curr["expected_depths"]
    assert len(depths) == len(ref.CURRICULUM_K)
    for i in range(len(depths) - 1):
        assert depths[i] < depths[i + 1]
    assert result["curriculum_no_collapse"] is True


def test_skip_curriculum_collapses() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    skipped = ref.evaluate_v6_protocol(skip_curriculum=True)
    curr = skipped["curriculum"]
    assert curr["skip_curriculum"] is True
    assert curr["did_stage_a"] is False
    assert curr["k_raised"] is False
    assert curr["halt_collapsed"] is True
    assert curr["curriculum_no_collapse"] is False
    assert skipped["curriculum_no_collapse"] is False
    assert skipped["meets_gates"] is False
    assert result["curriculum_no_collapse"] is True


def test_pin_halt_collapses() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    pinned = ref.evaluate_v6_protocol(pin_halt_first=True)
    curr = pinned["curriculum"]
    assert curr["pin_halt_first"] is True
    assert curr["halt_collapsed"] is True
    assert curr["curriculum_no_collapse"] is False
    assert pinned["meets_gates"] is False
    assert result["curriculum_no_collapse"] is True


# ---------------------------------------------------------------------------
# Held-out accuracy rises 1x → 8x; freeze-budget throwaway
# ---------------------------------------------------------------------------


def test_held_out_cannot_oneshot_accuracy_rises() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    proto = ref.evaluate_v6_protocol()
    budget = proto["budget"]
    assert budget["budgets"] == (1, 2, 4, 8)
    assert budget["required_hops"] == ref.REQUIRED_HOPS
    assert min(budget["required_hops"]) >= 2
    accs = budget["accuracies"]
    assert accs[0] == 0.0
    assert budget["cannot_oneshot"] is True
    assert budget["strictly_rises"] is True
    assert accs == (0.0, 0.25, 0.5, 1.0)
    assert budget["accuracy_rises_with_latent_budget"] is True
    assert result["accuracy_rises_with_latent_budget"] is True


def test_freeze_budget_accuracy_does_not_rise() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    frozen = ref.evaluate_v6_protocol(freeze_budget=True)
    budget = frozen["budget"]
    assert budget["freeze_budget"] is True
    accs = budget["accuracies"]
    assert accs[0] == accs[1] == accs[2] == accs[3]
    assert budget["strictly_rises"] is False
    assert budget["accuracy_rises_with_latent_budget"] is False
    assert frozen["accuracy_rises_with_latent_budget"] is False
    assert frozen["meets_gates"] is False
    assert result["accuracy_rises_with_latent_budget"] is True


# ---------------------------------------------------------------------------
# Thoughts decode after Jacobi; skip-decode / skip-Jacobi throwaways
# ---------------------------------------------------------------------------


def test_thoughts_decode_after_jacobi() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    proto = ref.evaluate_v6_protocol()
    decode = proto["decode"]
    assert decode["did_decode"] is True
    assert decode["did_jacobi"] is True
    assert decode["n_sweeps"] == ref.DEFAULT_JACOBI_SWEEPS == 4
    assert list(decode["teacher_ids"]) == list(decode["pred_ids"])
    assert decode["raw_match"] is True
    assert decode["thoughts_decode"] is True
    assert decode["thoughts_shape"] == (ref.TOY_CHUNK, ref.TOY_DIM)
    assert result["thoughts_decode"] is True


def test_skip_decode_thoughts_do_not_decode() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    skipped = ref.evaluate_v6_protocol(skip_decode=True)
    decode = skipped["decode"]
    assert decode["skip_decode"] is True
    assert decode["did_decode"] is False
    assert decode["thoughts_decode"] is False
    assert skipped["thoughts_decode"] is False
    assert skipped["meets_gates"] is False
    assert result["thoughts_decode"] is True


def test_skip_jacobi_decode_fails() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    skipped = ref.evaluate_v6_protocol(skip_jacobi=True)
    decode = skipped["decode"]
    assert decode["skip_jacobi"] is True
    assert decode["did_jacobi"] is False
    assert decode["raw_match"] is False
    assert decode["thoughts_decode"] is False
    assert skipped["thoughts_decode"] is False
    assert skipped["meets_gates"] is False
    assert result["thoughts_decode"] is True


# ---------------------------------------------------------------------------
# I5 glue: PonderNet, KL, Jacobi, noisy latent, Stage B loss
# ---------------------------------------------------------------------------


def test_ponder_golden_and_properties() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    assert ref.golden_ponder_ok() is True
    p = ref.ponder_distribution(np.array(ref.GOLDEN_LAMBDAS, dtype=np.float64))
    assert p.dtype == np.float64
    assert p.shape == (4,)
    np.testing.assert_allclose(p, np.array(ref.GOLDEN_PONDER_P), atol=1e-12, rtol=0.0)
    assert abs(float(np.sum(p)) - 1.0) <= 1e-12
    halt = ref.halt_from_logits(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64))
    assert halt.p.shape == (4,)
    assert abs(float(np.sum(halt.p)) - 1.0) <= 1e-12
    assert halt.lambdas.dtype == np.float64
    assert 0.0 <= halt.expected_depth <= 3.0 + 1e-12
    with pytest.raises(ValueError, match="empty"):
        ref.ponder_distribution(np.array([], dtype=np.float64))
    assert result["curriculum_no_collapse"] is True


def test_halt_kl_geometric_prior() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    g = ref.geometric_prior(4, ref.DEFAULT_LAMBDA_PRIOR)
    assert g.shape == (4,)
    assert g.dtype == np.float64
    assert abs(float(np.sum(g)) - 1.0) <= 1e-12
    assert all(float(x) > 0.0 for x in g)
    p = ref.ponder_distribution(np.array(ref.GOLDEN_LAMBDAS, dtype=np.float64))
    kl = ref.halt_kl(p)
    assert math.isfinite(kl)
    assert kl >= -1e-12
    g_kl = ref.halt_kl(g)
    assert math.isfinite(g_kl)
    loss = ref.halt_loss(p, 1)
    assert math.isfinite(loss)
    assert loss == pytest.approx(-math.log(float(p[1]) + ref.HALT_EPS))
    with pytest.raises(ValueError):
        ref.geometric_prior(0)
    with pytest.raises(ValueError):
        ref.geometric_prior(4, 0.0)
    assert result["curriculum_no_collapse"] is True


def test_jacobi_contraction() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    teacher = np.eye(ref.TOY_DIM, dtype=np.float64)[: ref.TOY_CHUNK]
    rng = np.random.default_rng(0)
    init = teacher + rng.normal(0.0, 1.0, teacher.shape)
    two = ref.jacobi_sweeps(init, ref.contract_toward(teacher), ref.TRUNCATED_SWEEPS)
    four = ref.jacobi_sweeps(init, ref.contract_toward(teacher), ref.DEFAULT_JACOBI_SWEEPS)
    d0 = float(np.linalg.norm(init - teacher))
    d2 = float(np.linalg.norm(two - teacher))
    d4 = float(np.linalg.norm(four - teacher))
    assert d2 < d0
    assert d4 < d2
    # mix=0.5: residual *= 0.5 per sweep.
    assert d2 == pytest.approx(d0 * (0.5**ref.TRUNCATED_SWEEPS), rel=1e-12)
    assert d4 == pytest.approx(d0 * (0.5**ref.DEFAULT_JACOBI_SWEEPS), rel=1e-12)
    with pytest.raises(ValueError, match="n_sweeps"):
        ref.jacobi_sweeps(init, ref.contract_toward(teacher), 0)
    assert result["thoughts_decode"] is True


def test_noisy_latent_formula_and_clamp() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    mu = np.array([0.0, 1.0], dtype=np.float64)
    sigma = np.array([0.05, 0.05], dtype=np.float64)
    eps = np.array([1.0, -1.0], dtype=np.float64)
    out = ref.noisy_latent(mu, sigma, eps)
    np.testing.assert_allclose(out.z, np.array([0.05, 0.95]), atol=1e-12)
    np.testing.assert_allclose(out.sigma, sigma)
    assert out.z.dtype == np.float64
    assert math.isfinite(out.log_density)
    # σ below min clamps up.
    clamped = ref.clamp_sigma(np.array([1e-9, 10.0], dtype=np.float64))
    assert float(clamped[0]) == ref.SIGMA_MIN
    assert float(clamped[1]) == ref.SIGMA_MAX
    with pytest.raises(ValueError, match="length mismatch"):
        ref.noisy_latent(mu, sigma[:1], eps)
    with pytest.raises(ValueError, match="empty"):
        ref.noisy_latent(np.array([], dtype=np.float64), sigma, eps)
    assert result["thoughts_decode"] is True


def test_stage_b_loss_is_weighted_sum() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    total = ref.stage_b_loss(1.0, 2.0, 3.0, 4.0, alpha=1.0, gamma=1.0, beta=0.01)
    assert total == pytest.approx(1.0 + 2.0 + 3.0 + 0.01 * 4.0)
    no_traj = ref.stage_b_loss(1.0, 2.0, 3.0, 4.0, alpha=0.0, gamma=1.0, beta=0.01)
    assert no_traj == pytest.approx(1.0 + 3.0 + 0.01 * 4.0)
    with pytest.raises(ValueError):
        ref.stage_b_loss(float("nan"), 0.0, 0.0, 0.0)
    assert ref.DEFAULT_ALPHA_TRAJ == 1.0
    assert ref.DEFAULT_GAMMA_HALT == 1.0
    assert ref.DEFAULT_BETA_KL == 0.01
    assert ref.DEFAULT_LAMBDA_PRIOR == 0.2
    assert ref.HALT_EPS == 1e-6
    assert result["curriculum_no_collapse"] is True


# ---------------------------------------------------------------------------
# Gradient checks (finite differences) and shapes / dtypes
# ---------------------------------------------------------------------------


def test_thought_decode_ce_finite_diff() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    logits = np.array(
        [
            [0.2, -0.1, 0.3, 0.0, -0.4, 0.5, -0.2, 0.1],
            [0.0, 0.4, -0.3, 0.2, 0.1, -0.5, 0.3, -0.1],
            [-0.2, 0.1, 0.5, -0.4, 0.0, 0.2, -0.3, 0.4],
            [0.3, -0.2, 0.1, 0.4, -0.1, 0.0, 0.2, -0.5],
        ],
        dtype=np.float64,
    )
    ids = ref.decode_teacher_ids()
    analytic = ref.thought_decode_ce_grad_logits(logits, ids)
    numeric = ref.finite_diff_thought_decode_ce(logits, ids)
    assert analytic.shape == logits.shape == numeric.shape
    assert analytic.dtype == np.float64
    assert numeric.dtype == np.float64
    assert ref.grad_match_ok(analytic, numeric) is True
    ce = ref.thought_decode_ce(logits, ids)
    assert math.isfinite(ce)
    assert ce > 0.0
    assert result["thoughts_decode"] is True


def test_halt_loss_finite_diff() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    logits = ref.teacher_halt_logits(2, ref.DEFAULT_MAX_THOUGHTS + 1)
    fd = ref.finite_diff_halt_loss(logits, 2)
    assert fd.shape == logits.shape
    assert fd.dtype == np.float64
    assert np.all(np.isfinite(fd))
    # Raising the teacher-slot logit must decrease halt_loss.
    assert float(fd[2]) < 0.0
    plus = logits.copy()
    plus[2] += 0.25
    minus = logits.copy()
    minus[2] -= 0.25
    assert ref.halt_loss_from_logits(plus, 2) < ref.halt_loss_from_logits(minus, 2)
    assert result["curriculum_no_collapse"] is True


def test_noisy_log_density_finite_diff() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    mu = np.array([0.0, 1.0, -0.5, 0.25], dtype=np.float64)
    sigma = np.array([0.05, 0.05, 0.08, 0.04], dtype=np.float64)
    z = mu + sigma * np.array([0.5, -1.0, 0.25, 1.5], dtype=np.float64)
    analytic = ref.analytic_log_density_grad_mu(z, mu, sigma)
    numeric = ref.finite_diff_log_density_mu(z, mu, sigma)
    assert analytic.shape == mu.shape == numeric.shape
    assert ref.grad_match_ok(analytic, numeric) is True
    assert ref.toy_grad_ok() is True
    assert result["thoughts_decode"] is True


def test_shapes_dtypes() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    xs, hops = ref.held_out_problems()
    assert xs.shape == (ref.TOY_N, ref.TOY_DIM)
    assert xs.dtype == np.float64
    assert hops == ref.REQUIRED_HOPS
    w = ref.hop_matrix()
    assert w.shape == (ref.TOY_DIM, ref.TOY_DIM)
    assert w.dtype == np.float64
    trained, losses = ref.train_stage_a()
    assert trained.shape == (ref.TOY_VOCAB, ref.TOY_VOCAB)
    assert trained.dtype == np.float64
    assert len(losses) == ref.TOY_STEPS_A
    proto = ref.evaluate_v6_protocol()
    decode = proto["decode"]
    assert decode["thoughts_dtype"] == "float64"
    assert proto["grad_ok"] is True
    assert result["curriculum_no_collapse"] is True
    assert result["accuracy_rises_with_latent_budget"] is True
    assert result["thoughts_decode"] is True


# ---------------------------------------------------------------------------
# Production must not import tests/; does not write v6.json
# ---------------------------------------------------------------------------


def test_production_source_does_not_import_tests() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    src = v6.__file__
    assert src is not None
    imported = _imported_names(src)
    for name in ("tests", "tests.reference", "tests.reference.v6_latent"):
        assert name not in imported, name
    assert result["curriculum_no_collapse"] is True


def test_reference_does_not_import_production_jax_torch() -> None:
    result = _assert_v6_result_shape(_cpu_run_v6())
    src = ref.__file__
    assert src is not None
    text = Path(src).read_text(encoding="utf-8")
    assert "prometheus.verify.v6_latent" not in text
    imported = _imported_names(src)
    for name in (
        "jax",
        "torch",
        "model",
        "train",
        "kernels",
        "prometheus",
        "prometheus.verify",
    ):
        assert name not in imported, name
    assert result["curriculum_no_collapse"] is True


def test_run_v6_does_not_write_v6_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VERIFY_OUT", str(tmp_path / "verify-out"))
    monkeypatch.setenv("VERIFY_OUTPUT_DIR", str(tmp_path / "verify-out"))
    monkeypatch.delenv("PROMETHEUS_ROOT", raising=False)
    result = _assert_v6_result_shape(v6.run_v6(gpus=1))
    assert result["curriculum_no_collapse"] is True
    assert result["accuracy_rises_with_latent_budget"] is True
    assert result["thoughts_decode"] is True
    assert list(tmp_path.rglob("v6.json")) == []
    assert not (tmp_path / "v6.json").exists()


# ---------------------------------------------------------------------------
# V6 GPU scale (spec 16.2: 4 to 8 H200). Skipped on CPU.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_v6_gpu_run_v6_gates() -> None:
    """V6: low end of spec (4 H200). Still analog; not V6 verified."""
    _require_gpu()
    result = _assert_v6_result_shape(v6.run_v6(gpus=4))
    assert result["curriculum_no_collapse"] is True
    assert result["accuracy_rises_with_latent_budget"] is True
    assert result["thoughts_decode"] is True
    assert ref.meets_v6_gates(result) is True


@pytest.mark.gpu
def test_v6_gpu_gpus_less_than_one_still_v6error() -> None:
    """V6: gpus < 1 is still V6Error on the GPU path; high end is 8 H200."""
    _require_gpu()
    with pytest.raises(v6.V6Error) as ei:
        v6.run_v6(gpus=0)
    assert type(ei.value) is v6.V6Error
    assert not isinstance(ei.value, NotImplementedError)
    result = _assert_v6_result_shape(v6.run_v6(gpus=8))
    assert result["curriculum_no_collapse"] is True
    assert result["accuracy_rises_with_latent_budget"] is True
    assert result["thoughts_decode"] is True
