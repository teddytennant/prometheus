"""Oracle tests for ``prometheus.verify.v9_lab.run_v9`` (spec 16.2 V9).

Every collected CPU test calls the public interface so the suite fails against
the ``NotImplementedError`` stub. GPU tests are marked ``gpu`` (V9 / 1–4 H200)
and skip cleanly without a device.

The CPU analog is one 14.6 cycle (pre-register, rung -1 through a Slurm
backend analog, replicate, ledger, eval-gate) glued to L4 genome-seed roles.
Passing these tests is not V9 verified.

Production is imported only here. ``tests.reference.v9_lab`` is plain NumPy
and must not import ``model/``, ``train/``, ``kernels/``, JAX, torch, or the
Rust crates.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prometheus.verify import v9_lab as v9
from tests.reference import v9_lab as ref

_CPU_RESULT: v9.V9Result | None = None
_CPU_ERROR: BaseException | None = None

V9_KEYS = (
    "planted_positive_found",
    "planted_positive_replicated",
    "planted_negative_recorded",
)


def _cpu_run_v9() -> v9.V9Result:
    """One ``run_v9(gpus=1)`` per process (CPU analog of V9); re-raise cached errors."""
    global _CPU_RESULT, _CPU_ERROR
    if _CPU_RESULT is not None:
        return _CPU_RESULT
    if _CPU_ERROR is not None:
        raise _CPU_ERROR
    try:
        out = v9.run_v9(gpus=1)
    except BaseException as exc:
        _CPU_ERROR = exc
        raise
    _CPU_RESULT = out
    return out


def _assert_v9_result_shape(result: Any) -> v9.V9Result:
    assert isinstance(result, dict), type(result)
    assert set(result) == set(V9_KEYS), set(result)
    found = result["planted_positive_found"]
    replicated = result["planted_positive_replicated"]
    recorded = result["planted_negative_recorded"]
    assert type(found) is bool, type(found)
    assert type(replicated) is bool, type(replicated)
    assert type(recorded) is bool, type(recorded)
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
    """Skip unless JAX sees a GPU. Marker ``gpu`` is the V9 stage tag."""
    jax = pytest.importorskip("jax")
    try:
        gpus = [d for d in jax.devices() if getattr(d, "platform", None) == "gpu"]
    except RuntimeError:
        gpus = []
    if not gpus:
        pytest.skip("no GPU")
    return jax


def _assert_v9error(gpus: int) -> None:
    with pytest.raises(v9.V9Error) as ei:
        v9.run_v9(gpus=gpus)
    assert type(ei.value) is v9.V9Error
    assert not isinstance(ei.value, NotImplementedError)


def _v9_json_hits(root: Path) -> list[Path]:
    return sorted(root.rglob("v9.json"))


# ---------------------------------------------------------------------------
# Faults: gpus < 1 -> V9Error (not NotImplementedError)
# ---------------------------------------------------------------------------


def test_gpus_zero_raises_v9error() -> None:
    with pytest.raises(ValueError, match="gpus"):
        ref.require_gpus(0)
    _assert_v9error(gpus=0)


def test_gpus_negative_raises_v9error() -> None:
    with pytest.raises(ValueError, match="gpus"):
        ref.require_gpus(-1)
    _assert_v9error(gpus=-1)
    _assert_v9error(gpus=-4)


# ---------------------------------------------------------------------------
# Result shape / types; all three bools true on a successful analog
# ---------------------------------------------------------------------------


def test_run_v9_result_shape_and_types() -> None:
    result = _assert_v9_result_shape(_cpu_run_v9())
    for key in V9_KEYS:
        assert result[key] is True, key
    payload = json.dumps(dict(result))
    loaded = json.loads(payload)
    assert set(loaded.keys()) == set(V9_KEYS)
    for key in V9_KEYS:
        assert loaded[key] is True
        assert type(loaded[key]) is bool


def test_successful_analog_all_three_bools_true() -> None:
    result = _assert_v9_result_shape(_cpu_run_v9())
    for key in V9_KEYS:
        assert result[key] is True, key
    assert ref.meets_v9_gates(result) is True
    proto = ref.evaluate_v9_protocol()
    assert proto["meets_gates"] is True
    assert proto["cycle_complete"] is True
    assert proto["planted_positive_found"] is True
    assert proto["planted_positive_replicated"] is True
    assert proto["planted_negative_recorded"] is True
    assert proto["rung"] == ref.RUNG_MINUS_ONE
    assert proto["grad_ok"] is True
    assert proto["fp32_parity_ok"] is True


def test_gpus_ge_one_accepted_cpu_analog() -> None:
    result = _assert_v9_result_shape(_cpu_run_v9())
    for key in V9_KEYS:
        assert result[key] is True, key
    assert v9.DEFAULT_GPUS == 1
    assert ref.DEFAULT_GPUS == 1
    assert v9.DEFAULT_GPUS == ref.DEFAULT_GPUS
    assert ref.DEFAULT_GPUS_MIN == 1
    assert ref.DEFAULT_GPUS_MAX == 4
    assert issubclass(v9.V9Error, Exception)
    assert not issubclass(v9.V9Error, NotImplementedError)
    four = _assert_v9_result_shape(v9.run_v9(gpus=4))
    for key in V9_KEYS:
        assert four[key] is True, key
    proto_four = ref.evaluate_v9_protocol(gpus=4)
    assert proto_four["meets_gates"] is True
    assert proto_four["gpus"] == 4


# ---------------------------------------------------------------------------
# Planted positive is actually found (rung -1 OLS analog, not hardcoded True)
# ---------------------------------------------------------------------------


def test_planted_positive_found_via_rung_minus_one() -> None:
    result = _assert_v9_result_shape(_cpu_run_v9())
    proto = ref.evaluate_v9_protocol()
    positive = proto["positive"]
    assert positive["planted"] is True
    assert positive["idea_id"] == ref.PLANTED_POSITIVE_ID
    assert positive["job_state"] == "completed"
    assert positive["delta_mse"] is not None
    assert positive["delta_mse"] > ref.KILL_EPS
    assert positive["beats_kill"] is True
    assert abs(positive["delta_mse"] - positive["predicted_delta_mse"]) <= ref.PREDICTION_TOL
    assert proto["ols"]["pos_mse_test"] < proto["ols"]["base_mse_test"]
    assert proto["planted_positive_found"] is True
    assert result["planted_positive_found"] is True


def test_fault_skip_plant_positive_found_false() -> None:
    _assert_v9_result_shape(_cpu_run_v9())
    proto = ref.evaluate_v9_protocol(plant_positive=False)
    assert proto["positive"]["planted"] is False
    assert proto["planted_positive_found"] is False
    assert proto["planted_positive_replicated"] is False
    assert proto["meets_gates"] is False
    assert _cpu_run_v9()["planted_positive_found"] is True


# ---------------------------------------------------------------------------
# Planted positive is actually replicated (independent seed)
# ---------------------------------------------------------------------------


def test_planted_positive_replicated_independent_seed() -> None:
    result = _assert_v9_result_shape(_cpu_run_v9())
    proto = ref.evaluate_v9_protocol()
    repl = proto["replication"]
    assert repl["skipped"] is False
    assert repl["status"] == "matched"
    assert repl["n"] == 2
    assert repl["seed"] == ref.TOY_REPLICATE_SEED
    assert repl["delta_mse"] is not None
    assert repl["delta_mse"] > ref.KILL_EPS
    assert repl["delta_mse"] != proto["positive"]["delta_mse"]
    assert proto["planted_positive_replicated"] is True
    assert result["planted_positive_replicated"] is True


def test_fault_skip_replicate_replicated_false() -> None:
    _assert_v9_result_shape(_cpu_run_v9())
    proto = ref.evaluate_v9_protocol(skip_replicate=True)
    assert proto["replication"]["skipped"] is True
    assert proto["replication"]["status"] == "unreplicated"
    assert proto["replication"]["n"] == 1
    assert proto["planted_positive_found"] is True
    assert proto["planted_positive_replicated"] is False
    assert proto["meets_gates"] is False
    assert _cpu_run_v9()["planted_positive_replicated"] is True


# ---------------------------------------------------------------------------
# Planted negative is recorded as negative in the ledger
# ---------------------------------------------------------------------------


def test_planted_negative_recorded_as_negative() -> None:
    result = _assert_v9_result_shape(_cpu_run_v9())
    proto = ref.evaluate_v9_protocol()
    negative = proto["negative"]
    assert negative["planted"] is True
    assert negative["idea_id"] == ref.PLANTED_NEGATIVE_ID
    assert negative["job_state"] == "completed"
    assert negative["delta_mse"] is not None
    assert negative["beats_kill"] is False
    assert negative["is_negative"] is True
    assert negative["recorded_outcome"] == "negative"
    assert proto["ledger"]["skipped"] is False
    assert "v9-rung-minus-1-planted-negative" in proto["ledger"]["ids"]
    assert ref.RUNG_MINUS_ONE in proto["ledger"]["rungs"]
    assert proto["planted_negative_recorded"] is True
    assert result["planted_negative_recorded"] is True


def test_fault_skip_ledger_write_negative_false() -> None:
    _assert_v9_result_shape(_cpu_run_v9())
    skipped = ref.evaluate_v9_protocol(skip_ledger_write=True)
    assert skipped["ledger"]["skipped"] is True
    assert skipped["ledger"]["n"] == 0
    assert skipped["negative"]["is_negative"] is True
    assert skipped["planted_negative_recorded"] is False
    assert skipped["meets_gates"] is False
    wrong_sign = ref.evaluate_v9_protocol(record_negative_as_positive=True)
    assert wrong_sign["negative"]["recorded_outcome"] == "positive"
    assert wrong_sign["planted_negative_recorded"] is False
    assert wrong_sign["meets_gates"] is False
    no_neg = ref.evaluate_v9_protocol(plant_negative=False)
    assert no_neg["negative"]["planted"] is False
    assert no_neg["planted_negative_recorded"] is False
    assert _cpu_run_v9()["planted_negative_recorded"] is True


# ---------------------------------------------------------------------------
# Cycle pieces: pre-register, Slurm analog, eval-gate, genome-seed roles
# ---------------------------------------------------------------------------


def test_cycle_preregister_slurm_eval_gate_roles() -> None:
    result = _assert_v9_result_shape(_cpu_run_v9())
    proto = ref.evaluate_v9_protocol()
    assert proto["roles"] == ref.GENOME_ROLES
    assert proto["cycle_steps"] == ref.CYCLE_STEPS
    prereg = proto["preregister"]
    assert prereg["skipped"] is False
    assert prereg["author_role"] == ref.AUTHOR_ROLE_RESEARCHER
    assert prereg["n"] == 2
    assert ref.PLANTED_POSITIVE_ID in prereg["ids"]
    assert ref.PLANTED_NEGATIVE_ID in prereg["ids"]
    assert "hypothesis" in prereg["fields"]
    assert "numeric_prediction" in prereg["fields"]
    slurm = proto["slurm"]
    assert slurm["skipped"] is False
    assert slurm["job_prefix"] == ref.JOB_PREFIX
    assert slurm["n_jobs"] == 3
    assert slurm["states"] == ("completed", "completed", "completed")
    assert all(name.startswith(f"{ref.JOB_PREFIX}-") for name in slurm["names"])
    gate = proto["eval_gate"]
    assert gate["skipped"] is False
    assert gate["caller"] == ref.AUTHOR_ROLE_TESTER
    assert gate["eval_gate_passed"] is True
    assert gate["leaked_task_text"] is False
    assert gate["suites"] == ref.HELD_OUT_SUITES
    assert gate["delta_rci_milli"] == ref.RCI_DELTA_MILLI
    for score in gate["scores"]:
        assert "prompt" not in score
        assert "task" not in score
        assert "answer" not in score
        assert "task_text" not in score

    no_reg = ref.evaluate_v9_protocol(skip_preregister=True)
    assert no_reg["preregister"]["n"] == 0
    assert no_reg["planted_positive_found"] is False
    assert no_reg["planted_negative_recorded"] is False
    assert no_reg["meets_gates"] is False

    no_slurm = ref.evaluate_v9_protocol(skip_slurm=True)
    assert no_slurm["slurm"]["skipped"] is True
    assert set(no_slurm["slurm"]["states"]) == {"pending"}
    assert no_slurm["planted_positive_found"] is False
    assert no_slurm["planted_positive_replicated"] is False
    assert no_slurm["planted_negative_recorded"] is False

    no_gate = ref.evaluate_v9_protocol(skip_eval_gate=True)
    assert no_gate["eval_gate"]["skipped"] is True
    assert no_gate["eval_gate"]["eval_gate_passed"] is False
    assert no_gate["cycle_complete"] is False
    assert result["planted_positive_found"] is True


# ---------------------------------------------------------------------------
# Toy OLS: shapes, dtypes, residual orthogonality, gradients, FP32 1e-5
# ---------------------------------------------------------------------------


def test_toy_ols_shapes_dtypes_gradients_fp32() -> None:
    result = _assert_v9_result_shape(_cpu_run_v9())
    proto = ref.evaluate_v9_protocol()
    ols = proto["ols"]
    n_tr = ref.TOY_N_TRAIN
    n_te = ref.TOY_N_TEST
    d_pos = 1 + ref.TOY_D_BASE + 1  # intercept + base + planted feature
    assert ols["X_train_shape"] == (n_tr, d_pos)
    assert ols["X_test_shape"] == (n_te, d_pos)
    assert ols["w_shape"] == (d_pos,)
    assert ols["dtype"] == "float64"
    assert ols["residual_orthogonality"] is True
    assert ols["nested_train_mse"] is True
    assert ols["pos_mse_train"] <= ols["base_mse_train"]
    assert proto["grad_ok"] is True
    assert proto["fp32_parity_ok"] is True
    assert ref.PREDICTION_TOL == 1e-5
    assert ref.toy_grad_ok() is True

    design = ref.make_design(ref.TOY_SEED)
    pos = ref.fit_idea(design, extra="pos")
    x_tr = pos["X_train"]
    y_tr = design["y_train"]
    w = pos["w"]
    assert x_tr.dtype == np.float64
    assert y_tr.dtype == np.float64
    assert w.dtype == np.float64
    m64 = ref.ols_mse_dtype(x_tr, y_tr, np.float64)
    m32 = ref.ols_mse_dtype(x_tr, y_tr, np.float32)
    assert abs(m32 - m64) <= ref.PREDICTION_TOL
    analytic = ref.analytic_mse_grad(x_tr, w, y_tr)
    fd = np.empty_like(w)
    eps = ref.TOY_FD_EPS
    for i in range(int(w.shape[0])):
        up = w.copy()
        dn = w.copy()
        up[i] += eps
        dn[i] -= eps
        fd[i] = (ref.mse_loss(x_tr, up, y_tr) - ref.mse_loss(x_tr, dn, y_tr)) / (
            2.0 * eps
        )
    assert np.allclose(analytic, fd, rtol=ref.TOY_GRAD_RTOL, atol=ref.TOY_GRAD_ATOL)
    assert result["planted_positive_found"] is True


# ---------------------------------------------------------------------------
# Production must not import tests/; reference stays plain NumPy
# ---------------------------------------------------------------------------


def test_production_source_does_not_import_tests() -> None:
    result = _assert_v9_result_shape(_cpu_run_v9())
    src = Path(v9.__file__).resolve()
    text = src.read_text(encoding="utf-8")
    names = _imported_names(str(src))
    assert "tests" not in names
    assert "tests.reference" not in text
    assert "tests.reference.v9_lab" not in text
    for key in V9_KEYS:
        assert result[key] is True, key


def test_reference_does_not_import_production_jax_torch() -> None:
    _assert_v9_result_shape(_cpu_run_v9())
    src = Path(__file__).resolve().parent / "reference" / "v9_lab.py"
    names = _imported_names(str(src))
    assert "jax" not in names
    assert "torch" not in names
    assert "prometheus" not in names
    assert "model" not in names
    assert "train" not in names
    assert "kernels" not in names


def test_run_v9_does_not_write_v9_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "verify-out"
    monkeypatch.setenv("VERIFY_OUT", str(out))
    monkeypatch.setenv("VERIFY_OUTPUT_DIR", str(out))
    result = v9.run_v9(gpus=1)
    _assert_v9_result_shape(result)
    assert _v9_json_hits(tmp_path) == []
    cwd_hit = Path("v9.json")
    assert not cwd_hit.exists()
    assert not (out / "v9.json").exists()


# ---------------------------------------------------------------------------
# GPU: V9 / 1–4 H200. Skipped without a device.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_v9_gpu_run_v9_gates() -> None:
    """V9 / 1–4 H200. Skipped without a device."""
    _require_gpu()
    result = _assert_v9_result_shape(v9.run_v9(gpus=ref.DEFAULT_GPUS_MAX))
    for key in V9_KEYS:
        assert result[key] is True, key
    assert ref.meets_v9_gates(result) is True


@pytest.mark.gpu
def test_v9_gpu_gpus_less_than_one_still_v9error() -> None:
    """V9 GPU analog still raises V9Error for gpus < 1."""
    _require_gpu()
    _assert_v9error(gpus=0)
