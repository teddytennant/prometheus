"""Oracle tests for V10 soak (spec 16.2 / 15.2 H11). Independent of the runner."""

from __future__ import annotations

import ast
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prometheus.verify import v10_soak as v10
from tests.reference import v10_soak as ref

V10_KEYS = ref.V10_RESULT_KEYS

_CPU_RESULT: v10.V10Result | None = None
_CPU_ERROR: BaseException | None = None


def _cpu_run_v10() -> v10.V10Result:
    """Cached ``run_v10(hours=1)``. Every CPU test must go through ``run_v10``."""
    global _CPU_RESULT, _CPU_ERROR
    if _CPU_RESULT is not None:
        return _CPU_RESULT
    if _CPU_ERROR is not None:
        raise _CPU_ERROR
    try:
        out = v10.run_v10(hours=1.0)
    except BaseException as exc:
        _CPU_ERROR = exc
        raise
    _CPU_RESULT = out
    return out


def _assert_v10error(*, hours: float) -> None:
    with pytest.raises(v10.V10Error) as ei:
        v10.run_v10(hours=hours)
    assert ei.type is v10.V10Error
    assert not isinstance(ei.value, NotImplementedError)


def _assert_v10_result_shape(result: Any, *, hours: float | None = None) -> v10.V10Result:
    assert isinstance(result, dict)
    assert set(result.keys()) == set(V10_KEYS)
    for key in V10_KEYS:
        assert type(result[key]) is float, key
        assert math.isfinite(float(result[key])), key
    if hours is not None:
        assert result["hours"] == float(hours)
    return result


def _assert_counts_zero(result: v10.V10Result) -> None:
    assert result["lost_tasks"] == 0.0
    assert result["duplicated_outputs"] == 0.0
    assert result["dead_tokens"] == 0.0


def _imported_top_level(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _v10_json_hits(root: Path) -> list[Path]:
    return sorted(root.rglob("v10.json"))


def _require_gpu():
    jax = pytest.importorskip("jax")
    try:
        devices = jax.devices()
    except Exception:
        pytest.skip("no JAX devices")
    kinds = {str(getattr(d, "platform", "")).lower() for d in devices}
    kinds |= {str(getattr(d, "device_kind", "")).lower() for d in devices}
    blob = " ".join(sorted(kinds))
    if not any(tok in blob for tok in ("gpu", "cuda", "rocm", "tpu")):
        pytest.skip("no GPU")
    return jax


# ---------------------------------------------------------------------------
# Faults: hours < 1 raises V10Error, not NotImplementedError
# ---------------------------------------------------------------------------


def test_hours_zero_raises_v10error() -> None:
    _assert_v10error(hours=0.0)


def test_hours_negative_raises_v10error() -> None:
    _assert_v10error(hours=-1.0)
    _assert_v10error(hours=-72.0)


def test_hours_below_one_raises_v10error() -> None:
    _assert_v10error(hours=0.5)
    _assert_v10error(hours=0.999)


def test_hours_nan_inf_raise_v10error() -> None:
    _assert_v10error(hours=float("nan"))
    _assert_v10error(hours=float("inf"))
    _assert_v10error(hours=float("-inf"))


# ---------------------------------------------------------------------------
# Public CPU analog (hours=1): exact four floats, counts 0, JSON-serializable
# ---------------------------------------------------------------------------


def test_run_v10_cpu_analog_keys_types_and_zeros() -> None:
    result = _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    _assert_counts_zero(result)
    assert ref.analog_ok(result) is True
    assert ref.meets_v10_gates(result) is False


def test_run_v10_result_json_serializable() -> None:
    result = _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    payload = json.dumps(dict(result))
    loaded = json.loads(payload)
    assert set(loaded.keys()) == set(V10_KEYS)
    for key in V10_KEYS:
        assert type(loaded[key]) is float
        assert loaded[key] == result[key]
    assert loaded["hours"] == 1.0
    assert loaded["lost_tasks"] == 0.0
    assert loaded["duplicated_outputs"] == 0.0
    assert loaded["dead_tokens"] == 0.0


def test_default_hours_constants() -> None:
    _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    assert v10.DEFAULT_HOURS == 72.0
    assert v10.SOAK_HOURS_MIN == 72.0
    assert v10.DEFAULT_HOURS == ref.DEFAULT_HOURS
    assert v10.SOAK_HOURS_MIN == ref.SOAK_HOURS_MIN
    assert ref.CPU_ANALOG_HOURS_MIN == 1.0
    assert ref.MS_PER_HOUR == 3_600_000


def test_v10error_is_exception_not_notimplemented() -> None:
    _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    assert issubclass(v10.V10Error, Exception)
    assert not issubclass(v10.V10Error, NotImplementedError)


def test_cpu_analog_returns_requested_hours() -> None:
    one = _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    _assert_counts_zero(one)
    one_half = _assert_v10_result_shape(v10.run_v10(hours=1.5), hours=1.5)
    _assert_counts_zero(one_half)


def test_cpu_hours_one_is_not_v10_verified() -> None:
    """hours=1 analog is allowed; F4 GPU gate still requires hours >= 72."""
    result = _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    _assert_counts_zero(result)
    assert result["hours"] < v10.SOAK_HOURS_MIN
    assert ref.meets_v10_gates(result) is False
    assert ref.analog_ok(result) is True


# ---------------------------------------------------------------------------
# Independent reference protocol (not hardcoded zeros)
# ---------------------------------------------------------------------------


def test_reference_protocol_happy_path() -> None:
    result = _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    proto = ref.evaluate_v10_protocol(hours=1.0)
    assert proto["result"]["hours"] == 1.0
    assert proto["result"]["lost_tasks"] == 0.0
    assert proto["result"]["duplicated_outputs"] == 0.0
    assert proto["result"]["dead_tokens"] == 0.0
    assert proto["analog_ok"] is True
    assert proto["meets_gates"] is False
    assert proto["did_kill"] is True
    assert proto["did_partition"] is True
    assert proto["did_replica_fault"] is True
    assert proto["did_recover"] is True
    assert proto["did_broker_death"] is True
    assert proto["did_token_expiry"] is True
    assert proto["chain_ok"] is True
    assert proto["n_replicas"] >= ref.MIN_REPLICAS
    assert proto["n_enqueued"] == ref.DEFAULT_TASKS
    assert proto["n_completed"] == ref.DEFAULT_TASKS
    assert proto["sim_ms"] == int(1.0 * ref.MS_PER_HOUR)
    assert proto["grad_ok"] is True
    _assert_counts_zero(result)


def test_reference_exact_values_vs_production() -> None:
    prod = _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    ref_res = ref.evaluate_v10_protocol(hours=1.0)["result"]
    for key in V10_KEYS:
        assert prod[key] == ref_res[key], key
        assert type(prod[key]) is float


def test_reference_drop_task_lost_tasks() -> None:
    _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    proto = ref.evaluate_v10_protocol(hours=1.0, drop_task=True)
    assert proto["result"]["lost_tasks"] == 1.0
    assert proto["result"]["duplicated_outputs"] == 0.0
    assert proto["result"]["dead_tokens"] == 0.0
    assert proto["analog_ok"] is False
    assert proto["dropped_task_id"] is not None
    assert proto["n_completed"] == ref.DEFAULT_TASKS - 1


def test_reference_duplicate_output() -> None:
    _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    proto = ref.evaluate_v10_protocol(hours=1.0, duplicate_output=True)
    assert proto["result"]["lost_tasks"] == 0.0
    assert proto["result"]["duplicated_outputs"] == 1.0
    assert proto["result"]["dead_tokens"] == 0.0
    assert proto["analog_ok"] is False


def test_reference_mint_dead_token() -> None:
    _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    proto = ref.evaluate_v10_protocol(hours=1.0, mint_dead_token=True)
    assert proto["result"]["lost_tasks"] == 0.0
    assert proto["result"]["duplicated_outputs"] == 0.0
    assert proto["result"]["dead_tokens"] == 1.0
    assert proto["analog_ok"] is False


def test_reference_kill_partition_replica_recover() -> None:
    _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    world = ref.World(seed=ref.DEFAULT_SEED)
    world.enqueue_task("task-0")
    world.kill_process("worker-0")
    assert any(not p.alive for p in world.processes)
    world.recover()
    assert all(p.alive for p in world.processes)
    world.partition(0, 1, asymmetric=True)
    assert (0, 1) in world.partitions
    world.heal()
    assert not world.partitions
    world.node_loss(1)
    assert world.replicas[1].lost is True
    world.recover()
    assert world.replicas[1].lost is False
    world.complete_task("task-0", attempt=1)
    lost, dup, dead = world.invariants()
    assert lost == 0.0
    assert dup == 0.0
    assert dead == 0.0
    assert "kill" in world.faults_applied
    assert "partition" in world.faults_applied
    assert "replica" in world.faults_applied
    assert "recover" in world.faults_applied


def test_reference_simulated_time_not_wall_clock() -> None:
    _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    t0 = time.monotonic()
    proto = ref.evaluate_v10_protocol(hours=72.0)
    elapsed = time.monotonic() - t0
    assert proto["result"]["hours"] == 72.0
    assert proto["sim_ms"] == int(72.0 * ref.MS_PER_HOUR)
    assert proto["meets_gates"] is True
    assert elapsed < 5.0


def test_reference_event_log_hash_chain() -> None:
    _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    proto = ref.evaluate_v10_protocol(hours=1.0)
    assert proto["chain_ok"] is True
    world = ref.World(seed=3)
    world.enqueue_task("task-0")
    world.complete_task("task-0", attempt=1)
    assert world.chain_ok(0) is True
    replica = world.replicas[0]
    assert replica.events, "expected analog events"
    replica.events[0].body["task_id"] = "mutated"
    assert world.chain_ok(0) is False


def test_reference_hours_below_one_valueerror() -> None:
    with pytest.raises(v10.V10Error):
        v10.run_v10(hours=0.0)
    with pytest.raises(ref.AnalogError):
        ref.evaluate_v10_protocol(hours=0.0)
    with pytest.raises(ref.AnalogError):
        ref.require_hours(0.5)
    with pytest.raises(ref.AnalogError):
        ref.require_hours(float("nan"))


def test_invariants_property_counts() -> None:
    _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    enqueued = ["a", "b", "c"]
    lost, dup, dead = ref.count_invariants(enqueued, {("a", 1): 1, ("b", 1): 1}, 0)
    assert lost == 1.0
    assert dup == 0.0
    assert dead == 0.0
    lost, dup, dead = ref.count_invariants(enqueued, {("a", 1): 2, ("b", 1): 1, ("c", 1): 1}, 0)
    assert lost == 0.0
    assert dup == 1.0
    lost, dup, dead = ref.count_invariants(["a"], {("a", 1): 1}, 4)
    assert lost == 0.0
    assert dup == 0.0
    assert dead == 4.0
    empty = ref.count_invariants([], {}, 0)
    assert empty == (0.0, 0.0, 0.0)


def test_meets_v10_gates_properties() -> None:
    _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    zeros = {
        "hours": 72.0,
        "lost_tasks": 0.0,
        "duplicated_outputs": 0.0,
        "dead_tokens": 0.0,
    }
    assert ref.meets_v10_gates(zeros) is True
    assert ref.analog_ok(zeros) is True
    short = dict(zeros)
    short["hours"] = 1.0
    assert ref.meets_v10_gates(short) is False
    assert ref.analog_ok(short) is True
    lost = dict(zeros)
    lost["lost_tasks"] = 1.0
    assert ref.meets_v10_gates(lost) is False
    dup = dict(zeros)
    dup["duplicated_outputs"] = 1.0
    assert ref.meets_v10_gates(dup) is False
    dead = dict(zeros)
    dead["dead_tokens"] = 1.0
    assert ref.meets_v10_gates(dead) is False
    assert ref.meets_v10_gates({"hours": 72.0}) is False


# ---------------------------------------------------------------------------
# Gradient checks (finite differences) and shapes/dtypes
# ---------------------------------------------------------------------------


def test_lease_remaining_finite_diff() -> None:
    _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    elapsed = 250.0
    analytic = ref.lease_d_elapsed()
    numeric = ref.lease_finite_diff(elapsed)
    assert analytic == -1.0
    assert numeric == pytest.approx(-1.0, abs=1e-9)
    assert ref.lease_grad_ok() is True
    assert ref.lease_expired(ref.LEASE_TTL_MS) is True
    assert ref.lease_expired(0.0) is False
    remaining = ref.lease_remaining_ms(100.0)
    assert remaining == ref.LEASE_TTL_MS - 100.0
    assert type(remaining) is float


def test_liveness_vector_shape_dtype() -> None:
    _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    world = ref.World()
    vec = world.liveness_vector()
    assert vec.shape == (ref.MIN_REPLICAS,)
    assert vec.dtype == np.float64
    assert np.all(vec == 1.0)
    world.node_loss(1)
    vec2 = world.liveness_vector()
    assert vec2[1] == 0.0
    assert vec2.dtype == np.float64


def test_clock_skew_bound() -> None:
    _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    world = ref.World()
    world.apply_clock_skew(ref.CLOCK_SKEW_MS)
    with pytest.raises(ref.AnalogError):
        world.apply_clock_skew(ref.CLOCK_SKEW_MS + 1)


# ---------------------------------------------------------------------------
# Production hygiene: no tests/ import; does not write v10.json
# ---------------------------------------------------------------------------


def test_reference_does_not_import_forbidden() -> None:
    _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    src = Path(__file__).resolve().parent / "reference" / "v10_soak.py"
    names = _imported_top_level(src)
    assert "jax" not in names
    assert "torch" not in names
    assert "prometheus" not in names
    assert "model" not in names
    assert "train" not in names
    assert "kernels" not in names


def test_production_source_does_not_import_tests() -> None:
    _assert_v10_result_shape(_cpu_run_v10(), hours=1.0)
    src = Path(__file__).resolve().parents[1] / "prometheus" / "verify" / "v10_soak.py"
    names = _imported_top_level(src)
    assert "tests" not in names


def test_run_v10_does_not_write_v10_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "verify-out"
    monkeypatch.setenv("VERIFY_OUT", str(out))
    monkeypatch.setenv("VERIFY_OUTPUT_DIR", str(out))
    result = _assert_v10_result_shape(v10.run_v10(hours=1.0), hours=1.0)
    _assert_counts_zero(result)
    assert _v10_json_hits(tmp_path) == []
    assert not Path("v10.json").exists()
    assert not (out / "v10.json").exists()


def test_cpu_analog_hours_72_is_simulated() -> None:
    t0 = time.monotonic()
    result = _assert_v10_result_shape(v10.run_v10(hours=72.0), hours=72.0)
    elapsed = time.monotonic() - t0
    _assert_counts_zero(result)
    assert ref.meets_v10_gates(result) is True
    assert elapsed < 30.0


# ---------------------------------------------------------------------------
# GPU: V10 / 72h soak. Skipped without a device.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_v10_gpu_run_v10_gates() -> None:
    """V10 / 72h soak. Skipped without a device."""
    _require_gpu()
    result = _assert_v10_result_shape(v10.run_v10(hours=v10.DEFAULT_HOURS), hours=v10.DEFAULT_HOURS)
    _assert_counts_zero(result)
    assert result["hours"] >= v10.SOAK_HOURS_MIN
    assert ref.meets_v10_gates(result) is True


@pytest.mark.gpu
def test_v10_gpu_hours_less_than_one_still_v10error() -> None:
    """V10 GPU analog still rejects hours < 1 with V10Error."""
    _require_gpu()
    _assert_v10error(hours=0.0)
    result = _assert_v10_result_shape(
        v10.run_v10(hours=v10.SOAK_HOURS_MIN), hours=v10.SOAK_HOURS_MIN
    )
    _assert_counts_zero(result)
    assert ref.meets_v10_gates(result) is True
