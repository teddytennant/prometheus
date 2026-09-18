"""Oracle tests for ``prometheus.verify.v7_rl.run_v7`` (spec 16.2 V7).

Every collected CPU test calls the public interface so the suite fails against
the ``NotImplementedError`` stub. GPU tests are marked ``gpu`` (V7 / 8 H200,
4 SGLang + 4 JAX) and skip cleanly without a device.

Production is imported only here. ``tests.reference.v7_rl`` is plain NumPy
and must not import ``prometheus.verify.v7_rl``, JAX, torch, or the Rust crates.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from prometheus.verify import v7_rl as v7
from tests.reference import v7_rl as ref

_CPU_RESULT: v7.V7Result | None = None
_CPU_ERROR: BaseException | None = None

V7_KEYS = ("reward_rises", "logprob_drift_halted", "planted_write_flagged")


def _cpu_run_v7() -> v7.V7Result:
    """One ``run_v7(gpus=1)`` per process (CPU analog of V7); re-raise cached errors."""
    global _CPU_RESULT, _CPU_ERROR
    if _CPU_RESULT is not None:
        return _CPU_RESULT
    if _CPU_ERROR is not None:
        raise _CPU_ERROR
    try:
        out = v7.run_v7(gpus=1)
    except BaseException as exc:
        _CPU_ERROR = exc
        raise
    _CPU_RESULT = out
    return out


def _assert_v7_result_shape(result: object) -> v7.V7Result:
    assert isinstance(result, dict), type(result)
    assert set(result.keys()) == set(V7_KEYS), result.keys()
    for key in V7_KEYS:
        assert type(result[key]) is bool, (key, type(result[key]), result[key])
    return result  # type: ignore[return-value]


def _assert_v7error(*, gpus: int) -> None:
    with pytest.raises(v7.V7Error) as ei:
        v7.run_v7(gpus=gpus)
    assert type(ei.value) is v7.V7Error
    assert not isinstance(ei.value, NotImplementedError)


def _require_gpu() -> None:
    jax = pytest.importorskip("jax")
    try:
        devices = jax.devices()
    except Exception:
        pytest.skip("no JAX devices")
    kinds = {str(getattr(d, "platform", "")).lower() for d in devices}
    kinds |= {str(getattr(d, "device_kind", "")).lower() for d in devices}
    blob = " ".join(sorted(kinds))
    if not any(tok in blob for tok in ("gpu", "cuda", "rocm", "tpu")):
        pytest.skip("no GPU/TPU JAX device")


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


def _v7_json_hits(root: Path) -> list[Path]:
    return sorted(root.rglob("v7.json"))


# ---------------------------------------------------------------------------
# Faults: gpus < 1 raises V7Error, not NotImplementedError
# ---------------------------------------------------------------------------


def test_gpus_zero_raises_v7error() -> None:
    _assert_v7error(gpus=0)


def test_gpus_negative_raises_v7error() -> None:
    _assert_v7error(gpus=-1)
    _assert_v7error(gpus=-8)


# ---------------------------------------------------------------------------
# Public CPU analog (gpus=1): exact three bools, all True, JSON-serializable
# ---------------------------------------------------------------------------


def test_run_v7_gpus_1_keys_exactly_three_bools_all_true() -> None:
    result = _assert_v7_result_shape(_cpu_run_v7())
    assert set(result.keys()) == set(V7_KEYS)
    for key in V7_KEYS:
        assert result[key] is True, key
    assert ref.meets_v7_gates(result) is True


def test_run_v7_result_json_serializable() -> None:
    result = _assert_v7_result_shape(_cpu_run_v7())
    payload = json.dumps(dict(result))
    loaded = json.loads(payload)
    assert set(loaded.keys()) == set(V7_KEYS)
    for key in V7_KEYS:
        assert loaded[key] is True
        assert type(loaded[key]) is bool


def test_default_gpus_constant() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    assert v7.DEFAULT_GPUS == 8
    assert v7.DEFAULT_GPUS == ref.DEFAULT_GPUS
    assert ref.DEFAULT_GPUS == 8


def test_v7error_is_exception_not_notimplemented() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    assert issubclass(v7.V7Error, Exception)
    assert not issubclass(v7.V7Error, NotImplementedError)


# ---------------------------------------------------------------------------
# Independent reference protocol (not hardcoded gates)
# ---------------------------------------------------------------------------


def test_reference_protocol_all_three_gates() -> None:
    result = _assert_v7_result_shape(_cpu_run_v7())
    proto = ref.evaluate_v7_protocol()
    assert proto["reward_rises"] is True
    assert proto["logprob_drift_halted"] is True
    assert proto["planted_write_flagged"] is True
    assert proto["meets_gates"] is True
    assert proto["grad_ok"] is True
    for key in V7_KEYS:
        assert result[key] is True, key


def test_reward_rises_vs_no_update_twin() -> None:
    """Reward actually rises vs a frozen twin; not a hardcoded True."""
    _assert_v7_result_shape(_cpu_run_v7())
    proto = ref.evaluate_v7_protocol(do_update=True)
    reward = proto["reward"]
    assert reward["did_update"] is True
    assert reward["trained_reward"] > reward["twin_reward"]
    assert proto["reward_rises"] is True
    assert _cpu_run_v7()["reward_rises"] is True


def test_skip_update_reward_does_not_rise() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    proto = ref.evaluate_v7_protocol(do_update=False)
    reward = proto["reward"]
    assert reward["did_update"] is False
    assert reward["trained_reward"] == reward["twin_reward"]
    assert proto["reward_rises"] is False
    assert proto["meets_gates"] is False


def test_injected_logprob_drift_halts() -> None:
    """Injected trainer-vs-engine drift actually exceeds the parity threshold."""
    _assert_v7_result_shape(_cpu_run_v7())
    proto = ref.evaluate_v7_protocol(inject_drift=True)
    parity = proto["parity"]
    assert parity["injected_drift"] is True
    assert parity["injected_drift_amount"] == ref.INJECTED_DRIFT
    assert parity["max_abs_logprob_diff"] > ref.PARITY_THRESHOLD
    assert parity["max_abs_logprob_diff"] >= ref.INJECTED_DRIFT
    assert proto["logprob_drift_halted"] is True
    assert _cpu_run_v7()["logprob_drift_halted"] is True


def test_no_injected_drift_does_not_halt() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    proto = ref.evaluate_v7_protocol(inject_drift=False)
    parity = proto["parity"]
    assert parity["injected_drift"] is False
    assert parity["max_abs_logprob_diff"] == 0.0
    assert proto["logprob_drift_halted"] is False
    assert proto["meets_gates"] is False


def test_planted_write_actually_flagged() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    proto = ref.evaluate_v7_protocol(plant_write=True)
    write = proto["write"]
    assert write["planted"] is True
    assert write["path"] == ref.PLANTED_WRITE_PATH
    assert write["is_grader_path"] is True
    assert proto["planted_write_flagged"] is True
    assert _cpu_run_v7()["planted_write_flagged"] is True


def test_no_planted_write_not_flagged() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    proto = ref.evaluate_v7_protocol(plant_write=False)
    assert proto["write"]["planted"] is False
    assert proto["planted_write_flagged"] is False
    assert proto["meets_gates"] is False


def test_planted_write_agent_path_not_flagged() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    proto = ref.evaluate_v7_protocol(plant_write=True, planted_path=ref.AGENT_WRITE_PATH)
    assert proto["write"]["is_grader_path"] is False
    assert proto["planted_write_flagged"] is False


# ---------------------------------------------------------------------------
# Protocol pieces: GSPO/DAPO, staleness, routing replay, weight sync
# ---------------------------------------------------------------------------


def test_mean_center_zero_mean() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    adv = ref.mean_center((1.0, 0.0, 1.0, 0.0))
    assert abs(sum(adv)) < 1e-12
    assert len(adv) == 4


def test_mean_center_all_equal_zero() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    adv = ref.mean_center((1.0, 1.0, 1.0))
    assert adv == (0.0, 0.0, 0.0)


def test_mean_center_empty_raises() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    with pytest.raises(ValueError):
        ref.mean_center(())


def test_clip_higher_bounds() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    lo = 1.0 - ref.CLIP_EPS_LOW
    hi = 1.0 + ref.CLIP_EPS_HIGH
    assert ref.clip_higher(1.0) == 1.0
    assert ref.clip_higher(0.0) == lo
    assert ref.clip_higher(10.0) == hi
    assert ref.CLIP_EPS_LOW == 0.2
    assert ref.CLIP_EPS_HIGH == 0.28


def test_sequence_ratio_one_token() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    assert ref.sequence_ratio((), ()) == 1.0
    ratio = ref.sequence_ratio((0.0,), (-1.0,))
    assert abs(ratio - float(np.exp(1.0))) < 1e-12
    with pytest.raises(ValueError):
        ref.sequence_ratio((0.0,), (0.0, 1.0))


def test_truncated_is_k0_is_one() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    assert ref.truncated_is(0, 0.5) == 1.0
    assert ref.truncated_is(1, 0.5) == 0.5
    assert ref.truncated_is(1, 2.0) == ref.TIS_CLIP
    with pytest.raises(ValueError):
        ref.truncated_is(-1, 1.0)
    with pytest.raises(ValueError):
        ref.truncated_is(ref.MAX_STALENESS + 1, 1.0)


def test_stale_k_gt_max_dropped() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    assert ref.MAX_STALENESS == 4
    assert ref.drop_stale(0) is False
    assert ref.drop_stale(4) is False
    assert ref.drop_stale(5) is True


def test_dynamic_sampling_drops_all_equal() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    assert ref.dynamic_sampling_keep((1.0, 1.0, 1.0)) is False
    assert ref.dynamic_sampling_keep((1.0, 0.0, 1.0)) is True
    assert ref.dynamic_sampling_keep(()) is False


def test_softmax_rows_sum_to_one() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    x, _y = ref.make_toy_task()
    w = ref.init_weights()
    p = ref.softmax(x @ w)
    assert p.shape == (ref.TOY_N, ref.TOY_ACTIONS)
    assert p.dtype == np.float64
    np.testing.assert_allclose(p.sum(axis=-1), 1.0, atol=1e-12)
    logp = ref.log_softmax(x @ w)
    np.testing.assert_allclose(np.exp(logp), p, atol=1e-12)


def test_routing_replay_forces_ids() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    proto = ref.evaluate_v7_protocol(skip_routing_replay=False)
    assert proto["reward"]["routing_replay_ok"] is True
    assert np.array_equal(proto["reward"]["rollout_ids"], proto["reward"]["trainer_ids"])
    assert proto["reward"]["rollout_ids"].shape == (ref.TOY_N,)
    assert ref.TOY_EXPERTS == 2
    assert ref.TOY_TOP_K == 1


def test_skip_routing_replay_mismatch() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    proto = ref.evaluate_v7_protocol(skip_routing_replay=True)
    assert proto["reward"]["routing_replay_ok"] is False


def test_weight_sync_copies() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    proto = ref.evaluate_v7_protocol(do_update=True, skip_sync=False)
    assert proto["reward"]["weights_synced"] is True
    assert proto["reward"]["engine_matches_trainer"] is True


def test_skip_weight_sync_diverges() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    proto = ref.evaluate_v7_protocol(do_update=True, skip_sync=True)
    assert proto["reward"]["weights_synced"] is False
    assert proto["reward"]["engine_matches_trainer"] is False


def test_parity_threshold_and_injected_drift_constants() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    assert ref.PARITY_THRESHOLD == 0.25
    assert ref.INJECTED_DRIFT > ref.PARITY_THRESHOLD
    assert ref.INJECTED_DRIFT == 1.0


def test_flag_test_file_write_paths() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    assert ref.is_test_file_write(ref.PLANTED_WRITE_PATH) is True
    assert ref.is_test_file_write(ref.GRADER_PREFIX) is True
    assert ref.is_test_file_write(ref.GRADER_PREFIX + "/") is True
    assert ref.is_test_file_write(ref.AGENT_WRITE_PATH) is False
    assert ref.is_test_file_write("/tmp/out.py") is False
    assert ref.flag_planted_write(plant=False, path=ref.PLANTED_WRITE_PATH) is False
    assert ref.flag_planted_write(plant=True, path=ref.PLANTED_WRITE_PATH) is True


def test_gspo_group_size_and_tis_constants() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    assert ref.GROUP_SIZE == 16
    assert ref.TOY_GROUP == 4
    assert ref.TOY_GROUP <= ref.GROUP_SIZE
    assert ref.ROLLOUT_NUMER == 65
    assert ref.ROLLOUT_DENOM == 100
    assert ref.TIS_CLIP == 1.0


# ---------------------------------------------------------------------------
# Gradient checks (finite differences) and shapes/dtypes
# ---------------------------------------------------------------------------


def test_softmax_ce_finite_diff() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    x, y = ref.make_toy_task()
    w = ref.init_weights()
    analytic = ref.softmax_ce_grad(x[0], w, int(y[0]))
    numeric = ref.finite_diff_ce_grad(x[0], w, int(y[0]))
    assert analytic.shape == (ref.TOY_DIM, ref.TOY_ACTIONS)
    assert numeric.shape == analytic.shape
    assert analytic.dtype == np.float64
    assert numeric.dtype == np.float64
    assert ref.grad_match_ok(analytic, numeric) is True
    assert ref.toy_grad_ok() is True


def test_logprob_grad_finite_diff() -> None:
    """d log π(a|x) / dw vs central FD on log-softmax."""
    _assert_v7_result_shape(_cpu_run_v7())
    x, _y = ref.make_toy_task()
    w = ref.init_weights()
    action = 1
    analytic = ref.logprob_grad(x[0], w, action)
    numeric = np.zeros_like(w)
    eps = ref.TOY_FD_EPS
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            plus = w.copy()
            minus = w.copy()
            plus[i, j] += eps
            minus[i, j] -= eps
            lp = float(ref.log_softmax(x[0] @ plus)[action])
            lm = float(ref.log_softmax(x[0] @ minus)[action])
            numeric[i, j] = (lp - lm) / (2.0 * eps)
    assert ref.grad_match_ok(analytic, numeric) is True


def test_toy_shapes_dtypes() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    x, y = ref.make_toy_task()
    w = ref.init_weights()
    assert x.shape == (ref.TOY_N, ref.TOY_DIM)
    assert y.shape == (ref.TOY_N,)
    assert w.shape == (ref.TOY_DIM, ref.TOY_ACTIONS)
    assert x.dtype == np.float64
    assert y.dtype == np.int64
    assert w.dtype == np.float64
    assert set(y.tolist()) == {0, 1}


# ---------------------------------------------------------------------------
# Production hygiene: no tests/ import; does not write v7.json
# ---------------------------------------------------------------------------


def test_production_does_not_import_tests() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    src = Path(__file__).resolve().parents[1] / "prometheus" / "verify" / "v7_rl.py"
    names = _imported_top_level(src)
    assert "tests" not in names


def test_reference_does_not_import_production_jax_torch() -> None:
    _assert_v7_result_shape(_cpu_run_v7())
    src = Path(__file__).resolve().parent / "reference" / "v7_rl.py"
    names = _imported_top_level(src)
    assert "jax" not in names
    assert "torch" not in names
    assert "prometheus" not in names
    assert "rl" not in names
    assert "audit" not in names
    assert "verifiers" not in names


def test_run_v7_does_not_write_v7_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "verify-out"
    monkeypatch.setenv("VERIFY_OUT", str(out))
    monkeypatch.setenv("VERIFY_OUTPUT_DIR", str(out))
    result = v7.run_v7(gpus=1)
    _assert_v7_result_shape(result)
    assert _v7_json_hits(tmp_path) == []
    cwd_hit = Path("v7.json")
    assert not cwd_hit.exists()
    assert not (out / "v7.json").exists()


# ---------------------------------------------------------------------------
# GPU: V7 / 8 H200. Skipped without a device.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_v7_gpu_run_v7_eight_h200_gates() -> None:
    """V7 / 8 H200 (4 SGLang + 4 JAX). Skipped without a device."""
    _require_gpu()
    result = _assert_v7_result_shape(v7.run_v7(gpus=v7.DEFAULT_GPUS))
    for key in V7_KEYS:
        assert result[key] is True, key
    assert ref.meets_v7_gates(result) is True


@pytest.mark.gpu
def test_v7_gpu_does_not_write_v7_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """V7 GPU analog still leaves v7.json to the ncshare template."""
    _require_gpu()
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "verify-out"
    monkeypatch.setenv("VERIFY_OUT", str(out))
    result = _assert_v7_result_shape(v7.run_v7(gpus=v7.DEFAULT_GPUS))
    for key in V7_KEYS:
        assert result[key] is True, key
    assert _v7_json_hits(tmp_path) == []
