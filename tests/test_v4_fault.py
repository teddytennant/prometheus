"""Oracle tests for ``prometheus.verify.v4_fault.run_v4`` (spec 16.2 V4).

Every collected CPU test calls the public interface so the suite fails against
the ``NotImplementedError`` stub. GPU tests are marked ``gpu`` (V4 / 4–8 H200)
and skip cleanly without a device.

Production is imported only here. ``tests.reference.v4_fault`` is plain NumPy
and must not import ``model/``, ``train/``, ``kernels/``,
``prometheus.verify.v4_fault``, JAX, torch, or the Rust crates.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from prometheus.verify import v4_fault as v4
from tests.reference import v4_fault as ref

_CPU_RESULT: v4.V4Result | None = None
_CPU_ERROR: BaseException | None = None

V4_KEYS = ("resumed_bitwise_equal", "sdc_caught_flip", "spike_rollback_skipped_shard")


def _cpu_run_v4() -> v4.V4Result:
    """One ``run_v4(gpus=1)`` per process (CPU analog of V4); re-raise cached errors."""
    global _CPU_RESULT, _CPU_ERROR
    if _CPU_RESULT is not None:
        return _CPU_RESULT
    if _CPU_ERROR is not None:
        raise _CPU_ERROR
    try:
        out = v4.run_v4(gpus=1)
    except BaseException as exc:
        _CPU_ERROR = exc
        raise
    _CPU_RESULT = out
    return out


def _require_gpu():
    """Skip unless JAX sees a GPU. Marker ``gpu`` is the V4 stage tag."""
    jax = pytest.importorskip("jax")
    gpus = [d for d in jax.devices() if getattr(d, "platform", None) == "gpu"]
    if not gpus:
        pytest.skip("V4 GPU test requires a GPU device")
    return jax


def _assert_v4_result_shape(result: object) -> v4.V4Result:
    assert isinstance(result, dict), "run_v4 must return a V4Result mapping"
    assert set(result.keys()) == set(V4_KEYS), f"V4Result keys must be exactly {V4_KEYS}"
    for key in V4_KEYS:
        assert type(result[key]) is bool, f"{key} must be a Python bool"
    return result  # type: ignore[return-value]


def _assert_v4error(gpus: int) -> None:
    with pytest.raises(v4.V4Error) as ei:
        v4.run_v4(gpus=gpus)
    assert type(ei.value) is v4.V4Error
    assert not isinstance(ei.value, NotImplementedError)


def _imported_names(path: str) -> set[str]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
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
# Faults: gpus < 1 -> V4Error (not a generic Exception-pass against the stub)
# ---------------------------------------------------------------------------


def test_gpus_zero_raises_v4error() -> None:
    _assert_v4error(gpus=0)


def test_gpus_negative_raises_v4error() -> None:
    _assert_v4error(gpus=-1)
    _assert_v4error(gpus=-4)
    _assert_v4error(gpus=-8)


# ---------------------------------------------------------------------------
# Public CPU analog (gpus=1): exact three bools, all True, JSON-serializable
# ---------------------------------------------------------------------------


def test_run_v4_gpus_1_keys_exactly_three_bools_all_true() -> None:
    result = _assert_v4_result_shape(_cpu_run_v4())
    assert list(result.keys()) == list(V4_KEYS) or set(result.keys()) == set(V4_KEYS)
    for key in V4_KEYS:
        assert result[key] is True, key
    assert ref.meets_v4_gates(result) is True


def test_run_v4_result_json_serializable() -> None:
    result = _assert_v4_result_shape(_cpu_run_v4())
    payload = json.dumps(dict(result))
    loaded = json.loads(payload)
    assert set(loaded.keys()) == set(V4_KEYS)
    for key in V4_KEYS:
        assert loaded[key] is True
        assert type(loaded[key]) is bool


def test_default_gpus_min_max_constants() -> None:
    _assert_v4_result_shape(_cpu_run_v4())
    assert v4.DEFAULT_GPUS_MIN == 4
    assert v4.DEFAULT_GPUS_MAX == 8
    assert v4.DEFAULT_GPUS_MIN == ref.DEFAULT_GPUS_MIN
    assert v4.DEFAULT_GPUS_MAX == ref.DEFAULT_GPUS_MAX
    assert ref.MEMORY_REPLICAS == 2
    assert ref.DEFAULT_GPUS_MIN <= ref.DEFAULT_GPUS_MAX


# ---------------------------------------------------------------------------
# Independent reference protocol
# ---------------------------------------------------------------------------


def test_reference_protocol_bitwise_sdc_spike_gates() -> None:
    """Happy path: restore matches twin; SDC catches a flip; spike skips shard."""
    result = _assert_v4_result_shape(_cpu_run_v4())
    proto = ref.evaluate_v4_protocol()
    assert proto["resumed_bitwise_equal"] is True
    assert proto["sdc_caught_flip"] is True
    assert proto["spike_rollback_skipped_shard"] is True
    assert proto["meets_gates"] is True
    assert proto["grad_ok"] is True

    resume = proto["resume"]
    assert resume["did_restore"] is True
    assert resume["raw_equal"] is True
    assert resume["weights_equal"] is True
    assert resume["opt_equal"] is True
    assert resume["rng_equal"] is True
    assert resume["replicas_filled"] is True
    assert resume["memory_replicas"] == 2
    assert resume["n_live_after_kill"] == ref.TOY_N_REPLICAS - 1
    assert resume["accum_after"] >= resume["accum_before"]
    assert resume["accum_after"] * resume["n_live_after_kill"] * ref.MICROBATCH_TOKENS >= (
        ref.TOKENS_PER_STEP
    )

    sdc = proto["sdc"]
    assert sdc["flip_injected"] is True
    assert sdc["mismatch_found"] is True
    assert sdc["within_n"] is True
    assert sdc["caught_at_step"] is not None
    assert int(sdc["caught_at_step"]) <= ref.SDC_CATCH_WITHIN_N
    assert sdc["clean_w0_hash"] != sdc["flipped_w0_hash"]

    spike = proto["spike"]
    assert spike["spike_handled"] is True
    assert ref.BAD_SHARD in spike["skipped_shards"]
    assert spike["bad_shard_consumed"] is False

    assert result["resumed_bitwise_equal"] is True
    assert result["sdc_caught_flip"] is True
    assert result["spike_rollback_skipped_shard"] is True
    assert ref.meets_v4_gates(result) is True


def test_deliberate_mismatches_fail_matching_gate() -> None:
    """Deliberate mismatches fail the matching gate (independent path)."""
    result = _assert_v4_result_shape(_cpu_run_v4())
    happy = ref.evaluate_v4_protocol()
    assert happy["meets_gates"] is True

    no_restore = ref.evaluate_v4_protocol(do_restore=False)
    assert no_restore["resumed_bitwise_equal"] is False
    assert no_restore["resume"]["did_restore"] is False
    assert not ref.meets_v4_gates(no_restore)

    no_flip = ref.evaluate_v4_protocol(inject_flip=False)
    assert no_flip["sdc_caught_flip"] is False
    assert no_flip["sdc"]["flip_injected"] is False
    assert not ref.meets_v4_gates(no_flip)

    no_spike = ref.evaluate_v4_protocol(inject_bad_shard=False)
    assert no_spike["spike_rollback_skipped_shard"] is False
    assert no_spike["spike"]["spike_handled"] is False
    assert not ref.meets_v4_gates(no_spike)

    # Production analog must sit on the matching side; the independent knobs move.
    assert result["resumed_bitwise_equal"] is not no_restore["resumed_bitwise_equal"]
    assert result["sdc_caught_flip"] is not no_flip["sdc_caught_flip"]
    assert result["spike_rollback_skipped_shard"] is not no_spike["spike_rollback_skipped_shard"]


# ---------------------------------------------------------------------------
# Fault injection on the independent path
# ---------------------------------------------------------------------------


def test_flip_not_reported_sdc_caught_flip_false() -> None:
    _assert_v4_result_shape(_cpu_run_v4())
    proto = ref.evaluate_v4_protocol(report_flipped_hash=False)
    assert proto["sdc"]["flip_injected"] is True
    assert proto["sdc"]["report_flipped_hash"] is False
    assert proto["sdc"]["mismatch_found"] is False
    assert proto["sdc_caught_flip"] is False
    assert not ref.meets_v4_gates(proto)


def test_restore_different_bytes_resumed_bitwise_equal_false() -> None:
    _assert_v4_result_shape(_cpu_run_v4())
    proto = ref.evaluate_v4_protocol(restore_corrupt=True)
    assert proto["resume"]["did_restore"] is True
    assert proto["resume"]["restore_corrupt"] is True
    assert proto["resume"]["raw_equal"] is False
    assert proto["resume"]["weights_equal"] is False
    assert proto["resumed_bitwise_equal"] is False
    assert not ref.meets_v4_gates(proto)


def test_rollback_that_does_not_skip_shard_spike_false() -> None:
    _assert_v4_result_shape(_cpu_run_v4())
    proto = ref.evaluate_v4_protocol(skip_bad_shard=False)
    assert proto["spike"]["inject_bad_shard"] is True
    assert proto["spike"]["skip_bad_shard"] is False
    assert proto["spike"]["spike_handled"] is True
    assert ref.BAD_SHARD not in proto["spike"]["skipped_shards"]
    assert proto["spike"]["bad_shard_consumed"] is True
    assert proto["spike_rollback_skipped_shard"] is False
    assert not ref.meets_v4_gates(proto)


# ---------------------------------------------------------------------------
# Shapes, dtypes, elastic DP, host-RAM replicas, properties, gradients
# ---------------------------------------------------------------------------


def test_reference_shapes_dtypes_elastic_dp_host_ram() -> None:
    _assert_v4_result_shape(_cpu_run_v4())
    proto = ref.evaluate_v4_protocol()
    resume = proto["resume"]
    assert resume["weights_dtype"] == "float32"
    assert resume["momentum_dtype"] == "float32"
    assert resume["weights_shape"] == (ref.TOY_IN, ref.TOY_OUT)
    assert resume["momentum_shape"] == (ref.TOY_IN, ref.TOY_OUT)
    assert resume["run_step"] == ref.TOY_N_STEPS
    assert resume["twin_step"] == ref.TOY_N_STEPS

    state = ref.toy_init()
    assert state.weights.dtype == np.float32
    assert state.momentum.dtype == np.float32
    assert state.weights.shape == (ref.TOY_IN, ref.TOY_OUT)
    assert state.momentum.shape == (ref.TOY_IN, ref.TOY_OUT)
    assert ref.states_bitwise_equal(state, state.clone()) is True

    ckpt = ref.HostRamCheckpointer()
    ckpt.save_in_memory(state)
    assert len(ckpt.replicas) == ref.MEMORY_REPLICAS == 2
    restored = ckpt.restore_latest_memory()
    assert ref.states_bitwise_equal(state, restored) is True
    assert ckpt.replicas[0] is not state
    assert ckpt.replicas[1] is not state
    assert ckpt.replicas[0] is not ckpt.replicas[1]

    n4 = ref.min_grad_accumulation(4, ref.MICROBATCH_TOKENS, ref.TOKENS_PER_STEP)
    n3 = ref.min_grad_accumulation(3, ref.MICROBATCH_TOKENS, ref.TOKENS_PER_STEP)
    assert n4 >= 1
    assert n3 >= n4
    assert 4 * ref.MICROBATCH_TOKENS * n4 >= ref.TOKENS_PER_STEP
    assert 3 * ref.MICROBATCH_TOKENS * n3 >= ref.TOKENS_PER_STEP
    assert ref.min_grad_accumulation(0, 2, 16) == 0
    assert ref.min_grad_accumulation(4, 2, 0) == 0


def test_reference_properties_bit_flip_hash_and_equality() -> None:
    _assert_v4_result_shape(_cpu_run_v4())
    state = ref.toy_init()
    other = state.clone()
    assert ref.states_bitwise_equal(state, other) is True
    other.weights = other.weights.copy()
    other.weights.reshape(-1)[0] = np.float32(other.weights.reshape(-1)[0] + 1.0)
    assert ref.states_bitwise_equal(state, other) is False

    raw = b"\x00\x01\x02\x03"
    flipped = ref.flip_bit(raw, bit_index=0)
    assert flipped != raw
    assert len(flipped) == len(raw)
    assert ref.shard_hash_hex(flipped) != ref.shard_hash_hex(raw)
    assert len(ref.shard_hash_hex(raw)) == 64

    reports_ok = [("r0", "aa"), ("r1", "aa"), ("r2", "aa"), ("r3", "aa")]
    reports_flip = [("r0", "bb"), ("r1", "aa"), ("r2", "aa"), ("r3", "aa")]
    assert ref.check_sdc_hashes(reports_ok) is False
    assert ref.check_sdc_hashes(reports_flip) is True
    assert ref.check_sdc_hashes([("r0", "aa")]) is False

    empty = np.zeros((0, 0), dtype=np.float64)
    assert ref.mse_mean(empty, empty) == 0.0
    with pytest.raises(ValueError, match="shape mismatch"):
        ref.mse_mean(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="empty"):
        ref.flip_bit(b"")


def test_toy_gradient_finite_differences() -> None:
    _assert_v4_result_shape(_cpu_run_v4())
    assert ref.toy_grad_ok() is True
    rng = ref.make_rng(ref.TOY_SEED + 11)
    x = rng.standard_normal((ref.TOY_BATCH, ref.TOY_IN)).astype(np.float32)
    y = rng.standard_normal((ref.TOY_BATCH, ref.TOY_OUT)).astype(np.float32)
    scale = np.float32(0.02)
    w = (rng.standard_normal((ref.TOY_IN, ref.TOY_OUT)).astype(np.float32) * scale).astype(
        np.float32
    )
    analytic, numeric = ref.toy_finite_diff_w_row0(x, y, w)
    assert analytic.shape == numeric.shape == (ref.TOY_OUT,)
    assert analytic.dtype == np.float32
    assert numeric.dtype == np.float32
    assert ref.grad_match_ok(analytic, numeric) is True

    # Central FD on a single scalar MSE vs analytic ``dL/dw[0,0]``.
    eps = 1e-3
    w64 = w.astype(np.float64)
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    plus = w64.copy()
    minus = w64.copy()
    plus[0, 0] += eps
    minus[0, 0] -= eps
    fd = (ref.mse_mean(x64 @ plus, y64) - ref.mse_mean(x64 @ minus, y64)) / (2.0 * eps)
    g = ref.linear_mse_grad(x64, y64, w64)
    assert g.shape == w.shape
    assert g.dtype == np.float64
    assert g[0, 0] == pytest.approx(fd, rel=5e-2, abs=5e-3)


def test_reference_does_not_import_production() -> None:
    _assert_v4_result_shape(_cpu_run_v4())
    src = ref.__file__
    assert src is not None
    imported = _imported_names(src)
    for name in (
        "jax",
        "torch",
        "model",
        "train",
        "kernels",
        "prometheus",
        "prometheus.verify",
        "prometheus.verify.v4_fault",
    ):
        assert name not in imported, name


def test_production_source_does_not_import_tests() -> None:
    _assert_v4_result_shape(_cpu_run_v4())
    src = v4.__file__
    assert src is not None
    imported = _imported_names(src)
    for name in ("tests", "tests.reference", "tests.reference.v4_fault"):
        assert name not in imported, name


# ---------------------------------------------------------------------------
# V4 GPU scale (spec 16.2: 4 to 8). Skipped on CPU.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_v4_gpu_run_v4_gates() -> None:
    """V4: low end of spec (4 GPUs). Still analog; not V4 verified."""
    _require_gpu()
    result = _assert_v4_result_shape(v4.run_v4(gpus=4))
    assert result["resumed_bitwise_equal"] is True
    assert result["sdc_caught_flip"] is True
    assert result["spike_rollback_skipped_shard"] is True
    assert ref.meets_v4_gates(result) is True


@pytest.mark.gpu
def test_v4_gpu_gpus_less_than_one_still_v4error() -> None:
    _require_gpu()
    with pytest.raises(v4.V4Error) as ei:
        v4.run_v4(gpus=0)
    assert type(ei.value) is v4.V4Error
    assert not isinstance(ei.value, NotImplementedError)
    result = _assert_v4_result_shape(v4.run_v4(gpus=4))
    assert result["resumed_bitwise_equal"] is True
    assert result["sdc_caught_flip"] is True
    assert result["spike_rollback_skipped_shard"] is True
