"""Oracle tests for ``prometheus.verify.v8_serving.run_v8`` (spec 16.2 V8).

Every collected CPU test calls the public interface so the suite fails against
the ``NotImplementedError`` stub. GPU tests are marked ``gpu`` (V8 / 1–4 H200)
and skip cleanly without a device.

The CPU analog is a tiny checkpoint (host stand-in for SGLang). Spec V8 is
1 to 4 H200; passing these tests is not V8 verified.

Production is imported only here. ``tests.reference.v8_serving`` is plain NumPy
and must not import ``model/``, ``train/``, ``kernels/``, JAX, torch, or the
Rust crates.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prometheus.verify import v8_serving as v8
from tests.reference import v8_serving as ref

_CPU_RESULT: v8.V8Result | None = None
_CPU_ERROR: BaseException | None = None

V8_KEYS = ("logprob_within_threshold", "tiered_restore_matches")


def _cpu_run_v8() -> v8.V8Result:
    """One ``run_v8(gpus=1)`` per process (CPU analog of V8); re-raise cached errors."""
    global _CPU_RESULT, _CPU_ERROR
    if _CPU_RESULT is not None:
        return _CPU_RESULT
    if _CPU_ERROR is not None:
        raise _CPU_ERROR
    try:
        out = v8.run_v8(gpus=1)
    except BaseException as exc:
        _CPU_ERROR = exc
        raise
    _CPU_RESULT = out
    return out


def _assert_v8_result_shape(result: Any) -> v8.V8Result:
    assert isinstance(result, dict), type(result)
    assert set(result) == set(V8_KEYS), set(result)
    logprob_ok = result["logprob_within_threshold"]
    restore_ok = result["tiered_restore_matches"]
    assert type(logprob_ok) is bool, type(logprob_ok)
    assert type(restore_ok) is bool, type(restore_ok)
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
    """Skip unless JAX sees a GPU. Marker ``gpu`` is the V8 stage tag."""
    jax = pytest.importorskip("jax")
    try:
        gpus = [d for d in jax.devices() if getattr(d, "platform", None) == "gpu"]
    except RuntimeError:
        gpus = []
    if not gpus:
        pytest.skip("no GPU")
    return jax


def _assert_v8error(gpus: int) -> None:
    with pytest.raises(v8.V8Error) as ei:
        v8.run_v8(gpus=gpus)
    assert type(ei.value) is v8.V8Error
    assert not isinstance(ei.value, NotImplementedError)


# ---------------------------------------------------------------------------
# Faults: gpus < 1 -> V8Error (not NotImplementedError)
# ---------------------------------------------------------------------------


def test_gpus_zero_raises_v8error() -> None:
    _assert_v8error(gpus=0)


def test_gpus_negative_raises_v8error() -> None:
    _assert_v8error(gpus=-1)
    _assert_v8error(gpus=-8)


# ---------------------------------------------------------------------------
# Result shape / types; both bools true on a successful analog
# ---------------------------------------------------------------------------


def test_run_v8_result_shape_and_types() -> None:
    result = _assert_v8_result_shape(_cpu_run_v8())
    assert result["logprob_within_threshold"] is True
    assert result["tiered_restore_matches"] is True


def test_successful_analog_both_bools_true() -> None:
    result = _assert_v8_result_shape(_cpu_run_v8())
    assert result["logprob_within_threshold"] is True
    assert result["tiered_restore_matches"] is True
    assert ref.meets_v8_gates(result) is True
    proto = ref.evaluate_v8_protocol()
    assert proto["meets_gates"] is True
    assert proto["logprob_within_threshold"] is True
    assert proto["tiered_restore_matches"] is True


def test_default_gpus_is_four() -> None:
    result = _assert_v8_result_shape(_cpu_run_v8())
    assert result["logprob_within_threshold"] is True
    assert v8.DEFAULT_GPUS == 4
    assert ref.DEFAULT_GPUS == 4
    assert v8.DEFAULT_GPUS == ref.DEFAULT_GPUS
    assert ref.DEFAULT_GPUS_MIN == 1
    assert ref.DEFAULT_GPUS_MAX == 4
    assert ref.LOGPROB_THRESHOLD == 1e-5


# ---------------------------------------------------------------------------
# Log-probs actually compared (not self-compare)
# ---------------------------------------------------------------------------


def test_logprobs_actually_compared_not_self_compare() -> None:
    result = _assert_v8_result_shape(_cpu_run_v8())
    proto = ref.evaluate_v8_protocol()
    logprob = proto["logprob"]
    assert logprob["distinct_backends"] is True
    assert logprob["self_compared"] is False
    assert logprob["same_object"] is False
    assert logprob["backend_a"] == "jax"
    assert logprob["backend_b"] == "sglang"
    assert logprob["jax_pack"] is not logprob["sgl_pack"]
    # Same-backend compare would pass numerically; that is not the gate.
    jax_pack = logprob["jax_pack"]
    self_diff = float(np.max(np.abs(jax_pack - jax_pack)))
    assert self_diff == ref.GOLDEN_MATCH_DIFF
    assert ref.meets_logprob_gate(self_diff) is True
    assert logprob["diff"] <= ref.LOGPROB_THRESHOLD
    assert logprob["logprob_within_threshold"] is True
    # Perturbing the SGLang path must fail the 1e-5 gate.
    shifted = ref.evaluate_v8_protocol(shift_sglang=ref.GOLDEN_SHIFT_ABOVE_GATE)
    assert shifted["logprob"]["distinct_backends"] is True
    assert shifted["logprob"]["diff"] > ref.LOGPROB_THRESHOLD
    assert shifted["logprob_within_threshold"] is False
    # Self-compare of JAX vs JAX is forbidden even when the numeric gap is 0.
    selfed = ref.evaluate_v8_protocol(self_compare=True)
    assert selfed["logprob"]["self_compared"] is True
    assert selfed["logprob"]["distinct_backends"] is False
    assert selfed["logprob_within_threshold"] is False
    assert result["logprob_within_threshold"] is True


def test_logprob_shapes_dtypes_and_properties() -> None:
    result = _assert_v8_result_shape(_cpu_run_v8())
    proto = ref.evaluate_v8_protocol()
    logprob = proto["logprob"]
    assert logprob["jax_logits_shape"] == (ref.TOY_BATCH, ref.TOY_SEQ, ref.TOY_VOCAB)
    assert logprob["sgl_logits_shape"] == (ref.TOY_BATCH, ref.TOY_SEQ, ref.TOY_VOCAB)
    assert logprob["jax_logits_dtype"] == "float32"
    assert logprob["sgl_logits_dtype"] == "float32"
    assert logprob["tokens_shape"] == (ref.TOY_BATCH, ref.TOY_SEQ)
    assert logprob["tokens_dtype"] == "int32"
    assert logprob["threshold"] == 1e-5
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((2, 3), dtype=np.float32)
    assert ref.logprob_max_abs_diff(a, b) == 0.0
    assert ref.logprob_max_abs_diff(np.zeros((0, 3)), np.zeros((0, 3))) == 0.0
    with pytest.raises(ValueError, match="shape mismatch"):
        ref.logprob_max_abs_diff(np.zeros((2, 3)), np.zeros((2, 4)))
    nan_diff = ref.logprob_max_abs_diff(
        np.array([[np.nan, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0]], dtype=np.float32),
    )
    assert not np.isfinite(nan_diff)
    assert ref.meets_logprob_gate(nan_diff) is False
    assert result["logprob_within_threshold"] is True


def test_fault_shifted_logprobs_fail_threshold() -> None:
    result = _assert_v8_result_shape(_cpu_run_v8())
    shifted = ref.run_logprob_experiment(shift_sglang=ref.GOLDEN_SHIFT_ABOVE_GATE)
    assert shifted["distinct_backends"] is True
    assert shifted["diff"] > ref.LOGPROB_THRESHOLD
    assert shifted["logprob_within_threshold"] is False
    happy = ref.run_logprob_experiment()
    assert happy["logprob_within_threshold"] is True
    assert result["logprob_within_threshold"] is True


# ---------------------------------------------------------------------------
# Tiered restore actually matches an unswapped twin
# ---------------------------------------------------------------------------


def test_tiered_restore_matches_unswapped_twin() -> None:
    result = _assert_v8_result_shape(_cpu_run_v8())
    proto = ref.evaluate_v8_protocol()
    restore = proto["restore"]
    assert restore["did_swap"] is True
    assert restore["went_through_grace"] is True
    assert restore["went_through_nvme"] is True
    assert restore["did_restore"] is True
    assert restore["restore_corrupt"] is False
    assert restore["compared_to_twin"] is True
    assert restore["compared_to_self"] is False
    assert restore["raw_equal"] is True
    assert restore["tiered_restore_matches"] is True
    assert restore["kv_tiers"] == ref.KV_TIER_ORDER
    assert restore["final_tier"] == ref.KV_TIER_HBM
    assert restore["prefix_len"] == ref.TOY_PREFIX
    assert restore["suffix_len"] == ref.TOY_SEQ - ref.TOY_PREFIX
    # Comparing the swapped session to itself is not the gate.
    selfed = ref.evaluate_v8_protocol(compare_restore_to_self=True)
    assert selfed["restore"]["compared_to_self"] is True
    assert selfed["restore"]["compared_to_twin"] is False
    assert selfed["tiered_restore_matches"] is False
    assert result["tiered_restore_matches"] is True


def test_fault_skip_restore_tiered_false() -> None:
    result = _assert_v8_result_shape(_cpu_run_v8())
    skipped = ref.run_tiered_restore_experiment(do_restore=False)
    assert skipped["did_restore"] is False
    assert skipped["raw_equal"] is False
    assert skipped["tiered_restore_matches"] is False
    no_swap = ref.run_tiered_restore_experiment(skip_swap=True)
    assert no_swap["did_swap"] is False
    assert no_swap["tiered_restore_matches"] is False
    no_nvme = ref.run_tiered_restore_experiment(skip_nvme=True)
    assert no_nvme["went_through_nvme"] is False
    assert no_nvme["tiered_restore_matches"] is False
    assert result["tiered_restore_matches"] is True


def test_fault_corrupt_restore_tiered_false() -> None:
    result = _assert_v8_result_shape(_cpu_run_v8())
    corrupt = ref.run_tiered_restore_experiment(restore_corrupt=True)
    assert corrupt["did_restore"] is True
    assert corrupt["went_through_nvme"] is True
    assert corrupt["restore_corrupt"] is True
    assert corrupt["tiered_restore_matches"] is False
    assert result["tiered_restore_matches"] is True


# ---------------------------------------------------------------------------
# Latent decode, recurrence buckets, MTP; gradient check
# ---------------------------------------------------------------------------


def test_latent_recurrence_mtp_and_gradients() -> None:
    result = _assert_v8_result_shape(_cpu_run_v8())
    proto = ref.evaluate_v8_protocol()
    logprob = proto["logprob"]
    assert logprob["used_latent_decode"] is True
    assert logprob["used_mtp"] is True
    assert logprob["mtp_heads"] == ref.MTP_HEADS == 2
    assert logprob["r_requested"] == ref.TOY_R == 3
    assert logprob["r_bucketed"] == 4
    assert ref.bucket_r(1) == 1
    assert ref.bucket_r(2) == 2
    assert ref.bucket_r(3) == 4
    assert ref.bucket_r(4) == 4
    assert ref.bucket_r(5) == 8
    assert ref.bucket_r(16) == 16
    assert ref.bucket_r(3, budget=2) == 2
    with pytest.raises(ValueError):
        ref.bucket_r(0)
    assert ref.next_kv_tier(ref.KV_TIER_HBM) == ref.KV_TIER_GRACE
    assert ref.next_kv_tier(ref.KV_TIER_GRACE) == ref.KV_TIER_NVME
    assert ref.next_kv_tier(ref.KV_TIER_NVME) is None
    assert proto["grad_ok"] is True
    assert ref.toy_grad_ok() is True
    assert result["logprob_within_threshold"] is True
    assert result["tiered_restore_matches"] is True


# ---------------------------------------------------------------------------
# Production must not import tests/; does not write v8.json
# ---------------------------------------------------------------------------


def test_production_source_does_not_import_tests() -> None:
    result = _assert_v8_result_shape(_cpu_run_v8())
    src = v8.__file__
    assert src is not None
    imported = _imported_names(src)
    for name in ("tests", "tests.reference", "tests.reference.v8_serving"):
        assert name not in imported, name
    assert result["logprob_within_threshold"] is True


def test_reference_does_not_import_production_jax_torch() -> None:
    result = _assert_v8_result_shape(_cpu_run_v8())
    src = ref.__file__
    assert src is not None
    text = Path(src).read_text(encoding="utf-8")
    # Substring scan: the production dotted path must not appear at all.
    assert "prometheus.verify.v8_serving" not in text
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
    assert result["logprob_within_threshold"] is True


def test_run_v8_does_not_write_v8_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VERIFY_OUT", str(tmp_path / "verify-out"))
    monkeypatch.setenv("VERIFY_OUTPUT_DIR", str(tmp_path / "verify-out"))
    monkeypatch.delenv("PROMETHEUS_ROOT", raising=False)
    result = _assert_v8_result_shape(v8.run_v8(gpus=1))
    assert result["logprob_within_threshold"] is True
    assert result["tiered_restore_matches"] is True
    assert list(tmp_path.rglob("v8.json")) == []
    assert not (tmp_path / "v8.json").exists()


# ---------------------------------------------------------------------------
# V8 GPU scale (spec 16.2: 1 to 4 H200). Skipped on CPU.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_v8_gpu_run_v8_gates() -> None:
    """V8: high end of spec (4 H200). Still analog; not V8 verified."""
    _require_gpu()
    result = _assert_v8_result_shape(v8.run_v8(gpus=4))
    assert result["logprob_within_threshold"] is True
    assert result["tiered_restore_matches"] is True
    assert ref.meets_v8_gates(result) is True


@pytest.mark.gpu
def test_v8_gpu_gpus_less_than_one_still_v8error() -> None:
    _require_gpu()
    with pytest.raises(v8.V8Error) as ei:
        v8.run_v8(gpus=0)
    assert type(ei.value) is v8.V8Error
    assert not isinstance(ei.value, NotImplementedError)
    result = _assert_v8_result_shape(v8.run_v8(gpus=4))
    assert result["logprob_within_threshold"] is True
    assert result["tiered_restore_matches"] is True
