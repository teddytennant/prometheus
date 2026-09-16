"""Implementer-facing tests for the ``evals`` public API (spec 11, F3).

These import production ``evals``. Constants, enums, and dataclasses are real
and membership tests may pass against the stub. Every test that calls
``require_suite``, ``result_envelope``, ``DecontamIndex.add``,
``DecontamIndex.is_clean``, ``DecontamIndex.method_hash``, ``Runner.__init__``,
``Runner.load_items``, or ``Runner.run`` must FAIL on the stub.

Groups:
- Constants: PUBLIC_SUITES, FORECASTING_METRICS, EFFICIENCY_METRICS, schema id.
- require_suite: known slug returned; unknown slug → UnknownSuiteError (KeyError).
- result_envelope: F1-valid eval_result dict; blank harness_version errors.
- Runner: harness_version required; unknown suite; run envelope + per_item_hash.
- DecontamIndex: fail-closed empty index; n-gram near-copy dirty; unrelated clean.
- Probes: rephrased questions and canaries are suite items, addable to the index.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pytest

import contracts
import evals

# Lowercase hex SHA-256 used when the test supplies hashes (golden eval_result).
_SHA256 = "2c9a93d84ad4df28371eab9ca3e915b257ae530b5d77e66a548628892464c228"
_RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)

# Distinctive 16-token span: long enough for 8-gram and 13-gram overlap.
_CORE = (
    "xenon hexafluoride hydrolysis under anhydrous hydrogen fluoride "
    "with palladium catalyst distinctive prometheus eval phrase"
)
_PROMPT = f"Question: {_CORE} What is the product formed?"
_NEAR_COPY = f"Rephrased probe: a student asked about {_CORE} please explain."
_UNRELATED = (
    "The 1998 Chicago Bulls starting lineup and free-throw percentages "
    "have nothing in common with noble-gas chemistry."
)
_CANARY = (
    "PROMETHEUS_EVAL_CANARY_TOKEN_v1 this exact sentence is a contamination "
    "probe and must never appear in pretraining shards or fine-tune mixtures"
)


def _decontam(**overrides: Any) -> evals.Decontamination:
    kwargs: dict[str, Any] = {
        "status": evals.DecontamStatus.CLEAN,
        "against": ("gpqa",),
        "method_hash": _SHA256,
    }
    kwargs.update(overrides)
    return evals.Decontamination(**kwargs)


def _envelope_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "result_id": "eval-gpqa-r0",
        "suite": "gpqa",
        "split": "test",
        "checkpoint_id": "r0-step-1",
        "metrics": {"pass_at_1": 0.0},
        "n_items": 0,
        "harness_version": "oracle-test",
        "decontamination": _decontam(),
        "created_at": "2026-09-16T12:00:00Z",
    }
    for key, value in overrides.items():
        if value is _OMIT:
            kwargs.pop(key, None)
        else:
            kwargs[key] = value
    return kwargs


class _Omit:
    pass


_OMIT = _Omit()


def _accept_eval_result(payload: Any) -> None:
    """``result_envelope`` must build a dict F1 ``contracts.validate`` accepts."""
    assert isinstance(payload, dict)
    assert payload["schema_id"] == "prometheus.eval_result"
    assert payload["schema_version"] == 1
    # F1 selects the schema from payload["schema_id"] / schema_version.
    contracts.validate(payload)


def _eval_item(
    *,
    item_id: str = "item-1",
    suite: str = "gpqa",
    split: str = "test",
    prompt: str = _PROMPT,
    answer: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> evals.EvalItem:
    return evals.EvalItem(
        item_id=item_id,
        suite=suite,
        split=split,
        prompt=prompt,
        answer=answer,
        metadata=dict(metadata or {}),
    )


def _filled_index(*items: evals.EvalItem) -> evals.DecontamIndex:
    index = evals.DecontamIndex()
    for item in items:
        index.add(item)
    return index


def _assert_rfc3339(value: str) -> None:
    assert _RFC3339.match(value), f"not RFC3339: {value!r}"
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    datetime.fromisoformat(text)


# ---------------------------------------------------------------------------
# Constants / enums (real on the stub; these may pass)
# ---------------------------------------------------------------------------


def test_public_suites_is_exactly_the_spec_11_slugs() -> None:
    assert evals.PUBLIC_SUITES == (
        "re_bench",
        "mle_bench",
        "paperbench",
        "speedrun",
        "metr_time_horizon",
        "swe_bench_verified",
        "swe_bench_pro",
        "terminal_bench",
        "competitive_programming",
        "aime",
        "hmmt",
        "frontiermath",
        "minif2f",
        "putnambench",
        "arc_agi_1",
        "arc_agi_2",
        "arc_agi_3",
        "hle",
        "gpqa",
        "long_context_128k",
        "long_context_1m",
        "forecasting",
    )
    assert len(evals.PUBLIC_SUITES) == len(set(evals.PUBLIC_SUITES))


def test_forecasting_metrics_exact() -> None:
    assert evals.FORECASTING_METRICS == ("brier", "log_score", "paper_pnl", "leak_probe")


def test_efficiency_metrics_exact() -> None:
    assert evals.EFFICIENCY_METRICS == (
        "tokens_to_solve",
        "latent_steps",
        "recurrence_iterations",
    )


def test_schema_constants() -> None:
    assert evals.SCHEMA_ID == "prometheus.eval_result"
    assert evals.SCHEMA_VERSION == 1


def test_split_enum_values() -> None:
    assert set(evals.Split) == {
        evals.Split.TRAIN,
        evals.Split.VAL,
        evals.Split.TEST,
        evals.Split.HELD_OUT,
    }
    assert {s.value for s in evals.Split} == {"train", "val", "test", "held_out"}


def test_decontam_status_enum_values() -> None:
    assert {s.value for s in evals.DecontamStatus} == {
        "pending",
        "clean",
        "flagged",
        "not_applicable",
    }


def test_unknown_suite_error_is_keyerror() -> None:
    assert issubclass(evals.UnknownSuiteError, KeyError)


def test_harness_version_error_is_valueerror() -> None:
    assert issubclass(evals.HarnessVersionError, ValueError)


def test_decontam_error_is_valueerror() -> None:
    assert issubclass(evals.DecontamError, ValueError)


# ---------------------------------------------------------------------------
# require_suite
# ---------------------------------------------------------------------------


def test_require_suite_known_returns_slug() -> None:
    for slug in evals.PUBLIC_SUITES:
        assert evals.require_suite(slug) == slug


def test_require_suite_unknown_raises_unknown_suite_error() -> None:
    for slug in ("gsm8k", "GPQA", "not_a_suite", "swe-bench-verified", ""):
        with pytest.raises(evals.UnknownSuiteError):
            evals.require_suite(slug)


# ---------------------------------------------------------------------------
# result_envelope
# ---------------------------------------------------------------------------


def test_result_envelope_validates_against_f1() -> None:
    payload = evals.result_envelope(**_envelope_kwargs())
    _accept_eval_result(payload)
    assert payload["result_id"] == "eval-gpqa-r0"
    assert payload["suite"] == "gpqa"
    assert payload["split"] == "test"
    assert payload["checkpoint_id"] == "r0-step-1"
    assert payload["n_items"] == 0
    assert isinstance(payload["n_items"], int) and not isinstance(payload["n_items"], bool)
    assert payload["metrics"]["pass_at_1"] == 0.0
    assert payload["decontamination"]["status"] in {s.value for s in evals.DecontamStatus}
    _assert_rfc3339(payload["created_at"])
    assert "harness_version" not in payload


def test_result_envelope_all_legal_splits() -> None:
    for split in ("train", "val", "test", "held_out"):
        payload = evals.result_envelope(**_envelope_kwargs(split=split))
        _accept_eval_result(payload)
        assert payload["split"] == split
        assert payload["split"] in {s.value for s in evals.Split}


@pytest.mark.parametrize("status", list(evals.DecontamStatus))
def test_result_envelope_decontam_status_in_enum(status: evals.DecontamStatus) -> None:
    payload = evals.result_envelope(
        **_envelope_kwargs(decontamination=_decontam(status=status))
    )
    _accept_eval_result(payload)
    assert payload["decontamination"]["status"] == status


def test_result_envelope_blank_harness_version_is_error_not_empty_string() -> None:
    with pytest.raises(evals.HarnessVersionError):
        evals.result_envelope(**_envelope_kwargs(harness_version=""))


def test_result_envelope_generated_created_at_is_rfc3339() -> None:
    payload = evals.result_envelope(**_envelope_kwargs(created_at=_OMIT))
    _accept_eval_result(payload)
    _assert_rfc3339(payload["created_at"])


def test_result_envelope_n_items_zero_and_positive() -> None:
    zero = evals.result_envelope(**_envelope_kwargs(n_items=0))
    _accept_eval_result(zero)
    assert zero["n_items"] == 0
    one = evals.result_envelope(**_envelope_kwargs(n_items=1, metrics={"pass_at_1": 1.0}))
    _accept_eval_result(one)
    assert one["n_items"] == 1


def test_result_envelope_efficiency_metrics_live_on_the_result() -> None:
    metrics = {"pass_at_1": 0.25}
    for i, name in enumerate(evals.EFFICIENCY_METRICS, start=1):
        metrics[name] = float(i)
    payload = evals.result_envelope(**_envelope_kwargs(metrics=metrics, n_items=4))
    _accept_eval_result(payload)
    for name in evals.EFFICIENCY_METRICS:
        assert payload["metrics"][name] == metrics[name]


def test_result_envelope_forecasting_metrics_live_on_the_result() -> None:
    metrics = {name: 0.1 * (i + 1) for i, name in enumerate(evals.FORECASTING_METRICS)}
    payload = evals.result_envelope(
        **_envelope_kwargs(suite="forecasting", metrics=metrics, n_items=8)
    )
    _accept_eval_result(payload)
    for name in evals.FORECASTING_METRICS:
        assert payload["metrics"][name] == metrics[name]


def test_result_envelope_optional_hashes_round_trip() -> None:
    payload = evals.result_envelope(
        **_envelope_kwargs(
            per_item_hash=_SHA256,
            config_hash=_SHA256,
            job_id="job-1",
        )
    )
    _accept_eval_result(payload)
    assert payload["per_item_hash"] == _SHA256
    assert payload["config_hash"] == _SHA256
    assert payload["job_id"] == "job-1"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def test_runner_blank_harness_version_raises() -> None:
    cfg = evals.EvalConfig(
        suite="gpqa",
        split="test",
        checkpoint_id="ckpt",
        harness_version="",
    )
    with pytest.raises(evals.HarnessVersionError):
        evals.Runner(cfg)


def test_runner_unknown_suite_raises() -> None:
    cfg = evals.EvalConfig(
        suite="not_a_public_suite",
        split="test",
        checkpoint_id="ckpt",
        harness_version="oracle-test",
    )
    with pytest.raises(evals.UnknownSuiteError):
        runner = evals.Runner(cfg)
        runner.load_items()


def test_runner_load_items_n_items_zero() -> None:
    cfg = evals.EvalConfig(
        suite="gpqa",
        split="test",
        checkpoint_id="ckpt",
        harness_version="oracle-test",
        n_items=0,
    )
    runner = evals.Runner(cfg, index=_filled_index(_eval_item()))
    items = runner.load_items()
    assert isinstance(items, list)
    assert len(items) == 0
    assert all(isinstance(item, evals.EvalItem) for item in items)


def test_runner_run_returns_envelope_n_items_and_per_item_hash() -> None:
    cfg = evals.EvalConfig(
        suite="gpqa",
        split="test",
        checkpoint_id="ckpt-run",
        harness_version="oracle-test",
        n_items=0,
        seed=0,
    )
    runner = evals.Runner(cfg, index=_filled_index(_eval_item()))
    items = runner.load_items()
    payload = runner.run(lambda item: item.answer or "")
    _accept_eval_result(payload)
    assert payload["n_items"] == len(items)
    assert payload["n_items"] == 0
    assert payload["suite"] == "gpqa"
    assert payload["split"] == "test"
    assert payload["checkpoint_id"] == "ckpt-run"
    assert "per_item_hash" in payload
    assert re.fullmatch(r"[0-9a-f]{64}", payload["per_item_hash"])
    assert isinstance(payload["metrics"], dict)
    assert payload["metrics"]
    for value in payload["metrics"].values():
        assert isinstance(value, (int, float)) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# DecontamIndex
# ---------------------------------------------------------------------------


def test_decontam_fail_closed_empty_or_method_hash_unset() -> None:
    index = evals.DecontamIndex()
    with pytest.raises(evals.DecontamError):
        index.is_clean(_UNRELATED)


def test_decontam_near_copy_not_clean_unrelated_is_clean() -> None:
    index = _filled_index(_eval_item(prompt=_PROMPT))
    assert index.is_clean(_NEAR_COPY) is False
    assert index.is_clean(_PROMPT) is False
    assert index.is_clean(_UNRELATED) is True


def test_decontam_method_hash_is_sha256_after_add() -> None:
    index = evals.DecontamIndex()
    index.add(_eval_item())
    digest = index.method_hash()
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_decontam_dirty_stays_dirty_after_more_adds() -> None:
    index = _filled_index(_eval_item(item_id="a", prompt=_PROMPT))
    assert index.is_clean(_NEAR_COPY) is False
    index.add(_eval_item(item_id="b", prompt=_UNRELATED, suite="hle"))
    assert index.is_clean(_NEAR_COPY) is False
    assert index.is_clean(_PROMPT) is False


def test_canary_probe_is_a_suite_item_and_can_be_added() -> None:
    canary = _eval_item(
        item_id="canary-1",
        suite="gpqa",
        split="held_out",
        prompt=_CANARY,
        metadata={"probe": "canary"},
    )
    index = evals.DecontamIndex()
    index.add(canary)
    assert index.is_clean(_CANARY) is False
    assert index.is_clean(_UNRELATED) is True


def test_rephrased_probe_is_a_suite_item_not_a_side_channel() -> None:
    original = _eval_item(item_id="gpqa-orig", prompt=_PROMPT, split="test")
    rephrased = _eval_item(
        item_id="gpqa-rephrased",
        prompt=_NEAR_COPY,
        split="held_out",
        metadata={"probe": "rephrased"},
    )
    index = evals.DecontamIndex()
    index.add(original)
    index.add(rephrased)
    assert index.is_clean(_NEAR_COPY) is False
    assert index.is_clean(_PROMPT) is False
    assert index.is_clean(_UNRELATED) is True
