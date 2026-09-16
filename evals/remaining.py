"""Forecasting scores, latent scaling, efficiency, canaries (spec 11)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any


def brier(p: float, y: float) -> float:
    return (float(p) - float(y)) ** 2


def log_score(p: float, y: float) -> float:
    p = min(max(float(p), 1e-15), 1.0 - 1e-15)
    y = float(y)
    return y * math.log(p) + (1.0 - y) * math.log(1.0 - p)


def paper_pnl(price: float, settle: float, stake: float = 1.0) -> float:
    return float(stake) * (float(settle) - float(price))


def leak_probe(p_clean: float, p_leaked: float) -> float:
    """Absolute jump after seeing leaked future information."""
    return abs(float(p_leaked) - float(p_clean))


def efficiency_metrics(
    tokens_to_solve: float = 0.0,
    latent_steps: float = 0.0,
    recurrence_iterations: float = 0.0,
) -> dict[str, float]:
    return {
        "tokens_to_solve": float(tokens_to_solve),
        "latent_steps": float(latent_steps),
        "recurrence_iterations": float(recurrence_iterations),
    }


def attach_efficiency(
    payload: dict[str, Any],
    *,
    tokens_to_solve: float = 0.0,
    latent_steps: float = 0.0,
    recurrence_iterations: float = 0.0,
) -> dict[str, Any]:
    out = dict(payload)
    metrics = dict(payload.get("metrics") or {})
    metrics.update(
        efficiency_metrics(
            tokens_to_solve=tokens_to_solve,
            latent_steps=latent_steps,
            recurrence_iterations=recurrence_iterations,
        )
    )
    out["metrics"] = metrics
    return out


def latent_scaling(
    acc_by_budget: dict[int, float],
    discrete_acc: float,
    flop_by_budget: dict[int, float],
    flop_discrete: float,
) -> dict[str, Any]:
    """Accuracy vs recurrence budget 1..16 vs discrete CoT at matched FLOP."""
    budgets = [1, 2, 4, 8, 16]
    acc = {b: float(acc_by_budget[b]) for b in budgets}
    flops = {b: float(flop_by_budget[b]) for b in budgets}
    matched = min(budgets, key=lambda b: abs(flops[b] - float(flop_discrete)))
    beats = acc[matched] >= float(discrete_acc)
    return {
        "budgets": budgets,
        "accuracy": acc,
        "discrete_acc": float(discrete_acc),
        "matched_budget": matched,
        "beats_discrete_at_matched_flop": beats,
        "rises_with_budget": acc[16] >= acc[1],
        "flop_by_budget": flops,
        "flop_discrete": float(flop_discrete),
    }


CANARY_PREFIX = "PROMETHEUS_EVAL_CANARY_TOKEN_v1"


def scan_canaries(corpus: str, tokens: Sequence[str] | None = None) -> list[str]:
    toks = list(tokens) if tokens is not None else [CANARY_PREFIX]
    return [t for t in toks if t and t in corpus]


def run_forecasting(
    items: Sequence[Any],
    predict: Callable[[Any], float],
    *,
    config: Any,
    index: Any | None = None,
) -> dict[str, Any]:
    from evals import (
        Decontamination,
        DecontamStatus,
        _hash_config,
        _hash_items,
        _item_text,
        _require_harness_version,
        require_suite,
        result_envelope,
    )

    _require_harness_version(config.harness_version)
    require_suite("forecasting")
    if config.suite != "forecasting":
        raise ValueError("Forecasting runner requires suite='forecasting'")
    briers: list[float] = []
    logs: list[float] = []
    pnls: list[float] = []
    leaks: list[float] = []
    flagged = False
    item_list = list(items)
    for item in item_list:
        p = float(predict(item))
        p = min(1.0, max(0.0, p))
        meta = getattr(item, "metadata", {}) or {}
        y = meta.get("outcome", item.answer)
        y = float(y) if y is not None else 0.0
        price = float(meta.get("price", p))
        briers.append(brier(p, y))
        logs.append(log_score(p, y))
        pnls.append(paper_pnl(price, y))
        if "p_clean" in meta:
            leaks.append(leak_probe(float(meta["p_clean"]), p))
        if index is not None and not index.is_clean(
            _item_text(item.prompt, item.answer), against=(item.suite,)
        ):
            flagged = True
    n = len(item_list)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    metrics = {
        "brier": _mean(briers),
        "log_score": _mean(logs),
        "paper_pnl": _mean(pnls),
        "leak_probe": _mean(leaks),
    }
    if index is None:
        decontamination = Decontamination(status=DecontamStatus.PENDING)
    else:
        decontamination = Decontamination(
            status=DecontamStatus.FLAGGED if flagged else DecontamStatus.CLEAN,
            against=("forecasting",),
            method_hash=index.method_hash(),
        )
    return result_envelope(
        result_id=f"eval-forecasting-{config.checkpoint_id}",
        suite="forecasting",
        split=config.split,
        checkpoint_id=config.checkpoint_id,
        metrics=metrics,
        n_items=n,
        harness_version=config.harness_version,
        decontamination=decontamination,
        per_item_hash=_hash_items(item_list) if item_list else None,
        config_hash=_hash_config(config),
    )


def contamination_canaries(n: int = 3, suite: str = "gpqa") -> list[Any]:
    from evals import EvalItem

    items = []
    for i in range(n):
        token = f"{CANARY_PREFIX}_{i:04d}"
        items.append(
            EvalItem(
                item_id=f"canary-{i:04d}",
                suite=suite,
                split="held_out",
                prompt=f"{token} unused eval item must never appear in training",
                answer=token,
                metadata={"probe": "canary", "token": token},
            )
        )
    return items
