"""Lab v0 research cycle (spec 14.6).

Pre-register idea, method, success criterion, compute bound. Run. Replicate
on a held-out seed. Eval-gate. Write the sign to the SQLite ledger. A
positive that does not replicate is recorded as negative.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Experiment:
    experiment_id: str
    idea: str
    method: str
    success: str
    compute_bound: str
    author_role: str = "worker"
    rung: int = 0
    registered: bool = False
    result: dict[str, Any] | None = None
    replica: dict[str, Any] | None = None
    sign: str | None = None
    replicated: bool = False
    recorded: str | None = None
    eval_gate_pass: bool | None = None


@dataclass
class LabCycle:
    ledger_path: Path
    experiments: dict[str, Experiment] = field(default_factory=dict)

    def pre_register(
        self,
        experiment_id: str,
        idea: str,
        method: str,
        success: str,
        compute_bound: str,
        author_role: str = "worker",
        rung: int = 0,
    ) -> Experiment:
        if experiment_id in self.experiments:
            raise ValueError(f"already registered: {experiment_id}")
        exp = Experiment(
            experiment_id=experiment_id,
            idea=idea,
            method=method,
            success=success,
            compute_bound=compute_bound,
            author_role=author_role,
            rung=rung,
            registered=True,
        )
        self.experiments[experiment_id] = exp
        return exp

    def run_experiment(
        self, experiment_id: str, fn: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        exp = self.experiments[experiment_id]
        if not exp.registered:
            raise ValueError("run before pre-register")
        result = dict(fn())
        exp.result = result
        return result

    def replicate(
        self, experiment_id: str, fn: Callable[[], dict[str, Any]]
    ) -> bool:
        exp = self.experiments[experiment_id]
        if exp.result is None:
            raise ValueError("replicate before run")
        replica = dict(fn())
        exp.replica = replica
        exp.replicated = _same_sign(exp.result, replica)
        return exp.replicated

    def eval_gate(self, experiment_id: str, metrics: dict[str, float] | None = None) -> bool:
        exp = self.experiments[experiment_id]
        metrics = metrics if metrics is not None else (exp.result or {})
        held_out = float(metrics.get("held_out", metrics.get("score", 0.0)))
        baseline = float(metrics.get("baseline", 0.0))
        exp.eval_gate_pass = bool(held_out > baseline)
        return exp.eval_gate_pass

    def record(self, experiment_id: str) -> dict[str, Any]:
        from harness.ledger import Ledger

        exp = self.experiments[experiment_id]
        if exp.result is None:
            raise ValueError("record before run")
        if exp.eval_gate_pass is None:
            self.eval_gate(experiment_id)
        if exp.replica is None:
            exp.replicated = False
        if exp.eval_gate_pass and exp.replicated:
            exp.sign = "positive"
            exp.recorded = "positive"
        else:
            exp.sign = "negative"
            exp.recorded = "negative"
        led = Ledger(self.ledger_path)
        row = {
            "experiment_id": exp.experiment_id,
            "author_role": exp.author_role,
            "rung": exp.rung,
            "sign": exp.sign,
            "replicated": exp.replicated,
            "recorded": exp.recorded,
            "idea": exp.idea,
            "method": exp.method,
            "success": exp.success,
            "compute_bound": exp.compute_bound,
        }
        led.append(row)
        led.close()
        return row

    def dump(self, path: Path) -> None:
        path.write_text(
            json.dumps({k: asdict(v) for k, v in self.experiments.items()}, indent=2)
        )


def _same_sign(a: dict[str, Any], b: dict[str, Any]) -> bool:
    def _sign(d: dict[str, Any]) -> str:
        if d.get("sign"):
            return str(d["sign"])
        return "positive" if float(d.get("score", 0)) > float(d.get("baseline", 0)) else "negative"

    sa = _sign(a)
    sb = _sign(b)
    return sa == sb


def planted_cycle(ledger_path: Path) -> dict[str, Any]:
    """V9 dry run: one planted positive that replicates, one planted negative."""
    lab = LabCycle(ledger_path)
    lab.pre_register(
        "plant-pos",
        idea="muon-clip",
        method="QK clip at 10",
        success="logit explosion stops",
        compute_bound="cpu",
    )
    pos = {"score": 1.0, "baseline": 0.0, "sign": "positive", "held_out": 1.0}
    lab.run_experiment("plant-pos", lambda: pos)
    lab.replicate("plant-pos", lambda: pos)
    lab.eval_gate("plant-pos")
    lab.record("plant-pos")

    lab.pre_register(
        "plant-neg",
        idea="drop-qk-norm",
        method="remove QK-norm",
        success="loss drops",
        compute_bound="cpu",
    )
    neg = {"score": 0.1, "baseline": 0.5, "sign": "negative", "held_out": 0.1}
    lab.run_experiment("plant-neg", lambda: neg)
    lab.replicate("plant-neg", lambda: neg)
    lab.eval_gate("plant-neg")
    lab.record("plant-neg")
    return {
        "planted_positive_replicated": lab.experiments["plant-pos"].replicated
        and lab.experiments["plant-pos"].recorded == "positive",
        "planted_negative_recorded": lab.experiments["plant-neg"].recorded == "negative",
        "ledger_path": str(ledger_path),
        "n_experiments": len(lab.experiments),
        "config_hash": hashlib.sha256(b"lab-v0").hexdigest()[:16],
    }
