"""Independent CPU analog of V9 lab dry-run (spec 16.2 V9, 14.6 cycle).

Plain NumPy, slow and obvious. Host analog of L4 genome-seed roles plus one
14.6 cycle: researchers pre-register, run rung -1 experiments through a Slurm
backend analog, replicate, write the ledger, pass eval-gate. Glue of that
cycle, not a second lab. Passing this analog is not V9 verified.

Must not import ``prometheus.verify.v9_lab``, ``model/``, ``train/``,
``kernels/``, JAX, torch, or the Rust crates.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Array = np.ndarray

# Spec 16.2: 1 to 4 H200. Template ``v9.sh`` passes ``gpus``. CPU analog
# accepts ``gpus >= 1``. Default is the bottom of that range (spec DEFAULT_GPUS).
DEFAULT_GPUS = 1
DEFAULT_GPUS_MIN = 1
DEFAULT_GPUS_MAX = 4

# Spec 14.6 / 16.2 analog constants.
RUNG_MINUS_ONE = -1
JOB_PREFIX = "fv"
METRIC_NAME = "mse"
KILL_EPS = 0.05
PREDICTION_TOL = 1e-5  # FP32 parity / prediction calibration
AUTHOR_ROLE_RESEARCHER = "researcher"
AUTHOR_ROLE_TESTER = "tester"
GENOME_ROLES: tuple[str, ...] = (
    "director",
    "researcher",
    "implementer",
    "reviewer",
    "tester",
)
CYCLE_STEPS: tuple[str, ...] = (
    "preregister",
    "rung_minus_one",
    "replicate",
    "ledger",
    "eval_gate",
)
HELD_OUT_SUITES: tuple[str, ...] = (
    "re_bench",
    "mle_bench",
    "paperbench",
    "speedrun",
    "metr_time_horizon",
)

PLANTED_POSITIVE_ID = "planted-positive-ols-feature"
PLANTED_NEGATIVE_ID = "planted-negative-signflip-targets"
POSITIVE_HYPOTHESIS = (
    "Including the planted true-signal feature reduces held-out MSE vs baseline."
)
NEGATIVE_HYPOTHESIS = (
    "Fitting OLS on sign-flipped training targets reduces held-out MSE vs baseline."
)
KILL_CRITERION = f"held-out delta_mse <= {KILL_EPS}"
BASELINE_NAME = "ols-base-features-plus-intercept"

# Tiny rung -1 analog. Not a real lab / not V9 verified.
TOY_SEED = 9
TOY_REPLICATE_SEED = 26
TOY_N_TRAIN = 24
TOY_N_TEST = 24
TOY_D_BASE = 3
POSITIVE_WEIGHT = 2.0
NOISE_STD = 0.1
BUDGET_GPU_HOURS = 0.01
RCI_BASE_MILLI = 1000
RCI_DELTA_MILLI = 50
TOY_FD_EPS = 1e-6
TOY_GRAD_RTOL = 1e-5
TOY_GRAD_ATOL = 1e-8
GOLDEN_MATCH_DIFF = 0.0

V9_KEYS: tuple[str, ...] = (
    "planted_positive_found",
    "planted_positive_replicated",
    "planted_negative_recorded",
)


def require_gpus(gpus: int) -> None:
    """CPU analog of the production ``gpus < 1`` check (raises ``ValueError``)."""
    if gpus < 1:
        raise ValueError("gpus must be >= 1")


def add_intercept(features: Array) -> Array:
    n = int(features.shape[0])
    ones = np.ones((n, 1), dtype=np.float64)
    feat = np.asarray(features, dtype=np.float64)
    return np.concatenate([ones, feat], axis=1)


def ols_weights(x: Array, y: Array) -> Array:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    w, *_ = np.linalg.lstsq(x64, y64, rcond=None)
    return np.asarray(w, dtype=np.float64).reshape(-1)


def mse(x: Array, y: Array, w: Array) -> float:
    pred = np.asarray(x, dtype=np.float64) @ np.asarray(w, dtype=np.float64)
    err = pred - np.asarray(y, dtype=np.float64)
    return float(np.mean(err * err))


def mse_loss(x: Array, w: Array, y: Array) -> float:
    return mse(x, y, w)


def analytic_mse_grad(x: Array, w: Array, y: Array) -> Array:
    x64 = np.asarray(x, dtype=np.float64)
    w64 = np.asarray(w, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    err = x64 @ w64 - y64
    n = float(err.shape[0])
    return (2.0 / n) * (x64.T @ err)


def toy_grad_ok(
    *,
    eps: float = TOY_FD_EPS,
    rtol: float = TOY_GRAD_RTOL,
    atol: float = TOY_GRAD_ATOL,
) -> bool:
    """OLS held-out MSE: analytic ``dL/dw`` vs central finite differences."""
    design = make_design(TOY_SEED)
    fitted = fit_idea(design, extra="pos")
    x = fitted["X_train"]
    y = design["y_train"]
    w = fitted["w"]
    analytic = analytic_mse_grad(x, w, y)
    fd = np.empty_like(w)
    for i in range(int(w.shape[0])):
        up = w.copy()
        dn = w.copy()
        up[i] += eps
        dn[i] -= eps
        fd[i] = (mse_loss(x, up, y) - mse_loss(x, dn, y)) / (2.0 * eps)
    return bool(np.allclose(analytic, fd, rtol=rtol, atol=atol))


def ols_mse_dtype(x: Array, y: Array, dtype: np.dtype) -> float:
    xd = np.asarray(x, dtype=dtype)
    yd = np.asarray(y, dtype=dtype)
    w, *_ = np.linalg.lstsq(xd, yd, rcond=None)
    w = np.asarray(w, dtype=dtype).reshape(-1)
    pred = xd @ w
    err = pred.astype(np.float64) - yd.astype(np.float64)
    return float(np.mean(err * err))


def fp32_parity_ok(x: Array, y: Array, *, tol: float = PREDICTION_TOL) -> bool:
    m64 = ols_mse_dtype(x, y, np.float64)
    m32 = ols_mse_dtype(x, y, np.float32)
    return bool(abs(m32 - m64) <= tol)


def make_design(seed: int) -> dict[str, Array]:
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    n = TOY_N_TRAIN + TOY_N_TEST
    x_base = rng.standard_normal((n, TOY_D_BASE)).astype(np.float64)
    x_pos = rng.standard_normal((n,)).astype(np.float64)
    x_neg = rng.standard_normal((n,)).astype(np.float64)
    w_base = rng.standard_normal((TOY_D_BASE,)).astype(np.float64)
    noise = NOISE_STD * rng.standard_normal((n,)).astype(np.float64)
    y = x_base @ w_base + POSITIVE_WEIGHT * x_pos + noise
    tr = slice(0, TOY_N_TRAIN)
    te = slice(TOY_N_TRAIN, n)
    return {
        "x_base_train": x_base[tr],
        "x_base_test": x_base[te],
        "x_pos_train": x_pos[tr],
        "x_pos_test": x_pos[te],
        "x_neg_train": x_neg[tr],
        "x_neg_test": x_neg[te],
        "y_train": y[tr],
        "y_test": y[te],
    }


def _with_extra(base: Array, extra: Array | None) -> Array:
    intercept = add_intercept(base)
    if extra is None:
        return intercept
    return np.concatenate([intercept, np.asarray(extra, dtype=np.float64)[:, None]], axis=1)


def fit_idea(design: Mapping[str, Array], *, extra: str) -> dict[str, Any]:
    if extra == "pos":
        x_tr = _with_extra(design["x_base_train"], design["x_pos_train"])
        x_te = _with_extra(design["x_base_test"], design["x_pos_test"])
        y_fit = design["y_train"]
    elif extra == "neg":
        # Planted known-negative idea: fit the baseline design to -y.
        x_tr = _with_extra(design["x_base_train"], None)
        x_te = _with_extra(design["x_base_test"], None)
        y_fit = -np.asarray(design["y_train"], dtype=np.float64)
    elif extra == "none":
        x_tr = _with_extra(design["x_base_train"], None)
        x_te = _with_extra(design["x_base_test"], None)
        y_fit = design["y_train"]
    else:
        raise ValueError(f"unknown extra {extra!r}")
    w = ols_weights(x_tr, y_fit)
    return {
        "w": w,
        "X_train": x_tr,
        "X_test": x_te,
        "mse_train": mse(x_tr, design["y_train"], w),
        "mse_test": mse(x_te, design["y_test"], w),
        "w_shape": tuple(int(s) for s in w.shape),
        "X_train_shape": tuple(int(s) for s in x_tr.shape),
        "X_test_shape": tuple(int(s) for s in x_te.shape),
        "dtype": "float64",
    }


def residual_orthogonality(x: Array, y: Array, w: Array, *, atol: float = 1e-8) -> bool:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    w64 = np.asarray(w, dtype=np.float64)
    resid = y64 - x64 @ w64
    xt_r = x64.T @ resid
    return bool(np.allclose(xt_r, 0.0, atol=atol))


def config_hash(idea_id: str, seed: int) -> str:
    blob = (
        f"{idea_id}|{seed}|{RUNG_MINUS_ONE}|{TOY_N_TRAIN}|{TOY_N_TEST}|{TOY_D_BASE}"
    ).encode()
    return hashlib.sha256(blob).hexdigest()


@dataclass
class Preregistration:
    idea_id: str
    hypothesis: str
    metric: str
    baseline: str
    kill_criterion: str
    budget_gpu_hours: float
    predicted_delta_mse: float
    predicted_ci: float
    rung: int
    author_role: str


@dataclass
class SlurmJob:
    job_id: str
    job_name: str
    idea_id: str
    seed: int
    state: str
    mse_test: float | None
    delta_mse: float | None


@dataclass
class LedgerRecord:
    experiment_id: str
    idea_id: str
    hypothesis: str
    prediction: str
    config_hash: str
    author_role: str
    rung: int
    outcome: str
    delta_mse: float
    delta_rci: float | None
    replication_status: str
    gpu_hours: float
    results: dict[str, Any] = field(default_factory=dict)


class LedgerAnalog:
    """Append-only host analog of the self-improvement ledger."""

    def __init__(self) -> None:
        self.rows: list[LedgerRecord] = []

    def append(self, rec: LedgerRecord) -> None:
        if any(row.experiment_id == rec.experiment_id for row in self.rows):
            raise ValueError(f"duplicate experiment_id {rec.experiment_id}")
        self.rows.append(rec)

    def find(self, idea_id: str) -> LedgerRecord | None:
        for row in self.rows:
            if row.idea_id == idea_id:
                return row
        return None


class SlurmAnalog:
    """Host analog of the Slurm backend: jobs actually run the OLS experiment."""

    def __init__(self) -> None:
        self.jobs: dict[str, SlurmJob] = {}
        self._next = 1

    def submit(
        self,
        *,
        idea_id: str,
        seed: int,
        extra: str,
        baseline_mse: float,
        run: bool,
    ) -> SlurmJob:
        job_id = str(self._next)
        self._next += 1
        name = f"{JOB_PREFIX}-v9-{idea_id}-s{seed}"
        if not run:
            job = SlurmJob(
                job_id=job_id,
                job_name=name,
                idea_id=idea_id,
                seed=seed,
                state="pending",
                mse_test=None,
                delta_mse=None,
            )
            self.jobs[job_id] = job
            return job
        design = make_design(seed)
        fitted = fit_idea(design, extra=extra)
        mse_test = float(fitted["mse_test"])
        job = SlurmJob(
            job_id=job_id,
            job_name=name,
            idea_id=idea_id,
            seed=seed,
            state="completed",
            mse_test=mse_test,
            delta_mse=float(baseline_mse - mse_test),
        )
        self.jobs[job_id] = job
        return job


def _baseline_mse(seed: int) -> float:
    design = make_design(seed)
    return float(fit_idea(design, extra="none")["mse_test"])


def _idea_delta(seed: int, extra: str) -> float:
    design = make_design(seed)
    base = float(fit_idea(design, extra="none")["mse_test"])
    idea = float(fit_idea(design, extra=extra)["mse_test"])
    return float(base - idea)


def beats_kill(delta_mse: float) -> bool:
    return bool(delta_mse > KILL_EPS)


def run_eval_gate(*, applied_positive: bool, skip: bool) -> dict[str, Any]:
    """Held-out analog: scores only. Task text never leaves the gate."""
    if skip:
        return {
            "eval_gate_passed": False,
            "skipped": True,
            "caller": AUTHOR_ROLE_TESTER,
            "scores": [],
            "delta_rci_milli": 0,
            "leaked_task_text": False,
            "suites": HELD_OUT_SUITES,
        }
    delta = RCI_DELTA_MILLI if applied_positive else 0
    scores = []
    for suite in HELD_OUT_SUITES:
        scores.append(
            {
                "suite": suite,
                "rci_milli": RCI_BASE_MILLI + delta,
                "n_items": TOY_N_TEST,
                "suite_hash": hashlib.sha256(suite.encode()).hexdigest(),
            }
        )
    blob = {"scores": scores, "delta_rci_milli": delta}
    leaked = any(
        k in blob or k in score
        for score in scores
        for k in ("prompt", "task", "answer", "item", "task_text")
    )
    return {
        "eval_gate_passed": (not leaked) and delta >= 0,
        "skipped": False,
        "caller": AUTHOR_ROLE_TESTER,
        "scores": scores,
        "delta_rci_milli": delta,
        "leaked_task_text": leaked,
        "suites": HELD_OUT_SUITES,
    }


def evaluate_v9_protocol(
    *,
    plant_positive: bool = True,
    plant_negative: bool = True,
    skip_preregister: bool = False,
    skip_slurm: bool = False,
    skip_replicate: bool = False,
    skip_ledger_write: bool = False,
    skip_eval_gate: bool = False,
    record_negative_as_positive: bool = False,
    gpus: int = DEFAULT_GPUS,
) -> dict[str, Any]:
    """Run one 14.6-cycle analog and report the F4 V9 gates.

    Reviewer throwaways: skip plant / skip replicate / skip ledger write
    must flip the corresponding bool to False.
    """
    require_gpus(gpus)
    slurm = SlurmAnalog()
    ledger = LedgerAnalog()
    prereg: list[Preregistration] = []

    predicted_pos = _idea_delta(TOY_SEED, "pos")
    predicted_neg = _idea_delta(TOY_SEED, "neg")
    baseline_mse = _baseline_mse(TOY_SEED)

    if not skip_preregister:
        if plant_positive:
            prereg.append(
                Preregistration(
                    idea_id=PLANTED_POSITIVE_ID,
                    hypothesis=POSITIVE_HYPOTHESIS,
                    metric=METRIC_NAME,
                    baseline=BASELINE_NAME,
                    kill_criterion=KILL_CRITERION,
                    budget_gpu_hours=BUDGET_GPU_HOURS,
                    predicted_delta_mse=predicted_pos,
                    predicted_ci=PREDICTION_TOL,
                    rung=RUNG_MINUS_ONE,
                    author_role=AUTHOR_ROLE_RESEARCHER,
                )
            )
        if plant_negative:
            prereg.append(
                Preregistration(
                    idea_id=PLANTED_NEGATIVE_ID,
                    hypothesis=NEGATIVE_HYPOTHESIS,
                    metric=METRIC_NAME,
                    baseline=BASELINE_NAME,
                    kill_criterion=KILL_CRITERION,
                    budget_gpu_hours=BUDGET_GPU_HOURS,
                    predicted_delta_mse=predicted_neg,
                    predicted_ci=PREDICTION_TOL,
                    rung=RUNG_MINUS_ONE,
                    author_role=AUTHOR_ROLE_RESEARCHER,
                )
            )

    registered_ids = {p.idea_id for p in prereg}
    run_jobs = not skip_slurm

    pos_job: SlurmJob | None = None
    neg_job: SlurmJob | None = None
    repl_job: SlurmJob | None = None

    if plant_positive:
        pos_job = slurm.submit(
            idea_id=PLANTED_POSITIVE_ID,
            seed=TOY_SEED,
            extra="pos",
            baseline_mse=baseline_mse,
            run=run_jobs,
        )
        if not skip_replicate:
            repl_baseline = _baseline_mse(TOY_REPLICATE_SEED)
            repl_job = slurm.submit(
                idea_id=PLANTED_POSITIVE_ID,
                seed=TOY_REPLICATE_SEED,
                extra="pos",
                baseline_mse=repl_baseline,
                run=run_jobs,
            )
    if plant_negative:
        neg_job = slurm.submit(
            idea_id=PLANTED_NEGATIVE_ID,
            seed=TOY_SEED,
            extra="neg",
            baseline_mse=baseline_mse,
            run=run_jobs,
        )

    pos_observed = pos_job.delta_mse if pos_job is not None else None
    pos_found = bool(
        plant_positive
        and PLANTED_POSITIVE_ID in registered_ids
        and pos_job is not None
        and pos_job.state == "completed"
        and pos_observed is not None
        and beats_kill(pos_observed)
        and abs(pos_observed - predicted_pos) <= PREDICTION_TOL
    )

    repl_observed = repl_job.delta_mse if repl_job is not None else None
    pos_replicated = bool(
        pos_found
        and (not skip_replicate)
        and repl_job is not None
        and repl_job.state == "completed"
        and repl_observed is not None
        and beats_kill(repl_observed)
    )
    replication_status = (
        "matched" if pos_replicated else ("unreplicated" if skip_replicate else "failed")
    )
    replication_n = 2 if pos_replicated else (1 if pos_found else 0)

    neg_observed = neg_job.delta_mse if neg_job is not None else None
    neg_is_negative = bool(
        plant_negative
        and PLANTED_NEGATIVE_ID in registered_ids
        and neg_job is not None
        and neg_job.state == "completed"
        and neg_observed is not None
        and (not beats_kill(neg_observed))
    )
    recorded_outcome = (
        "positive" if record_negative_as_positive else "negative"
    )

    if not skip_ledger_write:
        if pos_found and pos_observed is not None:
            ledger.append(
                LedgerRecord(
                    experiment_id="v9-rung-minus-1-planted-positive",
                    idea_id=PLANTED_POSITIVE_ID,
                    hypothesis=POSITIVE_HYPOTHESIS,
                    prediction=str(predicted_pos),
                    config_hash=config_hash(PLANTED_POSITIVE_ID, TOY_SEED),
                    author_role=AUTHOR_ROLE_RESEARCHER,
                    rung=RUNG_MINUS_ONE,
                    outcome="positive",
                    delta_mse=pos_observed,
                    delta_rci=None,
                    replication_status=replication_status,
                    gpu_hours=BUDGET_GPU_HOURS,
                    results={"mse_test": pos_job.mse_test if pos_job else None},
                )
            )
        if neg_is_negative and neg_observed is not None:
            ledger.append(
                LedgerRecord(
                    experiment_id="v9-rung-minus-1-planted-negative",
                    idea_id=PLANTED_NEGATIVE_ID,
                    hypothesis=NEGATIVE_HYPOTHESIS,
                    prediction=str(predicted_neg),
                    config_hash=config_hash(PLANTED_NEGATIVE_ID, TOY_SEED),
                    author_role=AUTHOR_ROLE_RESEARCHER,
                    rung=RUNG_MINUS_ONE,
                    outcome=recorded_outcome,
                    delta_mse=neg_observed,
                    delta_rci=None,
                    replication_status="unreplicated",
                    gpu_hours=BUDGET_GPU_HOURS,
                    results={"mse_test": neg_job.mse_test if neg_job else None},
                )
            )

    neg_row = ledger.find(PLANTED_NEGATIVE_ID)
    planted_negative_recorded = bool(
        neg_is_negative
        and neg_row is not None
        and neg_row.outcome == "negative"
        and neg_row.rung == RUNG_MINUS_ONE
    )

    eval_gate = run_eval_gate(applied_positive=pos_found, skip=skip_eval_gate)
    if (not skip_ledger_write) and pos_found:
        pos_row = ledger.find(PLANTED_POSITIVE_ID)
        if pos_row is not None:
            pos_row.delta_rci = float(eval_gate["delta_rci_milli"]) / 1000.0

    design = make_design(TOY_SEED)
    base_fit = fit_idea(design, extra="none")
    pos_fit = fit_idea(design, extra="pos")
    neg_fit = fit_idea(design, extra="neg")
    grad_ok = toy_grad_ok()
    parity_ok = fp32_parity_ok(pos_fit["X_train"], design["y_train"])

    result: dict[str, Any] = {
        "planted_positive_found": pos_found,
        "planted_positive_replicated": pos_replicated,
        "planted_negative_recorded": planted_negative_recorded,
        "gpus": int(gpus),
        "rung": RUNG_MINUS_ONE,
        "roles": GENOME_ROLES,
        "cycle_steps": CYCLE_STEPS,
        "preregister": {
            "skipped": skip_preregister,
            "n": len(prereg),
            "ids": tuple(p.idea_id for p in prereg),
            "author_role": AUTHOR_ROLE_RESEARCHER,
            "fields": (
                "hypothesis",
                "metric",
                "baseline",
                "kill_criterion",
                "budget",
                "numeric_prediction",
            ),
            "predicted_positive_delta_mse": predicted_pos,
            "predicted_negative_delta_mse": predicted_neg,
        },
        "slurm": {
            "skipped": skip_slurm,
            "job_prefix": JOB_PREFIX,
            "n_jobs": len(slurm.jobs),
            "states": tuple(j.state for j in slurm.jobs.values()),
            "names": tuple(j.job_name for j in slurm.jobs.values()),
        },
        "positive": {
            "planted": plant_positive,
            "idea_id": PLANTED_POSITIVE_ID,
            "delta_mse": pos_observed,
            "predicted_delta_mse": predicted_pos,
            "beats_kill": (
                beats_kill(pos_observed) if pos_observed is not None else False
            ),
            "job_state": pos_job.state if pos_job is not None else None,
        },
        "negative": {
            "planted": plant_negative,
            "idea_id": PLANTED_NEGATIVE_ID,
            "delta_mse": neg_observed,
            "predicted_delta_mse": predicted_neg,
            "beats_kill": (
                beats_kill(neg_observed) if neg_observed is not None else False
            ),
            "is_negative": neg_is_negative,
            "recorded_outcome": recorded_outcome if neg_row is not None else None,
            "job_state": neg_job.state if neg_job is not None else None,
        },
        "replication": {
            "skipped": skip_replicate,
            "status": replication_status,
            "n": replication_n,
            "delta_mse": repl_observed,
            "seed": TOY_REPLICATE_SEED,
        },
        "ledger": {
            "skipped": skip_ledger_write,
            "n": len(ledger.rows),
            "ids": tuple(r.experiment_id for r in ledger.rows),
            "rungs": tuple(r.rung for r in ledger.rows),
            "outcomes": tuple(r.outcome for r in ledger.rows),
        },
        "eval_gate": eval_gate,
        "ols": {
            "base_mse_test": float(base_fit["mse_test"]),
            "pos_mse_test": float(pos_fit["mse_test"]),
            "neg_mse_test": float(neg_fit["mse_test"]),
            "base_mse_train": float(base_fit["mse_train"]),
            "pos_mse_train": float(pos_fit["mse_train"]),
            "X_train_shape": pos_fit["X_train_shape"],
            "X_test_shape": pos_fit["X_test_shape"],
            "w_shape": pos_fit["w_shape"],
            "dtype": pos_fit["dtype"],
            "residual_orthogonality": residual_orthogonality(
                pos_fit["X_train"], design["y_train"], pos_fit["w"]
            ),
            "nested_train_mse": bool(pos_fit["mse_train"] <= base_fit["mse_train"]),
        },
        "grad_ok": grad_ok,
        "fp32_parity_ok": parity_ok,
        "skip_plant_positive": not plant_positive,
        "skip_plant_negative": not plant_negative,
        "record_negative_as_positive": record_negative_as_positive,
    }
    result["meets_gates"] = meets_v9_gates(result)
    result["cycle_complete"] = bool(
        result["meets_gates"]
        and (not skip_preregister)
        and (not skip_slurm)
        and (not skip_eval_gate)
        and eval_gate["eval_gate_passed"]
    )
    return result


def meets_v9_gates(result: Mapping[str, Any]) -> bool:
    return bool(
        result["planted_positive_found"]
        and result["planted_positive_replicated"]
        and result["planted_negative_recorded"]
    )
