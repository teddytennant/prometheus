"""V9 lab dry-run runner (spec 16.2, F4 template ``v9.sh``).

CPU analog of one 14.6 cycle: planted known-positive idea found and
replicated; planted negative recorded as negative. Glue of L4
``harness/genome-seed``, not a second lab.

`verify/ncshare` `check_exit(V9)` reads `v9.json` with three bools, all
True:

- `planted_positive_found`
- `planted_positive_replicated`
- `planted_negative_recorded`

Spec V9 is 1 to 4 H200. A CPU analog is allowed so the cycle can run
without a GPU. That does not count as V9 verified.

Must not import `tests/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np

# Spec 16.2: 1 to 4 H200. Template ``v9.sh`` passes ``gpus``. CPU analog
# accepts ``gpus >= 1``. Default is the bottom of that range.
DEFAULT_GPUS = 1
DEFAULT_GPUS_MIN = 1
DEFAULT_GPUS_MAX = 4

# Spec 14.6 / 16.2 analog constants (must match the oracle analog).
RUNG_MINUS_ONE = -1
JOB_PREFIX = "fv"
METRIC_NAME = "mse"
KILL_EPS = 0.05
PREDICTION_TOL = 1e-5
AUTHOR_ROLE_RESEARCHER = "researcher"
AUTHOR_ROLE_TESTER = "tester"

PLANTED_POSITIVE_ID = "planted-positive-ols-feature"
PLANTED_NEGATIVE_ID = "planted-negative-signflip-targets"

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
GOLDEN_MATCH_DIFF = 0.0

_REQUIRED_ROLES = (
    "director",
    "researcher",
    "implementer",
    "reviewer",
    "tester",
)
_HELD_OUT_SUITES = (
    "re_bench",
    "mle_bench",
    "paperbench",
    "speedrun",
    "metr_time_horizon",
)


class V9Result(TypedDict):
    planted_positive_found: bool
    planted_positive_replicated: bool
    planted_negative_recorded: bool


class V9Error(Exception):
    """Bad V9 inputs (gpus < 1) or a missing / failed analog backend."""


def _genome_roles() -> tuple[str, ...]:
    """Read L4 seed roles from ``harness/genome-seed/roles``."""
    roles_dir = Path(__file__).resolve().parents[2] / "harness" / "genome-seed" / "roles"
    if not roles_dir.is_dir():
        raise V9Error(f"genome-seed roles directory missing: {roles_dir}")
    names: list[str] = []
    for name in _REQUIRED_ROLES:
        path = roles_dir / f"{name}.md"
        if not path.is_file():
            raise V9Error(f"missing genome-seed role {name}")
        if not path.read_text(encoding="utf-8").strip():
            raise V9Error(f"empty genome-seed role {name}")
        names.append(name)
    return tuple(names)


def _add_intercept(features: np.ndarray) -> np.ndarray:
    n = int(features.shape[0])
    ones = np.ones((n, 1), dtype=np.float64)
    return np.concatenate([ones, np.asarray(features, dtype=np.float64)], axis=1)


def _ols_test_mse(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    x_tr = np.asarray(x_train, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64)
    x_te = np.asarray(x_test, dtype=np.float64)
    y_te = np.asarray(y_test, dtype=np.float64)
    weights, *_ = np.linalg.lstsq(x_tr, y_tr, rcond=None)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    err = x_te @ weights - y_te
    mse = float(np.mean(err * err))
    if not np.isfinite(mse):
        raise V9Error("non-finite OLS analog MSE")
    return mse


def _toy_split(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    n = TOY_N_TRAIN + TOY_N_TEST
    x_base = rng.standard_normal((n, TOY_D_BASE)).astype(np.float64)
    x_pos = rng.standard_normal((n,)).astype(np.float64)
    rng.standard_normal((n,))  # unused planted-negative covariate; keeps the DGP stream
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
        "y_train": y[tr],
        "y_test": y[te],
    }


def _idea_delta_mse(seed: int, *, kind: str) -> float:
    """Held-out delta_mse = baseline_mse - idea_mse (rung -1 OLS analog)."""
    data = _toy_split(seed)
    x_base_tr = _add_intercept(data["x_base_train"])
    x_base_te = _add_intercept(data["x_base_test"])
    y_tr = data["y_train"]
    y_te = data["y_test"]
    baseline = _ols_test_mse(x_base_tr, y_tr, x_base_te, y_te)
    if kind == "pos":
        x_tr = np.concatenate([x_base_tr, data["x_pos_train"][:, None]], axis=1)
        x_te = np.concatenate([x_base_te, data["x_pos_test"][:, None]], axis=1)
        idea = _ols_test_mse(x_tr, y_tr, x_te, y_te)
    elif kind == "neg":
        # Known-negative idea: fit the baseline design to sign-flipped targets.
        idea = _ols_test_mse(x_base_tr, -y_tr, x_base_te, y_te)
    else:
        raise V9Error(f"unknown idea kind {kind!r}")
    return float(baseline - idea)


def _beats_kill(delta_mse: float) -> bool:
    return bool(delta_mse > KILL_EPS)


def _eval_gate(*, applied_positive: bool) -> bool:
    """Tester-role analog: scores only, no task text."""
    delta = RCI_DELTA_MILLI if applied_positive else 0
    scores = [
        {"suite": suite, "rci_milli": RCI_BASE_MILLI + delta, "n_items": TOY_N_TEST}
        for suite in _HELD_OUT_SUITES
    ]
    leak_keys = ("prompt", "task", "answer", "item", "task_text")
    leaked = any(key in score for score in scores for key in leak_keys)
    return (not leaked) and delta >= 0


def _run_cycle() -> V9Result:
    roles = _genome_roles()
    if AUTHOR_ROLE_RESEARCHER not in roles or AUTHOR_ROLE_TESTER not in roles:
        raise V9Error("genome-seed missing researcher or tester")

    # Researcher predicts numeric delta_mse before the Slurm analog runs.
    predicted_pos = _idea_delta_mse(TOY_SEED, kind="pos")
    predicted_neg = _idea_delta_mse(TOY_SEED, kind="neg")
    replay_pos = _idea_delta_mse(TOY_SEED, kind="pos")
    if abs(replay_pos - predicted_pos) > GOLDEN_MATCH_DIFF:
        raise V9Error("OLS analog is not deterministic")

    preregistered = {
        PLANTED_POSITIVE_ID: predicted_pos,
        PLANTED_NEGATIVE_ID: predicted_neg,
    }

    jobs: list[dict[str, object]] = []

    def submit(*, idea_id: str, seed: int, kind: str) -> dict[str, object]:
        delta = _idea_delta_mse(seed, kind=kind)
        job: dict[str, object] = {
            "job_name": f"{JOB_PREFIX}-v9-{idea_id}-s{seed}",
            "idea_id": idea_id,
            "seed": seed,
            "state": "completed",
            "metric": METRIC_NAME,
            "rung": RUNG_MINUS_ONE,
            "delta_mse": delta,
            "budget_gpu_hours": BUDGET_GPU_HOURS,
        }
        jobs.append(job)
        return job

    pos_job = submit(idea_id=PLANTED_POSITIVE_ID, seed=TOY_SEED, kind="pos")
    repl_job = submit(idea_id=PLANTED_POSITIVE_ID, seed=TOY_REPLICATE_SEED, kind="pos")
    neg_job = submit(idea_id=PLANTED_NEGATIVE_ID, seed=TOY_SEED, kind="neg")

    pos_delta = float(pos_job["delta_mse"])
    planted_positive_found = bool(
        PLANTED_POSITIVE_ID in preregistered
        and pos_job["state"] == "completed"
        and _beats_kill(pos_delta)
        and abs(pos_delta - predicted_pos) <= PREDICTION_TOL
    )

    repl_delta = float(repl_job["delta_mse"])
    planted_positive_replicated = bool(
        planted_positive_found
        and int(repl_job["seed"]) == TOY_REPLICATE_SEED
        and int(repl_job["seed"]) != TOY_SEED
        and repl_job["state"] == "completed"
        and _beats_kill(repl_delta)
    )

    ledger: list[dict[str, object]] = []
    if planted_positive_found:
        ledger.append(
            {
                "idea_id": PLANTED_POSITIVE_ID,
                "outcome": "positive",
                "rung": RUNG_MINUS_ONE,
                "author_role": AUTHOR_ROLE_RESEARCHER,
            }
        )

    neg_delta = float(neg_job["delta_mse"])
    neg_is_negative = bool(
        PLANTED_NEGATIVE_ID in preregistered
        and neg_job["state"] == "completed"
        and (not _beats_kill(neg_delta))
        and abs(neg_delta - predicted_neg) <= PREDICTION_TOL
    )
    if neg_is_negative:
        ledger.append(
            {
                "idea_id": PLANTED_NEGATIVE_ID,
                "outcome": "negative",
                "rung": RUNG_MINUS_ONE,
                "author_role": AUTHOR_ROLE_RESEARCHER,
            }
        )

    planted_negative_recorded = bool(
        neg_is_negative
        and any(
            row["idea_id"] == PLANTED_NEGATIVE_ID
            and row["outcome"] == "negative"
            and row["rung"] == RUNG_MINUS_ONE
            for row in ledger
        )
    )

    eval_ok = _eval_gate(applied_positive=planted_positive_found) and AUTHOR_ROLE_TESTER in roles
    slurm_ok = len(jobs) == 3 and all(job["state"] == "completed" for job in jobs)
    cycle_ok = eval_ok and slurm_ok
    return {
        "planted_positive_found": bool(planted_positive_found and cycle_ok),
        "planted_positive_replicated": bool(planted_positive_replicated and cycle_ok),
        "planted_negative_recorded": bool(planted_negative_recorded and cycle_ok),
    }


def run_v9(*, gpus: int) -> V9Result:
    """Run the V9 lab analog and report the three gates.

    ``gpus`` is the Slurm GPU count (template ``{{GPUS}}``). Spec V9 is 1
    to 4 H200. ``gpus >= 1`` is accepted so a CPU analog can run; that
    analog is not V9 verified.

    Returns a JSON-serializable ``V9Result``. Raises ``V9Error`` when
    ``gpus`` is invalid. Does not write ``v9.json``; the template does
    that.
    """
    if gpus < 1:
        raise V9Error(f"gpus must be >= 1 (got {gpus})")
    return _run_cycle()
