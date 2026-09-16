"""S1/S2 tree completeness (spec 17). Mirrors verify/ncshare/check.sh trees."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TREES = [
    "contracts",
    "model",
    "train",
    "parallel",
    "kernels",
    "ckpt",
    "control",
    "data",
    "tokenizer",
    "synth",
    "serve",
    "sglang-fork",
    "verifiers",
    "rl/coordinator",
    "rl/loss",
    "rl/rewards",
    "rl/envs",
    "rl/tasks",
    "rl/multiagent",
    "arc",
    "evals",
    "audit",
    "obs",
    "harness/swarm",
    "harness/providers",
    "harness/slurm",
    "harness/ledger",
    "harness/pipeline",
    "harness/chaos",
    "harness/ops",
    "harness/kernel",
    "harness/eval-gate",
    "harness/monitors",
    "harness/genome-seed",
    "verify/ncshare",
    "data/extract",
    "data/loader",
    "data/dedup",
    "data/classifiers",
    "data/decontam",
    "data/mixture",
]


def test_required_trees_exist_and_have_source():
    missing = []
    empty = []
    for t in TREES:
        p = ROOT / t
        if not p.is_dir():
            missing.append(t)
            continue
        files = [
            f
            for f in p.rglob("*")
            if f.is_file() and f.suffix in {".py", ".rs", ".sh", ".md"} and "target" not in f.parts
        ]
        if not files:
            empty.append(t)
    assert missing == [], missing
    assert empty == [], empty


def test_v_templates_exist():
    for i in range(11):
        assert (ROOT / "verify" / "ncshare" / f"V{i}.sh").is_file()
    assert (ROOT / "verify" / "ncshare" / "checkers.py").is_file()
    assert (ROOT / "verify" / "ncshare" / "check.sh").is_file()
