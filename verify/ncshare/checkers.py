"""Exit-criterion checkers for V0 to V10 (spec 16.2).

Each checker reads a result JSON written by the job, not an agent summary.
CPU stand-ins (identity mesh, numpy NCCL, tiny_standin soak) are rejected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class CheckError(Exception):
    pass


def load(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise CheckError("result is not an object")
    return data


def require_h200(r: dict[str, Any]) -> None:
    if r.get("tiny_standin"):
        raise CheckError("tiny_standin rejected")
    if r.get("standin"):
        raise CheckError("CPU stand-in rejected")
    name = str(r.get("gpu_name", ""))
    if "H200" not in name.upper():
        raise CheckError("not an H200 result")
    if not r.get("slurm_job_id"):
        raise CheckError("missing slurm_job_id")


def check_v0(r: dict[str, Any]) -> None:
    require_h200(r)
    backend = str(r.get("nccl_backend", "")).lower()
    if backend in ("", "numpy", "cpu"):
        raise CheckError("numpy/CPU NCCL stand-in")
    if int(r.get("n_devices", 0)) < 2:
        raise CheckError("NCCL all_reduce needs >=2 GPUs")
    if "bus_bandwidth_gbps" not in r:
        raise CheckError("missing bus_bandwidth_gbps")
    if float(r["bus_bandwidth_gbps"]) <= 0:
        raise CheckError("bandwidth not measured")
    if "kvm_present" not in r:
        raise CheckError("missing kvm_present")
    if not r.get("nccl_intra_ok"):
        raise CheckError("NCCL intra-node all_reduce failed")


def check_v1(r: dict[str, Any]) -> None:
    require_h200(r)
    if float(r.get("logit_max_abs_err", 1)) > 1e-5:
        raise CheckError("logits miss 1e-5")
    if not r.get("grad_check"):
        raise CheckError("grad check failed")
    if not r.get("overfit_one_batch"):
        raise CheckError("did not overfit one batch")
    if r.get("toy_embed"):
        raise CheckError("V1 toy embedding stand-in")
    if float(r.get("n_params") or 0) < 1e6:
        raise CheckError("V1 model not ~10M flagship shape")
    if not r.get("flagship_shape"):
        raise CheckError("V1 missing flagship discrete choices")


def check_v2(r: dict[str, Any]) -> None:
    require_h200(r)
    if r.get("identity_mesh"):
        raise CheckError("identity mesh rejected")
    if int(r.get("mesh_size", 1)) <= 1:
        raise CheckError("identity mesh rejected")
    if int(r.get("n_devices", 0)) < 2:
        raise CheckError("parallel equivalence needs >=2 GPUs")
    if float(r.get("relative_loss_err", 1)) > 1e-6:
        raise CheckError("parallel loss mismatch")
    if not r.get("routing_identical"):
        raise CheckError("routing not identical")
    if int(r.get("n_steps") or 0) < 200:
        raise CheckError("V2 too few steps")
    if int(r.get("ep") or 0) < 2:
        raise CheckError("V2 EP not sharded")


def check_v3(r: dict[str, Any]) -> None:
    require_h200(r)
    rel = float(r.get("fp8_vs_bf16_rel", 1))
    if rel > 0.005:
        raise CheckError("FP8 loss not within 0.5% of BF16")
    if "nvfp4_numerics_ok" not in r:
        raise CheckError("NVFP4 numerics missing")
    if int(r.get("n_steps") or 0) < 2000:
        raise CheckError("V3 too few steps")
    if float(r.get("n_params") or 0) < 1e6:
        raise CheckError("V3 is not the model")


def check_v4(r: dict[str, Any]) -> None:
    require_h200(r)
    if not r.get("resume_bitwise_equal"):
        raise CheckError("resume not bitwise-equal")
    if not r.get("sdc_caught_flip"):
        raise CheckError("SDC missed the flip")
    if not r.get("spike_skipped_shard"):
        raise CheckError("spike rollback did not skip shard")
    if int(r.get("n_param_leaves") or 0) < 8:
        raise CheckError("V4 ckpt is not a param tree")
    if int(r.get("ckpt_bytes") or 0) < 64:
        raise CheckError("V4 ckpt too small")


def check_v5(r: dict[str, Any]) -> None:
    require_h200(r)
    if r.get("tiny") or r.get("tiny_standin"):
        raise CheckError("V5 tiny stand-in")
    if int(r.get("n_devices") or 0) < 8:
        raise CheckError("V5 needs 8 GPUs")
    if float(r.get("tokens_seen") or 0) < 2e10:
        raise CheckError("V5 needs 20B tokens")
    jobs = r.get("ckpt_job_ids")
    if not isinstance(jobs, list) or len({str(j) for j in jobs}) < 2:
        raise CheckError("ckpt/resume across jobs failed")
    if not r.get("ckpt_resume_across_jobs"):
        raise CheckError("ckpt/resume across jobs failed")
    losses = r.get("losses") or []
    if not isinstance(losses, list) or len(losses) < 8:
        raise CheckError("V5 loss curve is two points, not a fit")
    if not r.get("loss_matches_ladder"):
        raise CheckError("loss curve missed ladder fit")
    n_params = float(r.get("active_params") or r.get("n_params") or 0)
    if n_params < 5e7:
        raise CheckError("rung0 too small")


def check_v6(r: dict[str, Any]) -> None:
    require_h200(r)
    if not r.get("no_collapse"):
        raise CheckError("curriculum collapsed")
    if not r.get("accuracy_rises_with_budget"):
        raise CheckError("latent budget did not help")
    if not r.get("thoughts_decode"):
        raise CheckError("thoughts did not decode")
    if not r.get("adapter_trained"):
        raise CheckError("V6 thought decode was not trained")
    if float(r.get("decode_loss_end") or 1) >= float(r.get("decode_loss_start") or 0):
        raise CheckError("V6 decode loss did not drop")


def check_v7(r: dict[str, Any]) -> None:
    require_h200(r)
    if not r.get("reward_rose"):
        raise CheckError("reward did not rise")
    if not r.get("drift_halted"):
        raise CheckError("log-prob drift did not halt")
    if not r.get("planted_write_flagged"):
        raise CheckError("planted test-file write not flagged")
    if not r.get("used_gspo"):
        raise CheckError("V7 did not use GSPO")


def check_v8(r: dict[str, Any]) -> None:
    require_h200(r)
    if float(r.get("logprob_max_abs_err", 1)) > 1e-3:
        raise CheckError("SGLang vs JAX log-probs off")
    if not r.get("tiered_restore_match"):
        raise CheckError("tiered restore mismatch")
    if not r.get("independent_ref"):
        raise CheckError("V8 serving path is not independent")


def check_v9(r: dict[str, Any]) -> None:
    require_h200(r)
    if not r.get("planted_positive_replicated"):
        raise CheckError("planted positive not replicated")
    if not r.get("planted_negative_recorded"):
        raise CheckError("planted negative not recorded")
    if int(r.get("ledger_rows") or 0) < 2:
        raise CheckError("V9 ledger empty")


def check_v10(r: dict[str, Any]) -> None:
    require_h200(r)
    if r.get("tiny_standin"):
        raise CheckError("tiny_standin rejected")
    if r.get("lost_tasks", 1) != 0:
        raise CheckError("lost tasks")
    if r.get("duplicated_outputs", 1) != 0:
        raise CheckError("duplicated outputs")
    if r.get("dead_tokens", 1) != 0:
        raise CheckError("dead tokens")
    hours = float(r.get("hours", 0))
    if hours < 72:
        raise CheckError("soak under 72h")
    if int(r.get("faults_injected") or 0) < 1:
        raise CheckError("V10 injected no faults")
    if int(r.get("tasks") or 0) < 1:
        raise CheckError("V10 ran no tasks")


CHECKERS = {
    0: check_v0,
    1: check_v1,
    2: check_v2,
    3: check_v3,
    4: check_v4,
    5: check_v5,
    6: check_v6,
    7: check_v7,
    8: check_v8,
    9: check_v9,
    10: check_v10,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--result", required=True)
    args = p.parse_args(argv)
    if args.stage not in CHECKERS:
        print(f"unknown stage {args.stage}", file=sys.stderr)
        return 2
    try:
        CHECKERS[args.stage](load(args.result))
    except CheckError as exc:
        print(f"V{args.stage} FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"V{args.stage} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
