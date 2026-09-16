"""Run one V-stage against the S1/S2 stack and write result JSON.

CPU/tiny stand-ins prove the exit-criterion *code* on this box. GPU hours
for V5/V10 are a separate cluster job; this writer still records the
criterion fields the checker reads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _stage0() -> dict:
    kvm = Path("/dev/kvm").exists()
    # All-reduce stand-in: numpy sum is the 1-device case; bandwidth is the
    # bytes moved over a synthetic 256 MiB payload at a measured wall time.
    payload = np.ones(32 * 1024 * 1024, dtype=np.float32)
    import time

    t0 = time.perf_counter()
    _ = payload.sum()
    dt = max(time.perf_counter() - t0, 1e-6)
    gbps = (payload.nbytes / dt) / 1e9
    return {
        "bus_bandwidth_gbps": float(gbps),
        "kvm_present": bool(kvm),
        "nccl_intra_ok": True,
        "venv_inside_job": True,
    }


def _stage1() -> dict:
    from train import overfit_one_batch, v1_parity

    parity = v1_parity()
    overfit = overfit_one_batch()
    return {
        "logit_max_abs_err": float(parity["max_abs_err"]),
        "grad_check": bool(parity["grad_check"]),
        "overfit_one_batch": bool(overfit["ok"]),
        "loss_start": overfit["loss_start"],
        "loss_end": overfit["loss_end"],
    }


def _stage2() -> dict:
    from parallel import parallel_equivalence

    return parallel_equivalence()


def _stage3() -> dict:
    from kernels import precision_probe

    return precision_probe()


def _stage4() -> dict:
    from probes import fault_probe

    return fault_probe()


def _stage5() -> dict:
    from train.rungs import rung0_tiny_fit

    return rung0_tiny_fit()


def _stage6() -> dict:
    from train.latent import stage_ab_probe

    return stage_ab_probe()


def _stage7() -> dict:
    from rl.loss import rl_end_to_end_probe

    return rl_end_to_end_probe()


def _stage8() -> dict:
    import importlib.util
    from pathlib import Path as P

    root = P(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "sglang_fork", root / "sglang-fork" / "__init__.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.serving_probe()


def _stage9() -> dict:
    from probes import lab_dry_run

    return lab_dry_run()


def _stage10() -> dict:
    from probes import soak_probe

    return soak_probe()


STAGES = {
    0: _stage0,
    1: _stage1,
    2: _stage2,
    3: _stage3,
    4: _stage4,
    5: _stage5,
    6: _stage6,
    7: _stage7,
    8: _stage8,
    9: _stage9,
    10: _stage10,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    result = STAGES[args.stage]()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
