"""Run one V-stage on an NCShare H200 allocation and write result JSON.

Refuses to run outside Slurm or without a GPU. Numpy NCCL, identity mesh,
and tiny_standin soaks are not produced here.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# GPU matmul defaults to TF32, which blows the V1 1e-5 numpy check.
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


def _gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"nvidia-smi failed: {exc}") from exc
    names = sorted({line.strip() for line in out.splitlines() if line.strip()})
    if not names:
        raise RuntimeError("nvidia-smi returned no GPU names")
    return ", ".join(names)


def h200_meta() -> dict:
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise RuntimeError("V stages must run under Slurm on NCShare H200s")
    name = _gpu_name()
    if "H200" not in name.upper():
        raise RuntimeError(f"not an H200: {name}")
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    backend = jax.default_backend()
    if backend != "gpu":
        raise RuntimeError(f"jax backend is {backend}, not gpu")
    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("no JAX GPU devices")
    return {
        "slurm_job_id": str(job),
        "gpu_name": name,
        "jax_backend": backend,
        "n_devices": len(devices),
        "node": os.environ.get("SLURMD_NODENAME", ""),
        "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "standin": False,
        "tiny_standin": False,
    }


def _stage0() -> dict:
    import jax
    import jax.numpy as jnp

    devices = jax.devices("gpu")
    n = len(devices)
    if n < 2:
        raise RuntimeError(f"V0 NCCL all_reduce needs >=2 GPUs, got {n}")
    # 256 MiB per rank, matching spec 16.2 / nccl-tests default payload.
    nbytes = 256 * 1024 * 1024
    elems = nbytes // 4
    x = jnp.ones((n, elems), dtype=jnp.float32)

    def _allreduce(v):
        return jax.lax.psum(v, axis_name="i")

    allreduce = jax.pmap(_allreduce, axis_name="i")
    y = allreduce(x)
    y.block_until_ready()
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        y = allreduce(x)
    y.block_until_ready()
    dt = max((time.perf_counter() - t0) / reps, 1e-9)
    # nccl-tests busbw for all_reduce: 2*(n-1)/n * size / time
    busbw = (2.0 * (n - 1) / n) * nbytes / dt / 1e9
    ok = bool(jnp.allclose(y, jnp.full((n, elems), float(n), dtype=jnp.float32)))
    return {
        "bus_bandwidth_gbps": float(busbw),
        "kvm_present": Path("/dev/kvm").exists(),
        "nccl_intra_ok": ok,
        "nccl_backend": "nccl",
        "venv_inside_job": True,
        "payload_bytes": nbytes,
        "allreduce_seconds": dt,
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
    from parallel import parallel_equivalence_gpu

    return parallel_equivalence_gpu()


def _stage3() -> dict:
    from kernels import precision_probe

    return precision_probe()


def _stage4() -> dict:
    from probes import fault_probe

    return fault_probe()


def _stage5() -> dict:
    from verify.ncshare.probes import v5_rung0

    return v5_rung0()


def _stage6() -> dict:
    from train.latent import stage_ab_probe

    return stage_ab_probe()


def _stage7() -> dict:
    from rl.loss import rl_end_to_end_probe

    return rl_end_to_end_probe()


def _stage8() -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sglang_fork", _ROOT / "sglang-fork" / "__init__.py"
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
    meta = h200_meta()
    result = STAGES[args.stage]()
    result = {**result, **meta}
    if args.stage != 10:
        result["tiny_standin"] = False
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
