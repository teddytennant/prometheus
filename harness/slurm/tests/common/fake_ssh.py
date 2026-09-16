#!/usr/bin/env python3
"""Fake ssh/scp driven by files under $PROMETHEUS_SLURM_MOCK_DIR.

Used by prometheus-slurm integration tests. The mock cluster state lives in
the test process's temp dir; this helper is only a subprocess shim so the
production Client can keep spawning `ssh` (see tests/client.rs).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path

TAKES_ARG = {
    "-b",
    "-c",
    "-D",
    "-E",
    "-e",
    "-F",
    "-I",
    "-i",
    "-J",
    "-L",
    "-l",
    "-m",
    "-O",
    "-o",
    "-p",
    "-Q",
    "-R",
    "-S",
    "-W",
    "-w",
}


def parse_ssh_argv(argv: list[str]) -> tuple[str, list[str]]:
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--":
            i += 1
            break
        if a.startswith("-") and a != "-":
            key = a[:2] if len(a) >= 2 else a
            if a in TAKES_ARG:
                i += 2
                continue
            if key in TAKES_ARG and len(a) > 2:
                i += 1
                continue
            i += 1
            continue
        break
    host = argv[i] if i < len(argv) else ""
    rest = argv[i + 1 :] if i + 1 <= len(argv) else []
    return host, rest


def log(mock: Path, host: str, rest: list[str]) -> None:
    mock.mkdir(parents=True, exist_ok=True)
    rec = {
        "host": host,
        "remote": " ".join(rest),
        "argv": sys.argv[1:],
        "prog": Path(sys.argv[0]).name,
    }
    with (mock / "commands.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    with (mock / "remote.log").open("a", encoding="utf-8") as f:
        f.write(" ".join(rest) + "\n")


def snapshot_state(mock: Path) -> None:
    state = os.environ.get("PROMETHEUS_SLURM_STATE_DIR")
    if not state or not os.path.isdir(state):
        (mock / "state_at_sbatch.missing").write_text("no state dir\n")
        return
    dest = mock / "state_at_sbatch"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(state, dest)


def job_id_from_cmd(cmd: str) -> str | None:
    m = re.search(r"(?:-j|--job(?:id)?=)\s*(\d+)", cmd)
    if m:
        return m.group(1)
    m = re.search(r"(?:-j|--job(?:id)?)\s+(\d+)", cmd)
    if m:
        return m.group(1)
    nums = re.findall(r"\b(\d+)\b", cmd)
    return nums[-1] if nums else None


def maybe_hang(mock: Path, kind: str) -> None:
    flag = mock / "hang_queries"
    if not flag.exists():
        return
    kinds = flag.read_text(encoding="utf-8").strip()
    if kinds and kinds not in ("1", "true", "all"):
        if kind not in kinds.split(","):
            return
    time.sleep(3.0)


def handle_sbatch(mock: Path, cmd: str) -> int:
    snapshot_state(mock)
    mode = "ok"
    mode_path = mock / "sbatch_mode"
    if mode_path.exists():
        mode = mode_path.read_text(encoding="utf-8").strip() or "ok"
    if mode == "hang":
        time.sleep(3.0)
    if mode == "fail":
        sys.stderr.write("sbatch: simulated failure\n")
        return 1
    if mode == "panic":
        sys.stderr.write("sbatch: simulated panic\n")
        os.kill(os.getpid(), signal.SIGABRT)
        return 99
    jobid = "4242"
    jpath = mock / "sbatch_jobid"
    if jpath.exists():
        jobid = jpath.read_text(encoding="utf-8").strip() or jobid
    sys.stdout.write(jobid + "\n")
    return 0


def handle_squeue(mock: Path) -> int:
    maybe_hang(mock, "squeue")
    q = mock / "squeue_out"
    if q.exists():
        sys.stdout.write(q.read_text(encoding="utf-8"))
    return 0


def handle_sacct(mock: Path, cmd: str) -> int:
    maybe_hang(mock, "sacct")
    jid = job_id_from_cmd(cmd)
    if jid:
        p = mock / "states" / jid
        if p.exists():
            sys.stdout.write(p.read_text(encoding="utf-8").rstrip() + "\n")
            return 0
    d = mock / "default_state"
    if d.exists():
        sys.stdout.write(d.read_text(encoding="utf-8").rstrip() + "\n")
    return 0


def handle_scancel(mock: Path, rest: list[str], cmd: str) -> int:
    with (mock / "scancel.log").open("a", encoding="utf-8") as f:
        f.write(cmd + "\n")
    toks = cmd.split()
    if any(t == "-u" or t == "--user" or t.startswith("--user=") for t in toks):
        (mock / "scancel_u").write_text("yes\n")
    ids = [t for t in rest if re.fullmatch(r"\d+", t)]
    ids += re.findall(r"\b(\d+)\b", cmd)
    with (mock / "cancelled_ids").open("a", encoding="utf-8") as f:
        for i in ids:
            f.write(i + "\n")
    return 0


def main() -> int:
    mock_root = os.environ.get("PROMETHEUS_SLURM_MOCK_DIR")
    if not mock_root:
        sys.stderr.write("PROMETHEUS_SLURM_MOCK_DIR unset\n")
        return 2
    mock = Path(mock_root)
    prog = Path(sys.argv[0]).name
    if prog == "scp":
        log(mock, "", sys.argv[1:])
        return 0
    host, rest = parse_ssh_argv(sys.argv[1:])
    log(mock, host, rest)
    cmd = " ".join(rest)
    low = cmd.lower()
    if not cmd.strip():
        return 0
    if re.search(r"\bsbatch\b", low):
        return handle_sbatch(mock, cmd)
    if re.search(r"\bsqueue\b", low):
        return handle_squeue(mock)
    if re.search(r"\bsacct\b", low):
        return handle_sacct(mock, cmd)
    if re.search(r"\bscancel\b", low):
        return handle_scancel(mock, rest, cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
