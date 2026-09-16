#!/usr/bin/env python3
"""Fake rsync for fetch() tests. Maps host:remote_root/... to $MOCK/remote/."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    mock_root = os.environ.get("PROMETHEUS_SLURM_MOCK_DIR")
    if not mock_root:
        return 0
    mock = Path(mock_root)
    mock.mkdir(parents=True, exist_ok=True)
    with (mock / "commands.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps({"prog": "rsync", "argv": sys.argv[1:], "remote": " ".join(sys.argv[1:])})
            + "\n"
        )
    argv = sys.argv[1:]
    positional: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-e", "--rsh") and i + 1 < len(argv):
            i += 2
            continue
        if a.startswith("-"):
            i += 1
            continue
        positional.append(a)
        i += 1
    if len(positional) < 2:
        return 0
    src, dest = positional[-2], positional[-1]
    if ":" in src:
        src = src.split(":", 1)[1]
    remote_root = os.environ.get("PROMETHEUS_SLURM_REMOTE_ROOT", "")
    rel = src
    if remote_root and src.startswith(remote_root):
        rel = src[len(remote_root) :].lstrip("/")
    real_src = mock / "remote" / rel
    dest_p = Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    if real_src.exists():
        if real_src.is_dir():
            shutil.copytree(real_src, dest_p, dirs_exist_ok=True)
        else:
            if dest.endswith("/"):
                dest_p.mkdir(parents=True, exist_ok=True)
                shutil.copy2(real_src, dest_p / real_src.name)
            else:
                shutil.copy2(real_src, dest_p)
    else:
        dest_p.mkdir(parents=True, exist_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
