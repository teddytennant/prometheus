"""Idempotent task queue for the V10 soak (spec 16.2). Write-once outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


class TaskQueue:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.out = self.root / "out"
        self.out.mkdir(parents=True, exist_ok=True)
        self.completed: set[str] = {p.stem for p in self.out.glob("*.json")}
        self.dup_prevented = 0
        self.lost = 0
        self.killed_tmp = 0

    def recover_incomplete(self) -> int:
        n = 0
        for tmp in self.out.glob("*.tmp"):
            tmp.unlink()
            n += 1
            self.killed_tmp += 1
        return n

    def run(self, task_id: str, fn: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], str]:
        dest = self.out / f"{task_id}.json"
        if dest.exists():
            self.dup_prevented += 1
            return json.loads(dest.read_text()), "dup_prevented"
        tmp = dest.with_suffix(".tmp")
        result = fn()
        tmp.write_text(json.dumps(result))
        tmp.replace(dest)
        self.completed.add(task_id)
        return result, "ok"

    def get(self, task_id: str) -> dict[str, Any] | None:
        dest = self.out / f"{task_id}.json"
        if not dest.exists():
            return None
        return json.loads(dest.read_text())
