"""Slow, obvious L3 monitors / kill-switch reference (Python twin).

Not prometheus-monitors. Production never imports this file.

Freeze file: JSON object with keys frozen, frozen_at, reason, violations.
Each violation is {boundary, at_ms, detail}. boundary is the serde unit-variant
name: Kernel, Graders, HeldOut, Monitors.

Absent freeze path => not frozen. Present corrupt file => fail closed.

Grader hashes: exact string equality. Unknown id is a mismatch.

NowMs is injected. This module does not read the wall clock.
"""

from __future__ import annotations

import json
import os
from typing import Any

BOUNDARIES = ["kernel", "graders", "held_out", "monitors"]

BOUNDARY_FROM_NAME = {
    "kernel": "Kernel",
    "graders": "Graders",
    "held_out": "HeldOut",
    "monitors": "Monitors",
}

FREEZE_REASON = {
    "Kernel": "kernel escape",
    "Graders": "grader hash mismatch",
    "HeldOut": "held-out access blocked",
    "Monitors": "monitor self-check failed",
}

TRIP_ERR = {
    "Kernel": ("Message", "kernel escape"),
    "Graders": ("GraderHashMismatch", None),
    "HeldOut": ("HeldOutAccess", None),
    "Monitors": ("MonitorSelfCheck", None),
}


def empty_snapshot() -> dict[str, Any]:
    return {
        "frozen": False,
        "frozen_at": None,
        "reason": None,
        "violations": [],
    }


def parse_boundary(name: str) -> str:
    if name in BOUNDARY_FROM_NAME:
        return BOUNDARY_FROM_NAME[name]
    raise ValueError(("UnknownBoundary", name))


class RefMonitors:
    def __init__(self, freeze_path: str, grader_hashes: dict[str, str]):
        self.freeze_path = freeze_path
        self.grader_hashes = dict(grader_hashes)
        self.snapshot = empty_snapshot()

    @classmethod
    def open(cls, freeze_path: str, grader_hashes: dict[str, str]) -> RefMonitors:
        m = cls(freeze_path, grader_hashes)
        m.snapshot = load_freeze(freeze_path)
        return m

    def frozen(self) -> bool:
        return bool(self.snapshot["frozen"])

    def note_grader_hash(self, grader_id: str, sha256_hex: str, now: int):
        expected = self.grader_hashes.get(grader_id)
        if expected is not None and expected == sha256_hex:
            return None
        return self.trip("Graders", grader_id, now)

    def note_held_out_access(self, who: str, now: int):
        return self.trip("HeldOut", who, now)

    def note_kernel_escape(self, capability: str, now: int):
        return self.trip("Kernel", capability, now)

    def note_self_check(self, detail: str, now: int):
        return self.trip("Monitors", detail, now)

    def plant(self, boundary: str, detail: str, now: int):
        return self.trip(boundary, detail, now)

    def kill(self, reason: str, now: int):
        if self.snapshot["frozen"]:
            return ("AlreadyFrozen", None)
        self._apply_kill(reason, now)
        return ("Ok", dict(self.snapshot))

    def trip(self, boundary: str, detail: str, now: int):
        self.snapshot["violations"].append(
            {"boundary": boundary, "at_ms": now, "detail": detail}
        )
        if not self.snapshot["frozen"]:
            self._apply_kill(FREEZE_REASON[boundary], now)
        else:
            write_freeze(self.freeze_path, self.snapshot)
        return TRIP_ERR[boundary]

    def _apply_kill(self, reason: str, now: int) -> None:
        self.snapshot["frozen"] = True
        self.snapshot["frozen_at"] = now
        self.snapshot["reason"] = reason
        write_freeze(self.freeze_path, self.snapshot)


def load_freeze(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return empty_snapshot()
    if os.path.isdir(path):
        raise ValueError("freeze path is a directory")
    with open(path, "rb") as f:
        raw = f.read()
    try:
        snap = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(str(e)) from e
    if not isinstance(snap, dict):
        raise ValueError("freeze file is not an object")
    for key in ("frozen", "frozen_at", "reason", "violations"):
        if key not in snap:
            raise ValueError(f"missing {key}")
    return snap


def write_freeze(path: str, snapshot: dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, separators=(",", ":"))
