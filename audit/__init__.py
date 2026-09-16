"""Tamper-evident audit log, WORM persist, thought probes (spec 12, 14.10, I9)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@dataclass
class AuditLog:
    entries: list[dict] = field(default_factory=list)
    last_hash: str = "0" * 64
    path: str | None = None

    def persist(self, path: str | Path) -> None:
        """Bind this log to an append-only file. Existing bytes are never rewritten."""
        self.path = str(path)
        p = Path(self.path)
        if p.exists() and p.stat().st_size > 0:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        for rec in self.entries:
            self._worm_write(rec)

    def _worm_write(self, rec: dict) -> None:
        if not self.path:
            return
        line = json.dumps(rec, sort_keys=True) + "\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode())
            os.fsync(fd)
        finally:
            os.close(fd)

    def append(self, event: dict) -> str:
        event = dict(event)
        event.setdefault("time", time.time())
        payload = json.dumps(event, sort_keys=True) + self.last_hash
        h = _sha(payload)
        rec = {"event": event, "hash": h, "prev": self.last_hash}
        self.entries.append(rec)
        self.last_hash = h
        self._worm_write(rec)
        return h

    def export(self) -> list[dict]:
        return json.loads(json.dumps(self.entries))

    def discrete_vs_latent(
        self,
        *,
        problem_id: str,
        discrete: str,
        latent: str,
        **extra: Any,
    ) -> str:
        return self.append(
            {
                "kind": "discrete_vs_latent",
                "problem_id": problem_id,
                "discrete": discrete,
                "latent": latent,
                "match": discrete == latent,
                **extra,
            }
        )

    def thought_decode(
        self,
        *,
        thought_id: str,
        decoded: str,
        gold: str | None = None,
        **extra: Any,
    ) -> str:
        rec: dict[str, Any] = {
            "kind": "thought_decode",
            "thought_id": thought_id,
            "decoded": decoded,
        }
        if gold is not None:
            rec["gold"] = gold
            rec["correct"] = decoded.strip() == str(gold).strip()
        rec.update(extra)
        return self.append(rec)

    def verify(self) -> bool:
        prev = "0" * 64
        for rec in self.entries:
            payload = json.dumps(rec["event"], sort_keys=True) + prev
            if rec["hash"] != _sha(payload) or rec["prev"] != prev:
                return False
            prev = rec["hash"]
        return prev == self.last_hash

    def query(
        self,
        key: str | None = None,
        value: object = None,
        *,
        kind: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict]:
        out = self.entries
        if key is not None:
            out = [e for e in out if e["event"].get(key) == value]
        if kind is not None:
            out = [e for e in out if e["event"].get("kind") == kind]
        if since is not None:
            out = [e for e in out if float(e["event"].get("time", 0.0)) >= since]
        if until is not None:
            out = [e for e in out if float(e["event"].get("time", 0.0)) <= until]
        return list(out)
