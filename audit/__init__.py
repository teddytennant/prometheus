"""WORM audit log: hash chain, query, export (spec 12, 15.5 E1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


def _digest(prev: str, body: str) -> str:
    h = hashlib.sha256()
    h.update(prev.encode())
    h.update(b"\n")
    h.update(body.encode())
    return h.hexdigest()


@dataclass
class AuditLog:
    entries: list[dict] = field(default_factory=list)
    last_hash: str = "0" * 64

    def append(self, event: dict) -> str:
        body = json.dumps(event, sort_keys=True, separators=(",", ":"))
        digest = _digest(self.last_hash, body)
        rec = {"event": event, "prev": self.last_hash, "hash": digest}
        self.entries.append(rec)
        self.last_hash = digest
        return digest

    def verify(self) -> bool:
        prev = "0" * 64
        for rec in self.entries:
            body = json.dumps(rec["event"], sort_keys=True, separators=(",", ":"))
            if rec["prev"] != prev:
                return False
            if _digest(prev, body) != rec["hash"]:
                return False
            prev = rec["hash"]
        return True

    def query(self, key: str, value) -> list[dict]:
        return [e for e in self.entries if e["event"].get(key) == value]
