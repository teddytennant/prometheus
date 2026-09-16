"""SQLite experiment ledger (spec 14.6, 16.2 V9). Matches the Rust schema."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    experiment_id TEXT PRIMARY KEY NOT NULL,
    author_role TEXT NOT NULL,
    rung INTEGER,
    record_json TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def append(self, record: dict[str, Any]) -> None:
        eid = str(record["experiment_id"])
        role = str(record.get("author_role") or record.get("role") or "unknown")
        rung = record.get("rung")
        self.conn.execute(
            "INSERT OR REPLACE INTO records(experiment_id, author_role, rung, record_json) VALUES (?,?,?,?)",
            (eid, role, rung, json.dumps(record)),
        )
        self.conn.commit()

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT record_json FROM records WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])

    def close(self) -> None:
        self.conn.close()
