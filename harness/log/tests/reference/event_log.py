"""Independent Python twin of H2 hashing (hashlib). Not imported by production.

Canonical JSON: sort_keys, compact separators, ensure_ascii=False (UTF-8, not
\\uXXXX), integers without a decimal point.

payload_hash = SHA-256 of those UTF-8 bytes, lowercase hex.
event_hash = SHA-256 of utf-8 "{seq}|{prev_hash}|{payload_hash}|{timestamp}|{event_type}"
with seq as decimal and no leading zeros.

First log seq is 0; prev_hash of seq 0 is 64 zero hex chars.

The F1 golden file contracts/goldens/v1/event_log.default.json is a shape
fixture. Its payload_hash/hash strings are not used as hash oracles.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

GENESIS_PREV_HASH = "0" * 64

F1_TIMESTAMP = "2026-09-16T12:00:00Z"
F1_EVENT_TYPE = "session.start"


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def payload_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def event_hash(
    seq: int,
    prev_hash: str,
    payload_hash_hex: str,
    timestamp: str,
    event_type: str,
) -> str:
    concat = f"{seq}|{prev_hash}|{payload_hash_hex}|{timestamp}|{event_type}"
    return hashlib.sha256(concat.encode("utf-8")).hexdigest()


def f1_genesis_hashes() -> tuple[str, str]:
    ph = payload_hash({"cycle": 1})
    eh = event_hash(0, GENESIS_PREV_HASH, ph, F1_TIMESTAMP, F1_EVENT_TYPE)
    return ph, eh


if __name__ == "__main__":
    ph, eh = f1_genesis_hashes()
    assert ph == "9ea9926526cd16b40c70cd8988bfa74bb6c857dbb64faf55ea20a1c42d5ab122", ph
    assert eh == "c975ddd2d64d8781552891f9b1b5241208b703554c6f323a47336d2b5de5d74c", eh
    assert payload_hash({}) == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    assert (
        payload_hash({"msg": "hello", "n": 1, "ok": True})
        == "2fa4793603fe17e52b611317579148d3b47e527b3f6f8163240c0d734ca6273c"
    )
    assert (
        payload_hash({"note": "ΔRCI café"})
        == "e8d13b8f2f569be2e3ef0a2803761f39ed4d93cf167636d3e3c7a6d3c7562e00"
    )
    print("ok", ph, eh)
