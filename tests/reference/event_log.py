"""Independent SHA-256 reference for F2 payload_hash / event_hash.

Slow and obvious: `json.dumps` with sorted keys, then `hashlib.sha256`.
Must match `obs/tests/reference/mod.rs`. Cargo tests under `obs/tests/`
compare production `prometheus_obs` hashes to this algorithm.

Canonical JSON
--------------
- Compact separators `(',', ':')`, `sort_keys=True` (recursive).
- `ensure_ascii=False` so non-ASCII is UTF-8, not `\\uXXXX`.
- Integers have no decimal point (`1` not `1.0`).

payload_hash
------------
SHA-256 of those UTF-8 bytes, lowercase hex (64 chars).

event_hash
----------
SHA-256 of the UTF-8 concatenation::

    f"{seq}|{prev_hash}|{payload_hash}|{timestamp}|{event_type}"

where ``seq`` is the decimal integer with no leading zeros.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

GENESIS_HASH = "0" * 64

# Locked goldens (also embedded in obs/tests/reference/mod.rs).
GOLDEN_FIXED_PAYLOAD = {"msg": "hello", "n": 1, "ok": True}
GOLDEN_FIXED_PAYLOAD_HASH = (
    "2fa4793603fe17e52b611317579148d3b47e527b3f6f8163240c0d734ca6273c"
)
GOLDEN_FIXED_TIMESTAMP = "2026-09-16T12:00:00Z"
GOLDEN_FIXED_EVENT_TYPE = "session.start"
GOLDEN_FIXED_EVENT_HASH = (
    "167b92870604c9cb29fff1d41e47af8b8de79f7618939267b2e6b586e7616272"
)
GOLDEN_EMPTY_OBJECT_HASH = (
    "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
)
GOLDEN_UNICODE_PAYLOAD_HASH = (
    "e8d13b8f2f569be2e3ef0a2803761f39ed4d93cf167636d3e3c7a6d3c7562e00"
)


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


if __name__ == "__main__":
    ph = payload_hash(GOLDEN_FIXED_PAYLOAD)
    eh = event_hash(
        1, GENESIS_HASH, ph, GOLDEN_FIXED_TIMESTAMP, GOLDEN_FIXED_EVENT_TYPE
    )
    assert ph == GOLDEN_FIXED_PAYLOAD_HASH, ph
    assert eh == GOLDEN_FIXED_EVENT_HASH, eh
    assert payload_hash({}) == GOLDEN_EMPTY_OBJECT_HASH
    assert payload_hash({"note": "ΔRCI café"}) == GOLDEN_UNICODE_PAYLOAD_HASH
    print("event_log.py goldens ok")
