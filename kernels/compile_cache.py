"""XLA compile cache keyed by program content hash (spec 5.1).

Shared storage is a directory. The key is SHA-256 of the serialized program
(StableHLO or HLO bytes), not of the compiled artifact and not of Python
source. A hit returns the stored artifact and does not call the compiler.
A miss calls the compiler once, stores the artifact, and returns it.

A reader never observes a partial artifact. Two caches opened on the same
directory see each other's completed stores. The <10 min cold-start figure
is a full-scale V-stage target, not a CPU assertion.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

# magic + uint64 length + payload + SHA-256(payload). A 4-byte prefix or a
# planted short file cannot satisfy the length, so it is a miss, not a hit.
_MAGIC = b"CC01"
_HEADER = 4 + 8
_CHECKSUM = 32


class CacheError(ValueError):
    """Program bytes, artifact, or cache root violated the compile-cache contract."""


def program_hash(program: bytes) -> str:
    """SHA-256 hex of `program`. 64 lowercase hex characters.

    Empty `program` raises CacheError. A non-bytes value raises CacheError.
    The hash is of the program, not of the compiled artifact.
    """
    if not isinstance(program, bytes):
        raise CacheError(f"program must be bytes, got {type(program).__name__}")
    if len(program) == 0:
        raise CacheError("empty program")
    return hashlib.sha256(program).hexdigest()


def _encode(payload: bytes) -> bytes:
    return _MAGIC + len(payload).to_bytes(8, "big") + payload + hashlib.sha256(payload).digest()


def _decode(blob: bytes) -> bytes | None:
    """Payload of a completed record, or None if `blob` is truncated or garbage."""
    if len(blob) < _HEADER + _CHECKSUM or blob[:4] != _MAGIC:
        return None
    length = int.from_bytes(blob[4:12], "big")
    end = _HEADER + length
    if length <= 0 or len(blob) != end + _CHECKSUM:
        return None
    payload = blob[_HEADER:end]
    if hashlib.sha256(payload).digest() != blob[end : end + _CHECKSUM]:
        return None
    return payload


def _atomic_write(path: Path, data: bytes) -> None:
    """Publish `data` at `path` via same-directory rename.

    The temp name is not the hash, so a reader sees the previous complete file
    or the new one, never a partial write under the final name.
    """
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


class CompileCache:
    """Directory of compiled artifacts keyed by `program_hash`.

    `root` is the shared-storage stand-in. It is created if missing. A path
    that exists and is not a directory raises CacheError.
    """

    def __init__(self, root: Path) -> None:
        root = Path(root)
        if root.exists() and not root.is_dir():
            raise CacheError(f"cache root is not a directory: {root}")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root

    def lookup(self, program: bytes) -> bytes | None:
        """Artifact for `program`, or None if absent. Does not compile.

        A hit is bitwise equal to the bytes `store` wrote. A name that is
        not a completed store (a partial write, a temp file, a truncated
        file left by a crash) is a miss, not a hit of garbage. Empty
        `program` raises CacheError.
        """
        path = self.root / program_hash(program)
        if not path.is_file():
            return None
        return _decode(path.read_bytes())

    def store(self, program: bytes, compiled: bytes) -> str:
        """Write `compiled` under `program_hash(program)`. Return the hash.

        Empty `program` or empty `compiled` raises CacheError and writes
        nothing. The same program and the same artifact is idempotent and
        returns the same hash. The same program and a different artifact
        raises CacheError and leaves the first artifact in place.

        Does not return until another CompileCache opened on the same root
        would see the full artifact or nothing. No partial file is visible
        under the final name.
        """
        digest = program_hash(program)
        if not isinstance(compiled, bytes):
            raise CacheError(f"compiled must be bytes, got {type(compiled).__name__}")
        if len(compiled) == 0:
            raise CacheError("empty compiled artifact")
        path = self.root / digest
        if path.is_file():
            existing = _decode(path.read_bytes())
            if existing == compiled:
                return digest
            if existing is not None:
                raise CacheError("program already stored with a different artifact")
        _atomic_write(path, _encode(compiled))
        return digest

    def get_or_compile(
        self, program: bytes, compile: Callable[[bytes], bytes]
    ) -> tuple[bytes, bool]:
        """Return `(artifact, hit)`.

        `compile` is called only on a miss, exactly once, with `program`.
        The returned artifact is what `compile` returned, and a later
        `lookup` hits. If `compile` raises, nothing is stored and the
        exception propagates. If `compile` returns empty bytes, CacheError
        and nothing is stored. A hit does not call `compile`.
        """
        found = self.lookup(program)
        if found is not None:
            return found, True
        artifact = compile(program)
        if not isinstance(artifact, bytes) or len(artifact) == 0:
            raise CacheError("compile returned an empty artifact")
        self.store(program, artifact)
        return artifact, False
