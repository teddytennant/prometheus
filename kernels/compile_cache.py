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

from collections.abc import Callable
from pathlib import Path


class CacheError(ValueError):
    """Program bytes, artifact, or cache root violated the compile-cache contract."""


def program_hash(program: bytes) -> str:
    """SHA-256 hex of `program`. 64 lowercase hex characters.

    Empty `program` raises CacheError. A non-bytes value raises CacheError.
    The hash is of the program, not of the compiled artifact.
    """
    raise NotImplementedError("compile cache")


class CompileCache:
    """Directory of compiled artifacts keyed by `program_hash`.

    `root` is the shared-storage stand-in. It is created if missing. A path
    that exists and is not a directory raises CacheError.
    """

    def __init__(self, root: Path) -> None:
        raise NotImplementedError("compile cache")

    def lookup(self, program: bytes) -> bytes | None:
        """Artifact for `program`, or None if absent. Does not compile.

        A hit is bitwise equal to the bytes `store` wrote. A name that is
        not a completed store (a partial write, a temp file, a truncated
        file left by a crash) is a miss, not a hit of garbage. Empty
        `program` raises CacheError.
        """
        raise NotImplementedError("compile cache")

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
        raise NotImplementedError("compile cache")

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
        raise NotImplementedError("compile cache")
