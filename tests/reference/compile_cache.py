"""Independent reference for the XLA compile cache (spec 5.1).

Slow and obvious: ``hashlib.sha256`` of the program bytes, one file per
program in a directory, published with a same-directory atomic rename.
A completed record is self-describing (magic, length, payload, payload
checksum) so a truncated file at the final hash name is a miss, not a
short hit of garbage.

Production ``kernels.compile_cache`` must not import this module.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

# magic + uint64 big-endian payload length
_MAGIC = b"CC01"
_HEADER = 4 + 8
_CHECKSUM = 32

# NIST FIPS 180-4 / FIPS 180-2 SHA-256("abc"). Locked here so this file can
# be checked without importing production.
NIST_SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


class CacheError(ValueError):
    """Program bytes, artifact, or cache root violated the compile-cache contract."""


def program_hash(program: bytes) -> str:
    """SHA-256 hex of ``program``. 64 lowercase hex characters.

    Empty ``program`` raises CacheError. A non-bytes value raises CacheError.
    The hash is of the program, not of the compiled artifact.
    """
    if not isinstance(program, bytes):
        raise CacheError(f"program must be bytes, got {type(program).__name__}")
    if len(program) == 0:
        raise CacheError("empty program")
    return hashlib.sha256(program).hexdigest()


def _encode(payload: bytes) -> bytes:
    digest = hashlib.sha256(payload).digest()
    return _MAGIC + len(payload).to_bytes(8, "big") + payload + digest


def _decode(blob: bytes) -> bytes | None:
    """Payload of a completed record, or None if ``blob`` is partial or garbage."""
    if len(blob) < _HEADER + _CHECKSUM:
        return None
    if blob[:4] != _MAGIC:
        return None
    length = int.from_bytes(blob[4:12], "big")
    end = _HEADER + length
    if length <= 0 or len(blob) != end + _CHECKSUM:
        return None
    payload = blob[_HEADER:end]
    checksum = blob[end : end + _CHECKSUM]
    if hashlib.sha256(payload).digest() != checksum:
        return None
    return payload


def _atomic_write(path: Path, data: bytes) -> None:
    """Publish ``data`` at ``path`` via same-directory rename.

    A reader sees the previous complete file or the new complete file, never
    a partial write under the final name. The temp name is not the hash.
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
    """Directory of compiled artifacts keyed by ``program_hash``.

    ``root`` is created if missing. A path that exists and is not a directory
    raises CacheError. The final file name is the hex digest, nothing else.
    """

    def __init__(self, root: Path) -> None:
        root = Path(root)
        if root.exists() and not root.is_dir():
            raise CacheError(f"cache root is not a directory: {root}")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root

    def _final_path(self, digest: str) -> Path:
        return self.root / digest

    def lookup(self, program: bytes) -> bytes | None:
        """Artifact for ``program``, or None. Does not compile.

        Only a file named exactly ``program_hash(program)`` that decodes as a
        completed record is a hit. A temp name, a partial name, a wrong name,
        or a truncated file at the final name is a miss.
        """
        digest = program_hash(program)
        path = self._final_path(digest)
        if not path.is_file():
            return None
        return _decode(path.read_bytes())

    def store(self, program: bytes, compiled: bytes) -> str:
        """Write ``compiled`` under ``program_hash(program)``. Return the hash.

        Empty program or empty compiled raises CacheError and writes nothing.
        The same program and the same artifact is idempotent. The same program
        and a different completed artifact raises CacheError and leaves the
        first artifact in place. A truncated file at the final name is not a
        completed store, so a later store may replace it.
        """
        digest = program_hash(program)
        if not isinstance(compiled, bytes):
            raise CacheError(f"compiled must be bytes, got {type(compiled).__name__}")
        if len(compiled) == 0:
            raise CacheError("empty compiled artifact")
        path = self._final_path(digest)
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
        """Return ``(artifact, hit)``.

        ``compile`` is called only on a miss, exactly once, with ``program``.
        If ``compile`` raises, nothing is stored and the exception propagates.
        If ``compile`` returns empty bytes, CacheError and nothing is stored.
        A hit does not call ``compile``.
        """
        found = self.lookup(program)
        if found is not None:
            return found, True
        artifact = compile(program)
        if not isinstance(artifact, bytes) or len(artifact) == 0:
            raise CacheError("compile returned an empty artifact")
        self.store(program, artifact)
        return artifact, False


if __name__ == "__main__":
    import tempfile as _tempfile

    assert program_hash(b"abc") == NIST_SHA256_ABC
    try:
        program_hash(b"")
    except CacheError:
        pass
    else:
        raise SystemExit("empty program must raise")
    try:
        program_hash("abc")  # type: ignore[arg-type]
    except CacheError:
        pass
    else:
        raise SystemExit("non-bytes must raise")

    with _tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "cache"
        cache = CompileCache(root)
        assert root.is_dir()
        program = b"abc"
        artifact = b"compiled-artifact"
        assert cache.lookup(program) is None
        digest = cache.store(program, artifact)
        assert digest == NIST_SHA256_ABC
        assert cache.lookup(program) == artifact
        assert cache.store(program, artifact) == digest
        assert list(root.iterdir()) == [root / digest]
        other = CompileCache(root)
        assert other.lookup(program) == artifact
        try:
            cache.store(program, b"other-artifact")
        except CacheError:
            pass
        else:
            raise SystemExit("conflict must raise")
        assert cache.lookup(program) == artifact
        raw = (root / digest).read_bytes()
        (root / digest).write_bytes(raw[:4])
        assert cache.lookup(program) is None
        assert cache.store(program, artifact) == digest
        assert cache.lookup(program) == artifact
        (root / f".{digest}.partial").write_bytes(b"partial-not-artifact")
        (root / f"{digest}.tmp").write_bytes(b"tmp-not-artifact")
        assert cache.lookup(program) == artifact
        calls: list[bytes] = []

        def _compile(got: bytes) -> bytes:
            calls.append(got)
            return b"from-compile"

        other_program = b"other-program"
        out, hit = cache.get_or_compile(other_program, _compile)
        assert out == b"from-compile" and hit is False and calls == [other_program]
        out2, hit2 = cache.get_or_compile(other_program, _compile)
        assert out2 == b"from-compile" and hit2 is True and calls == [other_program]
        assert program_hash(program) != program_hash(other_program)
        third = b"third-program"
        assert cache.store(third, artifact) == program_hash(third)
        assert cache.lookup(third) == artifact
        assert program_hash(third) != program_hash(program)

    print("compile_cache reference ok")
