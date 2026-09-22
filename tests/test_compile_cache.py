"""Implementer-facing tests for the XLA compile cache (spec 5.1).

These call production ``kernels.compile_cache``. The reference in
``tests.reference.compile_cache`` is an independent oracle (hashlib SHA-256,
atomic rename, self-describing records). Production must not import it.
Expected hashes are compared to that reference and to the NIST SHA-256 of
``b"abc"``.

Every test calls the stub and must fail. ``NotImplementedError`` is not
success. There is no differentiable math here, so there are no gradient
checks. The <10 min cold-start figure is a full-scale V-stage target, not a
CPU assertion; nothing in this file is marked ``gpu``.

Groups
------
- program_hash: NIST vector, reference parity, hex shape, empty / non-bytes.
- store / lookup: miss, bitwise hit, returned hash, idempotent one file,
  conflicting artifact leaves the first, empty artifact writes nothing.
- fault injection: wrong name, temp/partial sibling, truncated final name
  is a miss (not a short hit), other cache on the same root sees a full store.
- get_or_compile: one compile on miss, hit skips compile, exception
  propagates and stores nothing, empty artifact is CacheError.
- root: a file path raises, a missing directory is created.
- key: hash is of the program, not of the artifact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import kernels.compile_cache as compile_cache
from tests.reference import compile_cache as ref_cache

# NIST FIPS 180-2 / 180-4 SHA-256 of the three bytes b"abc". Not imported
# from the reference so this golden stays independent of that module.
NIST_SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def _files(root: Path) -> dict[str, bytes]:
    """Relative path -> bytes for every file under ``root`` (none if missing)."""
    if not root.exists():
        return {}
    found: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            found[str(path.relative_to(root))] = path.read_bytes()
    return found


def _assert_hex64(digest: str) -> None:
    assert type(digest) is str
    assert len(digest) == 64
    assert digest == digest.lower()
    assert set(digest) <= set("0123456789abcdef")


def test_program_hash_abc_is_nist_sha256() -> None:
    """program_hash(b"abc") is the NIST SHA-256, matching the reference."""
    got = compile_cache.program_hash(b"abc")
    _assert_hex64(got)
    assert got == NIST_SHA256_ABC
    assert got == ref_cache.program_hash(b"abc")
    assert got == ref_cache.NIST_SHA256_ABC


def test_program_hash_matches_reference_on_varied_bytes() -> None:
    """Hash is SHA-256 of the program bytes, including NULs and high bytes."""
    programs = [
        b"\x00",
        b"\x00\x00",
        b"a",
        b"abc",
        b" ",
        b"abc\x00def",
        bytes(range(256)),
        b"\xff" * 128,
        b"stablehlo" + bytes(1000),
    ]
    seen: set[str] = set()
    for program in programs:
        got = compile_cache.program_hash(program)
        _assert_hex64(got)
        assert got == ref_cache.program_hash(program)
        assert got not in seen
        seen.add(got)
    # Equal contents, freshly copied object: same hash. Not the artifact hash.
    copied = bytes(bytearray(b"abc"))
    assert copied == b"abc"
    assert compile_cache.program_hash(copied) == NIST_SHA256_ABC


def test_program_hash_empty_raises_cache_error() -> None:
    with pytest.raises(compile_cache.CacheError):
        compile_cache.program_hash(b"")


@pytest.mark.parametrize(
    "bad",
    ["abc", None, 0, bytearray(b"abc"), memoryview(b"abc")],
    ids=["str", "none", "int", "bytearray", "memoryview"],
)
def test_program_hash_non_bytes_raises_cache_error(bad: object) -> None:
    with pytest.raises(compile_cache.CacheError):
        compile_cache.program_hash(bad)  # type: ignore[arg-type]


def test_lookup_before_store_is_none(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = compile_cache.CompileCache(root)
    assert cache.lookup(b"abc") is None
    assert cache.lookup(b"\x00\x01") is None


def test_store_then_lookup_is_bitwise_equal_and_returns_hash(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = compile_cache.CompileCache(root)
    program = b"abc"
    artifact = b"compiled-xla-program"
    digest = cache.store(program, artifact)
    assert type(digest) is str
    assert digest == NIST_SHA256_ABC
    assert digest == compile_cache.program_hash(program)
    assert digest == ref_cache.program_hash(program)
    got = cache.lookup(program)
    assert type(got) is bytes
    assert got == artifact
    # Another cache on the same root sees the same bytes (not a copy that differs).
    other = compile_cache.CompileCache(root)
    other_got = other.lookup(program)
    assert type(other_got) is bytes
    assert other_got == artifact


def test_store_same_artifact_twice_is_one_file_same_hash(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = compile_cache.CompileCache(root)
    program = b"program-bytes"
    artifact = b"same-artifact"
    first = cache.store(program, artifact)
    second = cache.store(program, artifact)
    assert first == second
    assert first == compile_cache.program_hash(program)
    assert first == ref_cache.program_hash(program)
    children = list(root.iterdir())
    assert len(children) == 1
    assert children[0].is_file()
    assert children[0].name == first
    assert cache.lookup(program) == artifact


def test_store_different_artifact_raises_and_keeps_first(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = compile_cache.CompileCache(root)
    program = b"program-bytes"
    first_artifact = b"artifact-v1"
    digest = cache.store(program, first_artifact)
    names_before = sorted(p.name for p in root.iterdir())
    with pytest.raises(compile_cache.CacheError):
        cache.store(program, b"artifact-v2-different")
    assert sorted(p.name for p in root.iterdir()) == names_before
    assert names_before == [digest]
    assert cache.lookup(program) == first_artifact
    assert compile_cache.CompileCache(root).lookup(program) == first_artifact


def test_store_empty_compiled_raises_and_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = compile_cache.CompileCache(root)
    before = _files(root)
    with pytest.raises(compile_cache.CacheError):
        cache.store(b"program-bytes", b"")
    assert _files(root) == before
    assert cache.lookup(b"program-bytes") is None


def test_store_empty_program_raises_and_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = compile_cache.CompileCache(root)
    before = _files(root)
    with pytest.raises(compile_cache.CacheError):
        cache.store(b"", b"compiled-artifact")
    assert _files(root) == before


def test_lookup_empty_program_raises_cache_error(tmp_path: Path) -> None:
    cache = compile_cache.CompileCache(tmp_path / "cache")
    with pytest.raises(compile_cache.CacheError):
        cache.lookup(b"")


def test_lookup_ignores_wrong_name_and_temp_partial(tmp_path: Path) -> None:
    """A file not named the hash, and temp/partial names, are not a hit."""
    root = tmp_path / "cache"
    cache = compile_cache.CompileCache(root)
    program = b"abc"
    digest = compile_cache.program_hash(program)
    assert digest == NIST_SHA256_ABC
    (root / "not-the-hash").write_bytes(b"wrong-name-bytes")
    (root / digest.upper()).write_bytes(b"upper-hash-bytes")
    (root / f".{digest}.partial").write_bytes(b"partial-bytes-not-artifact")
    (root / f"{digest}.tmp").write_bytes(b"tmp-bytes-not-artifact")
    # Final name is absent. None of the siblings may be returned.
    assert cache.lookup(program) is None
    assert compile_cache.CompileCache(root).lookup(program) is None


def test_truncated_file_at_hash_name_is_miss_not_short_hit(tmp_path: Path) -> None:
    """A crash-truncated file at the final hash name is a miss.

    The final name is the program hash (docstring: write under
    ``program_hash``). Framing is not prescribed; whatever ``store`` wrote,
    a shortened file at that name must not come back as a short hit. A later
    store of the original artifact repairs it. A direct short plant, with no
    prior completed store, is also a miss.
    """
    root = tmp_path / "cache"
    cache = compile_cache.CompileCache(root)
    program = b"stablehlo-program"
    artifact = bytes(range(64))
    digest = cache.store(program, artifact)
    assert digest == compile_cache.program_hash(program)
    path = root / digest
    assert path.is_file()
    raw = path.read_bytes()
    assert len(raw) >= 8
    truncated = raw[:4]
    path.write_bytes(truncated)
    got = cache.lookup(program)
    assert got is None
    assert got != truncated
    assert got != artifact
    assert compile_cache.CompileCache(root).lookup(program) is None
    # Not a completed store, so storing the original artifact again succeeds.
    assert cache.store(program, artifact) == digest
    assert cache.lookup(program) == artifact

    other_program = b"other-stablehlo"
    other_digest = compile_cache.program_hash(other_program)
    planted = b"short"
    (root / other_digest).write_bytes(planted)
    assert cache.lookup(other_program) is None
    assert cache.lookup(other_program) != planted


def test_partial_sibling_is_ignored_when_final_store_exists(tmp_path: Path) -> None:
    """``.{hash}.partial`` and ``{hash}.tmp`` are not the stored artifact."""
    root = tmp_path / "cache"
    cache = compile_cache.CompileCache(root)
    program = b"abc"
    artifact = b"full-compiled-artifact"
    digest = cache.store(program, artifact)
    partial = b"partial-different-bytes"
    tmp = b"tmp-different-bytes"
    (root / f".{digest}.partial").write_bytes(partial)
    (root / f"{digest}.tmp").write_bytes(tmp)
    got = cache.lookup(program)
    assert type(got) is bytes
    assert got == artifact
    assert got != partial
    assert got != tmp
    other = compile_cache.CompileCache(root)
    other_got = other.lookup(program)
    assert type(other_got) is bytes
    assert other_got == artifact


def test_two_caches_same_directory_store_then_lookup(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    writer = compile_cache.CompileCache(root)
    reader = compile_cache.CompileCache(root)
    program = b"shared-program"
    artifact = b"shared-artifact-bytes"
    digest = writer.store(program, artifact)
    assert digest == compile_cache.program_hash(program)
    assert reader.lookup(program) == artifact
    # Opened after the store, not only a cache that already existed.
    late = compile_cache.CompileCache(root)
    assert late.lookup(program) == artifact


def test_get_or_compile_miss_calls_compile_once_then_hit_skips(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = compile_cache.CompileCache(root)
    program = b"to-compile"
    artifact = b"compiled-once"
    calls: list[bytes] = []

    def compile_fn(got: bytes) -> bytes:
        calls.append(got)
        return artifact

    assert cache.lookup(program) is None
    out, hit = cache.get_or_compile(program, compile_fn)
    assert calls == [program]
    assert type(out) is bytes
    assert out == artifact
    assert hit is False
    assert cache.lookup(program) == artifact
    # Reference agrees on the public result (separate directory, own callback).
    ref_calls: list[bytes] = []

    def ref_compile(got: bytes) -> bytes:
        ref_calls.append(got)
        return artifact

    ref = ref_cache.CompileCache(tmp_path / "ref")
    ref_out, ref_hit = ref.get_or_compile(program, ref_compile)
    assert ref_calls == [program]
    assert (out, hit) == (ref_out, ref_hit) == (artifact, False)

    def must_not_run(got: bytes) -> bytes:
        calls.append(got)
        raise AssertionError("compile must not run on a hit")

    out2, hit2 = cache.get_or_compile(program, must_not_run)
    assert calls == [program]
    assert out2 == artifact
    assert hit2 is True
    assert type(hit2) is bool


def test_get_or_compile_after_store_is_hit_and_does_not_compile(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = compile_cache.CompileCache(root)
    program = b"already-stored"
    artifact = b"pre-stored-artifact"
    cache.store(program, artifact)
    calls: list[bytes] = []

    def compile_fn(got: bytes) -> bytes:
        calls.append(got)
        return b"should-not-be-used"

    out, hit = cache.get_or_compile(program, compile_fn)
    assert calls == []
    assert out == artifact
    assert hit is True


def test_get_or_compile_compile_exception_propagates_and_stores_nothing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    cache = compile_cache.CompileCache(root)
    program = b"fails-to-compile"
    before = _files(root)
    err = RuntimeError("xla compiler failed")

    def compile_fn(got: bytes) -> bytes:
        assert got == program
        raise err

    with pytest.raises(RuntimeError) as caught:
        cache.get_or_compile(program, compile_fn)
    assert caught.value is err
    assert _files(root) == before
    assert cache.lookup(program) is None


def test_get_or_compile_empty_artifact_raises_and_stores_nothing(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = compile_cache.CompileCache(root)
    program = b"compiles-empty"
    calls: list[bytes] = []

    def compile_fn(got: bytes) -> bytes:
        calls.append(got)
        return b""

    before = _files(root)
    with pytest.raises(compile_cache.CacheError):
        cache.get_or_compile(program, compile_fn)
    assert calls == [program]
    assert _files(root) == before
    assert cache.lookup(program) is None


def test_root_that_is_a_file_raises_cache_error(tmp_path: Path) -> None:
    path = tmp_path / "not-a-directory"
    path.write_bytes(b"keep-me")
    with pytest.raises(compile_cache.CacheError):
        compile_cache.CompileCache(path)
    assert path.is_file()
    assert path.read_bytes() == b"keep-me"


def test_missing_directory_is_created(tmp_path: Path) -> None:
    root = tmp_path / "missing-cache"
    assert not root.exists()
    cache = compile_cache.CompileCache(root)
    assert root.is_dir()
    assert cache.lookup(b"abc") is None
    # Opening again does not require the directory to be absent.
    again = compile_cache.CompileCache(root)
    assert again.lookup(b"abc") is None


def test_hash_is_of_program_not_artifact(tmp_path: Path) -> None:
    """Two programs that compile to the same bytes have different hashes."""
    root = tmp_path / "cache"
    cache = compile_cache.CompileCache(root)
    artifact = b"identical-compiled-bytes"
    program_a = b"program-a"
    program_b = b"program-b"
    hash_a = cache.store(program_a, artifact)
    hash_b = cache.store(program_b, artifact)
    assert hash_a == compile_cache.program_hash(program_a)
    assert hash_b == compile_cache.program_hash(program_b)
    assert hash_a == ref_cache.program_hash(program_a)
    assert hash_b == ref_cache.program_hash(program_b)
    assert hash_a != hash_b
    assert hash_a != compile_cache.program_hash(artifact)
    assert hash_b != compile_cache.program_hash(artifact)
    assert cache.lookup(program_a) == artifact
    assert cache.lookup(program_b) == artifact
    names = sorted(p.name for p in root.iterdir() if p.is_file())
    assert names == sorted([hash_a, hash_b])

    calls: list[bytes] = []

    def compile_fn(got: bytes) -> bytes:
        calls.append(got)
        return artifact

    other = compile_cache.CompileCache(tmp_path / "other")
    out_a, hit_a = other.get_or_compile(program_a, compile_fn)
    out_b, hit_b = other.get_or_compile(program_b, compile_fn)
    assert out_a == out_b == artifact
    assert hit_a is False and hit_b is False
    assert calls == [program_a, program_b]
    assert compile_cache.program_hash(program_a) != compile_cache.program_hash(program_b)


def test_store_lookup_matches_reference(tmp_path: Path) -> None:
    """Public store/lookup results match the independent reference."""
    program = b"parity-program\x00\xff"
    artifact = b"parity-artifact"
    prod = compile_cache.CompileCache(tmp_path / "prod")
    reference = ref_cache.CompileCache(tmp_path / "ref")
    assert prod.lookup(program) is None
    assert prod.lookup(program) == reference.lookup(program)
    prod_hash = prod.store(program, artifact)
    ref_hash = reference.store(program, artifact)
    assert prod_hash == ref_hash == compile_cache.program_hash(program)
    assert prod.lookup(program) == reference.lookup(program) == artifact
