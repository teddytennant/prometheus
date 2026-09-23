"""Repo invariants CI enforces on top of ruff, pytest, rustfmt, clippy and cargo test.

    python3 .github/ci/checks.py tree           # run from the repo root
    python3 .github/ci/checks.py skips junit.xml

tree:  production code must not import tests/, and every crate must be a
       workspace member (a crate left out is never built, linted or tested).
skips: the only acceptable pytest skip on a CPU runner is a missing GPU. Any
       other skip means a test stopped running without anyone deciding it should.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

# Same patterns as prometheus-watch. Day one, the sglang fork served logits out
# of tests/reference and the suite passed because the oracle was the
# implementation.
_TEST_IMPORT_PATTERNS = {
    "*.py": r"^[[:space:]]*(from|import)[[:space:]]+tests([.[:space:]]|$)",
    "*.rs": r"(include!|#\[path[[:space:]]*=)[^)]*tests/",
    "*.toml": r'^[[:space:]]*path[[:space:]]*=[[:space:]]*"[^"]*tests/',
}

_GPU_SKIP = re.compile(r"\b(GPU|TPU)\b")


def _git_grep(pattern: str, glob: str) -> list[str]:
    out = subprocess.run(
        [
            "git",
            "grep",
            "-nE",
            pattern,
            "--",
            glob,
            ":(exclude)tests/**",
            ":(exclude)*/tests/**",
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode not in (0, 1):
        raise SystemExit(f"git grep failed: {out.stderr.strip()}")
    return [line for line in out.stdout.splitlines() if line]


def check_test_imports() -> list[str]:
    errors = []
    for glob, pattern in _TEST_IMPORT_PATTERNS.items():
        for hit in _git_grep(pattern, glob):
            errors.append(f"production code reaches into tests/: {hit}")
    return errors


def check_workspace_members() -> list[str]:
    root = tomllib.loads(Path("Cargo.toml").read_text())
    members = set(root.get("workspace", {}).get("members", []))
    exclude = set(root.get("workspace", {}).get("exclude", []))
    tracked = subprocess.run(
        ["git", "ls-files", "*Cargo.toml"], capture_output=True, text=True, check=True
    ).stdout.split()
    errors = []
    for manifest in tracked:
        crate = str(Path(manifest).parent)
        if crate == ".":
            continue
        if "/tests/" in f"/{crate}/":
            continue
        if crate not in members and crate not in exclude:
            errors.append(f"{manifest} is not a workspace member, so CI never builds it")
    for member in sorted(members):
        if not Path(member, "Cargo.toml").is_file():
            errors.append(f"workspace member {member} has no Cargo.toml")
    return errors


def check_skips(junit: Path) -> list[str]:
    errors = []
    for case in ET.parse(junit).iter("testcase"):
        skipped = case.find("skipped")
        if skipped is None:
            continue
        reason = skipped.get("message", "")
        if _GPU_SKIP.search(reason):
            continue
        name = f"{case.get('classname')}::{case.get('name')}"
        errors.append(f"unexpected skip {name}: {reason or '(no reason)'}")
    return errors


def main(argv: list[str]) -> int:
    if argv[:1] == ["tree"]:
        errors = check_test_imports() + check_workspace_members()
    elif len(argv) == 2 and argv[0] == "skips":
        errors = check_skips(Path(argv[1]))
    else:
        print(__doc__, file=sys.stderr)
        return 2
    for err in errors:
        print(f"::error::{err}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
