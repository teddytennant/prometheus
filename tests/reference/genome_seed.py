"""Independent seed-tree loader for L4 (spec 14.3, 14.4, 15.5 L4).

Slow and obvious: list each of roles/, programs/, tools/, skills/, keep
``*.md`` files, name = stem, body = UTF-8 text. No glob tricks, no caching.

Do not import production ``genome_seed`` from this file. Types here are a
local mirror so tests can compare field-by-field without sharing code.

Required roles (spec 14.4 order as frozen on the L4 interface)
--------------------------------------------------------------
director, researcher, implementer, reviewer, tester.

``load_seed`` reads Markdown under ``root`` (default: repo
``harness/genome-seed``). Missing required role -> ``GenomeError``.
``role_names`` returns present names with those five first, in that order,
then any extra role names in the order they appeared on the seed.
``require_roles`` checks the five names and does not look at programs.

tools/ and skills/ may be empty. memory/ and search/ are layout, not loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

# tests/reference/genome_seed.py -> repo root is parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_ROOT = _REPO_ROOT / "harness" / "genome-seed"

ROLES = (
    "director",
    "researcher",
    "implementer",
    "reviewer",
    "tester",
)


class GenomeError(ValueError):
    """Seed is missing a required role, program, or file."""


@dataclass(frozen=True)
class Role:
    name: str
    body: str


@dataclass(frozen=True)
class Program:
    name: str
    body: str


@dataclass(frozen=True)
class ToolSpec:
    name: str
    body: str


@dataclass(frozen=True)
class Skill:
    name: str
    body: str


@dataclass(frozen=True)
class GenomeSeed:
    """Immutable snapshot of the seed tree."""

    roles: tuple[Role, ...]
    programs: tuple[Program, ...]
    tools: tuple[ToolSpec, ...]
    skills: tuple[Skill, ...]
    root: Path


T = TypeVar("T")


def seed_root() -> Path:
    """Directory that holds roles/, programs/, tools/, skills/."""
    return DEFAULT_SEED_ROOT


def _read_markdown_dir(directory: Path, cls: type[T]) -> tuple[T, ...]:
    """Return one ``cls(name=stem, body=text)`` per ``*.md`` file, name-sorted."""
    if not directory.is_dir():
        return ()
    items: list[T] = []
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        if path.suffix != ".md":
            continue
        body = path.read_text(encoding="utf-8")
        items.append(cls(name=path.stem, body=body))  # type: ignore[call-arg]
    return tuple(items)


def _order_roles(roles: tuple[Role, ...]) -> tuple[Role, ...]:
    """Required names in ROLES order, then extras in the incoming order."""
    by_name = {role.name: role for role in roles}
    ordered: list[Role] = []
    seen: set[str] = set()
    for name in ROLES:
        role = by_name.get(name)
        if role is not None:
            ordered.append(role)
            seen.add(name)
    for role in roles:
        if role.name not in seen:
            ordered.append(role)
            seen.add(role.name)
    return tuple(ordered)


def role_names(seed: Any) -> tuple[str, ...]:
    """Names in spec 14.4 order (then extras, if any)."""
    present = [role.name for role in seed.roles]
    required = [name for name in ROLES if name in present]
    extras = [name for name in present if name not in ROLES]
    return tuple(required + extras)


def require_roles(seed: Any) -> None:
    """Raise GenomeError unless every name in ROLES is present."""
    present = {role.name for role in seed.roles}
    missing = [name for name in ROLES if name not in present]
    if missing:
        raise GenomeError("missing required role(s): " + ", ".join(missing))


def load_seed(root: Path | None = None) -> GenomeSeed:
    """Read Markdown files. Raises GenomeError if a required role is missing."""
    if root is None:
        root = DEFAULT_SEED_ROOT
    root = Path(root).resolve()
    if not root.is_dir():
        raise GenomeError(f"seed root is not a directory: {root}")
    roles = _order_roles(_read_markdown_dir(root / "roles", Role))
    programs = _read_markdown_dir(root / "programs", Program)
    tools = _read_markdown_dir(root / "tools", ToolSpec)
    skills = _read_markdown_dir(root / "skills", Skill)
    seed = GenomeSeed(
        roles=roles,
        programs=programs,
        tools=tools,
        skills=skills,
        root=root,
    )
    require_roles(seed)
    return seed


if __name__ == "__main__":
    loaded = load_seed()
    assert role_names(loaded) == ROLES, role_names(loaded)
    assert any(program.name == "loop" for program in loaded.programs)
    require_roles(loaded)
    print("genome_seed.py reference ok", loaded.root)
