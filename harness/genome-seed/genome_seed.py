"""Seed genome: roles, programs, tools, skills (spec 14.3, 14.4, 15.5 L4).

The genome is Markdown plus Python, versioned in git. This module loads
the seed checked in next to it. The lab grows it after L1; the kernel
promotes a new genome only through the capability API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

SEED_DIR = Path(__file__).resolve().parent

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


def seed_root() -> Path:
    """Directory that holds roles/, programs/, tools/, skills/."""
    return SEED_DIR


T = TypeVar("T")


def _read_markdown_dir(directory: Path, cls: type[T]) -> tuple[T, ...]:
    if not directory.is_dir():
        return ()
    items: list[T] = []
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
        if not path.is_file() or path.suffix != ".md":
            continue
        body = path.read_text(encoding="utf-8")
        items.append(cls(name=path.stem, body=body))  # type: ignore[call-arg]
    return tuple(items)


def _order_roles(roles: tuple[Role, ...]) -> tuple[Role, ...]:
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


def load_seed(root: Path | None = None) -> GenomeSeed:
    """Read Markdown files. Raises GenomeError if a required role is missing."""
    if root is None:
        root = SEED_DIR
    root = Path(root).resolve()
    if not root.is_dir():
        raise GenomeError(f"seed root is not a directory: {root}")
    seed = GenomeSeed(
        roles=_order_roles(_read_markdown_dir(root / "roles", Role)),
        programs=_read_markdown_dir(root / "programs", Program),
        tools=_read_markdown_dir(root / "tools", ToolSpec),
        skills=_read_markdown_dir(root / "skills", Skill),
        root=root,
    )
    require_roles(seed)
    return seed


def role_names(seed: GenomeSeed) -> tuple[str, ...]:
    """Names in spec 14.4 order."""
    present = [role.name for role in seed.roles]
    required = [name for name in ROLES if name in present]
    extras = [name for name in present if name not in ROLES]
    return tuple(required + extras)


def require_roles(seed: GenomeSeed) -> None:
    """Raise GenomeError unless every name in ROLES is present."""
    present = {role.name for role in seed.roles}
    missing = [name for name in ROLES if name not in present]
    if missing:
        raise GenomeError("missing required role(s): " + ", ".join(missing))
