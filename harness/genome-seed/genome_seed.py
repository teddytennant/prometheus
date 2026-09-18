"""Seed genome: roles, programs, tools, skills (spec 14.3, 14.4, 15.5 L4).

The genome is Markdown plus Python, versioned in git. This module loads
the seed checked in next to it. The lab grows it after L1; the kernel
promotes a new genome only through the capability API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def load_seed(root: Path | None = None) -> GenomeSeed:
    """Read Markdown files. Raises GenomeError if a required role is missing."""
    raise NotImplementedError("L4 load_seed")


def role_names(seed: GenomeSeed) -> tuple[str, ...]:
    """Names in spec 14.4 order."""
    raise NotImplementedError("L4 role_names")


def require_roles(seed: GenomeSeed) -> None:
    """Raise GenomeError unless every name in ROLES is present."""
    raise NotImplementedError("L4 require_roles")
