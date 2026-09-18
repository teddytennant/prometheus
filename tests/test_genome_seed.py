"""Oracle tests for L4 ``harness/genome-seed`` (spec 14.3, 14.4, 15.5 L4).

Constants, dataclasses, and ``seed_root()`` (implemented as ``SEED_DIR``) may
pass against the committed stubs. Every test that calls ``load_seed``,
``role_names``, or ``require_roles`` must FAIL on the stub with
``NotImplementedError``. No GPU tests; V9 is later and blocked on F4.

Coverage
--------
seed_root: directory containing roles/, programs/, tools/, skills/; also
    memory/ and search/ (may be empty).
ROLES: director, researcher, implementer, reviewer, tester.
load_seed: five roles in that order via role_names; programs includes loop;
    tools/ and skills/ load as tuples (empty on the seed); bodies match the
    reference and the frozen goldens; ``root=`` reads that tree, not only
    SEED_DIR.
require_roles: ok on the seed; GenomeError if a required role is absent
    (constructed seed and a temp copy with a role file deleted).
GenomeError: load_seed on a temp copy missing a required role file.
edges: extra role markdown is loaded; non-markdown tools are ignored;
    empty role body is still a present role; missing root -> GenomeError.

Frozen goldens live in this file and were computed from
``tests.reference.genome_seed``, not from production.
"""

from __future__ import annotations

import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import genome_seed
import pytest

from tests.reference import genome_seed as ref

GOLDEN_ROLE_NAMES = (
    "director",
    "researcher",
    "implementer",
    "reviewer",
    "tester",
)

# Locked from tests.reference.genome_seed.load_seed on the committed seed tree.
GOLDEN_ROLE_BODIES = {
    "director": (
        "# Director\n"
        "\n"
        "Holds the goal and the budget. Assigns work to the other roles. Does not\n"
        "edit code or run evals itself. Talks to the kernel through the capability\n"
        "API only.\n"
    ),
    "researcher": (
        "# Researcher\n"
        "\n"
        "Reads the repo, the ledger, and public papers. Writes a plan the\n"
        "implementer can follow. Does not merge and does not call the eval-gate.\n"
    ),
    "implementer": (
        "# Implementer\n"
        "\n"
        "Writes code against the researcher's plan and the oracle tests. Does not\n"
        "edit tests. Does not promote weights or genomes.\n"
    ),
    "reviewer": (
        "# Reviewer\n"
        "\n"
        "Reads the implementer's diff against the spec. Picks one candidate or\n"
        "rejects all. A reject is final for that round.\n"
    ),
    "tester": (
        "# Tester\n"
        "\n"
        "Runs the CPU suite and the eval-gate through the kernel. Reports numbers\n"
        "from output it saw. Does not weaken tests.\n"
    ),
}

GOLDEN_LOOP_BODY = (
    "# Seed loop\n"
    "\n"
    "1. Director picks a step whose Needs are merged.\n"
    "2. Researcher writes the interface notes.\n"
    "3. Implementer writes the code.\n"
    "4. Reviewer accepts or rejects.\n"
    "5. Tester runs the suite through the kernel.\n"
    "6. Promote only on a pass.\n"
)


def _bodies(items: tuple[object, ...]) -> dict[str, str]:
    return {item.name: item.body for item in items}  # type: ignore[attr-defined]


def _copy_seed(tmp_path: Path) -> Path:
    dest = tmp_path / "seed"
    shutil.copytree(
        genome_seed.seed_root(),
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return dest


def _complete_seed(
    *,
    skip: str | None = None,
    extra: tuple[genome_seed.Role, ...] = (),
    root: Path | None = None,
) -> genome_seed.GenomeSeed:
    roles = tuple(
        genome_seed.Role(name=name, body=GOLDEN_ROLE_BODIES[name])
        for name in genome_seed.ROLES
        if name != skip
    )
    roles = roles + extra
    return genome_seed.GenomeSeed(
        roles=roles,
        programs=(genome_seed.Program(name="loop", body=GOLDEN_LOOP_BODY),),
        tools=(),
        skills=(),
        root=root if root is not None else genome_seed.SEED_DIR,
    )


# ---------------------------------------------------------------------------
# seed_root / layout / constants (may pass on the stub)
# ---------------------------------------------------------------------------
def test_roles_constant_is_spec_order() -> None:
    assert genome_seed.ROLES == GOLDEN_ROLE_NAMES
    assert genome_seed.ROLES == ref.ROLES


def test_genome_error_is_value_error() -> None:
    assert issubclass(genome_seed.GenomeError, ValueError)


def test_seed_types_are_frozen() -> None:
    role = genome_seed.Role(name="director", body="x")
    program = genome_seed.Program(name="loop", body="y")
    with pytest.raises(FrozenInstanceError):
        role.body = "z"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        program.body = "z"  # type: ignore[misc]


def test_seed_root_is_directory_with_roles_programs_tools_skills() -> None:
    root = genome_seed.seed_root()
    assert root == genome_seed.SEED_DIR
    assert root == ref.seed_root()
    assert root.is_dir()
    assert root.is_absolute()
    for name in ("roles", "programs", "tools", "skills"):
        assert (root / name).is_dir(), name
    for name in genome_seed.ROLES:
        assert (root / "roles" / f"{name}.md").is_file(), name
    assert (root / "programs" / "loop.md").is_file()


def test_tools_skills_memory_search_exist_and_may_be_empty() -> None:
    root = genome_seed.seed_root()
    for name in ("tools", "skills", "memory", "search"):
        path = root / name
        assert path.is_dir(), name
        markdown = [p for p in path.iterdir() if p.is_file() and p.suffix == ".md"]
        assert markdown == []


# ---------------------------------------------------------------------------
# load_seed
# ---------------------------------------------------------------------------
def test_load_seed_five_roles_in_spec_order_via_role_names() -> None:
    got = genome_seed.load_seed()
    want = ref.load_seed(root=genome_seed.seed_root())
    assert genome_seed.role_names(got) == GOLDEN_ROLE_NAMES
    assert genome_seed.role_names(got) == ref.role_names(want)
    assert _bodies(got.roles) == GOLDEN_ROLE_BODIES
    assert _bodies(got.roles) == _bodies(want.roles)
    assert isinstance(got, genome_seed.GenomeSeed)
    assert Path(got.root).resolve() == genome_seed.seed_root().resolve()
    assert all(isinstance(role, genome_seed.Role) for role in got.roles)


def test_load_seed_programs_includes_loop() -> None:
    got = genome_seed.load_seed()
    want = ref.load_seed(root=genome_seed.seed_root())
    assert "loop" in _bodies(got.programs)
    assert _bodies(got.programs)["loop"] == GOLDEN_LOOP_BODY
    assert _bodies(got.programs) == _bodies(want.programs)
    assert all(isinstance(program, genome_seed.Program) for program in got.programs)


def test_load_seed_tools_and_skills_are_empty_tuples_on_the_seed() -> None:
    got = genome_seed.load_seed()
    want = ref.load_seed(root=genome_seed.seed_root())
    assert got.tools == ()
    assert got.skills == ()
    assert _bodies(got.tools) == _bodies(want.tools) == {}
    assert _bodies(got.skills) == _bodies(want.skills) == {}


def test_load_seed_default_root_is_seed_root() -> None:
    got = genome_seed.load_seed()
    explicit = genome_seed.load_seed(root=genome_seed.seed_root())
    assert _bodies(got.roles) == _bodies(explicit.roles)
    assert _bodies(got.programs) == _bodies(explicit.programs)
    assert Path(got.root).resolve() == Path(explicit.root).resolve()
    assert Path(got.root).resolve() == genome_seed.seed_root().resolve()


def test_load_seed_root_temp_reads_given_tree_not_only_seed_dir(
    tmp_path: Path,
) -> None:
    dest = _copy_seed(tmp_path)
    mutated = "# Director\n\nmutated-for-oracle\n"
    (dest / "roles" / "director.md").write_text(mutated, encoding="utf-8")
    got = genome_seed.load_seed(root=dest)
    want = ref.load_seed(root=dest)
    assert _bodies(got.roles)["director"] == mutated
    assert _bodies(got.roles) == _bodies(want.roles)
    assert Path(got.root).resolve() == dest.resolve()
    default = genome_seed.load_seed()
    assert _bodies(default.roles)["director"] == GOLDEN_ROLE_BODIES["director"]
    assert "mutated-for-oracle" not in _bodies(default.roles)["director"]


@pytest.mark.parametrize("missing", GOLDEN_ROLE_NAMES)
def test_load_seed_genome_error_on_missing_required_role_file(tmp_path: Path, missing: str) -> None:
    dest = _copy_seed(tmp_path)
    (dest / "roles" / f"{missing}.md").unlink()
    with pytest.raises(genome_seed.GenomeError):
        genome_seed.load_seed(root=dest)


def test_load_seed_missing_root_is_genome_error(tmp_path: Path) -> None:
    with pytest.raises(genome_seed.GenomeError):
        genome_seed.load_seed(root=tmp_path / "no-such-seed")


def test_load_seed_extra_role_markdown(tmp_path: Path) -> None:
    dest = _copy_seed(tmp_path)
    extra_body = "# Watcher\n\nextra role\n"
    (dest / "roles" / "watcher.md").write_text(extra_body, encoding="utf-8")
    got = genome_seed.load_seed(root=dest)
    want = ref.load_seed(root=dest)
    assert _bodies(got.roles)["watcher"] == extra_body
    assert _bodies(got.roles) == _bodies(want.roles)
    names = genome_seed.role_names(got)
    assert tuple(n for n in names if n in genome_seed.ROLES) == genome_seed.ROLES
    assert names == ref.role_names(want)


def test_load_seed_ignores_non_markdown_in_tools(tmp_path: Path) -> None:
    dest = _copy_seed(tmp_path)
    (dest / "tools" / "skip.py").write_text("x = 1\n", encoding="utf-8")
    (dest / "tools" / "note.md").write_text("# Note\n", encoding="utf-8")
    got = genome_seed.load_seed(root=dest)
    want = ref.load_seed(root=dest)
    assert _bodies(got.tools) == {"note": "# Note\n"}
    assert _bodies(got.tools) == _bodies(want.tools)
    assert all(isinstance(tool, genome_seed.ToolSpec) for tool in got.tools)


def test_load_seed_empty_role_body_still_counts(tmp_path: Path) -> None:
    dest = _copy_seed(tmp_path)
    (dest / "roles" / "tester.md").write_text("", encoding="utf-8")
    got = genome_seed.load_seed(root=dest)
    assert _bodies(got.roles)["tester"] == ""
    assert genome_seed.role_names(got) == GOLDEN_ROLE_NAMES
    genome_seed.require_roles(got)


# ---------------------------------------------------------------------------
# role_names
# ---------------------------------------------------------------------------
def test_role_names_spec_order_on_constructed_seed() -> None:
    # Filesystem order would put implementer before researcher; spec order does not.
    shuffled = (
        genome_seed.Role(name="tester", body=""),
        genome_seed.Role(name="implementer", body=""),
        genome_seed.Role(name="director", body=""),
        genome_seed.Role(name="reviewer", body=""),
        genome_seed.Role(name="researcher", body=""),
    )
    seed = genome_seed.GenomeSeed(
        roles=shuffled,
        programs=(),
        tools=(),
        skills=(),
        root=genome_seed.SEED_DIR,
    )
    assert genome_seed.role_names(seed) == GOLDEN_ROLE_NAMES
    assert genome_seed.role_names(seed) == ref.role_names(seed)


def test_role_names_keeps_required_order_when_extras_present() -> None:
    seed = _complete_seed(extra=(genome_seed.Role(name="watcher", body="w"),))
    names = genome_seed.role_names(seed)
    assert tuple(n for n in names if n in genome_seed.ROLES) == GOLDEN_ROLE_NAMES
    assert names == ref.role_names(seed)


# ---------------------------------------------------------------------------
# require_roles
# ---------------------------------------------------------------------------
def test_require_roles_ok_on_the_seed() -> None:
    seed = genome_seed.load_seed()
    assert genome_seed.require_roles(seed) is None
    ref.require_roles(seed)


def test_require_roles_ok_on_constructed_complete_seed() -> None:
    seed = _complete_seed()
    assert genome_seed.require_roles(seed) is None


@pytest.mark.parametrize("missing", GOLDEN_ROLE_NAMES)
def test_require_roles_genome_error_on_missing_name(missing: str) -> None:
    seed = _complete_seed(skip=missing)
    with pytest.raises(genome_seed.GenomeError):
        genome_seed.require_roles(seed)


@pytest.mark.parametrize("missing", GOLDEN_ROLE_NAMES)
def test_require_roles_genome_error_if_role_file_missing_in_temp_copy(
    tmp_path: Path, missing: str
) -> None:
    dest = _copy_seed(tmp_path)
    (dest / "roles" / f"{missing}.md").unlink()
    with pytest.raises(genome_seed.GenomeError):
        genome_seed.load_seed(root=dest)
    incomplete = _complete_seed(skip=missing, root=dest)
    with pytest.raises(genome_seed.GenomeError):
        genome_seed.require_roles(incomplete)
