from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_workspace_manifests_exist():
    assert (ROOT / "pyproject.toml").is_file()
    assert (ROOT / "Cargo.toml").is_file()
