"""Shared assertions for the contracts public API.

Imported by implementer-facing tests (against ``contracts``) and by
``tests/test_oracle_reference.py`` (against ``oracle_reference``). This
module is not collected by pytest.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = ROOT / "contracts" / "schemas"
GOLDEN_ROOT = ROOT / "contracts" / "goldens"
FIXTURE_ROOT = ROOT / "tests" / "fixtures"


def short_name(schema_id: str) -> str:
    return str(schema_id).removeprefix("prometheus.").replace(".", "_")


def pascal_name(schema_id: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in short_name(schema_id).split("_") if p)


def disk_schema_path(schema_id: str, version: int = 1) -> Path:
    return SCHEMA_ROOT / f"v{version}" / f"{short_name(schema_id)}.schema.json"


def disk_golden_path(schema_id: str, version: int = 1, name: str = "default") -> Path:
    return GOLDEN_ROOT / f"v{version}" / f"{short_name(schema_id)}.{name}.json"


def load_disk_schema(schema_id: str, version: int = 1) -> dict[str, Any]:
    return json.loads(disk_schema_path(schema_id, version).read_text(encoding="utf-8"))


def load_disk_golden(
    schema_id: str, version: int = 1, name: str = "default"
) -> dict[str, Any]:
    return json.loads(disk_golden_path(schema_id, version, name).read_text(encoding="utf-8"))


def first_required_field(schema: dict[str, Any], *skip: str) -> str:
    for key in schema.get("required", []):
        if key not in skip:
            return key
    raise AssertionError(f"no required field besides {skip} in {schema.get('$id')}")


def first_emptyable_field(schema: dict[str, Any]) -> str:
    """A required string-ish field that is empty-invalid (minLength or sha256)."""
    props = schema.get("properties", {})
    skip = {"schema_id", "schema_version"}
    for key in schema.get("required", []):
        if key in skip:
            continue
        spec = props.get(key, {})
        if spec.get("type") == "string":
            return key
        ref = spec.get("$ref", "")
        if ref.endswith("sha256") or ref.endswith("datetime"):
            return key
    for key, spec in props.items():
        if key in skip:
            continue
        if spec.get("type") == "string" or str(spec.get("$ref", "")).endswith("sha256"):
            return key
    raise AssertionError(f"no emptyable string field in {schema.get('$id')}")


def iter_pydantic_models(out_dir: Path) -> list[type]:
    """Import generated modules as a real package so relative imports work."""
    pkg_name = f"_contracts_gen_{abs(hash(str(out_dir)))}"
    init_path = out_dir / "__init__.py"
    if init_path.is_file():
        spec = importlib.util.spec_from_file_location(
            pkg_name,
            init_path,
            submodule_search_locations=[str(out_dir)],
        )
        if spec is None or spec.loader is None:
            raise AssertionError(f"cannot import generated package {out_dir}")
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = pkg
        spec.loader.exec_module(pkg)
        modules = [pkg]
        for py_path in sorted(out_dir.glob("*.py")):
            mod_name = f"{pkg_name}.{py_path.stem}"
            if mod_name in sys.modules:
                modules.append(sys.modules[mod_name])
                continue
            sub = importlib.util.spec_from_file_location(mod_name, py_path)
            if sub is None or sub.loader is None:
                raise AssertionError(f"cannot import generated module {py_path}")
            module = importlib.util.module_from_spec(sub)
            sys.modules[mod_name] = module
            sub.loader.exec_module(module)
            modules.append(module)
    else:
        modules = []
        for py_path in sorted(out_dir.rglob("*.py")):
            mod_name = f"{pkg_name}_{py_path.stem}"
            spec = importlib.util.spec_from_file_location(mod_name, py_path)
            if spec is None or spec.loader is None:
                raise AssertionError(f"cannot import generated module {py_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
            modules.append(module)

    models: list[type] = []
    for module in modules:
        for obj in vars(module).values():
            if not isinstance(obj, type):
                continue
            if obj.__name__ in {"BaseModel", "ConfigDict", "Field"}:
                continue
            if hasattr(obj, "model_validate") and hasattr(obj, "model_fields"):
                models.append(obj)
    return models


def parse_with_generated(models: list[type], payload: dict[str, Any]) -> None:
    errors: list[str] = []
    for model in models:
        try:
            model.model_validate(payload)
            return
        except Exception as exc:  # noqa: BLE001 — collect per-model failures
            errors.append(f"{model.__name__}: {exc}")
    preview = "; ".join(errors[:8]) if errors else "no pydantic models found"
    raise AssertionError(
        f"no generated model parsed payload schema_id={payload.get('schema_id')!r}: {preview}"
    )


def rust_sources(out_dir: Path) -> str:
    files = sorted(out_dir.rglob("*.rs"))
    if not files:
        raise AssertionError(f"generate_rust wrote no .rs files under {out_dir}")
    return "\n".join(path.read_text(encoding="utf-8") for path in files)


def assert_rust_has_schema_types(combined: str, schema_ids: list[Any]) -> None:
    if "Serialize" not in combined or "Deserialize" not in combined:
        raise AssertionError("generated Rust is missing serde Serialize/Deserialize derives")
    for schema_id in schema_ids:
        name = pascal_name(str(schema_id))
        if not re.search(rf"\b(struct|enum)\s+{name}\b", combined):
            raise AssertionError(f"generated Rust missing struct/enum {name}")
