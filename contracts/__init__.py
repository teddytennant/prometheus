"""Versioned schemas for every module boundary.

Spec 15.4: every format that crosses a module boundary is a versioned schema
in this package. Schemas in ``contracts/schemas/`` generate the Python and
Rust types. Readers accept every past version. A schema change needs a
migration, a compatibility test, and a human sign-off.

Do not hand-write payload dataclasses on either side of a boundary. Call
``generate_python`` / ``generate_rust`` and import the emitted types.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from ._generate import write_python, write_rust

__all__ = [
    "CONTRACTS_ROOT",
    "GOLDEN_DIR",
    "SCHEMA_DIR",
    "MigrationError",
    "SchemaId",
    "UnknownSchemaError",
    "ValidationError",
    "current_version",
    "generate_all",
    "generate_python",
    "generate_rust",
    "load_golden",
    "load_schema",
    "migrate",
    "schema_path",
    "supported_versions",
    "validate",
]

CONTRACTS_ROOT = Path(__file__).resolve().parent
SCHEMA_DIR = CONTRACTS_ROOT / "schemas"
GOLDEN_DIR = CONTRACTS_ROOT / "goldens"


class SchemaId(StrEnum):
    """Stable names for every cross-module payload (spec 15.4)."""

    TOKENIZER = "prometheus.tokenizer"
    VOCAB = "prometheus.vocab"
    DATA_SHARD = "prometheus.data_shard"
    PACKING = "prometheus.packing"
    LOADER_STATE = "prometheus.loader_state"
    CHECKPOINT = "prometheus.checkpoint"
    WEIGHT_EXPORT = "prometheus.weight_export"
    ROLLOUT = "prometheus.rollout"
    TASK_SPEC = "prometheus.task_spec"
    REWARD_REQUEST = "prometheus.reward_request"
    REWARD_RESPONSE = "prometheus.reward_response"
    ENV_TOOL_REQUEST = "prometheus.env_tool_request"
    ENV_TOOL_RESPONSE = "prometheus.env_tool_response"
    EVAL_RESULT = "prometheus.eval_result"
    LEDGER_RECORD = "prometheus.ledger_record"
    EVENT_LOG = "prometheus.event_log"


class ValidationError(ValueError):
    """Raised when a payload does not match its schema."""


class UnknownSchemaError(KeyError):
    """Raised when ``schema_id`` or ``schema_version`` is not in the registry."""


class MigrationError(ValueError):
    """Raised when no migration path exists between two schema versions."""


def _short_name(schema_id: str) -> str:
    return str(schema_id).removeprefix("prometheus.")


def _coerce_schema_id(schema_id: str) -> SchemaId:
    try:
        return SchemaId(str(schema_id))
    except ValueError as exc:
        raise UnknownSchemaError(str(schema_id)) from exc


def _coerce_version(version: int) -> int:
    if isinstance(version, bool) or not isinstance(version, int):
        raise UnknownSchemaError(str(version))
    return version


def schema_path(schema_id: str, version: int) -> Path:
    """Return the JSON Schema path for ``schema_id`` at ``version``.

    Raises ``UnknownSchemaError`` if that pair is not on disk.
    """
    sid = _coerce_schema_id(schema_id)
    ver = _coerce_version(version)
    path = SCHEMA_DIR / f"v{ver}" / f"{_short_name(sid)}.schema.json"
    if not path.is_file():
        raise UnknownSchemaError(f"{sid}@v{ver}")
    return path


def load_schema(schema_id: str, version: int) -> dict[str, Any]:
    """Load and return the JSON Schema object for ``schema_id`` at ``version``."""
    path = schema_path(schema_id, version)
    return json.loads(path.read_text(encoding="utf-8"))


def _is_datetime(instance: object) -> bool:
    if not isinstance(instance, str):
        return True
    text = instance[:-1] + "+00:00" if instance.endswith("Z") else instance
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return False
    return True


@lru_cache(maxsize=1)
def _format_checker() -> FormatChecker:
    checker = Draft202012Validator.FORMAT_CHECKER
    if "date-time" in checker.checkers:
        return checker
    merged = FormatChecker()
    merged.checkers = dict(checker.checkers)
    merged.checkers["date-time"] = (_is_datetime, ())
    return merged


@cache
def _registry(version: int) -> Registry:
    version_dir = SCHEMA_DIR / f"v{version}"
    resources: list[tuple[str, Resource]] = []
    if version_dir.is_dir():
        for path in sorted(version_dir.glob("*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            schema_uri = doc.get("$id")
            if isinstance(schema_uri, str):
                resources.append((schema_uri, Resource.from_contents(doc)))
    return Registry().with_resources(resources)


def _payload_schema_ref(payload: Mapping[str, Any]) -> tuple[SchemaId, int]:
    if "schema_id" not in payload:
        raise ValidationError("missing schema_id")
    schema_id = payload["schema_id"]
    if not isinstance(schema_id, str):
        raise ValidationError("schema_id must be a string")
    try:
        sid = SchemaId(schema_id)
    except ValueError as exc:
        raise UnknownSchemaError(schema_id) from exc

    if "schema_version" not in payload:
        raise ValidationError("missing schema_version")
    version = payload["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValidationError("schema_version must be an integer")
    if version not in supported_versions(sid):
        raise UnknownSchemaError(f"{sid}@v{version}")
    return sid, version


def validate(payload: Mapping[str, Any]) -> None:
    """Raise ``ValidationError`` if ``payload`` does not match its schema.

    ``schema_id`` and ``schema_version`` select the schema. Unknown pairs raise
    ``UnknownSchemaError``.
    """
    if not isinstance(payload, Mapping):
        raise ValidationError("payload must be an object")
    sid, version = _payload_schema_ref(payload)
    schema = load_schema(sid, version)
    validator = Draft202012Validator(
        schema,
        registry=_registry(version),
        format_checker=_format_checker(),
    )
    errors = list(validator.iter_errors(payload))
    if errors:
        messages = "; ".join(error.message for error in errors)
        raise ValidationError(messages)


def migrate(payload: Mapping[str, Any], to_version: int) -> dict[str, Any]:
    """Return a new payload at ``to_version``, leaving ``payload`` unchanged.

    Identity when ``payload['schema_version'] == to_version``. Raises
    ``MigrationError`` if no path exists. v1 is the first version, so the
    only defined path at F1 is v1 -> v1.
    """
    if not isinstance(payload, Mapping):
        raise ValidationError("payload must be an object")
    from_version = payload.get("schema_version")
    if isinstance(from_version, bool) or not isinstance(from_version, int):
        raise ValidationError("schema_version must be an integer")
    if from_version == to_version:
        return copy.deepcopy(dict(payload))
    raise MigrationError(f"no migration from v{from_version} to v{to_version}")


def current_version(schema_id: str) -> int:
    """Highest schema_version that has a schema file for ``schema_id``."""
    return max(supported_versions(schema_id))


def supported_versions(schema_id: str) -> list[int]:
    """Sorted list of schema_version values that can be loaded for ``schema_id``."""
    sid = _coerce_schema_id(schema_id)
    short = _short_name(sid)
    found: list[int] = []
    if SCHEMA_DIR.is_dir():
        for child in SCHEMA_DIR.iterdir():
            if not child.is_dir() or not child.name.startswith("v"):
                continue
            suffix = child.name[1:]
            if not suffix.isdigit():
                continue
            if (child / f"{short}.schema.json").is_file():
                found.append(int(suffix))
    if not found:
        raise UnknownSchemaError(str(sid))
    return sorted(found)


def load_golden(schema_id: str, version: int, name: str = "default") -> dict[str, Any]:
    """Load a golden payload from ``goldens/v{version}/{short_name}.{name}.json``."""
    sid = _coerce_schema_id(schema_id)
    ver = _coerce_version(version)
    path = GOLDEN_DIR / f"v{ver}" / f"{_short_name(sid)}.{name}.json"
    if not path.is_file():
        raise UnknownSchemaError(f"{sid}@v{ver}/{name}")
    return json.loads(path.read_text(encoding="utf-8"))


def generate_python(out_dir: Path) -> None:
    """Emit Pydantic models for every schema into ``out_dir``.

    Generated modules are the only place payload types are defined. Callers
    import those modules; they do not hand-write dataclasses that mirror
    the schemas.
    """
    write_python(Path(out_dir))


def generate_rust(out_dir: Path) -> None:
    """Emit serde structs for every schema into ``out_dir``.

    Generated files are the only place Rust payload types are defined.
    """
    write_rust(Path(out_dir))


def generate_all() -> None:
    """Generate Python types under ``contracts/generated`` and Rust types
    under ``crates/contracts/src/generated``.
    """
    generate_python(CONTRACTS_ROOT / "generated")
    generate_rust(CONTRACTS_ROOT.parent / "crates" / "contracts" / "src" / "generated")
