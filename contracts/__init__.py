"""Versioned schemas for every module boundary.

Spec 15.4: every format that crosses a module boundary is a versioned schema
in this package. Schemas in ``contracts/schemas/`` generate the Python and
Rust types. Readers accept every past version. A schema change needs a
migration, a compatibility test, and a human sign-off.

Do not hand-write payload dataclasses on either side of a boundary. Call
``generate_python`` / ``generate_rust`` and import the emitted types.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

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


class ValidationError(Exception):
    """Payload does not match the schema named by its schema_id and schema_version."""


class UnknownSchemaError(Exception):
    """schema_id or schema_version is not in the registry."""


class MigrationError(Exception):
    """No migration path from the payload's version to the requested version."""


def schema_path(schema_id: str, version: int) -> Path:
    """Return the JSON Schema file for ``schema_id`` at ``version``.

    File layout: ``schemas/v{version}/{short_name}.schema.json`` where
    ``short_name`` is ``schema_id`` with the ``prometheus.`` prefix stripped
    and dots replaced by underscores.
    """
    raise NotImplementedError


def load_schema(schema_id: str, version: int) -> dict[str, Any]:
    """Load and return the JSON Schema document for ``schema_id`` at ``version``."""
    raise NotImplementedError


def validate(payload: Mapping[str, Any]) -> None:
    """Validate ``payload`` against the schema it names.

    ``payload`` must contain ``schema_id`` (a ``SchemaId`` value) and
    ``schema_version`` (a positive integer). Readers accept every past
    version that still has a schema file. Raises ``ValidationError`` on
    mismatch, ``UnknownSchemaError`` if the pair is not registered.
    """
    raise NotImplementedError


def migrate(payload: Mapping[str, Any], to_version: int) -> dict[str, Any]:
    """Return a new payload at ``to_version``, leaving ``payload`` unchanged.

    Identity when ``payload['schema_version'] == to_version``. Raises
    ``MigrationError`` if no path exists. v1 is the first version, so the
    only defined path at F1 is v1 -> v1.
    """
    raise NotImplementedError


def current_version(schema_id: str) -> int:
    """Highest schema_version that has a schema file for ``schema_id``."""
    raise NotImplementedError


def supported_versions(schema_id: str) -> list[int]:
    """Sorted list of schema_version values that can be loaded for ``schema_id``."""
    raise NotImplementedError


def load_golden(schema_id: str, version: int, name: str = "default") -> dict[str, Any]:
    """Load a golden payload from ``goldens/v{version}/{short_name}.{name}.json``."""
    raise NotImplementedError


def generate_python(out_dir: Path) -> None:
    """Emit Pydantic models for every schema into ``out_dir``.

    Generated modules are the only place payload types are defined. Callers
    import those modules; they do not hand-write dataclasses that mirror
    the schemas.
    """
    raise NotImplementedError


def generate_rust(out_dir: Path) -> None:
    """Emit serde structs for every schema into ``out_dir``.

    Generated files are the only place Rust payload types are defined.
    """
    raise NotImplementedError


def generate_all() -> None:
    """Generate Python types under ``contracts/generated`` and Rust types
    under ``crates/contracts/src/generated``.
    """
    raise NotImplementedError
