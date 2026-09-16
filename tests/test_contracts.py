"""Implementer-facing tests for the ``contracts`` public API.

These import production ``contracts`` (every function body is
``raise NotImplementedError`` today) and must FAIL against that stub.
They must PASS once the API is implemented. They do not import
``oracle_reference``.

Groups:
- Files on disk via the API: every SchemaId has v1 schema + default golden.
- validate() accepts goldens; rejects wrong const / missing / extra / hashes /
  datetimes / empty required strings.
- Unknown schema_id or version → UnknownSchemaError.
- migrate v1→v1 is identity (equal, not same object); v1→99 → MigrationError.
- current_version is 1; supported_versions is [1].
- load_schema / schema_path / load_golden round-trip the files on disk.
- generate_python / generate_rust emit usable types.
- Event-log genesis hashes; JSON round-trip property; readers accept v1.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import contracts
from tests.impl_contracts import (
    FIXTURE_ROOT,
    SCHEMA_ROOT,
    assert_rust_has_schema_types,
    disk_golden_path,
    disk_schema_path,
    first_emptyable_field,
    first_required_field,
    iter_pydantic_models,
    load_disk_golden,
    load_disk_schema,
    parse_with_generated,
    rust_sources,
)

SCHEMA_IDS = list(contracts.SchemaId)


@pytest.mark.parametrize("schema_id", SCHEMA_IDS)
def test_schema_path_and_load_schema_roundtrip(schema_id: str) -> None:
    path = contracts.schema_path(schema_id, 1)
    assert path.is_file()
    assert path.resolve() == disk_schema_path(schema_id).resolve()
    loaded = contracts.load_schema(schema_id, 1)
    assert loaded == load_disk_schema(schema_id)


@pytest.mark.parametrize("schema_id", SCHEMA_IDS)
def test_load_golden_roundtrip(schema_id: str) -> None:
    loaded = contracts.load_golden(schema_id, 1, "default")
    assert loaded == load_disk_golden(schema_id)
    on_disk = disk_golden_path(schema_id)
    assert on_disk.is_file()


@pytest.mark.parametrize("schema_id", SCHEMA_IDS)
def test_validate_accepts_golden(schema_id: str) -> None:
    payload = contracts.load_golden(schema_id, 1)
    assert contracts.validate(payload) is None


@pytest.mark.parametrize("schema_id", SCHEMA_IDS)
def test_validate_rejects_wrong_schema_id_const(schema_id: str) -> None:
    payload = dict(contracts.load_golden(schema_id, 1))
    other = next(s for s in SCHEMA_IDS if s.value != str(schema_id))
    payload["schema_id"] = other.value
    with pytest.raises(contracts.ValidationError):
        contracts.validate(payload)


@pytest.mark.parametrize("schema_id", SCHEMA_IDS)
def test_validate_rejects_missing_required_field(schema_id: str) -> None:
    payload = dict(contracts.load_golden(schema_id, 1))
    schema = contracts.load_schema(schema_id, 1)
    field = first_required_field(schema, "schema_id", "schema_version")
    del payload[field]
    with pytest.raises(contracts.ValidationError):
        contracts.validate(payload)


@pytest.mark.parametrize("schema_id", SCHEMA_IDS)
def test_validate_rejects_extra_property(schema_id: str) -> None:
    payload = dict(contracts.load_golden(schema_id, 1))
    payload["unexpected_extra_field"] = "nope"
    with pytest.raises(contracts.ValidationError):
        contracts.validate(payload)


@pytest.mark.parametrize("schema_id", SCHEMA_IDS)
def test_validate_rejects_empty_required_string(schema_id: str) -> None:
    payload = copy.deepcopy(contracts.load_golden(schema_id, 1))
    schema = contracts.load_schema(schema_id, 1)
    field = first_emptyable_field(schema)
    payload[field] = ""
    with pytest.raises(contracts.ValidationError):
        contracts.validate(payload)


def test_validate_rejects_bad_sha256() -> None:
    payload = dict(contracts.load_golden(contracts.SchemaId.TOKENIZER, 1))
    payload["vocab_hash"] = "not-a-sha256"
    with pytest.raises(contracts.ValidationError):
        contracts.validate(payload)
    payload["vocab_hash"] = "ABCDEF" * 10 + "ABCD"  # 64 chars, uppercase
    with pytest.raises(contracts.ValidationError):
        contracts.validate(payload)


def test_validate_rejects_bad_datetime() -> None:
    payload = dict(contracts.load_golden(contracts.SchemaId.TOKENIZER, 1))
    payload["frozen_at"] = "not-a-datetime"
    with pytest.raises(contracts.ValidationError):
        contracts.validate(payload)


def test_validate_unknown_schema_id() -> None:
    payload = {
        "schema_id": "prometheus.not_registered",
        "schema_version": 1,
    }
    with pytest.raises(contracts.UnknownSchemaError):
        contracts.validate(payload)


def test_validate_unknown_version() -> None:
    payload = dict(contracts.load_golden(contracts.SchemaId.PACKING, 1))
    payload["schema_version"] = 99
    with pytest.raises(contracts.UnknownSchemaError):
        contracts.validate(payload)


def test_validate_unknown_from_fixture() -> None:
    payload = json.loads((FIXTURE_ROOT / "unknown_schema.json").read_text(encoding="utf-8"))
    with pytest.raises(contracts.UnknownSchemaError):
        contracts.validate(payload)


@pytest.mark.parametrize("schema_id", SCHEMA_IDS)
def test_migrate_v1_is_identity_not_same_object(schema_id: str) -> None:
    payload = contracts.load_golden(schema_id, 1)
    migrated = contracts.migrate(payload, 1)
    assert migrated == payload
    assert migrated is not payload
    migrated["schema_version"] = 999
    assert payload["schema_version"] == 1


@pytest.mark.parametrize("schema_id", SCHEMA_IDS)
def test_migrate_to_99_raises(schema_id: str) -> None:
    payload = contracts.load_golden(schema_id, 1)
    with pytest.raises(contracts.MigrationError):
        contracts.migrate(payload, 99)


@pytest.mark.parametrize("schema_id", SCHEMA_IDS)
def test_current_version_is_1(schema_id: str) -> None:
    assert contracts.current_version(schema_id) == 1


@pytest.mark.parametrize("schema_id", SCHEMA_IDS)
def test_supported_versions_is_1(schema_id: str) -> None:
    assert contracts.supported_versions(schema_id) == [1]


def test_current_version_unknown_schema() -> None:
    with pytest.raises(contracts.UnknownSchemaError):
        contracts.current_version("prometheus.does_not_exist")


def test_schema_path_unknown_version() -> None:
    with pytest.raises(contracts.UnknownSchemaError):
        contracts.schema_path(contracts.SchemaId.TOKENIZER, 99)


def test_event_log_genesis_hashes() -> None:
    payload = contracts.load_golden(contracts.SchemaId.EVENT_LOG, 1)
    assert payload["seq"] == 0
    assert payload["prev_hash"] == "0" * 64
    hex_chars = set("0123456789abcdef")
    for key in ("hash", "payload_hash", "prev_hash"):
        value = payload[key]
        assert isinstance(value, str)
        assert len(value) == 64
        assert set(value) <= hex_chars
    assert contracts.validate(payload) is None


@pytest.mark.parametrize("schema_id", SCHEMA_IDS)
def test_json_roundtrip_still_validates(schema_id: str) -> None:
    payload = contracts.load_golden(schema_id, 1)
    dumped = json.loads(json.dumps(payload))
    assert dumped == payload
    assert contracts.validate(dumped) is None


def test_readers_accept_past_versions_v1_only() -> None:
    """Readers accept every past version (spec 15.4).

    With only v1 schema files on disk, validate of a v1 payload must work.
    A future v2 MUST add a compatibility test that still loads v1 goldens
    (and migrates v1 → v2) so old writers keep working.
    """
    versions = contracts.supported_versions(contracts.SchemaId.EVENT_LOG)
    assert versions == [1]
    assert (SCHEMA_ROOT / "v1").is_dir()
    assert not (SCHEMA_ROOT / "v2").exists()
    payload = contracts.load_golden(contracts.SchemaId.EVENT_LOG, 1)
    assert payload["schema_version"] == 1
    contracts.validate(payload)


def test_generate_python_parses_each_golden(tmp_path: Path) -> None:
    contracts.generate_python(tmp_path)
    py_files = list(tmp_path.rglob("*.py"))
    assert py_files, "generate_python must emit at least one Python module"
    models = iter_pydantic_models(tmp_path)
    assert models, "generate_python must emit Pydantic models with model_validate"
    for schema_id in SCHEMA_IDS:
        payload = contracts.load_golden(schema_id, 1)
        parse_with_generated(models, payload)


def test_generate_rust_emits_serde_types(tmp_path: Path) -> None:
    contracts.generate_rust(tmp_path)
    combined = rust_sources(tmp_path)
    assert_rust_has_schema_types(combined, SCHEMA_IDS)


def test_invalid_fixtures_rejected() -> None:
    extra = json.loads(
        (FIXTURE_ROOT / "tokenizer_extra_property.json").read_text(encoding="utf-8")
    )
    missing = json.loads(
        (FIXTURE_ROOT / "tokenizer_missing_required.json").read_text(encoding="utf-8")
    )
    bad_hash = json.loads(
        (FIXTURE_ROOT / "tokenizer_bad_sha256.json").read_text(encoding="utf-8")
    )
    bad_dt = json.loads(
        (FIXTURE_ROOT / "tokenizer_bad_datetime.json").read_text(encoding="utf-8")
    )
    empty = json.loads(
        (FIXTURE_ROOT / "tokenizer_empty_id.json").read_text(encoding="utf-8")
    )
    for payload in (extra, missing, bad_hash, bad_dt, empty):
        with pytest.raises(contracts.ValidationError):
            contracts.validate(payload)
