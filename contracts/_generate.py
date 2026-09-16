"""Emit Pydantic and serde types from the versioned JSON Schemas."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"

_OBJECT_DEFS = ("mesh", "artifact", "provenance", "decontamination")
_RUST_KEYWORDS = {
    "as",
    "async",
    "await",
    "break",
    "const",
    "continue",
    "crate",
    "dyn",
    "else",
    "enum",
    "extern",
    "false",
    "fn",
    "for",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "match",
    "mod",
    "move",
    "mut",
    "pub",
    "ref",
    "return",
    "self",
    "Self",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "type",
    "unsafe",
    "use",
    "where",
    "while",
}


def _pascal(name: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", name)
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def _ref_def(ref: str) -> str | None:
    marker = "#/$defs/"
    if marker in ref:
        return ref.split(marker, 1)[1]
    return None


@dataclass
class IrType:
    kind: str
    literals: tuple[Any, ...] = ()
    model: str | None = None
    inner: IrType | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    nullable: bool = False


@dataclass
class IrField:
    name: str
    typ: IrType
    required: bool


@dataclass
class IrModel:
    name: str
    extra_forbid: bool
    fields: list[IrField]


class Builder:
    def __init__(self, defs: dict[str, Any]) -> None:
        self.defs = defs
        self.models: dict[str, IrModel] = {}
        self.order: list[str] = []

    def model(self, name: str, schema: dict[str, Any]) -> str:
        if name in self.models:
            return name
        self.models[name] = IrModel(name=name, extra_forbid=False, fields=[])
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        extra_forbid = schema.get("additionalProperties", True) is False
        fields: list[IrField] = []
        for fname, fschema in properties.items():
            if not isinstance(fschema, dict):
                fschema = {}
            hint = f"{name}{_pascal(fname)}"
            typ = self.typ(fschema, hint)
            fields.append(IrField(name=fname, typ=typ, required=fname in required))
        self.models[name] = IrModel(name=name, extra_forbid=extra_forbid, fields=fields)
        self.order.append(name)
        return name

    def typ(self, schema: dict[str, Any], hint: str) -> IrType:
        if "$ref" in schema:
            return self._ref(str(schema["$ref"]), hint)
        if "const" in schema:
            return IrType(kind="literal", literals=(schema["const"],))
        if "enum" in schema:
            return IrType(kind="literal", literals=tuple(schema["enum"]))

        raw_type = schema.get("type")
        nullable = False
        json_type: str | None
        if isinstance(raw_type, list):
            nullable = "null" in raw_type
            non_null = [t for t in raw_type if t != "null"]
            json_type = non_null[0] if len(non_null) == 1 else None
        else:
            json_type = raw_type if isinstance(raw_type, str) else None

        constraints: dict[str, Any] = {}
        if "minLength" in schema:
            constraints["min_length"] = schema["minLength"]
        if "maxLength" in schema:
            constraints["max_length"] = schema["maxLength"]
        if "minimum" in schema:
            constraints["ge"] = schema["minimum"]
        if "maximum" in schema:
            constraints["le"] = schema["maximum"]
        if "pattern" in schema:
            constraints["pattern"] = schema["pattern"]

        if json_type == "array" or "items" in schema:
            items = schema.get("items")
            if isinstance(items, dict):
                inner = self.typ(items, f"{hint}Item")
            else:
                inner = IrType(kind="any")
            list_constraints: dict[str, Any] = {}
            if "minItems" in schema:
                list_constraints["min_length"] = schema["minItems"]
            return IrType(
                kind="list", inner=inner, constraints=list_constraints, nullable=nullable
            )

        if json_type == "object" or "properties" in schema or "additionalProperties" in schema:
            properties = schema.get("properties")
            additional = schema.get("additionalProperties", True)
            if properties:
                return IrType(kind="model", model=self.model(hint, schema), nullable=nullable)
            if isinstance(additional, dict):
                inner = self.typ(additional, f"{hint}Value")
                return IrType(kind="dict", inner=inner, nullable=nullable)
            return IrType(kind="dict", inner=IrType(kind="any"), nullable=nullable)

        kind = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
        }.get(json_type or "", "any")
        return IrType(kind=kind, constraints=constraints, nullable=nullable)

    def _ref(self, ref: str, hint: str) -> IrType:
        name = _ref_def(ref)
        if name in _OBJECT_DEFS and name in self.defs:
            return IrType(kind="model", model=self.model(_pascal(name), self.defs[name]))
        if name == "sha256":
            return IrType(kind="str", constraints={"pattern": r"^[0-9a-f]{64}$"})
        if name in {"datetime", "date"}:
            return IrType(kind="str")
        if name == "pos_int":
            return IrType(kind="int", constraints={"ge": 1})
        if name == "nonneg_int":
            return IrType(kind="int", constraints={"ge": 0})
        if name == "dtype" and name in self.defs:
            return IrType(kind="literal", literals=tuple(self.defs[name].get("enum") or ()))
        if name and name in self.defs:
            return self.typ(self.defs[name], hint)
        return IrType(kind="any")


def _load_latest() -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    versions = []
    for child in _SCHEMA_DIR.iterdir():
        if child.is_dir() and child.name.startswith("v") and child.name[1:].isdigit():
            versions.append(int(child.name[1:]))
    if not versions:
        raise FileNotFoundError(f"no schema versions under {_SCHEMA_DIR}")
    version = max(versions)
    vdir = _SCHEMA_DIR / f"v{version}"
    defs_doc = json.loads((vdir / "_defs.schema.json").read_text(encoding="utf-8"))
    defs = defs_doc.get("$defs") or {}
    schemas: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(vdir.glob("*.schema.json")):
        if path.name.startswith("_"):
            continue
        short = path.name.removesuffix(".schema.json")
        schemas.append((short, json.loads(path.read_text(encoding="utf-8"))))
    return defs, schemas


def build_ir() -> list[IrModel]:
    defs, schemas = _load_latest()
    builder = Builder(defs)
    for name in _OBJECT_DEFS:
        if name in defs:
            builder.model(_pascal(name), defs[name])
    for short, schema in schemas:
        builder.model(_pascal(short), schema)
    return [builder.models[name] for name in builder.order if name in builder.models]


def _py_type(typ: IrType) -> str:
    expr: str
    if typ.kind == "literal":
        inner = ", ".join(repr(v) for v in typ.literals)
        expr = f"Literal[{inner}]"
    elif typ.kind == "model":
        expr = typ.model or "Any"
    elif typ.kind == "list":
        expr = f"list[{_py_type(typ.inner) if typ.inner else 'Any'}]"
    elif typ.kind == "dict":
        expr = f"dict[str, {_py_type(typ.inner) if typ.inner else 'Any'}]"
    elif typ.kind == "str":
        expr = "str"
    elif typ.kind == "int":
        expr = "int"
    elif typ.kind == "float":
        expr = "float"
    elif typ.kind == "bool":
        expr = "bool"
    else:
        expr = "Any"
    if typ.nullable:
        expr = f"{expr} | None"
    return expr


def _py_field(field: IrField) -> str:
    typ = field.typ
    expr = _py_type(typ)
    constraints = dict(typ.constraints)
    if not field.required:
        expr = f"{expr} | None" if not typ.nullable else expr
    parts: list[str] = []
    for key in ("min_length", "max_length", "ge", "le", "pattern"):
        if key in constraints:
            value = constraints[key]
            rendered = f"{key}={value!r}" if key == "pattern" else f"{key}={value}"
            parts.append(rendered)
    if not field.required:
        parts.append("default=None")
    if parts:
        return f"{field.name}: {expr} = Field({', '.join(parts)})"
    return f"{field.name}: {expr}"


def emit_python_source(models: list[IrModel]) -> str:
    lines = [
        '"""Auto-generated Prometheus contract types. Do not edit."""',
        "from __future__ import annotations",
        "",
        "from typing import Any, Literal",
        "",
        "from pydantic import BaseModel, ConfigDict, Field",
        "",
    ]
    for model in models:
        extra = "forbid" if model.extra_forbid else "ignore"
        lines.append(f"class {model.name}(BaseModel):")
        lines.append(f'    model_config = ConfigDict(extra="{extra}")')
        if not model.fields:
            lines.append("    pass")
        else:
            for item in model.fields:
                lines.append(f"    {_py_field(item)}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _rust_ident(name: str) -> str:
    if name in _RUST_KEYWORDS:
        return f"r#{name}"
    return name


def _rust_type(typ: IrType) -> str:
    if typ.kind == "literal":
        expr = "String"
    elif typ.kind == "model":
        expr = typ.model or "serde_json::Value"
    elif typ.kind == "list":
        inner = _rust_type(typ.inner) if typ.inner else "serde_json::Value"
        expr = f"Vec<{inner}>"
    elif typ.kind == "dict":
        inner = _rust_type(typ.inner) if typ.inner else "serde_json::Value"
        expr = f"std::collections::HashMap<String, {inner}>"
    elif typ.kind == "str":
        expr = "String"
    elif typ.kind == "int":
        expr = "i64"
    elif typ.kind == "float":
        expr = "f64"
    elif typ.kind == "bool":
        expr = "bool"
    else:
        expr = "serde_json::Value"
    if typ.nullable:
        expr = f"Option<{expr}>"
    return expr


def emit_rust_source(models: list[IrModel]) -> str:
    lines = [
        "// Auto-generated Prometheus contract types. Do not edit.",
        "use serde::{Deserialize, Serialize};",
        "",
    ]
    for model in models:
        lines.append("#[derive(Debug, Clone, Serialize, Deserialize)]")
        lines.append("pub struct " + model.name + " {")
        for item in model.fields:
            rust_ty = _rust_type(item.typ)
            if not item.required:
                if not rust_ty.startswith("Option<"):
                    rust_ty = f"Option<{rust_ty}>"
                lines.append("    #[serde(default, skip_serializing_if = \"Option::is_none\")]")
            ident = _rust_ident(item.name)
            lines.append(f"    pub {ident}: {rust_ty},")
        lines.append("}")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_python(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = emit_python_source(build_ir())
    (out_dir / "models.py").write_text(source, encoding="utf-8")


def write_rust(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = emit_rust_source(build_ir())
    (out_dir / "models.rs").write_text(source, encoding="utf-8")
