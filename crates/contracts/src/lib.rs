//! Versioned JSON Schema validation for Prometheus module boundaries.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("unknown schema {schema_id} v{version}")]
    UnknownSchema { schema_id: String, version: u32 },
    #[error("validation error: {0}")]
    Validation(String),
    #[error("no migration from v{from} to v{to}")]
    Migration { from: u32, to: u32 },
}

fn contracts_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

fn schemas_root() -> PathBuf {
    contracts_root().join("schemas")
}

fn short_name(schema_id: &str) -> Option<&str> {
    schema_id.strip_prefix("prometheus.")
}

fn version_dir(version: u32) -> PathBuf {
    schemas_root().join(format!("v{version}"))
}

fn schema_file(schema_id: &str, version: u32) -> Result<PathBuf, Error> {
    let short = short_name(schema_id).ok_or_else(|| Error::UnknownSchema {
        schema_id: schema_id.to_string(),
        version,
    })?;
    let path = version_dir(version).join(format!("{short}.schema.json"));
    if !path.is_file() {
        return Err(Error::UnknownSchema {
            schema_id: schema_id.to_string(),
            version,
        });
    }
    Ok(path)
}

fn read_json(path: &Path) -> Result<Value, Error> {
    let text = fs::read_to_string(path).map_err(|err| Error::Validation(err.to_string()))?;
    serde_json::from_str(&text).map_err(|err| Error::Validation(err.to_string()))
}

fn lookup_ref(reference: &str, defs: &Value) -> Result<Value, Error> {
    let fragment = reference
        .split_once('#')
        .map(|(_, frag)| frag)
        .unwrap_or("");
    let mut current = defs;
    for part in fragment.split('/').filter(|part| !part.is_empty()) {
        current = current
            .get(part)
            .ok_or_else(|| Error::Validation(format!("unresolved $ref {reference}")))?;
    }
    Ok(current.clone())
}

fn resolve_refs(node: &Value, defs: &Value) -> Result<Value, Error> {
    match node {
        Value::Object(map) => {
            if let Some(Value::String(reference)) = map.get("$ref") {
                let target = lookup_ref(reference, defs)?;
                return resolve_refs(&target, defs);
            }
            let mut out = serde_json::Map::new();
            for (key, value) in map {
                out.insert(key.clone(), resolve_refs(value, defs)?);
            }
            Ok(Value::Object(out))
        }
        Value::Array(items) => {
            let resolved = items
                .iter()
                .map(|item| resolve_refs(item, defs))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Value::Array(resolved))
        }
        other => Ok(other.clone()),
    }
}

fn compile(schema: &Value, defs: &Value) -> Result<jsonschema::Validator, Error> {
    let resolved = resolve_refs(schema, defs)?;
    jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(&resolved)
        .map_err(|err| Error::Validation(err.to_string()))
}

fn payload_ref(payload: &Value) -> Result<(String, u32), Error> {
    let object = payload
        .as_object()
        .ok_or_else(|| Error::Validation("payload must be an object".into()))?;
    let schema_id = match object.get("schema_id") {
        None => return Err(Error::Validation("missing schema_id".into())),
        Some(Value::String(value)) => value.clone(),
        Some(_) => return Err(Error::Validation("schema_id must be a string".into())),
    };
    let version = match object.get("schema_version") {
        None => return Err(Error::Validation("missing schema_version".into())),
        Some(Value::Number(number)) => {
            let parsed = number
                .as_u64()
                .ok_or_else(|| Error::Validation("schema_version must be a u32".into()))?;
            u32::try_from(parsed)
                .map_err(|_| Error::Validation("schema_version must be a u32".into()))?
        }
        Some(_) => {
            return Err(Error::Validation(
                "schema_version must be an integer".into(),
            ))
        }
    };
    Ok((schema_id, version))
}

/// Validate ``payload`` against the schema named by ``schema_id`` / ``schema_version``.
pub fn validate(payload: &Value) -> Result<(), Error> {
    let (schema_id, version) = payload_ref(payload)?;
    let path = schema_file(&schema_id, version)?;
    let schema = read_json(&path)?;
    let defs_path = version_dir(version).join("_defs.schema.json");
    let defs = read_json(&defs_path)?;
    let validator = compile(&schema, &defs)?;
    validator
        .validate(payload)
        .map_err(|err| Error::Validation(err.to_string()))
}

/// Identity-copy ``payload`` when it is already ``to_version``; otherwise error.
pub fn migrate(payload: &Value, to_version: u32) -> Result<Value, Error> {
    let from = payload
        .get("schema_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::Validation("schema_version must be an integer".into()))?;
    let from = u32::try_from(from)
        .map_err(|_| Error::Validation("schema_version must be a u32".into()))?;
    if from == to_version {
        return Ok(payload.clone());
    }
    Err(Error::Migration {
        from,
        to: to_version,
    })
}

/// Highest schema version on disk for ``schema_id``.
pub fn current_version(schema_id: &str) -> Result<u32, Error> {
    let short = short_name(schema_id).ok_or_else(|| Error::UnknownSchema {
        schema_id: schema_id.to_string(),
        version: 0,
    })?;
    let root = schemas_root();
    let entries = fs::read_dir(&root).map_err(|err| Error::Validation(err.to_string()))?;
    let mut found = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| Error::Validation(err.to_string()))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with('v') {
            continue;
        }
        let Ok(version) = name[1..].parse::<u32>() else {
            continue;
        };
        if entry.path().join(format!("{short}.schema.json")).is_file() {
            found.push(version);
        }
    }
    found.into_iter().max().ok_or_else(|| Error::UnknownSchema {
        schema_id: schema_id.to_string(),
        version: 0,
    })
}
