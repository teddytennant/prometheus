//! Versioned schemas for every module boundary (spec 15.4).
//!
//! JSON Schema files in `contracts/schemas/` are the source of truth.
//! `generate_rust` emits serde types into `src/generated/`. Do not
//! hand-write payload structs that mirror those schemas.

use serde_json::Value;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("payload does not match schema: {0}")]
    Validation(String),
    #[error("unknown schema {schema_id} version {version}")]
    UnknownSchema { schema_id: String, version: u32 },
    #[error("no migration path from v{from} to v{to} for {schema_id}")]
    Migration {
        schema_id: String,
        from: u32,
        to: u32,
    },
}

/// Validate `payload` against the schema named by its `schema_id` and
/// `schema_version` fields. Readers accept every past version.
pub fn validate(_payload: &Value) -> Result<(), Error> {
    unimplemented!("contracts::validate")
}

/// Return a new payload at `to_version`. Identity when already at that
/// version. v1 is the first version.
pub fn migrate(_payload: Value, _to_version: u32) -> Result<Value, Error> {
    unimplemented!("contracts::migrate")
}

/// Highest schema_version on disk for `schema_id`.
pub fn current_version(_schema_id: &str) -> Result<u32, Error> {
    unimplemented!("contracts::current_version")
}
