//! Append-only experiment ledger (spec 14.6, 15.5 H9).
//!
//! Every result, negative ones included, is a row. The store is SQL plus
//! embedding search so a researcher can ask "has this been tried?" before
//! spending GPUs. Rows are never updated or deleted; a correction is a new
//! row that points at `supersedes`.
//!
//! Payload shape matches `prometheus.ledger_record` in `contracts/` (F1).
//! This crate does not depend on the Python package; it speaks JSON that
//! validates against that schema once F1 is merged.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("record {0} not found")]
    NotFound(String),
    #[error("duplicate experiment_id {0}")]
    Duplicate(String),
    #[error("append-only: refused update of {0}")]
    Immutable(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RecordId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplicationStatus {
    Unreplicated,
    Matched,
    Failed,
    Running,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Replication {
    pub status: ReplicationStatus,
    pub n: u32,
    pub notes: Option<String>,
}

/// One ledger row. Field names match `contracts/schemas/v1/ledger_record.schema.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub experiment_id: String,
    pub hypothesis: String,
    pub prediction: String,
    pub config_hash: String,
    pub results: Option<Value>,
    pub delta_rci: Option<f64>,
    pub replication: Option<Replication>,
    pub gpu_hours: Option<f64>,
    pub author_role: String,
    pub rung: Option<u32>,
    pub created_at: String,
    pub closed_at: Option<String>,
    /// If set, this row corrects or follows `supersedes` without mutating it.
    pub supersedes: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub id: RecordId,
    pub record: Record,
    pub score: f32,
}

pub struct Ledger {
    _inner: (),
}

impl Ledger {
    /// Open (or create) the SQL store at `path`. Embedding index lives
    /// beside it.
    pub fn open(_path: &Path) -> Result<Self> {
        unimplemented!("Ledger::open")
    }

    /// Append a row. Refuses if `experiment_id` already exists unless
    /// `supersedes` is set (then `experiment_id` must be new and
    /// `supersedes` must exist).
    pub fn append(&self, _record: Record) -> Result<RecordId> {
        unimplemented!("Ledger::append")
    }

    pub fn get(&self, _id: &RecordId) -> Result<Record> {
        unimplemented!("Ledger::get")
    }

    pub fn get_by_experiment(&self, _experiment_id: &str) -> Result<Record> {
        unimplemented!("Ledger::get_by_experiment")
    }

    /// SQL filter over stored columns. `where_sql` is a parameterized
    /// WHERE body; implementations must not interpolate `params` into the
    /// string.
    pub fn query(&self, _where_sql: &str, _params: &[Value]) -> Result<Vec<Record>> {
        unimplemented!("Ledger::query")
    }

    /// Nearest neighbours of `text` in embedding space. Used to surface
    /// prior negatives before a new run.
    pub fn search_similar(&self, _text: &str, _k: usize) -> Result<Vec<SearchHit>> {
        unimplemented!("Ledger::search_similar")
    }

    pub fn list_negatives(&self) -> Result<Vec<Record>> {
        unimplemented!("Ledger::list_negatives")
    }

    /// Number of rows, including superseded ones.
    pub fn len(&self) -> Result<u64> {
        unimplemented!("Ledger::len")
    }

    pub fn is_empty(&self) -> Result<bool> {
        unimplemented!("Ledger::is_empty")
    }
}
