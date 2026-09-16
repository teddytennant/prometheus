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

use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Mutex;

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

impl From<rusqlite::Error> for Error {
    fn from(err: rusqlite::Error) -> Self {
        Error::Other(err.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Other(err.to_string())
    }
}

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
    conn: Mutex<Connection>,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS records (
    experiment_id TEXT PRIMARY KEY NOT NULL,
    author_role TEXT NOT NULL,
    rung INTEGER,
    record_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_author_role ON records(author_role);
CREATE INDEX IF NOT EXISTS idx_records_rung ON records(rung);
";

/// Hashing-trick dimension. Large enough that the test neighbourhoods don't collide.
const EMBED_DIM: usize = 256;

impl Ledger {
    /// Open (or create) the SQL store at `path`. Embedding index lives
    /// beside it.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Append a row. Refuses if `experiment_id` already exists unless
    /// `supersedes` is set (then `experiment_id` must be new and
    /// `supersedes` must exist).
    pub fn append(&self, record: Record) -> Result<RecordId> {
        let conn = self.lock()?;
        if row_exists(&conn, &record.experiment_id)? {
            return Err(Error::Duplicate(record.experiment_id));
        }
        if let Some(ref target) = record.supersedes {
            if !row_exists(&conn, target)? {
                return Err(Error::NotFound(target.clone()));
            }
        }
        let json = serde_json::to_string(&record)?;
        conn.execute(
            "INSERT INTO records (experiment_id, author_role, rung, record_json)
             VALUES (?1, ?2, ?3, ?4)",
            params![record.experiment_id, record.author_role, record.rung, json],
        )?;
        Ok(RecordId(record.experiment_id))
    }

    pub fn get(&self, id: &RecordId) -> Result<Record> {
        self.get_by_experiment(&id.0)
    }

    pub fn get_by_experiment(&self, experiment_id: &str) -> Result<Record> {
        let conn = self.lock()?;
        let json: Option<String> = conn
            .query_row(
                "SELECT record_json FROM records WHERE experiment_id = ?1",
                params![experiment_id],
                |row| row.get(0),
            )
            .optional()?;
        match json {
            Some(json) => Ok(serde_json::from_str(&json)?),
            None => Err(Error::NotFound(experiment_id.to_string())),
        }
    }

    /// SQL filter over stored columns. `where_sql` is a parameterized
    /// WHERE body; implementations must not interpolate `params` into the
    /// string.
    pub fn query(&self, where_sql: &str, params: &[Value]) -> Result<Vec<Record>> {
        let conn = self.lock()?;
        let sql = format!("SELECT record_json FROM records WHERE {where_sql}");
        let bound: Vec<rusqlite::types::Value> = params.iter().map(json_to_sql).collect();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(bound.iter()), |row| {
            row.get::<_, String>(0)
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row?)?);
        }
        Ok(out)
    }

    /// Nearest neighbours of `text` in embedding space. Used to surface
    /// prior negatives before a new run.
    pub fn search_similar(&self, text: &str, k: usize) -> Result<Vec<SearchHit>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let records = self.all_records()?;
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let query = embed(text);
        let mut hits: Vec<SearchHit> = records
            .into_iter()
            .map(|record| {
                let doc = embed(&format!("{} {}", record.hypothesis, record.prediction));
                let score = cosine(&query, &doc);
                SearchHit {
                    id: RecordId(record.experiment_id.clone()),
                    record,
                    score,
                }
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        Ok(hits)
    }

    pub fn list_negatives(&self) -> Result<Vec<Record>> {
        let records = self.all_records()?;
        Ok(records.into_iter().filter(is_negative).collect())
    }

    /// Number of rows, including superseded ones.
    pub fn len(&self) -> Result<u64> {
        let conn = self.lock()?;
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM records", [], |row| row.get(0))?;
        Ok(n as u64)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.conn
            .lock()
            .map_err(|e| Error::Other(format!("ledger mutex poisoned: {e}")))
    }

    fn all_records(&self) -> Result<Vec<Record>> {
        let conn = self.lock()?;
        let mut stmt = conn.prepare("SELECT record_json FROM records")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row?)?);
        }
        Ok(out)
    }
}

fn row_exists(conn: &Connection, experiment_id: &str) -> Result<bool> {
    let found: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM records WHERE experiment_id = ?1",
            params![experiment_id],
            |row| row.get(0),
        )
        .optional()?;
    Ok(found.is_some())
}

fn json_to_sql(value: &Value) -> rusqlite::types::Value {
    use rusqlite::types::Value as SqlValue;
    match value {
        Value::Null => SqlValue::Null,
        Value::Bool(b) => SqlValue::Integer(i64::from(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                SqlValue::Integer(i)
            } else if let Some(u) = n.as_u64() {
                SqlValue::Integer(u as i64)
            } else if let Some(f) = n.as_f64() {
                SqlValue::Real(f)
            } else {
                SqlValue::Null
            }
        }
        Value::String(s) => SqlValue::Text(s.clone()),
        other => SqlValue::Text(other.to_string()),
    }
}

fn is_negative(record: &Record) -> bool {
    if let Some(results) = &record.results {
        if results.get("passed") == Some(&Value::Bool(false)) {
            return true;
        }
        if results.get("negative") == Some(&Value::Bool(true)) {
            return true;
        }
    }
    matches!(
        record.replication.as_ref().map(|r| r.status),
        Some(ReplicationStatus::Failed)
    )
}

fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|tok| !tok.is_empty())
        .map(|tok| tok.to_ascii_lowercase())
}

/// Hashing-trick bag-of-words embedding, L2-normalized.
fn embed(text: &str) -> Vec<f32> {
    let mut vec = vec![0.0f32; EMBED_DIM];
    for token in tokenize(text) {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        token.hash(&mut hasher);
        let idx = (hasher.finish() as usize) % EMBED_DIM;
        vec[idx] += 1.0;
    }
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut vec {
            *x /= norm;
        }
    }
    vec
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
