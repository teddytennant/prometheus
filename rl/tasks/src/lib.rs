//! Task factories wave 1: code, SWE, math (spec 9.4, 15.5 D3).
//!
//! Each factory mints an F1 [`TaskSpec`], attaches a D2 exact verifier, and
//! drops tasks an injected solver always solves or never solves. Wave 2
//! (research, long-horizon, ARC-3, forecasting, open-ended) is I6.
//!
//! Solvers are a [`Solver`] so CPU tests inject [`ScriptedSolver`]. F5
//! serving can wrap this later; nothing here starts an engine.
//!
//! [`NowMs`] is injected. `created_at` is a caller-supplied ISO-8601 string.
//! Nothing here reads the wall clock.
//!
//! Gate: open-weights solve rate in (0, 100%). 0% and 100% are dropped.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use prometheus_envs::{Image, NowMs};
pub use prometheus_verifiers::{CodeTask, MathTask, VerifierId, VerifierKind};

pub const SCHEMA_TASK_SPEC: &str = "prometheus.task_spec";
pub const SCHEMA_VERSION: u32 = 1;

/// Curriculum starts here (spec 9.4). Raised when success at the current
/// horizon crosses [`RAISE_THRESHOLD`].
pub const START_TOOL_CALLS: u32 = 10;

/// Raise the horizon when the policy's success rate at the current cap
/// exceeds this.
pub const RAISE_THRESHOLD: f64 = 0.5;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FactoryId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub String);

/// F1 `prometheus.task_spec.domain`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskDomain {
    Math,
    Code,
    Science,
    Security,
    Arc,
    Agent,
    Other,
}

/// F1 `prometheus.task_spec.split`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Val,
    Test,
    Holdout,
}

/// F1 provenance. `source` is required.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_date: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
}

impl Provenance {
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            license: None,
            url_hash: None,
            snapshot_date: None,
            collection_id: None,
            author: None,
        }
    }
}

/// One catalog row a factory can mint from. `body` is the public statement
/// (math/code) or a repo snapshot identifier (SWE).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    pub id: String,
    pub provenance: Provenance,
    pub body: String,
}

impl Source {
    pub fn new(id: impl Into<String>, source: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            provenance: Provenance::new(source),
            body: body.into(),
        }
    }
}

/// F1 `prometheus.task_spec` v1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskSpec {
    pub schema_id: String,
    pub schema_version: u32,
    pub task_id: String,
    pub domain: TaskDomain,
    pub split: Split,
    pub statement_hash: String,
    pub hidden_tests_hash: String,
    pub verifier_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_image: Option<String>,
    pub horizon_s: u32,
    pub max_tool_calls: u32,
    pub provenance: Provenance,
    pub created_at: String,
}

impl TaskSpec {
    pub fn schema_ok(&self) -> Result<()> {
        if self.schema_id != SCHEMA_TASK_SPEC || self.schema_version != SCHEMA_VERSION {
            return Err(Error::Schema(format!(
                "{} v{}",
                self.schema_id, self.schema_version
            )));
        }
        Ok(())
    }
}

/// Minted task: F1 spec plus the D2 checker payload. Hidden tests live on
/// the [`CodeTask`] image / [`MathTask`] expected answer, never in the
/// public statement.
#[derive(Debug, Clone, PartialEq)]
pub struct MintedTask {
    pub spec: TaskSpec,
    pub verifier_kind: VerifierKind,
    pub verifier_id: VerifierId,
    pub statement: String,
    pub code: Option<CodeTask>,
    pub math: Option<MathTask>,
}

impl MintedTask {
    pub fn spec(&self) -> &TaskSpec {
        &self.spec
    }

    pub fn statement(&self) -> &str {
        &self.statement
    }
}

/// Scripted or remote attempt. CPU tests inject [`ScriptedSolver`].
pub trait Solver {
    fn attempt(&mut self, statement: &str, now: NowMs) -> Result<String>;
}

/// Exact map from statement to attempt. Missing statements error.
pub struct ScriptedSolver {
    replies: BTreeMap<String, String>,
}

impl ScriptedSolver {
    pub fn new() -> Self {
        Self {
            replies: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, statement: impl Into<String>, reply: impl Into<String>) {
        self.replies.insert(statement.into(), reply.into());
    }
}

impl Default for ScriptedSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for ScriptedSolver {
    fn attempt(&mut self, statement: &str, _now: NowMs) -> Result<String> {
        self.replies
            .get(statement)
            .cloned()
            .ok_or_else(|| Error::Unverifiable(format!("no scripted attempt for {statement}")))
    }
}

/// `passed` of `n` independent attempts. Rate is `passed / n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveRate {
    pub passed: u32,
    pub n: u32,
}

impl SolveRate {
    pub fn new(passed: u32, n: u32) -> Self {
        Self { passed, n }
    }

    /// `passed / n`. Zero when `n == 0`.
    pub fn rate(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            f64::from(self.passed) / f64::from(self.n)
        }
    }

    /// True iff 0 < passed < n.
    pub fn in_open_interval(&self) -> bool {
        self.n > 0 && self.passed > 0 && self.passed < self.n
    }
}

/// Drop 0% and 100% (spec 9.4, D3 gate).
pub struct SolveRateFilter;

impl SolveRateFilter {
    /// Run `n` attempts. A scripted solver that always returns the same
    /// reply still counts as `n` samples; the implementer scores each
    /// attempt with the attached D2 verifier.
    pub fn probe<S: Solver>(
        _solver: &mut S,
        _task: &MintedTask,
        _n: u32,
        _now: NowMs,
    ) -> Result<SolveRate> {
        unimplemented!("D3: SolveRateFilter::probe")
    }

    /// Keep only when [`SolveRate::in_open_interval`]. 0% is
    /// [`Error::Impossible`]. 100% is [`Error::Trivial`]. `n == 0` is
    /// [`Error::BadN`].
    pub fn keep(_rate: SolveRate) -> Result<SolveRate> {
        unimplemented!("D3: SolveRateFilter::keep")
    }
}

/// Tool-call curriculum (spec 9.4). Starts at [`START_TOOL_CALLS`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Horizon {
    tool_calls: u32,
}

impl Horizon {
    pub fn new() -> Self {
        Self {
            tool_calls: START_TOOL_CALLS,
        }
    }

    pub fn tool_calls(&self) -> u32 {
        self.tool_calls
    }

    /// Raise when `success_rate > RAISE_THRESHOLD`. No-op otherwise.
    pub fn raise_if(&mut self, _success_rate: f64) {
        unimplemented!("D3: Horizon::raise_if")
    }
}

impl Default for Horizon {
    fn default() -> Self {
        Self::new()
    }
}

/// Competitive-programming tasks with hidden tests and mutants.
pub struct CodeFactory {
    id: FactoryId,
    sources: Vec<Source>,
}

impl CodeFactory {
    pub fn new(id: impl Into<String>, sources: Vec<Source>) -> Self {
        Self {
            id: FactoryId(id.into()),
            sources,
        }
    }

    pub fn id(&self) -> &FactoryId {
        &self.id
    }

    pub fn sources(&self) -> &[Source] {
        &self.sources
    }

    pub fn mint(&mut self, _now: NowMs, _created_at: &str) -> Result<MintedTask> {
        unimplemented!("D3: CodeFactory::mint")
    }
}

/// Repo-level SWE tasks. Same D2 sandboxed-tests verifier, longer horizon,
/// `domain = code`.
pub struct SweFactory {
    id: FactoryId,
    sources: Vec<Source>,
}

impl SweFactory {
    pub fn new(id: impl Into<String>, sources: Vec<Source>) -> Self {
        Self {
            id: FactoryId(id.into()),
            sources,
        }
    }

    pub fn id(&self) -> &FactoryId {
        &self.id
    }

    pub fn sources(&self) -> &[Source] {
        &self.sources
    }

    pub fn mint(&mut self, _now: NowMs, _created_at: &str) -> Result<MintedTask> {
        unimplemented!("D3: SweFactory::mint")
    }
}

/// Symbolic-math tasks. D2 `VerifierKind::SymbolicMath`.
pub struct MathFactory {
    id: FactoryId,
    sources: Vec<Source>,
}

impl MathFactory {
    pub fn new(id: impl Into<String>, sources: Vec<Source>) -> Self {
        Self {
            id: FactoryId(id.into()),
            sources,
        }
    }

    pub fn id(&self) -> &FactoryId {
        &self.id
    }

    pub fn sources(&self) -> &[Source] {
        &self.sources
    }

    pub fn mint(&mut self, _now: NowMs, _created_at: &str) -> Result<MintedTask> {
        unimplemented!("D3: MathFactory::mint")
    }
}

pub trait Factory {
    fn id(&self) -> &FactoryId;
    fn domain(&self) -> TaskDomain;
    fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask>;
}

impl Factory for CodeFactory {
    fn id(&self) -> &FactoryId {
        CodeFactory::id(self)
    }

    fn domain(&self) -> TaskDomain {
        TaskDomain::Code
    }

    fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
        CodeFactory::mint(self, now, created_at)
    }
}

impl Factory for SweFactory {
    fn id(&self) -> &FactoryId {
        SweFactory::id(self)
    }

    fn domain(&self) -> TaskDomain {
        TaskDomain::Code
    }

    fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
        SweFactory::mint(self, now, created_at)
    }
}

impl Factory for MathFactory {
    fn id(&self) -> &FactoryId {
        MathFactory::id(self)
    }

    fn domain(&self) -> TaskDomain {
        TaskDomain::Math
    }

    fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
        MathFactory::mint(self, now, created_at)
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum Error {
    #[error("schema mismatch: {0}")]
    Schema(String),
    #[error("empty factory catalog")]
    EmptyCatalog,
    #[error("empty public statement")]
    EmptyStatement,
    #[error("solve-rate sample count must be > 0")]
    BadN,
    #[error("solve rate 0%")]
    Impossible,
    #[error("solve rate 100%")]
    Trivial,
    #[error("unverifiable: {0}")]
    Unverifiable(String),
}

pub type Result<T> = std::result::Result<T, Error>;
