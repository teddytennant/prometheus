//! Module pipeline from spec 15.3 (H10).
//!
//! Tests and implementation come from different agents. This crate enforces
//! the process: oracle tests must fail a stub, N implementers submit
//! candidates, a reviewer picks one or rejects all, a planted bad patch is
//! never merged. Work items live on an H3 [`Queue`]. `NowMs` is injected.

use prometheus_leases::{Queue, QueueConfig, TaskId, WorkItem, WorkerId};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

pub type NowMs = u64;

/// Bytes that mark a fixture patch the reviewer must never pick.
pub const PLANTED_BAD_MARKER: &[u8] = b"PLANTED_BAD_PATCH";

pub const DEFAULT_N_IMPLEMENTERS: u32 = 3;
pub const DEFAULT_MAX_ROUNDS: u32 = 3;

pub const ROLE_ORACLE: &str = "oracle";
pub const ROLE_IMPLEMENTER: &str = "implementer";
pub const ROLE_REVIEWER: &str = "reviewer";

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModuleId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CandidateId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stage {
    Interface,
    Oracle,
    StubMustFail,
    Implement,
    Review,
    Merged,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub n_implementers: u32,
    pub max_rounds: u32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            n_implementers: DEFAULT_N_IMPLEMENTERS,
            max_rounds: DEFAULT_MAX_ROUNDS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: CandidateId,
    pub angle: String,
    pub patch: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    Pick(CandidateId),
    RejectAll { defects: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergeRecord {
    pub module: ModuleId,
    pub spec_section: String,
    pub commit: String,
    pub tests_passed: u32,
    pub tests_failed: u32,
    pub gpu_job_ids: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("oracle tests passed the stub")]
    StubPassed,
    #[error("stub has not been shown to fail")]
    StubNotFailed,
    #[error("wrong stage: expected {expected:?}, at {actual:?}")]
    WrongStage { expected: Stage, actual: Stage },
    #[error("planted bad patch cannot be picked")]
    PlantedBad,
    #[error("candidate tests failed")]
    CandidateFailed,
    #[error("no such candidate: {0}")]
    NoCandidate(String),
    #[error("max rounds ({0}) exhausted")]
    MaxRounds(u32),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("{0}")]
    Queue(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<prometheus_leases::Error> for Error {
    fn from(err: prometheus_leases::Error) -> Self {
        Error::Queue(err.to_string())
    }
}

/// True iff `patch` contains [`PLANTED_BAD_MARKER`].
pub fn is_planted_bad(_patch: &[u8]) -> bool {
    unimplemented!("H10: is_planted_bad")
}

/// Pick the first candidate whose tests passed and whose patch is not planted
/// bad. Reject all if none qualify.
pub fn review(candidates: &[Candidate], tests_ok: &[bool]) -> Result<Verdict> {
    let _ = (candidates, tests_ok);
    unimplemented!("H10: review")
}

/// Refuse a pick of a planted-bad or failing candidate.
pub fn merge_allowed(verdict: &Verdict, candidates: &[Candidate], tests_ok: &[bool]) -> Result<()> {
    let _ = (verdict, candidates, tests_ok);
    unimplemented!("H10: merge_allowed")
}

pub struct Pipeline {
    dir: PathBuf,
    module: ModuleId,
    config: PipelineConfig,
    queue: Queue,
    stage: Stage,
    round: u32,
    stub_failed: bool,
}

impl Pipeline {
    pub fn create(
        dir: impl AsRef<Path>,
        module: ModuleId,
        config: PipelineConfig,
        queue_config: QueueConfig,
    ) -> Result<Self> {
        let _ = (dir, module, config, queue_config);
        unimplemented!("H10: Pipeline::create")
    }

    pub fn open(
        dir: impl AsRef<Path>,
        module: ModuleId,
        config: PipelineConfig,
        queue_config: QueueConfig,
    ) -> Result<Self> {
        let _ = (dir, module, config, queue_config);
        unimplemented!("H10: Pipeline::open")
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn module(&self) -> &ModuleId {
        &self.module
    }

    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn stage(&self) -> Stage {
        self.stage
    }

    pub fn round(&self) -> u32 {
        self.round
    }

    pub fn stub_failed(&self) -> bool {
        self.stub_failed
    }

    /// Interface is already written. Advance to Oracle and enqueue an oracle task.
    pub fn start_oracle(&mut self, _now: NowMs) -> Result<TaskId> {
        unimplemented!("H10: Pipeline::start_oracle")
    }

    /// Record oracle output. `passed_on_stub` true is [`Error::StubPassed`].
    /// False advances to StubMustFail then Implement.
    pub fn record_oracle(&mut self, _passed_on_stub: bool, _now: NowMs) -> Result<()> {
        unimplemented!("H10: Pipeline::record_oracle")
    }

    /// Enqueue `n_implementers` tasks. Requires stub already failed.
    pub fn start_implementers(&mut self, _now: NowMs) -> Result<Vec<TaskId>> {
        unimplemented!("H10: Pipeline::start_implementers")
    }

    pub fn submit_candidate(&mut self, _candidate: Candidate, _now: NowMs) -> Result<()> {
        unimplemented!("H10: Pipeline::submit_candidate")
    }

    pub fn start_review(&mut self, _now: NowMs) -> Result<TaskId> {
        unimplemented!("H10: Pipeline::start_review")
    }

    /// Apply a reviewer verdict. Pick of planted-bad is [`Error::PlantedBad`].
    /// RejectAll increments round; at max_rounds the stage is Blocked.
    pub fn record_verdict(&mut self, _verdict: Verdict, _now: NowMs) -> Result<Stage> {
        unimplemented!("H10: Pipeline::record_verdict")
    }

    /// Record a merge. Requires stage Review with a Pick that `merge_allowed`.
    pub fn record_merge(&mut self, _record: MergeRecord, _now: NowMs) -> Result<()> {
        unimplemented!("H10: Pipeline::record_merge")
    }

    pub fn candidates(&self) -> Result<Vec<Candidate>> {
        unimplemented!("H10: Pipeline::candidates")
    }
}

/// Payload helper for H3 work items. `role` is oracle/implementer/reviewer.
pub fn work_payload(module: &ModuleId, role: &str, extra: Value) -> WorkItem {
    let _ = WorkerId(String::new());
    let _ = (module, role, extra);
    unimplemented!("H10: work_payload")
}
