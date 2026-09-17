//! Module pipeline from spec 15.3 (H10).
//!
//! Tests and implementation come from different agents. This crate enforces
//! the process: oracle tests must fail a stub, N implementers submit
//! candidates, a reviewer picks one or rejects all, a planted bad patch is
//! never merged. Work items live on an H3 [`Queue`]. `NowMs` is injected.

use prometheus_leases::{Queue, QueueConfig, TaskId, WorkItem, EVENT_ENQUEUED};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub type NowMs = u64;

/// Bytes that mark a fixture patch the reviewer must never pick.
pub const PLANTED_BAD_MARKER: &[u8] = b"PLANTED_BAD_PATCH";

pub const DEFAULT_N_IMPLEMENTERS: u32 = 3;
pub const DEFAULT_MAX_ROUNDS: u32 = 3;

pub const ROLE_ORACLE: &str = "oracle";
pub const ROLE_IMPLEMENTER: &str = "implementer";
pub const ROLE_REVIEWER: &str = "reviewer";

const ROLE_PIPELINE: &str = "pipeline";
const OP_RECORD_ORACLE: &str = "record_oracle";
const OP_SUBMIT: &str = "submit_candidate";
const OP_VERDICT: &str = "record_verdict";
const OP_MERGE: &str = "record_merge";

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
pub fn is_planted_bad(patch: &[u8]) -> bool {
    let m = PLANTED_BAD_MARKER;
    if m.is_empty() {
        return true;
    }
    if patch.len() < m.len() {
        return false;
    }
    patch.windows(m.len()).any(|w| w == m)
}

/// Pick the first candidate whose tests passed and whose patch is not planted
/// bad. Reject all if none qualify.
pub fn review(candidates: &[Candidate], tests_ok: &[bool]) -> Result<Verdict> {
    if candidates.len() != tests_ok.len() {
        return Err(Error::Other(format!(
            "tests_ok length {} != candidates {}",
            tests_ok.len(),
            candidates.len()
        )));
    }
    for (c, ok) in candidates.iter().zip(tests_ok.iter()) {
        if *ok && !is_planted_bad(&c.patch) {
            return Ok(Verdict::Pick(c.id.clone()));
        }
    }
    let mut defects = Vec::new();
    if candidates.is_empty() {
        defects.push("no candidates".to_string());
    } else {
        for (c, ok) in candidates.iter().zip(tests_ok.iter()) {
            if is_planted_bad(&c.patch) {
                defects.push(format!("candidate {} is planted-bad", c.id.0));
            } else if !*ok {
                defects.push(format!("candidate {} tests failed", c.id.0));
            }
        }
        if defects.is_empty() {
            defects.push("no candidate qualified".to_string());
        }
    }
    Ok(Verdict::RejectAll { defects })
}

/// Refuse a pick of a planted-bad or failing candidate.
pub fn merge_allowed(verdict: &Verdict, candidates: &[Candidate], tests_ok: &[bool]) -> Result<()> {
    if candidates.len() != tests_ok.len() {
        return Err(Error::Other(format!(
            "tests_ok length {} != candidates {}",
            tests_ok.len(),
            candidates.len()
        )));
    }
    match verdict {
        Verdict::RejectAll { .. } => Err(Error::Other("verdict is RejectAll".into())),
        Verdict::Pick(id) => {
            let Some(idx) = candidates.iter().position(|c| &c.id == id) else {
                return Err(Error::NoCandidate(id.0.clone()));
            };
            if is_planted_bad(&candidates[idx].patch) {
                return Err(Error::PlantedBad);
            }
            if !tests_ok[idx] {
                return Err(Error::CandidateFailed);
            }
            Ok(())
        }
    }
}

pub struct Pipeline {
    dir: PathBuf,
    module: ModuleId,
    config: PipelineConfig,
    queue: Queue,
    stage: Stage,
    round: u32,
    stub_failed: bool,
    expected_ids: Vec<CandidateId>,
    submitted: Vec<Candidate>,
    last_verdict: Option<Verdict>,
}

impl Pipeline {
    pub fn create(
        dir: impl AsRef<Path>,
        module: ModuleId,
        config: PipelineConfig,
        queue_config: QueueConfig,
    ) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        if dir.exists() {
            return Err(Error::Other(format!("directory exists: {}", dir.display())));
        }
        std::fs::create_dir(&dir).map_err(|e| Error::Other(e.to_string()))?;
        let queue = Queue::create(dir.join("queue"), queue_config)?;
        Ok(Self {
            dir,
            module,
            config,
            queue,
            stage: Stage::Interface,
            round: 0,
            stub_failed: false,
            expected_ids: Vec::new(),
            submitted: Vec::new(),
            last_verdict: None,
        })
    }

    pub fn open(
        dir: impl AsRef<Path>,
        module: ModuleId,
        config: PipelineConfig,
        queue_config: QueueConfig,
    ) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        if !dir.is_dir() {
            return Err(Error::NotFound(format!(
                "pipeline dir missing: {}",
                dir.display()
            )));
        }
        let queue = Queue::open(dir.join("queue"), queue_config)?;
        let mut p = Self {
            dir,
            module,
            config,
            queue,
            stage: Stage::Interface,
            round: 0,
            stub_failed: false,
            expected_ids: Vec::new(),
            submitted: Vec::new(),
            last_verdict: None,
        };
        p.replay_from_queue_log()?;
        Ok(p)
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
    pub fn start_oracle(&mut self, now: NowMs) -> Result<TaskId> {
        self.require_stage(Stage::Interface)?;
        let item = work_payload(&self.module, ROLE_ORACLE, json!({ "round": self.round }));
        let id = self.enqueue(item, now)?;
        self.stage = Stage::Oracle;
        Ok(id)
    }

    /// Record oracle output. `passed_on_stub` true is [`Error::StubPassed`].
    /// False advances to StubMustFail then Implement.
    pub fn record_oracle(&mut self, passed_on_stub: bool, now: NowMs) -> Result<()> {
        self.require_stage(Stage::Oracle)?;
        if passed_on_stub {
            return Err(Error::StubPassed);
        }
        let item = work_payload(
            &self.module,
            ROLE_PIPELINE,
            json!({
                "round": self.round,
                "op": OP_RECORD_ORACLE,
                "passed_on_stub": false,
            }),
        );
        self.enqueue(item, now)?;
        self.stub_failed = true;
        self.stage = Stage::StubMustFail;
        Ok(())
    }

    /// Enqueue `n_implementers` tasks. Requires stub already failed.
    pub fn start_implementers(&mut self, now: NowMs) -> Result<Vec<TaskId>> {
        if !self.stub_failed {
            return Err(Error::StubNotFailed);
        }
        self.require_stage(Stage::StubMustFail)?;
        let ids = self.enqueue_implementers(now)?;
        self.stage = Stage::Implement;
        Ok(ids)
    }

    pub fn submit_candidate(&mut self, candidate: Candidate, now: NowMs) -> Result<()> {
        self.require_stage(Stage::Implement)?;
        if !self.expected_ids.iter().any(|id| id == &candidate.id) {
            return Err(Error::NoCandidate(candidate.id.0));
        }
        let mut item = work_payload(
            &self.module,
            ROLE_PIPELINE,
            json!({
                "round": self.round,
                "op": OP_SUBMIT,
                "candidate": candidate.id.0,
                "angle": candidate.angle,
                "patch": candidate.patch,
            }),
        );
        // Last write wins: Queue rejects duplicate task ids, so a resubmit of
        // the same candidate keeps the payload and gets a unique id.
        if self.queue.get(&item.task_id).is_some() {
            let base = item.task_id.0.clone();
            let mut n = 1u32;
            loop {
                let tid = TaskId(format!("{base}#{n}"));
                if self.queue.get(&tid).is_none() {
                    item.task_id = tid;
                    break;
                }
                n = n.saturating_add(1);
            }
        }
        self.enqueue(item, now)?;
        upsert_candidate(&mut self.submitted, candidate);
        Ok(())
    }

    pub fn start_review(&mut self, now: NowMs) -> Result<TaskId> {
        self.require_stage(Stage::Implement)?;
        let item = work_payload(&self.module, ROLE_REVIEWER, json!({ "round": self.round }));
        let id = self.enqueue(item, now)?;
        self.stage = Stage::Review;
        Ok(id)
    }

    /// Apply a reviewer verdict. Pick of planted-bad is [`Error::PlantedBad`].
    /// RejectAll increments round; at max_rounds the stage is Blocked.
    pub fn record_verdict(&mut self, verdict: Verdict, now: NowMs) -> Result<Stage> {
        self.require_stage(Stage::Review)?;
        match &verdict {
            Verdict::Pick(id) => {
                let Some(c) = self.submitted.iter().find(|c| &c.id == id) else {
                    return Err(Error::NoCandidate(id.0.clone()));
                };
                if is_planted_bad(&c.patch) {
                    return Err(Error::PlantedBad);
                }
            }
            Verdict::RejectAll { .. } => {}
        }
        let item = work_payload(
            &self.module,
            ROLE_PIPELINE,
            json!({
                "round": self.round,
                "op": OP_VERDICT,
                "verdict": verdict,
            }),
        );
        self.enqueue(item, now)?;
        self.apply_verdict(verdict, now)?;
        Ok(self.stage)
    }

    /// Record a merge. Requires stage Review with a Pick that `merge_allowed`.
    pub fn record_merge(&mut self, record: MergeRecord, now: NowMs) -> Result<()> {
        self.require_stage(Stage::Merged)?;
        if record.module != self.module {
            return Err(Error::Other("merge record module mismatch".into()));
        }
        match &self.last_verdict {
            Some(Verdict::Pick(id)) => {
                let Some(c) = self.submitted.iter().find(|c| &c.id == id) else {
                    return Err(Error::NoCandidate(id.0.clone()));
                };
                if is_planted_bad(&c.patch) {
                    return Err(Error::PlantedBad);
                }
            }
            _ => {
                return Err(Error::Other("record_merge requires a Pick verdict".into()));
            }
        }
        let item = work_payload(
            &self.module,
            ROLE_PIPELINE,
            json!({
                "round": self.round,
                "op": OP_MERGE,
                "record": record,
            }),
        );
        self.enqueue(item, now)?;
        Ok(())
    }

    pub fn candidates(&self) -> Result<Vec<Candidate>> {
        Ok(self.submitted.clone())
    }

    fn require_stage(&self, expected: Stage) -> Result<()> {
        if self.stage != expected {
            return Err(Error::WrongStage {
                expected,
                actual: self.stage,
            });
        }
        Ok(())
    }

    fn enqueue(&mut self, item: WorkItem, now: NowMs) -> Result<TaskId> {
        Ok(self.queue.enqueue(item, now)?)
    }

    fn enqueue_implementers(&mut self, now: NowMs) -> Result<Vec<TaskId>> {
        self.expected_ids.clear();
        self.submitted.clear();
        let n = self.config.n_implementers;
        let mut ids = Vec::with_capacity(n as usize);
        for i in 0..n {
            let cid = i.to_string();
            let item = work_payload(
                &self.module,
                ROLE_IMPLEMENTER,
                json!({ "round": self.round, "candidate": cid }),
            );
            ids.push(self.enqueue(item, now)?);
            self.expected_ids.push(CandidateId(cid));
        }
        Ok(ids)
    }

    fn apply_verdict(&mut self, verdict: Verdict, now: NowMs) -> Result<()> {
        match &verdict {
            Verdict::Pick(_) => {
                self.last_verdict = Some(verdict);
                self.stage = Stage::Merged;
            }
            Verdict::RejectAll { .. } => {
                self.last_verdict = Some(verdict);
                self.round = self.round.saturating_add(1);
                self.submitted.clear();
                self.expected_ids.clear();
                if self.round >= self.config.max_rounds {
                    self.stage = Stage::Blocked;
                } else {
                    self.enqueue_implementers(now)?;
                    self.stage = Stage::Implement;
                }
            }
        }
        Ok(())
    }

    fn replay_from_queue_log(&mut self) -> Result<()> {
        let events: Vec<_> = self
            .queue
            .log()
            .iter()
            .filter(|e| e.event_type == EVENT_ENQUEUED)
            .cloned()
            .collect();
        for ev in events {
            let p = &ev.payload;
            let role = p.get("role").and_then(Value::as_str).unwrap_or("");
            let op = p.get("op").and_then(Value::as_str).unwrap_or("");
            match (role, op) {
                (ROLE_ORACLE, _) => {
                    self.stage = Stage::Oracle;
                    if let Some(r) = p.get("round").and_then(Value::as_u64) {
                        self.round = r as u32;
                    }
                }
                (ROLE_PIPELINE, OP_RECORD_ORACLE) => {
                    let passed = p
                        .get("passed_on_stub")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    if !passed {
                        self.stub_failed = true;
                        self.stage = Stage::StubMustFail;
                    }
                }
                (ROLE_IMPLEMENTER, _) => {
                    if let Some(r) = p.get("round").and_then(Value::as_u64) {
                        let r = r as u32;
                        if r != self.round {
                            self.round = r;
                            self.expected_ids.clear();
                            self.submitted.clear();
                        }
                    }
                    if let Some(c) = p.get("candidate").and_then(Value::as_str) {
                        let id = CandidateId(c.to_string());
                        if !self.expected_ids.iter().any(|x| x == &id) {
                            self.expected_ids.push(id);
                        }
                    }
                    self.stage = Stage::Implement;
                }
                (ROLE_PIPELINE, OP_SUBMIT) => {
                    let id = p
                        .get("candidate")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let angle = p
                        .get("angle")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let patch = p
                        .get("patch")
                        .and_then(Value::as_array)
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|x| x.as_u64().map(|n| n as u8))
                                .collect()
                        })
                        .unwrap_or_default();
                    upsert_candidate(
                        &mut self.submitted,
                        Candidate {
                            id: CandidateId(id),
                            angle,
                            patch,
                        },
                    );
                }
                (ROLE_REVIEWER, _) => {
                    self.stage = Stage::Review;
                }
                (ROLE_PIPELINE, OP_VERDICT) => {
                    if let Some(v) = p.get("verdict") {
                        if let Ok(verdict) = serde_json::from_value::<Verdict>(v.clone()) {
                            match &verdict {
                                Verdict::Pick(_) => {
                                    self.last_verdict = Some(verdict);
                                    self.stage = Stage::Merged;
                                }
                                Verdict::RejectAll { .. } => {
                                    self.last_verdict = Some(verdict);
                                    self.round = self.round.saturating_add(1);
                                    self.submitted.clear();
                                    self.expected_ids.clear();
                                    if self.round >= self.config.max_rounds {
                                        self.stage = Stage::Blocked;
                                    } else {
                                        self.stage = Stage::Implement;
                                    }
                                }
                            }
                        }
                    }
                }
                (ROLE_PIPELINE, OP_MERGE) => {
                    self.stage = Stage::Merged;
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Payload helper for H3 work items. `role` is oracle/implementer/reviewer.
pub fn work_payload(module: &ModuleId, role: &str, extra: Value) -> WorkItem {
    let mut map = match extra {
        Value::Object(m) => m,
        other => {
            let mut m = serde_json::Map::new();
            m.insert("extra".into(), other);
            m
        }
    };
    map.insert("module".into(), json!(module.0));
    map.insert("role".into(), json!(role));
    let payload = Value::Object(map);
    let round = payload.get("round").and_then(Value::as_u64).unwrap_or(0);
    let op = payload.get("op").and_then(Value::as_str);
    let cand = payload.get("candidate").and_then(Value::as_str);
    let mut id = format!("{role}/{}/{round}", module.0);
    if let Some(op) = op {
        id.push('/');
        id.push_str(op);
    }
    if let Some(c) = cand {
        id.push('/');
        id.push_str(c);
    }
    WorkItem {
        task_id: TaskId(id),
        payload,
    }
}

fn upsert_candidate(submitted: &mut Vec<Candidate>, candidate: Candidate) {
    if let Some(slot) = submitted.iter_mut().find(|c| c.id == candidate.id) {
        *slot = candidate;
    } else {
        submitted.push(candidate);
    }
}
