//! Independent slow reference for H10 `prometheus-pipeline`.
//!
//! In-memory stage machine plus a real H3 `Queue` at `dir/queue` so replay is
//! the obvious "read every `task.enqueued` payload in order" loop. Production
//! (`harness/pipeline/src`) must never import this.

#![allow(dead_code)]

use prometheus_leases::{NowMs, Queue, QueueConfig, TaskId, WorkItem, EVENT_ENQUEUED};
use prometheus_pipeline::{
    Candidate, CandidateId, Error, MergeRecord, ModuleId, PipelineConfig, Result, Stage, Verdict,
    PLANTED_BAD_MARKER, ROLE_IMPLEMENTER, ROLE_ORACLE, ROLE_REVIEWER,
};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const ROLE_PIPELINE: &str = "pipeline";
const OP_RECORD_ORACLE: &str = "record_oracle";
const OP_SUBMIT: &str = "submit_candidate";
const OP_VERDICT: &str = "record_verdict";
const OP_MERGE: &str = "record_merge";

/// True iff `patch` contains `PLANTED_BAD_MARKER` as a contiguous slice.
pub fn ref_is_planted_bad(patch: &[u8]) -> bool {
    let m = PLANTED_BAD_MARKER;
    if m.is_empty() {
        return true;
    }
    if patch.len() < m.len() {
        return false;
    }
    patch.windows(m.len()).any(|w| w == m)
}

/// First passing, non-planted-bad candidate in slice order.
pub fn ref_review(candidates: &[Candidate], tests_ok: &[bool]) -> Result<Verdict> {
    if candidates.len() != tests_ok.len() {
        return Err(Error::Other(format!(
            "tests_ok length {} != candidates {}",
            tests_ok.len(),
            candidates.len()
        )));
    }
    for (c, ok) in candidates.iter().zip(tests_ok.iter()) {
        if *ok && !ref_is_planted_bad(&c.patch) {
            return Ok(Verdict::Pick(c.id.clone()));
        }
    }
    let mut defects = Vec::new();
    if candidates.is_empty() {
        defects.push("no candidates".to_string());
    } else {
        for (c, ok) in candidates.iter().zip(tests_ok.iter()) {
            if ref_is_planted_bad(&c.patch) {
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

pub fn ref_merge_allowed(
    verdict: &Verdict,
    candidates: &[Candidate],
    tests_ok: &[bool],
) -> Result<()> {
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
            let idx = candidates.iter().position(|c| &c.id == id);
            let Some(idx) = idx else {
                return Err(Error::NoCandidate(id.0.clone()));
            };
            if ref_is_planted_bad(&candidates[idx].patch) {
                return Err(Error::PlantedBad);
            }
            if !tests_ok[idx] {
                return Err(Error::CandidateFailed);
            }
            Ok(())
        }
    }
}

/// Deterministic work-item helper. See `tests/common/mod.rs` for the schema.
pub fn ref_work_payload(module: &ModuleId, role: &str, extra: Value) -> WorkItem {
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

/// Slow, obviously-correct pipeline. Owns a real `Queue` for durability.
pub struct RefPipeline {
    dir: PathBuf,
    module: ModuleId,
    config: PipelineConfig,
    queue: Queue,
    stage: Stage,
    round: u32,
    stub_failed: bool,
    /// Implementer candidate ids announced for the current round, in order.
    expected_ids: Vec<CandidateId>,
    /// Submitted this round, first-submit order (in-place update on duplicate id).
    submitted: Vec<Candidate>,
    last_verdict: Option<Verdict>,
}

impl std::fmt::Debug for RefPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefPipeline")
            .field("dir", &self.dir)
            .field("module", &self.module)
            .field("config", &self.config)
            .field("stage", &self.stage)
            .field("round", &self.round)
            .field("stub_failed", &self.stub_failed)
            .field("expected_ids", &self.expected_ids)
            .field("submitted", &self.submitted)
            .field("last_verdict", &self.last_verdict)
            .finish_non_exhaustive()
    }
}

impl RefPipeline {
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
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
    pub fn module(&self) -> &ModuleId {
        &self.module
    }
    pub fn config(&self) -> &PipelineConfig {
        &self.config
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

    pub fn start_oracle(&mut self, now: NowMs) -> Result<TaskId> {
        self.require_stage(Stage::Interface)?;
        let item = ref_work_payload(&self.module, ROLE_ORACLE, json!({ "round": self.round }));
        let id = self.enqueue(item, now)?;
        self.stage = Stage::Oracle;
        Ok(id)
    }

    pub fn record_oracle(&mut self, passed_on_stub: bool, now: NowMs) -> Result<()> {
        self.require_stage(Stage::Oracle)?;
        if passed_on_stub {
            return Err(Error::StubPassed);
        }
        let item = ref_work_payload(
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
        let mut item = ref_work_payload(
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
        item.task_id = self.uniquify_control_task_id(item.task_id);
        self.enqueue(item, now)?;
        upsert_candidate(&mut self.submitted, candidate);
        Ok(())
    }

    pub fn start_review(&mut self, now: NowMs) -> Result<TaskId> {
        self.require_stage(Stage::Implement)?;
        let item = ref_work_payload(&self.module, ROLE_REVIEWER, json!({ "round": self.round }));
        let id = self.enqueue(item, now)?;
        self.stage = Stage::Review;
        Ok(id)
    }

    pub fn record_verdict(&mut self, verdict: Verdict, now: NowMs) -> Result<Stage> {
        self.require_stage(Stage::Review)?;
        match &verdict {
            Verdict::Pick(id) => {
                let Some(c) = self.submitted.iter().find(|c| &c.id == id) else {
                    return Err(Error::NoCandidate(id.0.clone()));
                };
                if ref_is_planted_bad(&c.patch) {
                    return Err(Error::PlantedBad);
                }
            }
            Verdict::RejectAll { .. } => {}
        }
        let item = ref_work_payload(
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
                if ref_is_planted_bad(&c.patch) {
                    return Err(Error::PlantedBad);
                }
            }
            _ => {
                return Err(Error::Other("record_merge requires a Pick verdict".into()));
            }
        }
        let item = ref_work_payload(
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

    /// Queue task ids must be unique. A resubmit keeps the canonical
    /// `work_payload` extra (so `open` still sees `op` / candidate / patch) but
    /// suffixes `#1`, `#2`, … onto the control task id when the canonical id is
    /// already in the log. ROLE_* ids are never passed through here.
    fn uniquify_control_task_id(&self, canonical: TaskId) -> TaskId {
        if self.queue.get(&canonical).is_none() {
            return canonical;
        }
        let mut n = 1u64;
        loop {
            let tagged = TaskId(format!("{}#{n}", canonical.0));
            if self.queue.get(&tagged).is_none() {
                return tagged;
            }
            n = n.checked_add(1).expect("task id suffix overflow");
        }
    }

    fn enqueue_implementers(&mut self, now: NowMs) -> Result<Vec<TaskId>> {
        self.expected_ids.clear();
        self.submitted.clear();
        let n = self.config.n_implementers;
        let mut ids = Vec::with_capacity(n as usize);
        for i in 0..n {
            let cid = i.to_string();
            let item = ref_work_payload(
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
                            // Re-apply without re-enqueuing. Implementer tasks
                            // that follow a RejectAll are replayed separately.
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

fn upsert_candidate(submitted: &mut Vec<Candidate>, candidate: Candidate) {
    if let Some(slot) = submitted.iter_mut().find(|c| c.id == candidate.id) {
        *slot = candidate;
    } else {
        submitted.push(candidate);
    }
}
