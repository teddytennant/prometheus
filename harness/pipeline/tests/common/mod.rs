//! Shared builders and assertions for H10 `prometheus-pipeline` oracle tests.
//!
//! Documented rules the implementer must match (also see `tests/reference`):
//!
//! ## Layout
//! - `Pipeline::create(dir, module, config, queue_config)` fails if `dir` exists.
//!   It creates `dir` and an H3 `Queue` at `dir/queue` via `Queue::create`.
//!   Starts at `Stage::Interface`, `round = 0`, `stub_failed = false`, no candidates.
//!   Create writes no queue events.
//! - `Pipeline::open(dir, module, config, queue_config)` opens that queue and
//!   reconstructs `stage` / `round` / `stub_failed` / current-round candidates
//!   from the queue log. `module` and `config` are the values passed to `open`.
//! - `pipeline.dir()` is `dir`. `pipeline.queue().dir()` is `dir/queue`.
//!
//! ## Pure helpers
//! - `is_planted_bad(patch)` is true iff `patch` contains `PLANTED_BAD_MARKER`
//!   as a contiguous byte slice. Empty, unrelated, and any truncated marker
//!   (length < marker) are false. Case-sensitive raw bytes, not UTF-8 logic.
//! - `review(candidates, tests_ok)`: lengths must match or `Error::Other`.
//!   Walk the slice in order; pick the first whose `tests_ok[i]` is true and
//!   whose patch is not planted-bad. A planted-bad candidate is never picked
//!   even if tests passed. If none qualify, `Ok(RejectAll { defects })` with
//!   **non-empty** defects. Defect wording is not locked.
//! - `merge_allowed(verdict, candidates, tests_ok)`:
//!   - lengths must match or `Error::Other`
//!   - `RejectAll` → `Error::Other` (not Ok)
//!   - `Pick` of unknown id → `Error::NoCandidate`
//!   - `Pick` of planted-bad (even if tests passed / failed) → `Error::PlantedBad`
//!   - `Pick` of a candidate with `tests_ok[i] == false` → `Error::CandidateFailed`
//!   - otherwise `Ok(())`
//! - `work_payload(module, role, extra) -> WorkItem` (deterministic):
//!   Payload is a JSON object. If `extra` is an object its keys are copied
//!   first; otherwise `extra` is stored under `"extra"`. Then
//!   `"module" = module.0` and `"role" = role` are set (they win over extra).
//!   `task_id.0` is `{role}/{module}/{round}` plus `/{op}` if extra/payload
//!   has string `"op"`, plus `/{candidate}` if it has string `"candidate"`.
//!   `round` is payload `"round"` as u64, else 0.
//!
//! ## Stage machine (`WrongStage` on illegal calls; `actual` is the current stage)
//! ```text
//! Interface --start_oracle--> Oracle
//!   start_oracle enqueues one work item: work_payload(module, ROLE_ORACLE, {"round": round})
//! Oracle --record_oracle(passed_on_stub=false)--> StubMustFail, stub_failed=true
//!        --record_oracle(passed_on_stub=true)--> Err(StubPassed), stay Oracle, stub_failed stays false
//! start_implementers:
//!   if !stub_failed → Err(StubNotFailed) (even at Interface)
//!   else if stage != StubMustFail → WrongStage { expected: StubMustFail, actual }
//!   else enqueue n_implementers ROLE_IMPLEMENTER tasks, stage=Implement
//!   Candidate ids are decimal "0" .. "{n-1}" (no leading zeros).
//!   extra = {"round": round, "candidate": id}
//! Implement --submit_candidate--> still Implement
//!   Unknown id (not in this round's implementer ids) → NoCandidate, no record
//!   Duplicate id: last write wins, first-submit order kept
//!   planted_bad is derived from is_planted_bad(patch); submit still succeeds
//! Implement --start_review--> Review
//!   enqueues one ROLE_REVIEWER task, extra = {"round": round}
//!   allowed even if zero candidates submitted
//! Review --record_verdict(Pick of submitted, not planted-bad)--> Merged
//!        --record_verdict(Pick planted-bad)--> Err(PlantedBad), stay Review
//!        --record_verdict(Pick unknown)--> Err(NoCandidate), stay Review
//!        --record_verdict(RejectAll)--> round += 1; if round >= max_rounds then
//!             Blocked else Implement and enqueue a new batch of implementers
//!           current-round candidates cleared on RejectAll
//! Merged --record_merge--> stays Merged (records MergeRecord; module must match)
//! Blocked: every mutating call is WrongStage
//! ```
//! `record_oracle(false)` stops at `StubMustFail`; it does **not** auto-start
//! implementers. Default config: `n_implementers = 3`, `max_rounds = 3`
//! (three review attempts at rounds 0, 1, 2; a third RejectAll blocks).
//!
//! Persistence: every successful mutating call that must survive `open` is
//! reflected in the queue log (work items and/or control items). Control
//! items use role `"pipeline"` and `"op"` in `work_payload` extra; tests of
//! enqueue filter by `ROLE_*` and do not require a particular control schema
//! beyond production `open` reconstructing the same state.
//!
//! `candidates()` returns this round's submitted candidates in first-submit
//! order. After RejectAll / new round it is empty until new submits.
//!
//! No GPU coverage in H10 (no `gpu` marker / feature-gated tests).
//! Production `src/` must never import this module.

#![allow(dead_code)]

use prometheus_leases::{NowMs, Queue, QueueConfig, TaskId, EVENT_ENQUEUED};
use prometheus_pipeline::{
    Candidate, CandidateId, Error, MergeRecord, ModuleId, Pipeline, PipelineConfig, Result, Stage,
    Verdict, DEFAULT_MAX_ROUNDS, DEFAULT_N_IMPLEMENTERS,
};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use super::reference::RefPipeline;

pub const NOW: NowMs = 1_000;
pub const MODULE_NAME: &str = "mod-h10";

pub fn module() -> ModuleId {
    ModuleId(MODULE_NAME.to_string())
}

pub fn default_pipeline_config() -> PipelineConfig {
    PipelineConfig::default()
}

pub fn default_queue_config() -> QueueConfig {
    QueueConfig::default()
}

pub fn cfg(n_implementers: u32, max_rounds: u32) -> PipelineConfig {
    PipelineConfig {
        n_implementers,
        max_rounds,
    }
}

/// Parent temp dir plus not-yet-created `pipeline/` child.
pub fn fresh_dir() -> (tempfile::TempDir, PathBuf) {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("pipeline");
    (parent, dir)
}

pub fn queue_dir(pipeline_dir: &Path) -> PathBuf {
    pipeline_dir.join("queue")
}

pub fn cand(id: &str, angle: &str, patch: &[u8]) -> Candidate {
    Candidate {
        id: CandidateId(id.to_string()),
        angle: angle.to_string(),
        patch: patch.to_vec(),
    }
}

pub fn good_patch() -> &'static [u8] {
    b"fn implement() { /* honest */ }"
}

pub fn planted_patch() -> Vec<u8> {
    let mut v = b"fn evil() { ".to_vec();
    v.extend_from_slice(prometheus_pipeline::PLANTED_BAD_MARKER);
    v.extend_from_slice(b" }");
    v
}

pub fn merge_record(module: &ModuleId, passed: u32, failed: u32) -> MergeRecord {
    MergeRecord {
        module: module.clone(),
        spec_section: "15.5".to_string(),
        commit: "deadbeef".to_string(),
        tests_passed: passed,
        tests_failed: failed,
        gpu_job_ids: Vec::new(),
    }
}

pub fn unwrap_err<T>(r: Result<T>, what: &str) -> Error {
    match r {
        Ok(_) => panic!("expected error ({what}), got Ok"),
        Err(e) => e,
    }
}

pub fn assert_stub_passed(e: Error) {
    match e {
        Error::StubPassed => {}
        other => panic!("expected StubPassed, got {other:?}"),
    }
}

pub fn assert_stub_not_failed(e: Error) {
    match e {
        Error::StubNotFailed => {}
        other => panic!("expected StubNotFailed, got {other:?}"),
    }
}

pub fn assert_planted_bad(e: Error) {
    match e {
        Error::PlantedBad => {}
        other => panic!("expected PlantedBad, got {other:?}"),
    }
}

pub fn assert_candidate_failed(e: Error) {
    match e {
        Error::CandidateFailed => {}
        other => panic!("expected CandidateFailed, got {other:?}"),
    }
}

pub fn assert_no_candidate(e: Error, id: &str) {
    match e {
        Error::NoCandidate(got) => assert_eq!(got, id, "NoCandidate id"),
        other => panic!("expected NoCandidate({id}), got {other:?}"),
    }
}

pub fn assert_wrong_stage(e: Error, actual: Stage) {
    match e {
        Error::WrongStage { actual: got, .. } => {
            assert_eq!(got, actual, "WrongStage.actual")
        }
        other => panic!("expected WrongStage actual={actual:?}, got {other:?}"),
    }
}

pub fn assert_wrong_stage_expected(e: Error, expected: Stage, actual: Stage) {
    match e {
        Error::WrongStage {
            expected: exp,
            actual: got,
        } => {
            assert_eq!(got, actual, "WrongStage.actual");
            assert_eq!(exp, expected, "WrongStage.expected");
        }
        other => panic!(
            "expected WrongStage {{ expected: {expected:?}, actual: {actual:?} }}, got {other:?}"
        ),
    }
}

pub fn assert_other(e: Error) {
    match e {
        Error::Other(_) => {}
        other => panic!("expected Other, got {other:?}"),
    }
}

/// Same error *kind* (and id / stage fields where they matter). Messages on
/// `Other` / `Queue` / `NotFound` are not locked.
pub fn assert_same_error(prod: Error, refer: Error, ctx: &str) {
    match (prod, refer) {
        (Error::StubPassed, Error::StubPassed)
        | (Error::StubNotFailed, Error::StubNotFailed)
        | (Error::PlantedBad, Error::PlantedBad)
        | (Error::CandidateFailed, Error::CandidateFailed) => {}
        (
            Error::WrongStage {
                expected: e1,
                actual: a1,
            },
            Error::WrongStage {
                expected: e2,
                actual: a2,
            },
        ) => {
            assert_eq!(a1, a2, "{ctx} WrongStage actual");
            assert_eq!(e1, e2, "{ctx} WrongStage expected");
        }
        (Error::NoCandidate(a), Error::NoCandidate(b)) => {
            assert_eq!(a, b, "{ctx} NoCandidate")
        }
        (Error::MaxRounds(a), Error::MaxRounds(b)) => {
            assert_eq!(a, b, "{ctx} MaxRounds")
        }
        (Error::NotFound(_), Error::NotFound(_))
        | (Error::Queue(_), Error::Queue(_))
        | (Error::Other(_), Error::Other(_)) => {}
        (a, b) => panic!("{ctx}: production {a:?} vs reference {b:?}"),
    }
}

pub fn enqueued_payloads(queue: &Queue) -> Vec<Value> {
    queue
        .log()
        .iter()
        .filter(|e| e.event_type == EVENT_ENQUEUED)
        .map(|e| e.payload.clone())
        .collect()
}

pub fn payloads_with_role(queue: &Queue, role: &str) -> Vec<Value> {
    enqueued_payloads(queue)
        .into_iter()
        .filter(|p| p.get("role").and_then(Value::as_str) == Some(role))
        .collect()
}

pub fn enqueued_task_ids(queue: &Queue) -> Vec<String> {
    queue
        .log()
        .iter()
        .filter(|e| e.event_type == EVENT_ENQUEUED)
        .filter_map(|e| e.task_id.clone())
        .collect()
}

pub trait Pipe {
    fn dir(&self) -> &Path;
    fn queue(&self) -> &Queue;
    fn module(&self) -> &ModuleId;
    fn config(&self) -> &PipelineConfig;
    fn stage(&self) -> Stage;
    fn round(&self) -> u32;
    fn stub_failed(&self) -> bool;
    fn start_oracle(&mut self, now: NowMs) -> Result<TaskId>;
    fn record_oracle(&mut self, passed_on_stub: bool, now: NowMs) -> Result<()>;
    fn start_implementers(&mut self, now: NowMs) -> Result<Vec<TaskId>>;
    fn submit_candidate(&mut self, candidate: Candidate, now: NowMs) -> Result<()>;
    fn start_review(&mut self, now: NowMs) -> Result<TaskId>;
    fn record_verdict(&mut self, verdict: Verdict, now: NowMs) -> Result<Stage>;
    fn record_merge(&mut self, record: MergeRecord, now: NowMs) -> Result<()>;
    fn candidates(&self) -> Result<Vec<Candidate>>;
}

impl Pipe for Pipeline {
    fn dir(&self) -> &Path {
        Pipeline::dir(self)
    }
    fn queue(&self) -> &Queue {
        Pipeline::queue(self)
    }
    fn module(&self) -> &ModuleId {
        Pipeline::module(self)
    }
    fn config(&self) -> &PipelineConfig {
        Pipeline::config(self)
    }
    fn stage(&self) -> Stage {
        Pipeline::stage(self)
    }
    fn round(&self) -> u32 {
        Pipeline::round(self)
    }
    fn stub_failed(&self) -> bool {
        Pipeline::stub_failed(self)
    }
    fn start_oracle(&mut self, now: NowMs) -> Result<TaskId> {
        Pipeline::start_oracle(self, now)
    }
    fn record_oracle(&mut self, passed_on_stub: bool, now: NowMs) -> Result<()> {
        Pipeline::record_oracle(self, passed_on_stub, now)
    }
    fn start_implementers(&mut self, now: NowMs) -> Result<Vec<TaskId>> {
        Pipeline::start_implementers(self, now)
    }
    fn submit_candidate(&mut self, candidate: Candidate, now: NowMs) -> Result<()> {
        Pipeline::submit_candidate(self, candidate, now)
    }
    fn start_review(&mut self, now: NowMs) -> Result<TaskId> {
        Pipeline::start_review(self, now)
    }
    fn record_verdict(&mut self, verdict: Verdict, now: NowMs) -> Result<Stage> {
        Pipeline::record_verdict(self, verdict, now)
    }
    fn record_merge(&mut self, record: MergeRecord, now: NowMs) -> Result<()> {
        Pipeline::record_merge(self, record, now)
    }
    fn candidates(&self) -> Result<Vec<Candidate>> {
        Pipeline::candidates(self)
    }
}

impl Pipe for RefPipeline {
    fn dir(&self) -> &Path {
        RefPipeline::dir(self)
    }
    fn queue(&self) -> &Queue {
        RefPipeline::queue(self)
    }
    fn module(&self) -> &ModuleId {
        RefPipeline::module(self)
    }
    fn config(&self) -> &PipelineConfig {
        RefPipeline::config(self)
    }
    fn stage(&self) -> Stage {
        RefPipeline::stage(self)
    }
    fn round(&self) -> u32 {
        RefPipeline::round(self)
    }
    fn stub_failed(&self) -> bool {
        RefPipeline::stub_failed(self)
    }
    fn start_oracle(&mut self, now: NowMs) -> Result<TaskId> {
        RefPipeline::start_oracle(self, now)
    }
    fn record_oracle(&mut self, passed_on_stub: bool, now: NowMs) -> Result<()> {
        RefPipeline::record_oracle(self, passed_on_stub, now)
    }
    fn start_implementers(&mut self, now: NowMs) -> Result<Vec<TaskId>> {
        RefPipeline::start_implementers(self, now)
    }
    fn submit_candidate(&mut self, candidate: Candidate, now: NowMs) -> Result<()> {
        RefPipeline::submit_candidate(self, candidate, now)
    }
    fn start_review(&mut self, now: NowMs) -> Result<TaskId> {
        RefPipeline::start_review(self, now)
    }
    fn record_verdict(&mut self, verdict: Verdict, now: NowMs) -> Result<Stage> {
        RefPipeline::record_verdict(self, verdict, now)
    }
    fn record_merge(&mut self, record: MergeRecord, now: NowMs) -> Result<()> {
        RefPipeline::record_merge(self, record, now)
    }
    fn candidates(&self) -> Result<Vec<Candidate>> {
        RefPipeline::candidates(self)
    }
}

pub fn assert_state<P: Pipe, R: Pipe>(prod: &P, refer: &R, ctx: &str) {
    assert_eq!(prod.stage(), refer.stage(), "{ctx} stage");
    assert_eq!(prod.round(), refer.round(), "{ctx} round");
    assert_eq!(prod.stub_failed(), refer.stub_failed(), "{ctx} stub_failed");
    assert_eq!(prod.module(), refer.module(), "{ctx} module");
    assert_eq!(
        prod.config().n_implementers,
        refer.config().n_implementers,
        "{ctx} n_implementers"
    );
    assert_eq!(
        prod.config().max_rounds,
        refer.config().max_rounds,
        "{ctx} max_rounds"
    );
    let pc = prod.candidates().expect("{ctx} prod candidates");
    let rc = refer.candidates().expect("{ctx} ref candidates");
    assert_eq!(pc, rc, "{ctx} candidates");
}

pub fn assert_fresh_interface<P: Pipe>(
    p: &P,
    dir: &Path,
    module: &ModuleId,
    config: &PipelineConfig,
) {
    assert_eq!(p.dir(), dir);
    assert_eq!(p.queue().dir(), queue_dir(dir));
    assert_eq!(p.module(), module);
    assert_eq!(p.stage(), Stage::Interface);
    assert_eq!(p.round(), 0);
    assert!(!p.stub_failed());
    assert_eq!(p.config().n_implementers, config.n_implementers);
    assert_eq!(p.config().max_rounds, config.max_rounds);
    assert!(p.candidates().expect("candidates").is_empty());
}

pub fn drive_to_oracle<P: Pipe>(p: &mut P, now: NowMs) -> TaskId {
    p.start_oracle(now).expect("start_oracle")
}

pub fn drive_to_stub_must_fail<P: Pipe>(p: &mut P, now: NowMs) {
    drive_to_oracle(p, now);
    p.record_oracle(false, now).expect("record_oracle false");
    assert_eq!(p.stage(), Stage::StubMustFail);
    assert!(p.stub_failed());
}

pub fn drive_to_implement<P: Pipe>(p: &mut P, now: NowMs) -> Vec<TaskId> {
    drive_to_stub_must_fail(p, now);
    let ids = p.start_implementers(now).expect("start_implementers");
    assert_eq!(p.stage(), Stage::Implement);
    ids
}

pub fn drive_to_review<P: Pipe>(p: &mut P, now: NowMs, patches: &[&[u8]]) -> TaskId {
    let n = p.config().n_implementers as usize;
    assert_eq!(
        patches.len(),
        n,
        "drive_to_review: one patch per implementer"
    );
    let _ = drive_to_implement(p, now);
    for (i, patch) in patches.iter().enumerate() {
        p.submit_candidate(cand(&i.to_string(), &format!("angle-{i}"), patch), now)
            .expect("submit");
    }
    let tid = p.start_review(now).expect("start_review");
    assert_eq!(p.stage(), Stage::Review);
    tid
}

pub fn pick(id: &str) -> Verdict {
    Verdict::Pick(CandidateId(id.to_string()))
}

pub fn reject_all(defects: &[&str]) -> Verdict {
    Verdict::RejectAll {
        defects: defects.iter().map(|s| (*s).to_string()).collect(),
    }
}

pub fn defaults_are_three() {
    // Used after create so this is not a stub-passing constants-only test.
    let _ = DEFAULT_N_IMPLEMENTERS;
    let _ = DEFAULT_MAX_ROUNDS;
}

/// Oracle extra object locked for `start_oracle` / `start_review`.
pub fn round_extra(round: u32) -> Value {
    json!({ "round": round })
}

pub fn implementer_extra(round: u32, candidate: &str) -> Value {
    json!({ "round": round, "candidate": candidate })
}
