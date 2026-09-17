//! Exact verifiers for RL rewards and synth rejection sampling
//! (spec 9.3, 9.5, 15.5 D2).
//!
//! Five checkers: sandboxed tests plus mutation, symbolic math, Lean 4
//! kernel, ARC grid match, market resolution. Learned graders, contrastive
//! pairs, and tampering detectors are I3, not this crate.
//!
//! The reward API is F1 `prometheus.reward_request` /
//! `prometheus.reward_response`. [`NowMs`] is injected; nothing here reads
//! the wall clock. `scored_at` is a caller-supplied ISO-8601 string.
//!
//! Code checks run on a D1 [`prometheus_envs::Backend`]. Hidden tests stay
//! in the grader view. Writes to test files are flagged. Mutation testing
//! rejects a suite that a planted mutant still passes.
//!
//! Gate: planted wrong answers are rejected (`passed == false`). Markets
//! use a proper score against the market price, never live trading.

mod engine;
mod math;

use std::collections::BTreeMap;

use prometheus_envs::{Backend, Image, Pool};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Injected clock. Milliseconds since an arbitrary origin. Never wall time.
pub type NowMs = prometheus_envs::NowMs;

pub const SCHEMA_REWARD_REQUEST: &str = "prometheus.reward_request";
pub const SCHEMA_REWARD_RESPONSE: &str = "prometheus.reward_response";
pub const SCHEMA_VERSION: u32 = 1;

/// ARC pass@k (spec 9.3, 10). Two slots.
pub const GRID_PASS_K: usize = 2;

/// Probability clamp for log scores. Keeps `ln` defined.
pub const PROB_EPS: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VerifierId(pub String);

/// Spec 9.3 exact checkers. Not the I3 learned graders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifierKind {
    SandboxedTests,
    SymbolicMath,
    LeanKernel,
    GridMatch,
    MarketResolution,
}

/// F1 `prometheus.reward_response.flags` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Flag {
    Tampering,
    TestWrite,
    Timeout,
    EnvCrash,
    Unverifiable,
}

/// F1 `prometheus.reward_response.evidence.kind` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    HiddenTests,
    Mutation,
    Symbolic,
    Lean,
    Grid,
    Rubric,
    Human,
}

/// F1 evidence object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Evidence {
    pub kind: EvidenceKind,
    pub hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// F1 `prometheus.reward_request` v1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewardRequest {
    pub schema_id: String,
    pub schema_version: u32,
    pub request_id: String,
    pub task_id: String,
    pub rollout_id: String,
    pub trajectory_hash: String,
    pub verifier_id: String,
    pub env_snapshot_id: String,
}

impl RewardRequest {
    pub fn new(
        request_id: impl Into<String>,
        task_id: impl Into<String>,
        rollout_id: impl Into<String>,
        trajectory_hash: impl Into<String>,
        verifier_id: impl Into<String>,
        env_snapshot_id: impl Into<String>,
    ) -> Self {
        Self {
            schema_id: SCHEMA_REWARD_REQUEST.to_string(),
            schema_version: SCHEMA_VERSION,
            request_id: request_id.into(),
            task_id: task_id.into(),
            rollout_id: rollout_id.into(),
            trajectory_hash: trajectory_hash.into(),
            verifier_id: verifier_id.into(),
            env_snapshot_id: env_snapshot_id.into(),
        }
    }
}

/// F1 `prometheus.reward_response` v1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewardResponse {
    pub schema_id: String,
    pub schema_version: u32,
    pub request_id: String,
    pub score: f64,
    pub passed: bool,
    pub flags: Vec<Flag>,
    pub evidence: Vec<Evidence>,
    pub scored_at: String,
}

impl RewardResponse {
    pub fn new(request_id: impl Into<String>, scored_at: impl Into<String>) -> Self {
        Self {
            schema_id: SCHEMA_REWARD_RESPONSE.to_string(),
            schema_version: SCHEMA_VERSION,
            request_id: request_id.into(),
            score: 0.0,
            passed: false,
            flags: Vec::new(),
            evidence: Vec::new(),
            scored_at: scored_at.into(),
        }
    }
}

/// Submission for one of the five checkers.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// Files written into the agent view before hidden tests run.
    Code { files: BTreeMap<String, Vec<u8>> },
    /// Expression. Compared after canonicalize, then by symbolic equivalence.
    Math { latex: String },
    /// Proof artifact checked by a Lean 4 kernel.
    Lean { proof: String },
    /// Up to [`GRID_PASS_K`] predicted output grids. Pass if any matches.
    Grid { grids: Vec<Grid> },
    /// Model probability of the resolved-yes outcome, in `[0, 1]`.
    Forecast { p: f64 },
}

/// One ARC-style grid. Cell values are palette indices.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Grid {
    pub cells: Vec<Vec<u8>>,
}

impl Grid {
    pub fn new(cells: Vec<Vec<u8>>) -> Self {
        Self { cells }
    }

    pub fn rows(&self) -> usize {
        self.cells.len()
    }

    pub fn cols(&self) -> usize {
        self.cells.first().map(|r| r.len()).unwrap_or(0)
    }
}

/// Planted wrong solution used as a mutant (spec 9.3 mutation testing).
///
/// Hidden tests must fail every mutant. A mutant that still passes means
/// the suite is too weak: score is withheld (`Flag::Unverifiable`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mutant {
    pub id: String,
    pub files: BTreeMap<String, Vec<u8>>,
}

impl Mutant {
    pub fn new(id: impl Into<String>, files: BTreeMap<String, Vec<u8>>) -> Self {
        Self {
            id: id.into(),
            files,
        }
    }
}

/// Hidden-test code task. `image.hidden_tests` is the suite. `mutants` are
/// planted wrong answers the suite must reject.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeTask {
    pub image: Image,
    pub mutants: Vec<Mutant>,
    pub run: TestRun,
}

/// How hidden tests are invoked inside the sandbox.
///
/// A run passes iff the tool returns `ok` and stdout equals
/// `expected_stdout`. That is the pass/fail bit mutation testing uses.
#[derive(Debug, Clone, PartialEq)]
pub struct TestRun {
    pub tool: prometheus_envs::Tool,
    pub payload: serde_json::Value,
    pub timeout_s: u32,
    pub expected_stdout: Vec<u8>,
}

impl TestRun {
    pub fn python(payload: serde_json::Value, timeout_s: u32, expected_stdout: Vec<u8>) -> Self {
        Self {
            tool: prometheus_envs::Tool::Python,
            payload,
            timeout_s,
            expected_stdout,
        }
    }
}

impl CodeTask {
    pub fn new(image: Image, run: TestRun) -> Self {
        Self {
            image,
            mutants: Vec::new(),
            run,
        }
    }
}

/// Expected math answer. `expected` is latex or a canonical form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MathTask {
    pub expected: String,
}

impl MathTask {
    pub fn new(expected: impl Into<String>) -> Self {
        Self {
            expected: expected.into(),
        }
    }
}

/// Theorem statement plus the kernel that will check a proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeanTask {
    pub theorem: String,
}

impl LeanTask {
    pub fn new(theorem: impl Into<String>) -> Self {
        Self {
            theorem: theorem.into(),
        }
    }
}

/// Expected output grid. Predictions use pass@[`GRID_PASS_K`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridTask {
    pub expected: Grid,
}

impl GridTask {
    pub fn new(expected: Grid) -> Self {
        Self { expected }
    }
}

/// Resolved binary market. Reward is relative log score against `market_p`.
///
/// `ln p_model(outcome) - ln p_market(outcome)`, probabilities clamped to
/// `[PROB_EPS, 1 - PROB_EPS]`. `passed` is whether the relative score is
/// strictly positive (beats the market). No live trading.
#[derive(Debug, Clone, PartialEq)]
pub struct MarketTask {
    pub market_p: f64,
    pub outcome: bool,
}

impl MarketTask {
    pub fn new(market_p: f64, outcome: bool) -> Self {
        Self { market_p, outcome }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum Error {
    #[error("schema mismatch: {0}")]
    Schema(String),
    #[error("unknown verifier {0}")]
    UnknownVerifier(String),
    #[error("answer does not match verifier kind")]
    WrongKind,
    #[error("planted mutant {id} still passes hidden tests")]
    WeakTests { id: String },
    #[error("hidden tests leaked into the agent view")]
    HiddenTestsVisible,
    #[error("write to test file {path} blocked")]
    TestFileWrite { path: String },
    #[error("probability {0} is not in [0, 1]")]
    BadProbability(f64),
    #[error("grid is empty or ragged")]
    BadGrid,
    #[error("more than {GRID_PASS_K} grid predictions")]
    TooManyGrids,
    #[error("lean kernel rejected the artifact: {0}")]
    Lean(String),
    #[error("sandbox: {0}")]
    Sandbox(String),
    #[error("timeout")]
    Timeout,
    #[error("env crash: {0}")]
    EnvCrash(String),
    #[error("unverifiable: {0}")]
    Unverifiable(String),
}

impl From<prometheus_envs::Error> for Error {
    fn from(e: prometheus_envs::Error) -> Self {
        match e {
            prometheus_envs::Error::Timeout => Error::Timeout,
            prometheus_envs::Error::HiddenTestsVisible => Error::HiddenTestsVisible,
            prometheus_envs::Error::TestFileWrite { path } => Error::TestFileWrite { path },
            other => Error::Sandbox(other.to_string()),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Lean 4 kernel. CPU tests use [`TinyKernel`]. Production may wrap `lean`.
pub trait Kernel {
    fn check(&self, theorem: &str, proof: &str) -> Result<bool>;
}

/// Restricted CPU kernel. No `lean` binary. Accepts a tiny proof language
/// the oracle tests pin down.
pub struct TinyKernel;

impl TinyKernel {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TinyKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel for TinyKernel {
    fn check(&self, theorem: &str, proof: &str) -> Result<bool> {
        math::tiny_kernel_check(theorem, proof)
    }
}

/// Shell-out to a Lean 4 executable. Not used on the CPU path.
pub struct LeanCli {
    pub lean: std::path::PathBuf,
}

impl LeanCli {
    pub fn new(lean: std::path::PathBuf) -> Self {
        Self { lean }
    }
}

impl Kernel for LeanCli {
    fn check(&self, theorem: &str, proof: &str) -> Result<bool> {
        let th = math::collapse_ws(theorem);
        let pr = math::collapse_ws(proof);
        if th.is_empty() {
            return Err(Error::Lean("empty theorem".into()));
        }
        if pr.is_empty() {
            return Err(Error::Lean("empty proof".into()));
        }
        let src = format!("{th}\n{pr}\n");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "prom-d2-{}.lean",
            engine::sha256_hex(src.as_bytes())
        ));
        std::fs::write(&path, src.as_bytes()).map_err(|e| Error::Lean(e.to_string()))?;
        let output = std::process::Command::new(&self.lean).arg(&path).output();
        let _ = std::fs::remove_file(&path);
        let output = output.map_err(|e| Error::Lean(e.to_string()))?;
        Ok(output.status.success())
    }
}

/// Sandboxed tests plus mutation. Hidden tests run in the grader view.
pub struct SandboxedTests<B: Backend> {
    pub id: VerifierId,
    pub task: CodeTask,
    pool: Pool<B>,
}

impl<B: Backend> SandboxedTests<B> {
    pub fn new(id: impl Into<String>, task: CodeTask, pool: Pool<B>) -> Self {
        Self {
            id: VerifierId(id.into()),
            task,
            pool,
        }
    }

    pub fn kind(&self) -> VerifierKind {
        VerifierKind::SandboxedTests
    }

    pub fn pool(&self) -> &Pool<B> {
        &self.pool
    }

    pub fn pool_mut(&mut self) -> &mut Pool<B> {
        &mut self.pool
    }

    /// Score `answer`. Planted mutants must fail. A correct answer that
    /// passes while mutants fail scores 1. A wrong answer scores 0.
    pub fn verify(
        &mut self,
        req: &RewardRequest,
        answer: &Answer,
        scored_at: &str,
        now: NowMs,
    ) -> Result<RewardResponse> {
        engine::verify_sandboxed(&mut self.pool, &self.task, req, answer, scored_at, now)
    }
}

/// Exact match after canonicalize, then symbolic equivalence.
pub struct SymbolicMath {
    pub id: VerifierId,
    pub task: MathTask,
}

impl SymbolicMath {
    pub fn new(id: impl Into<String>, task: MathTask) -> Self {
        Self {
            id: VerifierId(id.into()),
            task,
        }
    }

    pub fn kind(&self) -> VerifierKind {
        VerifierKind::SymbolicMath
    }

    /// Strip latex noise and whitespace. Used before equivalence.
    pub fn canonicalize(input: &str) -> String {
        math::canonicalize(input)
    }

    pub fn equivalent(left: &str, right: &str) -> bool {
        math::equivalent(left, right)
    }

    pub fn verify(
        &self,
        req: &RewardRequest,
        answer: &Answer,
        scored_at: &str,
        _now: NowMs,
    ) -> Result<RewardResponse> {
        let Answer::Math { latex } = answer else {
            return Err(Error::WrongKind);
        };
        let ok = Self::equivalent(latex, &self.task.expected);
        let mut resp = engine::base_response(req, scored_at, if ok { 1.0 } else { 0.0 }, ok);
        resp.evidence
            .push(engine::evidence(EvidenceKind::Symbolic, latex));
        Ok(resp)
    }
}

/// Lean 4 kernel check.
pub struct LeanCheck<K: Kernel> {
    pub id: VerifierId,
    pub task: LeanTask,
    pub kernel: K,
}

impl<K: Kernel> LeanCheck<K> {
    pub fn new(id: impl Into<String>, task: LeanTask, kernel: K) -> Self {
        Self {
            id: VerifierId(id.into()),
            task,
            kernel,
        }
    }

    pub fn kind(&self) -> VerifierKind {
        VerifierKind::LeanKernel
    }

    pub fn verify(
        &self,
        req: &RewardRequest,
        answer: &Answer,
        scored_at: &str,
        _now: NowMs,
    ) -> Result<RewardResponse> {
        let Answer::Lean { proof } = answer else {
            return Err(Error::WrongKind);
        };
        let ok = self.kernel.check(&self.task.theorem, proof)?;
        let mut resp = engine::base_response(req, scored_at, if ok { 1.0 } else { 0.0 }, ok);
        resp.evidence
            .push(engine::evidence(EvidenceKind::Lean, proof));
        Ok(resp)
    }
}

fn grid_valid(g: &Grid) -> Result<()> {
    if g.cells.is_empty() || g.cells.iter().any(|r| r.is_empty()) {
        return Err(Error::BadGrid);
    }
    let w = g.cells[0].len();
    if g.cells.iter().any(|r| r.len() != w) {
        return Err(Error::BadGrid);
    }
    Ok(())
}

/// Exact grid match, pass@[`GRID_PASS_K`].
pub struct GridMatch {
    pub id: VerifierId,
    pub task: GridTask,
}

impl GridMatch {
    pub fn new(id: impl Into<String>, task: GridTask) -> Self {
        Self {
            id: VerifierId(id.into()),
            task,
        }
    }

    pub fn kind(&self) -> VerifierKind {
        VerifierKind::GridMatch
    }

    pub fn verify(
        &self,
        req: &RewardRequest,
        answer: &Answer,
        scored_at: &str,
        _now: NowMs,
    ) -> Result<RewardResponse> {
        let Answer::Grid { grids } = answer else {
            return Err(Error::WrongKind);
        };
        if grids.len() > GRID_PASS_K {
            return Err(Error::TooManyGrids);
        }
        grid_valid(&self.task.expected)?;
        for g in grids {
            grid_valid(g)?;
        }
        let ok = grids.iter().any(|g| g == &self.task.expected);
        let mut resp = engine::base_response(req, scored_at, if ok { 1.0 } else { 0.0 }, ok);
        resp.evidence.push(engine::evidence(
            EvidenceKind::Grid,
            &format!("{:?}", self.task.expected),
        ));
        Ok(resp)
    }
}

/// Relative log score against a resolved market price. No live trading.
pub struct MarketResolution {
    pub id: VerifierId,
    pub task: MarketTask,
}

impl MarketResolution {
    pub fn new(id: impl Into<String>, task: MarketTask) -> Self {
        Self {
            id: VerifierId(id.into()),
            task,
        }
    }

    pub fn kind(&self) -> VerifierKind {
        VerifierKind::MarketResolution
    }

    /// `ln p(outcome) - ln p_market(outcome)` with [`PROB_EPS`] clamp.
    pub fn relative_log_score(model_p: f64, market_p: f64, outcome: bool) -> Result<f64> {
        if !(0.0..=1.0).contains(&model_p) {
            return Err(Error::BadProbability(model_p));
        }
        if !(0.0..=1.0).contains(&market_p) {
            return Err(Error::BadProbability(market_p));
        }
        let p_out = if outcome { model_p } else { 1.0 - model_p };
        let m_out = if outcome { market_p } else { 1.0 - market_p };
        let pm = p_out.clamp(PROB_EPS, 1.0 - PROB_EPS);
        let pk = m_out.clamp(PROB_EPS, 1.0 - PROB_EPS);
        Ok(pm.ln() - pk.ln())
    }

    pub fn verify(
        &self,
        req: &RewardRequest,
        answer: &Answer,
        scored_at: &str,
        _now: NowMs,
    ) -> Result<RewardResponse> {
        let Answer::Forecast { p } = answer else {
            return Err(Error::WrongKind);
        };
        let score = Self::relative_log_score(*p, self.task.market_p, self.task.outcome)?;
        Ok(engine::base_response(req, scored_at, score, score > 0.0))
    }
}

/// Dispatch across the five checkers. Code holds the D1 pool.
pub enum AnyVerifier<B: Backend, K: Kernel = TinyKernel> {
    Code(Box<SandboxedTests<B>>),
    Math(SymbolicMath),
    Lean(LeanCheck<K>),
    Grid(GridMatch),
    Market(MarketResolution),
}

impl<B: Backend, K: Kernel> AnyVerifier<B, K> {
    pub fn id(&self) -> &VerifierId {
        match self {
            Self::Code(v) => &v.id,
            Self::Math(v) => &v.id,
            Self::Lean(v) => &v.id,
            Self::Grid(v) => &v.id,
            Self::Market(v) => &v.id,
        }
    }

    pub fn kind(&self) -> VerifierKind {
        match self {
            Self::Code(v) => v.kind(),
            Self::Math(v) => v.kind(),
            Self::Lean(v) => v.kind(),
            Self::Grid(v) => v.kind(),
            Self::Market(v) => v.kind(),
        }
    }

    pub fn verify(
        &mut self,
        req: &RewardRequest,
        answer: &Answer,
        scored_at: &str,
        now: NowMs,
    ) -> Result<RewardResponse> {
        match self {
            Self::Code(v) => v.verify(req, answer, scored_at, now),
            Self::Math(v) => v.verify(req, answer, scored_at, now),
            Self::Lean(v) => v.verify(req, answer, scored_at, now),
            Self::Grid(v) => v.verify(req, answer, scored_at, now),
            Self::Market(v) => v.verify(req, answer, scored_at, now),
        }
    }
}

/// Named verifiers. Lookup by F1 `verifier_id`.
pub struct Registry<B: Backend, K: Kernel = TinyKernel> {
    inner: BTreeMap<String, AnyVerifier<B, K>>,
}

impl<B: Backend, K: Kernel> Registry<B, K> {
    pub fn new() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, v: AnyVerifier<B, K>) -> Result<()> {
        let key = v.id().0.clone();
        if self.inner.contains_key(&key) {
            return Err(Error::Schema(format!("duplicate verifier {key}")));
        }
        self.inner.insert(key, v);
        Ok(())
    }

    pub fn get_mut(&mut self, id: &str) -> Result<&mut AnyVerifier<B, K>> {
        self.inner
            .get_mut(id)
            .ok_or_else(|| Error::UnknownVerifier(id.to_string()))
    }

    pub fn verify(
        &mut self,
        req: &RewardRequest,
        answer: &Answer,
        scored_at: &str,
        now: NowMs,
    ) -> Result<RewardResponse> {
        if req.schema_id != SCHEMA_REWARD_REQUEST || req.schema_version != SCHEMA_VERSION {
            return Err(Error::Schema(format!(
                "{} v{}",
                req.schema_id, req.schema_version
            )));
        }
        let v = self.get_mut(&req.verifier_id)?;
        v.verify(req, answer, scored_at, now)
    }
}

impl<B: Backend, K: Kernel> Default for Registry<B, K> {
    fn default() -> Self {
        Self::new()
    }
}
