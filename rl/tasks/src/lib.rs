//! Task factories: wave 1 (D3) and wave 2 (I6, spec 9.4, 15.5).
//!
//! Wave 1 mints an F1 [`TaskSpec`], attaches a D2 exact verifier, and drops
//! tasks an injected solver always solves or never solves. Wave 2 factories
//! mint research, long-horizon, ARC, forecast, and open-ended tasks.
//!
//! Solvers are a [`Solver`] so CPU tests inject [`ScriptedSolver`]. F5
//! serving can wrap this later; nothing here starts an engine.
//!
//! [`NowMs`] is injected. `created_at` is a caller-supplied ISO-8601 string.
//! Nothing here reads the wall clock.
//!
//! Gate: open-weights solve rate in (0, 100%). 0% and 100% are dropped.

use std::collections::BTreeMap;

use prometheus_envs::GRADER_ROOT;
use prometheus_verifiers::{Mutant, TestRun};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub use prometheus_envs::{Image, NowMs};
pub use prometheus_rewards::{Criterion, Rubric};
use prometheus_verifiers::PROB_EPS;
pub use prometheus_verifiers::{
    CodeTask, Grid, GridTask, MarketTask, MathTask, VerifierId, VerifierKind, GRID_PASS_K,
};

pub const SCHEMA_TASK_SPEC: &str = "prometheus.task_spec";
pub const SCHEMA_VERSION: u32 = 1;

/// Curriculum starts here (spec 9.4). Raised when success at the current
/// horizon crosses [`RAISE_THRESHOLD`].
pub const START_TOOL_CALLS: u32 = 10;

/// Raise the horizon when the policy's success rate at the current cap
/// exceeds this.
pub const RAISE_THRESHOLD: f64 = 0.5;

const CODE_HORIZON_S: u32 = 30;
const SWE_HORIZON_S: u32 = 60;
const MATH_HORIZON_S: u32 = 30;
const CODE_MAX_TOOL_CALLS: u32 = START_TOOL_CALLS;
const SWE_MAX_TOOL_CALLS: u32 = START_TOOL_CALLS * 2;
const MATH_MAX_TOOL_CALLS: u32 = START_TOOL_CALLS;
const CODE_TIMEOUT_S: u32 = 5;
const SWE_TIMEOUT_S: u32 = 10;
const ARC_HORIZON_S: u32 = 30;
const ARC_MAX_TOOL_CALLS: u32 = START_TOOL_CALLS;
const FORECAST_HORIZON_S: u32 = 30;
const FORECAST_MAX_TOOL_CALLS: u32 = START_TOOL_CALLS;
const OPEN_ENDED_HORIZON_S: u32 = 30;
const OPEN_ENDED_MAX_TOOL_CALLS: u32 = START_TOOL_CALLS;
const RESEARCH_HORIZON_S: u32 = 600;
const RESEARCH_MAX_TOOL_CALLS: u32 = START_TOOL_CALLS;
const LONG_HORIZON_S: u32 = 3600;
const LONG_HORIZON_MAX_TOOL_CALLS: u32 = 2000;

const MATH_VERIFIER_ID: &str = "symbolic-math";
const CODE_VERIFIER_ID: &str = "sandboxed-tests";
const ARC_VERIFIER_ID: &str = "grid-match";
const FORECAST_VERIFIER_ID: &str = "market-resolution";
const RESEARCH_VERIFIER_ID: &str = "research-numeric";
const LONG_HORIZON_VERIFIER_ID: &str = "long-horizon";
const OPEN_ENDED_VERIFIER_ID: &str = "open-ended-rubric";
const MAX_CELL: u8 = 9;

const HIDDEN_TEST_NAME: &str = "hidden.py";
const HIDDEN_TEST_BODY: &[u8] = b"# hidden\nprint(open('/workspace/out.txt').read())\n";
const PYTHON_CODE: &str = "print(open('/workspace/out.txt').read())";
const CODE_MUTANT_ID: &str = "wrong-stdout";
const SWE_MUTANT_ID: &str = "wrong-patch";
const MUTANT_PATH: &str = "/workspace/out.txt";
const CODE_MUTANT_BODY: &[u8] = b"WRONG\n";
const SWE_MUTANT_BODY: &[u8] = b"WRONG_SWE\n";
const SWE_REPO_PATH: &str = "/workspace/repo";

const HEX: &[u8; 16] = b"0123456789abcdef";

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

fn statement_hash(statement: &str) -> String {
    sha256_hex(statement.as_bytes())
}

/// D1 hidden-tests hash: sorted `(path, NUL, bytes)` concatenation.
fn hidden_files_hash(hidden: &BTreeMap<String, Vec<u8>>) -> String {
    let mut h = Sha256::new();
    for (path, bytes) in hidden {
        h.update(path.as_bytes());
        h.update([0u8]);
        h.update(bytes);
    }
    hex_lower(&h.finalize())
}

fn math_expected(body: &str) -> String {
    sha256_hex(format!("math-expected:{body}").as_bytes())
}

fn code_expected_stdout(body: &str) -> Vec<u8> {
    sha256_hex(format!("code-stdout:{body}").as_bytes()).into_bytes()
}

fn swe_expected_stdout(body: &str) -> Vec<u8> {
    sha256_hex(format!("swe-stdout:{body}").as_bytes()).into_bytes()
}

fn task_id(factory_id: &str, source_id: &str) -> String {
    format!("{factory_id}:{source_id}")
}

fn image_id(factory_id: &str, source_id: &str) -> String {
    format!("img:{factory_id}:{source_id}")
}

fn hidden_test_path() -> String {
    format!("{GRADER_ROOT}/{HIDDEN_TEST_NAME}")
}

fn python_payload() -> serde_json::Value {
    serde_json::json!({ "code": PYTHON_CODE })
}

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

/// Minted task: F1 spec plus the D2 / I3 checker payload. Hidden tests live
/// on the verifier payload, never in the public statement.
#[derive(Debug, Clone, PartialEq)]
pub struct MintedTask {
    pub spec: TaskSpec,
    pub verifier_kind: VerifierKind,
    pub verifier_id: VerifierId,
    pub statement: String,
    pub code: Option<CodeTask>,
    pub math: Option<MathTask>,
    pub grid: Option<GridTask>,
    pub market: Option<MarketTask>,
    pub research: Option<ResearchTask>,
    pub long_horizon: Option<LongHorizonTask>,
    pub open_ended: Option<OpenEndedTask>,
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
        solver: &mut S,
        task: &MintedTask,
        n: u32,
        now: NowMs,
    ) -> Result<SolveRate> {
        if n == 0 {
            return Err(Error::BadN);
        }
        let mut passed = 0u32;
        for _ in 0..n {
            let attempt = solver.attempt(task.statement(), now)?;
            if score_attempt(task, &attempt)? {
                passed = passed.saturating_add(1);
            }
        }
        Ok(SolveRate { passed, n })
    }

    /// Keep only when [`SolveRate::in_open_interval`]. 0% is
    /// [`Error::Impossible`]. 100% is [`Error::Trivial`]. `n == 0` is
    /// [`Error::BadN`].
    pub fn keep(rate: SolveRate) -> Result<SolveRate> {
        if rate.n == 0 {
            return Err(Error::BadN);
        }
        if rate.passed == 0 {
            return Err(Error::Impossible);
        }
        if rate.passed >= rate.n {
            return Err(Error::Trivial);
        }
        Ok(rate)
    }
}

/// Math: attempt equals `MathTask.expected`.
/// Code: attempt bytes equal `TestRun.expected_stdout`.
/// Wave-2 payloads: grid / market / research / long-horizon / open-ended.
/// No payload: [`Error::Unverifiable`]. Math wins if both math and code are set.
fn score_attempt(task: &MintedTask, attempt: &str) -> Result<bool> {
    if let Some(math) = &task.math {
        return Ok(attempt == math.expected);
    }
    if let Some(code) = &task.code {
        return Ok(attempt.as_bytes() == code.run.expected_stdout.as_slice());
    }
    if let Some(passed) = score_wave2(task, attempt) {
        return Ok(passed);
    }
    Err(Error::Unverifiable("no D2 payload".into()))
}

fn score_wave2(task: &MintedTask, attempt: &str) -> Option<bool> {
    if let Some(grid) = &task.grid {
        return Some(score_grid(&grid.expected, attempt));
    }
    if let Some(market) = &task.market {
        return Some(score_market(market, attempt));
    }
    if let Some(research) = &task.research {
        return Some(score_research(research, attempt));
    }
    if let Some(lh) = &task.long_horizon {
        return Some(score_long_horizon(lh, attempt));
    }
    if let Some(oe) = &task.open_ended {
        return Some(score_open_ended(oe, attempt));
    }
    None
}

#[derive(Deserialize)]
struct LhAttempt {
    checkpoints: Vec<String>,
    #[serde(rename = "final")]
    final_outcome: String,
}

fn parse_grids(attempt: &str) -> Option<Vec<Grid>> {
    let s = attempt.trim();
    if let Ok(cells3) = serde_json::from_str::<Vec<Vec<Vec<u8>>>>(s) {
        return Some(cells3.into_iter().map(Grid::new).collect());
    }
    if let Ok(cells2) = serde_json::from_str::<Vec<Vec<u8>>>(s) {
        return Some(vec![Grid::new(cells2)]);
    }
    None
}

fn grid_ok(g: &Grid) -> bool {
    if g.cells.is_empty() {
        return false;
    }
    let w = g.cells[0].len();
    if w == 0 {
        return false;
    }
    g.cells
        .iter()
        .all(|row| row.len() == w && row.iter().all(|&v| v <= MAX_CELL))
}

fn score_grid(expected: &Grid, attempt: &str) -> bool {
    let Some(predicted) = parse_grids(attempt) else {
        return false;
    };
    if predicted.len() > GRID_PASS_K {
        return false;
    }
    if !grid_ok(expected) || predicted.iter().any(|g| !grid_ok(g)) {
        return false;
    }
    predicted.iter().any(|g| g == expected)
}

fn clamp_prob(p: f64) -> f64 {
    p.clamp(PROB_EPS, 1.0 - PROB_EPS)
}

fn relative_log_score(p_model: f64, p_market: f64, outcome: bool) -> f64 {
    let p = clamp_prob(p_model);
    let m = clamp_prob(p_market);
    if outcome {
        p.ln() - m.ln()
    } else {
        (1.0 - p).ln() - (1.0 - m).ln()
    }
}

fn score_market(task: &MarketTask, attempt: &str) -> bool {
    let Ok(p) = attempt.trim().parse::<f64>() else {
        return false;
    };
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return false;
    }
    if !task.market_p.is_finite() || !(0.0..=1.0).contains(&task.market_p) {
        return false;
    }
    relative_log_score(p, task.market_p, task.outcome) > 0.0
}

#[allow(clippy::float_cmp)]
fn score_research(task: &ResearchTask, attempt: &str) -> bool {
    match attempt.trim().parse::<f64>() {
        Ok(x) if x.is_finite() && task.target.is_finite() => x == task.target,
        _ => false,
    }
}

fn score_payload_math_code_grid(
    math: &Option<MathTask>,
    code: &Option<CodeTask>,
    grid: &Option<GridTask>,
    attempt: &str,
) -> bool {
    if let Some(math) = math {
        return attempt == math.expected;
    }
    if let Some(code) = code {
        return attempt.as_bytes() == code.run.expected_stdout.as_slice();
    }
    if let Some(grid) = grid {
        return score_grid(&grid.expected, attempt);
    }
    false
}

fn score_long_horizon(task: &LongHorizonTask, attempt: &str) -> bool {
    let Ok(v) = serde_json::from_str::<LhAttempt>(attempt.trim()) else {
        return false;
    };
    if v.checkpoints.len() != task.checkpoints.len() {
        return false;
    }
    for (cp, ans) in task.checkpoints.iter().zip(&v.checkpoints) {
        if !score_payload_math_code_grid(&cp.math, &cp.code, &cp.grid, ans) {
            return false;
        }
    }
    score_payload_math_code_grid(&task.final_math, &task.final_code, &None, &v.final_outcome)
}

#[allow(clippy::float_cmp)]
fn weighted_mean(rubric: &Rubric, values: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (c, v) in rubric.criteria.iter().zip(values) {
        num += c.weight * v;
        den += c.weight;
    }
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

fn score_open_ended(task: &OpenEndedTask, attempt: &str) -> bool {
    let Ok(scores) = serde_json::from_str::<BTreeMap<String, f64>>(attempt.trim()) else {
        return false;
    };
    let mut values = Vec::new();
    for c in &task.rubric.criteria {
        let Some(&v) = scores.get(&c.id.0) else {
            return false;
        };
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return false;
        }
        values.push(v);
    }
    weighted_mean(&task.rubric, &values) > 0.5
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
    pub fn raise_if(&mut self, success_rate: f64) {
        if success_rate > RAISE_THRESHOLD {
            self.tool_calls = self.tool_calls.saturating_mul(2);
        }
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
    cursor: usize,
}

impl CodeFactory {
    pub fn new(id: impl Into<String>, sources: Vec<Source>) -> Self {
        Self {
            id: FactoryId(id.into()),
            sources,
            cursor: 0,
        }
    }

    pub fn id(&self) -> &FactoryId {
        &self.id
    }

    pub fn sources(&self) -> &[Source] {
        &self.sources
    }

    pub fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
        let _ = now;
        if self.sources.is_empty() {
            return Err(Error::EmptyCatalog);
        }
        let i = self.cursor;
        let source = &self.sources[i];
        if source.body.is_empty() {
            return Err(Error::EmptyStatement);
        }
        let minted = mint_code(&self.id.0, source, created_at);
        self.cursor = (i + 1) % self.sources.len();
        Ok(minted)
    }
}

/// Repo-level SWE tasks. Same D2 sandboxed-tests verifier, longer horizon,
/// `domain = code`.
pub struct SweFactory {
    id: FactoryId,
    sources: Vec<Source>,
    cursor: usize,
}

impl SweFactory {
    pub fn new(id: impl Into<String>, sources: Vec<Source>) -> Self {
        Self {
            id: FactoryId(id.into()),
            sources,
            cursor: 0,
        }
    }

    pub fn id(&self) -> &FactoryId {
        &self.id
    }

    pub fn sources(&self) -> &[Source] {
        &self.sources
    }

    pub fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
        let _ = now;
        if self.sources.is_empty() {
            return Err(Error::EmptyCatalog);
        }
        let i = self.cursor;
        let source = &self.sources[i];
        if source.body.is_empty() {
            return Err(Error::EmptyStatement);
        }
        let minted = mint_swe(&self.id.0, source, created_at);
        self.cursor = (i + 1) % self.sources.len();
        Ok(minted)
    }
}

/// Symbolic-math tasks. D2 `VerifierKind::SymbolicMath`.
pub struct MathFactory {
    id: FactoryId,
    sources: Vec<Source>,
    cursor: usize,
}

impl MathFactory {
    pub fn new(id: impl Into<String>, sources: Vec<Source>) -> Self {
        Self {
            id: FactoryId(id.into()),
            sources,
            cursor: 0,
        }
    }

    pub fn id(&self) -> &FactoryId {
        &self.id
    }

    pub fn sources(&self) -> &[Source] {
        &self.sources
    }

    pub fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
        let _ = now;
        if self.sources.is_empty() {
            return Err(Error::EmptyCatalog);
        }
        let i = self.cursor;
        let source = &self.sources[i];
        if source.body.is_empty() {
            return Err(Error::EmptyStatement);
        }
        let minted = mint_math(&self.id.0, source, created_at);
        self.cursor = (i + 1) % self.sources.len();
        Ok(minted)
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

/// Three research task shapes from spec 9.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResearchKind {
    /// Held-out val-loss after N GPU-minutes.
    Speedrun,
    /// Paper reproduction: I3 rubric plus a numeric match.
    PaperRepro,
    /// Kaggle-style held-out metric.
    Kaggle,
}

/// Numeric research target plus optional I3 rubric for paper reproduction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResearchTask {
    pub kind: ResearchKind,
    /// Held-out number: val-loss, paper figure, or Kaggle metric.
    pub target: f64,
    /// Speedrun GPU-minute budget. None for paper-repro / Kaggle.
    pub budget_gpu_minutes: Option<u32>,
    /// Paper-repro rubric. None for pure numeric kinds.
    pub rubric: Option<Rubric>,
}

impl ResearchTask {
    pub fn new(kind: ResearchKind, target: f64) -> Self {
        Self {
            kind,
            target,
            budget_gpu_minutes: None,
            rubric: None,
        }
    }
}

/// One verifiable subgoal on a long-horizon task (spec 9.1).
#[derive(Debug, Clone, PartialEq)]
pub struct Checkpoint {
    pub id: String,
    pub statement: String,
    pub math: Option<MathTask>,
    pub code: Option<CodeTask>,
    pub grid: Option<GridTask>,
}

/// Final outcome plus ordered subgoal checkpoints.
#[derive(Debug, Clone, PartialEq)]
pub struct LongHorizonTask {
    pub checkpoints: Vec<Checkpoint>,
    pub final_math: Option<MathTask>,
    pub final_code: Option<CodeTask>,
}

/// Open-ended task scored only by an I3 rubric (spec 9.1, 9.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenEndedTask {
    pub rubric: Rubric,
}

impl OpenEndedTask {
    pub fn new(rubric: Rubric) -> Self {
        Self { rubric }
    }
}

/// Catalog row for [`ResearchFactory`].
#[derive(Debug, Clone, PartialEq)]
pub struct ResearchSource {
    pub source: Source,
    pub kind: ResearchKind,
    pub target: f64,
    pub budget_gpu_minutes: Option<u32>,
    pub rubric: Option<Rubric>,
}

/// Catalog row for [`LongHorizonFactory`].
#[derive(Debug, Clone, PartialEq)]
pub struct LongHorizonSource {
    pub source: Source,
    pub checkpoints: Vec<Checkpoint>,
    pub final_math: Option<MathTask>,
    pub final_code: Option<CodeTask>,
}

/// Catalog row for [`ArcFactory`]. Expected grid is the hidden test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArcSource {
    pub source: Source,
    pub expected: Grid,
}

/// Catalog row for [`ForecastFactory`].
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastSource {
    pub source: Source,
    pub market: MarketTask,
}

/// Catalog row for [`OpenEndedFactory`].
#[derive(Debug, Clone, PartialEq)]
pub struct OpenEndedSource {
    pub source: Source,
    pub rubric: Rubric,
}

/// Research factory (spec 9.1, 9.4). Domain `science`.
pub struct ResearchFactory {
    id: FactoryId,
    sources: Vec<ResearchSource>,
    cursor: usize,
}

impl ResearchFactory {
    pub fn new(id: impl Into<String>, sources: Vec<ResearchSource>) -> Self {
        Self {
            id: FactoryId(id.into()),
            sources,
            cursor: 0,
        }
    }

    pub fn id(&self) -> &FactoryId {
        &self.id
    }

    pub fn sources(&self) -> &[ResearchSource] {
        &self.sources
    }

    pub fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
        let _ = now;
        if self.sources.is_empty() {
            return Err(Error::EmptyCatalog);
        }
        let i = self.cursor;
        let source = &self.sources[i];
        if source.source.body.is_empty() {
            return Err(Error::EmptyStatement);
        }
        let minted = mint_research(&self.id.0, source, created_at);
        self.cursor = (i + 1) % self.sources.len();
        Ok(minted)
    }
}

/// Long-horizon factory (spec 9.1, 9.4). Domain `agent`. Hours-scale horizon.
pub struct LongHorizonFactory {
    id: FactoryId,
    sources: Vec<LongHorizonSource>,
    cursor: usize,
}

impl LongHorizonFactory {
    pub fn new(id: impl Into<String>, sources: Vec<LongHorizonSource>) -> Self {
        Self {
            id: FactoryId(id.into()),
            sources,
            cursor: 0,
        }
    }

    pub fn id(&self) -> &FactoryId {
        &self.id
    }

    pub fn sources(&self) -> &[LongHorizonSource] {
        &self.sources
    }

    pub fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
        let _ = now;
        if self.sources.is_empty() {
            return Err(Error::EmptyCatalog);
        }
        let i = self.cursor;
        let source = &self.sources[i];
        if source.source.body.is_empty() {
            return Err(Error::EmptyStatement);
        }
        let minted = mint_long_horizon(&self.id.0, source, created_at);
        self.cursor = (i + 1) % self.sources.len();
        Ok(minted)
    }
}

/// ARC-3 factory (spec 9.1, 9.4). Domain `arc`. Exact grid match, pass@2.
pub struct ArcFactory {
    id: FactoryId,
    sources: Vec<ArcSource>,
    cursor: usize,
}

impl ArcFactory {
    pub fn new(id: impl Into<String>, sources: Vec<ArcSource>) -> Self {
        Self {
            id: FactoryId(id.into()),
            sources,
            cursor: 0,
        }
    }

    pub fn id(&self) -> &FactoryId {
        &self.id
    }

    pub fn sources(&self) -> &[ArcSource] {
        &self.sources
    }

    pub fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
        let _ = now;
        if self.sources.is_empty() {
            return Err(Error::EmptyCatalog);
        }
        let i = self.cursor;
        let source = &self.sources[i];
        if source.source.body.is_empty() {
            return Err(Error::EmptyStatement);
        }
        let minted = mint_arc(&self.id.0, source, created_at);
        self.cursor = (i + 1) % self.sources.len();
        Ok(minted)
    }
}

/// Forecasting factory (spec 9.1, 9.4). Domain `other`. Market log score.
pub struct ForecastFactory {
    id: FactoryId,
    sources: Vec<ForecastSource>,
    cursor: usize,
}

impl ForecastFactory {
    pub fn new(id: impl Into<String>, sources: Vec<ForecastSource>) -> Self {
        Self {
            id: FactoryId(id.into()),
            sources,
            cursor: 0,
        }
    }

    pub fn id(&self) -> &FactoryId {
        &self.id
    }

    pub fn sources(&self) -> &[ForecastSource] {
        &self.sources
    }

    pub fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
        let _ = now;
        if self.sources.is_empty() {
            return Err(Error::EmptyCatalog);
        }
        let i = self.cursor;
        let source = &self.sources[i];
        if source.source.body.is_empty() {
            return Err(Error::EmptyStatement);
        }
        let minted = mint_forecast(&self.id.0, source, created_at);
        self.cursor = (i + 1) % self.sources.len();
        Ok(minted)
    }
}

/// Open-ended factory (spec 9.1, 9.4, 9.5). Domain `other`. I3 rubric only.
pub struct OpenEndedFactory {
    id: FactoryId,
    sources: Vec<OpenEndedSource>,
    cursor: usize,
}

impl OpenEndedFactory {
    pub fn new(id: impl Into<String>, sources: Vec<OpenEndedSource>) -> Self {
        Self {
            id: FactoryId(id.into()),
            sources,
            cursor: 0,
        }
    }

    pub fn id(&self) -> &FactoryId {
        &self.id
    }

    pub fn sources(&self) -> &[OpenEndedSource] {
        &self.sources
    }

    pub fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
        let _ = now;
        if self.sources.is_empty() {
            return Err(Error::EmptyCatalog);
        }
        let i = self.cursor;
        let source = &self.sources[i];
        if source.source.body.is_empty() {
            return Err(Error::EmptyStatement);
        }
        let minted = mint_open_ended(&self.id.0, source, created_at);
        self.cursor = (i + 1) % self.sources.len();
        Ok(minted)
    }
}

impl Factory for ResearchFactory {
    fn id(&self) -> &FactoryId {
        ResearchFactory::id(self)
    }

    fn domain(&self) -> TaskDomain {
        TaskDomain::Science
    }

    fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
        ResearchFactory::mint(self, now, created_at)
    }
}

impl Factory for LongHorizonFactory {
    fn id(&self) -> &FactoryId {
        LongHorizonFactory::id(self)
    }

    fn domain(&self) -> TaskDomain {
        TaskDomain::Agent
    }

    fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
        LongHorizonFactory::mint(self, now, created_at)
    }
}

impl Factory for ArcFactory {
    fn id(&self) -> &FactoryId {
        ArcFactory::id(self)
    }

    fn domain(&self) -> TaskDomain {
        TaskDomain::Arc
    }

    fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
        ArcFactory::mint(self, now, created_at)
    }
}

impl Factory for ForecastFactory {
    fn id(&self) -> &FactoryId {
        ForecastFactory::id(self)
    }

    fn domain(&self) -> TaskDomain {
        TaskDomain::Other
    }

    fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
        ForecastFactory::mint(self, now, created_at)
    }
}

impl Factory for OpenEndedFactory {
    fn id(&self) -> &FactoryId {
        OpenEndedFactory::id(self)
    }

    fn domain(&self) -> TaskDomain {
        TaskDomain::Other
    }

    fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
        OpenEndedFactory::mint(self, now, created_at)
    }
}

#[allow(clippy::too_many_arguments)]
fn spec(
    factory_id: &str,
    source: &Source,
    domain: TaskDomain,
    statement: &str,
    hidden_tests_hash: String,
    verifier_id: &str,
    env_image: Option<String>,
    horizon_s: u32,
    max_tool_calls: u32,
    created_at: &str,
) -> TaskSpec {
    TaskSpec {
        schema_id: SCHEMA_TASK_SPEC.to_string(),
        schema_version: SCHEMA_VERSION,
        task_id: task_id(factory_id, &source.id),
        domain,
        split: Split::Train,
        statement_hash: statement_hash(statement),
        hidden_tests_hash,
        verifier_id: verifier_id.to_string(),
        env_image,
        horizon_s,
        max_tool_calls,
        provenance: source.provenance.clone(),
        created_at: created_at.to_string(),
    }
}

fn mint_math(factory_id: &str, source: &Source, created_at: &str) -> MintedTask {
    let statement = source.body.clone();
    let expected = math_expected(&source.body);
    let hidden = sha256_hex(expected.as_bytes());
    MintedTask {
        spec: spec(
            factory_id,
            source,
            TaskDomain::Math,
            &statement,
            hidden,
            MATH_VERIFIER_ID,
            None,
            MATH_HORIZON_S,
            MATH_MAX_TOOL_CALLS,
            created_at,
        ),
        verifier_kind: VerifierKind::SymbolicMath,
        verifier_id: VerifierId(MATH_VERIFIER_ID.to_string()),
        statement,
        code: None,
        math: Some(MathTask::new(expected)),
        grid: None,
        market: None,
        research: None,
        long_horizon: None,
        open_ended: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn mint_sandboxed(
    factory_id: &str,
    source: &Source,
    created_at: &str,
    horizon_s: u32,
    max_tool_calls: u32,
    timeout_s: u32,
    expected_stdout: Vec<u8>,
    mutant_id: &str,
    mutant_body: &[u8],
    extra_agent: Option<(&str, Vec<u8>)>,
) -> MintedTask {
    let statement = source.body.clone();
    let mut image = Image::new(image_id(factory_id, &source.id));
    image
        .hidden_tests
        .insert(hidden_test_path(), HIDDEN_TEST_BODY.to_vec());
    if let Some((path, bytes)) = extra_agent {
        image.agent_files.insert(path.to_string(), bytes);
    }
    image.hidden_tests_hash = hidden_files_hash(&image.hidden_tests);
    let hidden_hash = image.hidden_tests_hash.clone();
    let env_image = Some(image.id.0.clone());
    let run = TestRun::python(python_payload(), timeout_s, expected_stdout);
    let mut code = CodeTask::new(image, run);
    let mut files = BTreeMap::new();
    files.insert(MUTANT_PATH.to_string(), mutant_body.to_vec());
    code.mutants.push(Mutant::new(mutant_id, files));
    MintedTask {
        spec: spec(
            factory_id,
            source,
            TaskDomain::Code,
            &statement,
            hidden_hash,
            CODE_VERIFIER_ID,
            env_image,
            horizon_s,
            max_tool_calls,
            created_at,
        ),
        verifier_kind: VerifierKind::SandboxedTests,
        verifier_id: VerifierId(CODE_VERIFIER_ID.to_string()),
        statement,
        code: Some(code),
        math: None,
        grid: None,
        market: None,
        research: None,
        long_horizon: None,
        open_ended: None,
    }
}

fn mint_code(factory_id: &str, source: &Source, created_at: &str) -> MintedTask {
    mint_sandboxed(
        factory_id,
        source,
        created_at,
        CODE_HORIZON_S,
        CODE_MAX_TOOL_CALLS,
        CODE_TIMEOUT_S,
        code_expected_stdout(&source.body),
        CODE_MUTANT_ID,
        CODE_MUTANT_BODY,
        None,
    )
}

fn mint_swe(factory_id: &str, source: &Source, created_at: &str) -> MintedTask {
    mint_sandboxed(
        factory_id,
        source,
        created_at,
        SWE_HORIZON_S,
        SWE_MAX_TOOL_CALLS,
        SWE_TIMEOUT_S,
        swe_expected_stdout(&source.body),
        SWE_MUTANT_ID,
        SWE_MUTANT_BODY,
        Some((SWE_REPO_PATH, source.body.as_bytes().to_vec())),
    )
}

fn hidden_hash_grid(expected: &Grid) -> String {
    sha256_hex(&serde_json::to_vec(&expected.cells).expect("grid json"))
}

fn hidden_hash_market(market: &MarketTask) -> String {
    sha256_hex(format!("{}:{}", market.outcome, market.market_p).as_bytes())
}

fn hidden_hash_research(target: f64) -> String {
    sha256_hex(&target.to_le_bytes())
}

fn hidden_hash_long_horizon(task: &LongHorizonTask) -> String {
    let mut buf = Vec::new();
    for cp in &task.checkpoints {
        buf.extend_from_slice(checkpoint_hidden_bytes(cp).as_bytes());
        buf.push(0);
    }
    buf.extend_from_slice(b"final:");
    buf.extend_from_slice(final_hidden_bytes(task).as_bytes());
    sha256_hex(&buf)
}

fn hidden_hash_open_ended(rubric: &Rubric) -> String {
    let mut buf = Vec::new();
    for c in &rubric.criteria {
        buf.extend_from_slice(c.id.0.as_bytes());
        buf.push(0);
        buf.extend_from_slice(c.prompt.as_bytes());
        buf.push(0);
    }
    sha256_hex(&buf)
}

fn checkpoint_hidden_bytes(cp: &Checkpoint) -> String {
    if let Some(math) = &cp.math {
        return format!("math:{}", math.expected);
    }
    if let Some(code) = &cp.code {
        return format!(
            "code:{}",
            String::from_utf8_lossy(&code.run.expected_stdout)
        );
    }
    if let Some(grid) = &cp.grid {
        return format!("grid:{}", hidden_hash_grid(&grid.expected));
    }
    String::new()
}

fn final_hidden_bytes(task: &LongHorizonTask) -> String {
    if let Some(math) = &task.final_math {
        return format!("math:{}", math.expected);
    }
    if let Some(code) = &task.final_code {
        return format!(
            "code:{}",
            String::from_utf8_lossy(&code.run.expected_stdout)
        );
    }
    String::new()
}

fn mint_research(factory_id: &str, row: &ResearchSource, created_at: &str) -> MintedTask {
    let statement = row.source.body.clone();
    let research = ResearchTask {
        kind: row.kind,
        target: row.target,
        budget_gpu_minutes: row.budget_gpu_minutes,
        rubric: row.rubric.clone(),
    };
    let hidden = hidden_hash_research(row.target);
    MintedTask {
        spec: spec(
            factory_id,
            &row.source,
            TaskDomain::Science,
            &statement,
            hidden,
            RESEARCH_VERIFIER_ID,
            None,
            RESEARCH_HORIZON_S,
            RESEARCH_MAX_TOOL_CALLS,
            created_at,
        ),
        verifier_kind: VerifierKind::LeanKernel,
        verifier_id: VerifierId(RESEARCH_VERIFIER_ID.to_string()),
        statement,
        code: None,
        math: None,
        grid: None,
        market: None,
        research: Some(research),
        long_horizon: None,
        open_ended: None,
    }
}

fn mint_long_horizon(factory_id: &str, row: &LongHorizonSource, created_at: &str) -> MintedTask {
    let statement = row.source.body.clone();
    let task = LongHorizonTask {
        checkpoints: row.checkpoints.clone(),
        final_math: row.final_math.clone(),
        final_code: row.final_code.clone(),
    };
    let hidden = hidden_hash_long_horizon(&task);
    MintedTask {
        spec: spec(
            factory_id,
            &row.source,
            TaskDomain::Agent,
            &statement,
            hidden,
            LONG_HORIZON_VERIFIER_ID,
            None,
            LONG_HORIZON_S,
            LONG_HORIZON_MAX_TOOL_CALLS,
            created_at,
        ),
        verifier_kind: VerifierKind::LeanKernel,
        verifier_id: VerifierId(LONG_HORIZON_VERIFIER_ID.to_string()),
        statement,
        code: None,
        math: None,
        grid: None,
        market: None,
        research: None,
        long_horizon: Some(task),
        open_ended: None,
    }
}

fn mint_arc(factory_id: &str, row: &ArcSource, created_at: &str) -> MintedTask {
    let statement = row.source.body.clone();
    let expected = row.expected.clone();
    let hidden = hidden_hash_grid(&expected);
    MintedTask {
        spec: spec(
            factory_id,
            &row.source,
            TaskDomain::Arc,
            &statement,
            hidden,
            ARC_VERIFIER_ID,
            None,
            ARC_HORIZON_S,
            ARC_MAX_TOOL_CALLS,
            created_at,
        ),
        verifier_kind: VerifierKind::GridMatch,
        verifier_id: VerifierId(ARC_VERIFIER_ID.to_string()),
        statement,
        code: None,
        math: None,
        grid: Some(GridTask::new(expected)),
        market: None,
        research: None,
        long_horizon: None,
        open_ended: None,
    }
}

fn mint_forecast(factory_id: &str, row: &ForecastSource, created_at: &str) -> MintedTask {
    let statement = row.source.body.clone();
    let market = row.market.clone();
    let hidden = hidden_hash_market(&market);
    MintedTask {
        spec: spec(
            factory_id,
            &row.source,
            TaskDomain::Other,
            &statement,
            hidden,
            FORECAST_VERIFIER_ID,
            None,
            FORECAST_HORIZON_S,
            FORECAST_MAX_TOOL_CALLS,
            created_at,
        ),
        verifier_kind: VerifierKind::MarketResolution,
        verifier_id: VerifierId(FORECAST_VERIFIER_ID.to_string()),
        statement,
        code: None,
        math: None,
        grid: None,
        market: Some(market),
        research: None,
        long_horizon: None,
        open_ended: None,
    }
}

fn mint_open_ended(factory_id: &str, row: &OpenEndedSource, created_at: &str) -> MintedTask {
    let statement = row.source.body.clone();
    let rubric = row.rubric.clone();
    let hidden = hidden_hash_open_ended(&rubric);
    MintedTask {
        spec: spec(
            factory_id,
            &row.source,
            TaskDomain::Other,
            &statement,
            hidden,
            OPEN_ENDED_VERIFIER_ID,
            None,
            OPEN_ENDED_HORIZON_S,
            OPEN_ENDED_MAX_TOOL_CALLS,
            created_at,
        ),
        verifier_kind: VerifierKind::LeanKernel,
        verifier_id: VerifierId(OPEN_ENDED_VERIFIER_ID.to_string()),
        statement,
        code: None,
        math: None,
        grid: None,
        market: None,
        research: None,
        long_horizon: None,
        open_ended: Some(OpenEndedTask { rubric }),
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
