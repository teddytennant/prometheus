//! Independent I6 wave-2 reference: research, long-horizon, ARC, forecast,
//! open-ended mint and scoring.
#![allow(dead_code)]
//!
//! Slow and obvious. Production `src/` must not import this module.
//!
//! # Pins
//!
//! ## Horizons (explicit constants; SWE 2× is D3-only)
//! - ARC / forecast / open-ended: 30s / 10 tool calls (D3 math).
//! - Research: 600s / 10 tool calls (larger wall clock than D3 math).
//! - Long-horizon: 3600s (hours-scale, spec 9.1) / 2000 tool calls (spec 9.4).
//! - [`Horizon::raise_if`] is unchanged: doubles iff rate `> 0.5` (strict).
//!
//! ## F1
//! - `task_id` = `{factory_id}:{source.id}`.
//! - `created_at` is the caller string; `now` is unused (no wall clock).
//! - `statement` is exactly `source.body`.
//! - `statement_hash` = lowercase hex SHA-256 of the public statement.
//!
//! ## Domains
//! research=`science`, long-horizon=`agent`, ARC=`arc`, forecast=`other`,
//! open-ended=`other`.
//!
//! ## Verifier kinds / payloads
//! - ARC: `GridMatch` + `GridTask`; verifier id `grid-match`.
//! - Forecast: `MarketResolution` + `MarketTask`; verifier id `market-resolution`.
//! - Research / long-horizon / open-ended have no D2 exact checker. The
//!   required `MintedTask.verifier_kind` is pinned to `LeanKernel` as a
//!   non-claim; scoring uses the payload. Verifier ids: `research-numeric`,
//!   `long-horizon`, `open-ended-rubric`.
//!
//! ## Hidden tests
//! Never appear in the public statement. ARC expected grid, forecast `outcome`,
//! research `target`, long-horizon checkpoint/final expecteds, and rubric
//! grader prompts (grader-only) live on the payload.
//!
//! ## Catalog
//! Round-robin with wrap-around. Cursor advances only after a successful mint.
//! Empty catalog → `EmptyCatalog`. Empty `source.body` → `EmptyStatement`.
//!
//! ## Attempt encodings (string; no live sandbox / no live judge in probe)
//! - Grid: JSON `Vec<Vec<Vec<u8>>>` (k grids) or `Vec<Vec<u8>>` (one grid).
//!   Pass if `k ≤ GRID_PASS_K` (2) and any grid equals `expected`.
//! - Market: decimal probability in `[0, 1]`. Pass iff relative log score `> 0`
//!   (strictly beats the market; D2 clamp to `PROB_EPS`).
//! - Research: decimal matching `ResearchTask.target` (`f64` equality).
//! - Long-horizon: JSON `{"checkpoints":[<attempt>...],"final":<attempt>}`.
//!   Pass iff every checkpoint and the final pass (math/code/grid encodings).
//! - Open-ended: JSON object `{criterion_id: score}` in `[0, 1]`. Pass iff the
//!   I3 weighted mean is strictly `> 0.5` (same spirit as `Horizon::raise_if`).
//!   Malformed attempts are not solved (`false`), not `Unverifiable`.

use std::collections::BTreeMap;

use prometheus_envs::NowMs;
use prometheus_rewards::{Criterion, Judge, Rubric, ScriptedJudge};
use prometheus_tasks::{
    ArcFactory, ArcSource, Checkpoint, Error, FactoryId, ForecastFactory, ForecastSource, Grid,
    LongHorizonFactory, LongHorizonSource, LongHorizonTask, MintedTask, OpenEndedFactory,
    OpenEndedSource, OpenEndedTask, ResearchFactory, ResearchSource, ResearchTask, Result, Source,
    TaskDomain, VerifierId, GRID_PASS_K,
};
use prometheus_verifiers::{CodeTask, GridTask, MarketTask, MathTask, VerifierKind, PROB_EPS};

use super::{sha256_hex, spec};

pub const ARC_HORIZON_S: u32 = 30;
pub const ARC_MAX_TOOL_CALLS: u32 = 10;
pub const FORECAST_HORIZON_S: u32 = 30;
pub const FORECAST_MAX_TOOL_CALLS: u32 = 10;
pub const OPEN_ENDED_HORIZON_S: u32 = 30;
pub const OPEN_ENDED_MAX_TOOL_CALLS: u32 = 10;
pub const RESEARCH_HORIZON_S: u32 = 600;
pub const RESEARCH_MAX_TOOL_CALLS: u32 = 10;
pub const LONG_HORIZON_S: u32 = 3600;
pub const LONG_HORIZON_MAX_TOOL_CALLS: u32 = 2000;

pub const ARC_VERIFIER_ID: &str = "grid-match";
pub const FORECAST_VERIFIER_ID: &str = "market-resolution";
pub const RESEARCH_VERIFIER_ID: &str = "research-numeric";
pub const LONG_HORIZON_VERIFIER_ID: &str = "long-horizon";
pub const OPEN_ENDED_VERIFIER_ID: &str = "open-ended-rubric";

const MAX_CELL: u8 = 9;

#[derive(serde::Deserialize)]
struct LhAttempt {
    checkpoints: Vec<String>,
    #[serde(rename = "final")]
    final_outcome: String,
}

pub fn hidden_hash_grid(expected: &Grid) -> String {
    sha256_hex(&serde_json::to_vec(&expected.cells).expect("grid json"))
}

pub fn hidden_hash_market(market: &MarketTask) -> String {
    sha256_hex(format!("{}:{}", market.outcome, market.market_p).as_bytes())
}

pub fn hidden_hash_research(target: f64) -> String {
    sha256_hex(&target.to_le_bytes())
}

pub fn hidden_hash_long_horizon(task: &LongHorizonTask) -> String {
    let mut buf = Vec::new();
    for cp in &task.checkpoints {
        buf.extend_from_slice(checkpoint_hidden_bytes(cp).as_bytes());
        buf.push(0);
    }
    buf.extend_from_slice(b"final:");
    buf.extend_from_slice(final_hidden_bytes(task).as_bytes());
    sha256_hex(&buf)
}

pub fn hidden_hash_open_ended(rubric: &Rubric) -> String {
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

pub fn encode_grids(cells: &[Vec<Vec<u8>>]) -> String {
    serde_json::to_string(cells).expect("grid attempt json")
}

pub fn encode_market(p: f64) -> String {
    serde_json::to_string(&p).expect("market json")
}

pub fn encode_research(target: f64) -> String {
    serde_json::to_string(&target).expect("research json")
}

pub fn encode_long_horizon(checkpoints: &[String], final_outcome: &str) -> String {
    serde_json::json!({
        "checkpoints": checkpoints,
        "final": final_outcome,
    })
    .to_string()
}

pub fn encode_open_ended(scores: &BTreeMap<String, f64>) -> String {
    serde_json::to_string(scores).expect("open-ended json")
}

pub fn passing_attempt(task: &MintedTask) -> String {
    if let Some(grid) = &task.grid {
        return encode_grids(std::slice::from_ref(&grid.expected.cells));
    }
    if let Some(market) = &task.market {
        return if market.outcome {
            encode_market(1.0)
        } else {
            encode_market(0.0)
        };
    }
    if let Some(research) = &task.research {
        return encode_research(research.target);
    }
    if let Some(lh) = &task.long_horizon {
        let cps: Vec<String> = lh.checkpoints.iter().map(passing_checkpoint).collect();
        return encode_long_horizon(&cps, &passing_final(lh));
    }
    if let Some(oe) = &task.open_ended {
        let mut scores = BTreeMap::new();
        for c in &oe.rubric.criteria {
            scores.insert(c.id.0.clone(), 1.0);
        }
        return encode_open_ended(&scores);
    }
    panic!("passing_attempt: no wave-2 payload");
}

pub fn failing_attempt(task: &MintedTask) -> String {
    if task.grid.is_some() {
        return encode_grids(&[vec![vec![9, 9], vec![9, 9]]]);
    }
    if let Some(market) = &task.market {
        return if market.outcome {
            encode_market(0.0)
        } else {
            encode_market(1.0)
        };
    }
    if let Some(research) = &task.research {
        let x = if research.target == 0.0 { 1.0 } else { 0.0 };
        return encode_research(x);
    }
    if let Some(lh) = &task.long_horizon {
        let cps = vec!["WRONG".to_string(); lh.checkpoints.len()];
        return encode_long_horizon(&cps, "WRONG");
    }
    if let Some(oe) = &task.open_ended {
        let mut scores = BTreeMap::new();
        for c in &oe.rubric.criteria {
            scores.insert(c.id.0.clone(), 0.0);
        }
        return encode_open_ended(&scores);
    }
    "WRONG".into()
}

fn passing_checkpoint(cp: &Checkpoint) -> String {
    if let Some(math) = &cp.math {
        return math.expected.clone();
    }
    if let Some(code) = &cp.code {
        return String::from_utf8_lossy(&code.run.expected_stdout).into_owned();
    }
    if let Some(grid) = &cp.grid {
        return encode_grids(std::slice::from_ref(&grid.expected.cells));
    }
    String::new()
}

fn passing_final(task: &LongHorizonTask) -> String {
    if let Some(math) = &task.final_math {
        return math.expected.clone();
    }
    if let Some(code) = &task.final_code {
        return String::from_utf8_lossy(&code.run.expected_stdout).into_owned();
    }
    String::new()
}

pub fn score_wave2(task: &MintedTask, attempt: &str) -> Option<bool> {
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

pub fn clamp_prob(p: f64) -> f64 {
    p.clamp(PROB_EPS, 1.0 - PROB_EPS)
}

/// Independent D2 relative log score: `ln p_model(y) - ln p_market(y)`.
pub fn relative_log_score(p_model: f64, p_market: f64, outcome: bool) -> f64 {
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

pub fn weighted_mean(rubric: &Rubric, values: &[f64]) -> f64 {
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

pub fn grader_prompt(grader_id: &str, criterion: &Criterion, output: &str) -> String {
    format!(
        "grader:{}\ncriterion:{}\n{}\n\n{}",
        grader_id, criterion.id.0, criterion.prompt, output
    )
}

pub fn scripted_scores(
    rubric: &Rubric,
    grader_id: &str,
    output: &str,
    judge: &mut ScriptedJudge,
    now: NowMs,
) -> Result<BTreeMap<String, f64>> {
    let mut out = BTreeMap::new();
    for c in &rubric.criteria {
        let prompt = grader_prompt(grader_id, c, output);
        let reply = judge
            .complete(&prompt, now)
            .map_err(|e| Error::Unverifiable(e.to_string()))?;
        let v: f64 = reply
            .trim()
            .parse()
            .map_err(|_| Error::Unverifiable("judge reply is not a number".into()))?;
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(Error::Unverifiable("judge score out of [0, 1]".into()));
        }
        out.insert(c.id.0.clone(), v);
    }
    Ok(out)
}

pub fn scripted_weighted(
    rubric: &Rubric,
    grader_id: &str,
    output: &str,
    judge: &mut ScriptedJudge,
    now: NowMs,
) -> Result<f64> {
    let scores = scripted_scores(rubric, grader_id, output, judge, now)?;
    let values: Vec<f64> = rubric.criteria.iter().map(|c| scores[&c.id.0]).collect();
    Ok(weighted_mean(rubric, &values))
}

fn mint_base(
    factory_id: &str,
    source: &Source,
    domain: TaskDomain,
    statement: &str,
    hidden_tests_hash: String,
    verifier_id: &str,
    horizon_s: u32,
    max_tool_calls: u32,
    created_at: &str,
    verifier_kind: VerifierKind,
) -> MintedTask {
    MintedTask {
        spec: spec(
            factory_id,
            source,
            domain,
            statement,
            hidden_tests_hash,
            verifier_id,
            None,
            horizon_s,
            max_tool_calls,
            created_at,
        ),
        verifier_kind,
        verifier_id: VerifierId(verifier_id.to_string()),
        statement: statement.to_string(),
        code: None,
        math: None,
        grid: None,
        market: None,
        research: None,
        long_horizon: None,
        open_ended: None,
    }
}

fn empty_body(source: &Source) -> bool {
    source.body.is_empty()
}

pub fn mint_research(factory_id: &str, row: &ResearchSource, created_at: &str) -> MintedTask {
    let statement = row.source.body.clone();
    let hidden = hidden_hash_research(row.target);
    let mut t = mint_base(
        factory_id,
        &row.source,
        TaskDomain::Science,
        &statement,
        hidden,
        RESEARCH_VERIFIER_ID,
        RESEARCH_HORIZON_S,
        RESEARCH_MAX_TOOL_CALLS,
        created_at,
        VerifierKind::LeanKernel,
    );
    t.research = Some(ResearchTask {
        kind: row.kind,
        target: row.target,
        budget_gpu_minutes: row.budget_gpu_minutes,
        rubric: row.rubric.clone(),
    });
    t
}

pub fn mint_long_horizon(
    factory_id: &str,
    row: &LongHorizonSource,
    created_at: &str,
) -> MintedTask {
    let statement = row.source.body.clone();
    let lh = LongHorizonTask {
        checkpoints: row.checkpoints.clone(),
        final_math: row.final_math.clone(),
        final_code: row.final_code.clone(),
    };
    let hidden = hidden_hash_long_horizon(&lh);
    let mut t = mint_base(
        factory_id,
        &row.source,
        TaskDomain::Agent,
        &statement,
        hidden,
        LONG_HORIZON_VERIFIER_ID,
        LONG_HORIZON_S,
        LONG_HORIZON_MAX_TOOL_CALLS,
        created_at,
        VerifierKind::LeanKernel,
    );
    t.long_horizon = Some(lh);
    t
}

pub fn mint_arc(factory_id: &str, row: &ArcSource, created_at: &str) -> MintedTask {
    let statement = row.source.body.clone();
    let hidden = hidden_hash_grid(&row.expected);
    let mut t = mint_base(
        factory_id,
        &row.source,
        TaskDomain::Arc,
        &statement,
        hidden,
        ARC_VERIFIER_ID,
        ARC_HORIZON_S,
        ARC_MAX_TOOL_CALLS,
        created_at,
        VerifierKind::GridMatch,
    );
    t.grid = Some(GridTask::new(row.expected.clone()));
    t
}

pub fn mint_forecast(factory_id: &str, row: &ForecastSource, created_at: &str) -> MintedTask {
    let statement = row.source.body.clone();
    let hidden = hidden_hash_market(&row.market);
    let mut t = mint_base(
        factory_id,
        &row.source,
        TaskDomain::Other,
        &statement,
        hidden,
        FORECAST_VERIFIER_ID,
        FORECAST_HORIZON_S,
        FORECAST_MAX_TOOL_CALLS,
        created_at,
        VerifierKind::MarketResolution,
    );
    t.market = Some(row.market.clone());
    t
}

pub fn mint_open_ended(factory_id: &str, row: &OpenEndedSource, created_at: &str) -> MintedTask {
    let statement = row.source.body.clone();
    let hidden = hidden_hash_open_ended(&row.rubric);
    let mut t = mint_base(
        factory_id,
        &row.source,
        TaskDomain::Other,
        &statement,
        hidden,
        OPEN_ENDED_VERIFIER_ID,
        OPEN_ENDED_HORIZON_S,
        OPEN_ENDED_MAX_TOOL_CALLS,
        created_at,
        VerifierKind::LeanKernel,
    );
    t.open_ended = Some(OpenEndedTask {
        rubric: row.rubric.clone(),
    });
    t
}

macro_rules! wave2_factory {
    ($name:ident, $prod:ty, $row:ty, $mint:ident) => {
        pub struct $name {
            id: FactoryId,
            sources: Vec<$row>,
            cursor: usize,
        }

        impl $name {
            pub fn new(id: impl Into<String>, sources: Vec<$row>) -> Self {
                Self {
                    id: FactoryId(id.into()),
                    sources,
                    cursor: 0,
                }
            }

            pub fn from_prod(f: &$prod) -> Self {
                Self::new(f.id().0.clone(), f.sources().to_vec())
            }

            pub fn mint(&mut self, now: NowMs, created_at: &str) -> Result<MintedTask> {
                let _ = now;
                if self.sources.is_empty() {
                    return Err(Error::EmptyCatalog);
                }
                let i = self.cursor;
                let row = &self.sources[i];
                if empty_body(row_source(row)) {
                    return Err(Error::EmptyStatement);
                }
                let minted = $mint(&self.id.0, row, created_at);
                self.cursor = (i + 1) % self.sources.len();
                Ok(minted)
            }
        }
    };
}

fn row_source<T>(row: &T) -> &Source
where
    T: RowSource,
{
    row.source()
}

trait RowSource {
    fn source(&self) -> &Source;
}

impl RowSource for ResearchSource {
    fn source(&self) -> &Source {
        &self.source
    }
}
impl RowSource for LongHorizonSource {
    fn source(&self) -> &Source {
        &self.source
    }
}
impl RowSource for ArcSource {
    fn source(&self) -> &Source {
        &self.source
    }
}
impl RowSource for ForecastSource {
    fn source(&self) -> &Source {
        &self.source
    }
}
impl RowSource for OpenEndedSource {
    fn source(&self) -> &Source {
        &self.source
    }
}

wave2_factory!(RefResearch, ResearchFactory, ResearchSource, mint_research);
wave2_factory!(
    RefLongHorizon,
    LongHorizonFactory,
    LongHorizonSource,
    mint_long_horizon
);
wave2_factory!(RefArc, ArcFactory, ArcSource, mint_arc);
wave2_factory!(RefForecast, ForecastFactory, ForecastSource, mint_forecast);
wave2_factory!(
    RefOpenEnded,
    OpenEndedFactory,
    OpenEndedSource,
    mint_open_ended
);
