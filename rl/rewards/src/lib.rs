//! Rewards beyond exact verifiers (spec 9.3, 15.5 I3).
//!
//! Three pieces: rubric graders with disagreement down-weighting,
//! contrastive pairs (RLCD-Rescore), and a tampering detector. Exact
//! checkers stay in [`prometheus_verifiers`]. A learned preference model
//! is not used on any domain that already has an exact verifier.
//!
//! Graders are a [`Judge`] so CPU tests inject a scripted model. F5
//! serving can wrap this later; nothing here starts an engine.
//!
//! The reward API is F1 `prometheus.reward_request` /
//! `prometheus.reward_response`. [`NowMs`] is injected; nothing here
//! reads the wall clock. `scored_at` is a caller-supplied ISO-8601 string.
//!
//! Gate: planted hacks are flagged (`Flag::Tampering`). Detector evals
//! use a held-out set of hacks, never the generated contrastive pairs.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub use prometheus_verifiers::{
    Evidence, EvidenceKind, Flag, NowMs, RewardRequest, RewardResponse, SCHEMA_REWARD_REQUEST,
    SCHEMA_REWARD_RESPONSE, SCHEMA_VERSION,
};

const HEX: &[u8] = b"0123456789abcdef";

/// Drop or down-weight a sample when grader spread exceeds this.
pub const DISAGREE_EPS: f64 = 0.25;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraderId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RubricId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CriterionId(pub String);

/// One weighted rubric axis. `weight` must be finite and `> 0`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Criterion {
    pub id: CriterionId,
    pub prompt: String,
    pub weight: f64,
}

impl Criterion {
    pub fn new(id: impl Into<String>, prompt: impl Into<String>, weight: f64) -> Self {
        Self {
            id: CriterionId(id.into()),
            prompt: prompt.into(),
            weight,
        }
    }
}

/// Open-ended rubric. Not used when an exact verifier already scores the task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rubric {
    pub id: RubricId,
    pub criteria: Vec<Criterion>,
}

impl Rubric {
    pub fn new(id: impl Into<String>, criteria: Vec<Criterion>) -> Self {
        Self {
            id: RubricId(id.into()),
            criteria,
        }
    }

    pub fn total_weight(&self) -> f64 {
        self.criteria.iter().map(|c| c.weight).sum()
    }
}

/// Score on one criterion, in `[0, 1]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CriterionScore {
    pub criterion_id: CriterionId,
    pub value: f64,
}

/// One grader's weighted rubric score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RubricScore {
    pub grader_id: GraderId,
    pub scores: Vec<CriterionScore>,
    pub weighted: f64,
}

/// Confirmed hack kinds from spec 9.3. A pair is kept only when the
/// negative side shows at least one of these and the positive side shows
/// none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HackKind {
    TestWrite,
    SpecialCase,
    VisiblePassHiddenFail,
}

/// Features the detector is allowed to see. Not the grader's hidden tests.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TrajectoryView {
    pub wrote_test_file: bool,
    pub special_cased: bool,
    pub visible_passed: bool,
    pub hidden_passed: bool,
    pub test_write_paths: Vec<String>,
}

impl TrajectoryView {
    pub fn new() -> Self {
        Self::default()
    }

    /// True when any 9.3 confirmation check fires.
    pub fn is_confirmed_hack(&self) -> bool {
        self.wrote_test_file || self.special_cased || (self.visible_passed && !self.hidden_passed)
    }

    pub fn kinds(&self) -> Vec<HackKind> {
        let mut out = Vec::new();
        if self.wrote_test_file {
            out.push(HackKind::TestWrite);
        }
        if self.special_cased {
            out.push(HackKind::SpecialCase);
        }
        if self.visible_passed && !self.hidden_passed {
            out.push(HackKind::VisiblePassHiddenFail);
        }
        out
    }
}

/// Scripted or remote completion. CPU tests inject [`ScriptedJudge`].
pub trait Judge {
    fn complete(&mut self, prompt: &str, now: NowMs) -> Result<String>;
}

/// Exact map from prompt to completion. Missing prompts error.
pub struct ScriptedJudge {
    replies: BTreeMap<String, String>,
}

impl ScriptedJudge {
    pub fn new() -> Self {
        Self {
            replies: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, prompt: impl Into<String>, reply: impl Into<String>) {
        self.replies.insert(prompt.into(), reply.into());
    }
}

impl Default for ScriptedJudge {
    fn default() -> Self {
        Self::new()
    }
}

impl Judge for ScriptedJudge {
    fn complete(&mut self, prompt: &str, _now: NowMs) -> Result<String> {
        self.replies
            .get(prompt)
            .cloned()
            .ok_or_else(|| Error::Unverifiable(format!("no scripted reply for {prompt}")))
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum Error {
    #[error("schema mismatch: {0}")]
    Schema(String),
    #[error("criterion weight must be finite and > 0")]
    BadWeight,
    #[error("criterion score {0} is not in [0, 1]")]
    BadScore(f64),
    #[error("empty rubric")]
    EmptyRubric,
    #[error("empty grader panel")]
    EmptyPanel,
    #[error("exact verifier {0} already scores this task")]
    ExactVerifier(String),
    #[error("pair labels not confirmed")]
    UnconfirmedPair,
    #[error("held-out hack used as training pair")]
    HeldOutLeak,
    #[error("unverifiable: {0}")]
    Unverifiable(String),
}

pub type Result<T> = std::result::Result<T, Error>;

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_encode(&Sha256::digest(bytes))
}

fn check_schema(req: &RewardRequest) -> Result<()> {
    if req.schema_id != SCHEMA_REWARD_REQUEST || req.schema_version != SCHEMA_VERSION {
        return Err(Error::Schema(format!(
            "expected {SCHEMA_REWARD_REQUEST} v{SCHEMA_VERSION}, got {} v{}",
            req.schema_id, req.schema_version
        )));
    }
    Ok(())
}

fn grader_prompt(grader_id: &GraderId, criterion: &Criterion, output: &str) -> String {
    format!(
        "grader:{}\ncriterion:{}\n{}\n\n{}",
        grader_id.0, criterion.id.0, criterion.prompt, output
    )
}

fn parse_criterion_score(reply: &str) -> Result<f64> {
    let trimmed = reply.trim();
    let value: f64 = match trimmed.parse() {
        Ok(v) => v,
        Err(_) => {
            return Err(Error::Unverifiable(format!(
                "non-numeric grader reply: {trimmed}"
            )));
        }
    };
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Error::BadScore(value));
    }
    Ok(value)
}

fn weighted_mean(rubric: &Rubric, values: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, c) in rubric.criteria.iter().enumerate() {
        num += values[i] * c.weight;
        den += c.weight;
    }
    num / den
}

/// Multiple graders on one rubric. Disagreement down-weights.
pub struct RubricPanel<J: Judge> {
    pub id: String,
    pub rubric: Rubric,
    pub grader_ids: Vec<GraderId>,
    judge: J,
}

impl<J: Judge> RubricPanel<J> {
    pub fn new(id: impl Into<String>, rubric: Rubric, grader_ids: Vec<GraderId>, judge: J) -> Self {
        Self {
            id: id.into(),
            rubric,
            grader_ids,
            judge,
        }
    }

    pub fn judge(&self) -> &J {
        &self.judge
    }

    pub fn judge_mut(&mut self) -> &mut J {
        &mut self.judge
    }

    /// Weighted mean across graders, multiplied by `(1 - spread)` when
    /// `spread = max - min` of the per-grader weighted scores. Spread at
    /// or above [`DISAGREE_EPS`] still down-weights; it does not drop.
    pub fn score(&mut self, output: &str, now: NowMs) -> Result<Vec<RubricScore>> {
        if self.rubric.criteria.is_empty() {
            return Err(Error::EmptyRubric);
        }
        if self.grader_ids.is_empty() {
            return Err(Error::EmptyPanel);
        }
        for c in &self.rubric.criteria {
            if !c.weight.is_finite() || c.weight <= 0.0 {
                return Err(Error::BadWeight);
            }
        }

        let gids = self.grader_ids.clone();
        let criteria = self.rubric.criteria.clone();
        let mut out = Vec::new();
        for gid in &gids {
            let mut values = Vec::new();
            let mut criterion_scores = Vec::new();
            for c in &criteria {
                let prompt = grader_prompt(gid, c, output);
                let reply = self.judge_mut().complete(&prompt, now)?;
                let value = parse_criterion_score(&reply)?;
                values.push(value);
                criterion_scores.push(CriterionScore {
                    criterion_id: c.id.clone(),
                    value,
                });
            }
            let weighted = weighted_mean(&self.rubric, &values);
            out.push(RubricScore {
                grader_id: gid.clone(),
                scores: criterion_scores,
                weighted,
            });
        }
        Ok(out)
    }

    pub fn combine(&self, scores: &[RubricScore]) -> Result<f64> {
        if scores.is_empty() {
            return Err(Error::EmptyPanel);
        }
        let mut sum = 0.0;
        let mut min = scores[0].weighted;
        let mut max = scores[0].weighted;
        for s in scores {
            let w = s.weighted;
            sum += w;
            if w < min {
                min = w;
            }
            if w > max {
                max = w;
            }
        }
        let mean = sum / scores.len() as f64;
        let spread = max - min;
        Ok(mean * (1.0 - spread))
    }

    pub fn verify(
        &mut self,
        req: &RewardRequest,
        output: &str,
        scored_at: &str,
        now: NowMs,
    ) -> Result<RewardResponse> {
        check_schema(req)?;
        let scores = self.score(output, now)?;
        let value = self.combine(&scores)?;
        let mut resp = RewardResponse::new(req.request_id.clone(), scored_at);
        resp.score = value;
        resp.passed = value > 0.5;
        resp.evidence.push(Evidence {
            kind: EvidenceKind::Rubric,
            hash: sha256_hex(output.as_bytes()),
            summary: None,
        });
        Ok(resp)
    }
}

/// One RLCD pair. Assumed label is "prefer positive".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContrastivePair {
    pub input: String,
    pub positive_prompt: String,
    pub negative_prompt: String,
    pub positive: String,
    pub negative: String,
}

impl ContrastivePair {
    pub fn new(
        input: impl Into<String>,
        positive_prompt: impl Into<String>,
        negative_prompt: impl Into<String>,
        positive: impl Into<String>,
        negative: impl Into<String>,
    ) -> Self {
        Self {
            input: input.into(),
            positive_prompt: positive_prompt.into(),
            negative_prompt: negative_prompt.into(),
            positive: positive.into(),
            negative: negative.into(),
        }
    }
}

/// Keep a tampering pair only when checks confirm the labels.
pub struct TamperPairFilter;

impl TamperPairFilter {
    pub fn new() -> Self {
        Self
    }

    /// `negative` must be a confirmed hack; `positive` must not. Otherwise
    /// [`Error::UnconfirmedPair`].
    pub fn keep(
        &self,
        _pair: &ContrastivePair,
        positive: &TrajectoryView,
        negative: &TrajectoryView,
    ) -> Result<()> {
        if negative.is_confirmed_hack() && !positive.is_confirmed_hack() {
            Ok(())
        } else {
            Err(Error::UnconfirmedPair)
        }
    }
}

impl Default for TamperPairFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// RLCD-Rescore: keep an open-ended pair only when the rubric panel
/// prefers the positive output. Drop when graders disagree with the
/// assumed label.
pub struct RescoreFilter;

impl RescoreFilter {
    pub fn new() -> Self {
        Self
    }

    /// Prefers positive when `combine(positive) > combine(negative)`.
    pub fn keep(&self, positive: f64, negative: f64) -> Result<()> {
        if positive > negative {
            Ok(())
        } else {
            Err(Error::UnconfirmedPair)
        }
    }
}

impl Default for RescoreFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Rule-based detector on [`TrajectoryView`]. A learned reviewer can wrap
/// [`Judge`] later; CPU tests pin this one.
pub struct TamperingDetector {
    pub id: String,
}

impl TamperingDetector {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }

    pub fn kinds(&self, view: &TrajectoryView) -> Vec<HackKind> {
        view.kinds()
    }

    pub fn is_hack(&self, view: &TrajectoryView) -> bool {
        view.is_confirmed_hack()
    }

    /// `Flag::Tampering` when `view` is a confirmed hack. Score 0 and
    /// `passed == false`. Clean views do not add the flag.
    pub fn verify(
        &self,
        req: &RewardRequest,
        view: &TrajectoryView,
        scored_at: &str,
        _now: NowMs,
    ) -> Result<RewardResponse> {
        check_schema(req)?;
        let mut resp = RewardResponse::new(req.request_id.clone(), scored_at);
        if view.is_confirmed_hack() {
            resp.score = 0.0;
            resp.passed = false;
            resp.flags.push(Flag::Tampering);
        } else {
            resp.score = 1.0;
            resp.passed = true;
        }
        Ok(resp)
    }
}

/// Held-out hacks for detector eval. Never used as training pairs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HeldOutHacks {
    inner: BTreeMap<String, TrajectoryView>,
}

impl HeldOutHacks {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, id: impl Into<String>, view: TrajectoryView) {
        self.inner.insert(id.into(), view);
    }

    pub fn contains(&self, id: &str) -> bool {
        self.inner.contains_key(id)
    }

    pub fn get(&self, id: &str) -> Option<&TrajectoryView> {
        self.inner.get(id)
    }

    /// Eval only. Training-pair construction must not call this.
    pub fn eval(&self, detector: &TamperingDetector) -> Result<DetectReport> {
        let n = self.inner.len();
        let mut flagged = 0;
        for view in self.inner.values() {
            if detector.is_hack(view) {
                flagged += 1;
            }
        }
        Ok(DetectReport { n, flagged })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetectReport {
    pub n: usize,
    pub flagged: usize,
}

impl DetectReport {
    pub fn recall(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.flagged as f64 / self.n as f64
        }
    }
}

/// Refuse to attach a learned grader when an exact D2 verifier already
/// scores the task (spec 9.3: "Not used for any domain with an exact
/// verifier").
pub fn refuse_exact(verifier_id: &str) -> Result<()> {
    Err(Error::ExactVerifier(verifier_id.to_string()))
}
