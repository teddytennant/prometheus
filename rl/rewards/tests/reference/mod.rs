//! Independent slow I3 reference. Production must never call this module.
//!
//! Judge replies: trim whitespace, then parse the whole string as `f64`.
//! Non-numeric => `Error::Unverifiable`. Non-finite or outside `[0, 1]` =>
//! `Error::BadScore`.
//!
//! Rubric evidence hash is SHA-256 hex of the output bytes. Tests require a
//! `EvidenceKind::Rubric` entry with a non-empty hash, not byte-for-byte
//! equality with this formula.

#![allow(dead_code)]

use prometheus_rewards::{
    ContrastivePair, Criterion, CriterionScore, DetectReport, Error, Evidence, EvidenceKind, Flag,
    GraderId, HackKind, HeldOutHacks, Judge, Result, RewardRequest, RewardResponse, Rubric,
    RubricPanel, RubricScore, TrajectoryView,
};
use prometheus_verifiers::{NowMs, SCHEMA_REWARD_REQUEST, SCHEMA_VERSION};
use sha2::{Digest, Sha256};

const HEX: &[u8] = b"0123456789abcdef";

pub fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex_encode(&Sha256::digest(bytes))
}

pub fn check_schema(req: &RewardRequest) -> Result<()> {
    if req.schema_id != SCHEMA_REWARD_REQUEST || req.schema_version != SCHEMA_VERSION {
        return Err(Error::Schema(format!(
            "expected {SCHEMA_REWARD_REQUEST} v{SCHEMA_VERSION}, got {} v{}",
            req.schema_id, req.schema_version
        )));
    }
    Ok(())
}

pub fn grader_prompt(grader_id: &GraderId, criterion: &Criterion, output: &str) -> String {
    format!(
        "grader:{}\ncriterion:{}\n{}\n\n{}",
        grader_id.0, criterion.id.0, criterion.prompt, output
    )
}

pub fn parse_criterion_score(reply: &str) -> Result<f64> {
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

pub fn is_confirmed_hack(view: &TrajectoryView) -> bool {
    view.wrote_test_file || view.special_cased || (view.visible_passed && !view.hidden_passed)
}

pub fn kinds(view: &TrajectoryView) -> Vec<HackKind> {
    let mut out = Vec::new();
    if view.wrote_test_file {
        out.push(HackKind::TestWrite);
    }
    if view.special_cased {
        out.push(HackKind::SpecialCase);
    }
    if view.visible_passed && !view.hidden_passed {
        out.push(HackKind::VisiblePassHiddenFail);
    }
    out
}

pub fn keep_tamper(
    _pair: &ContrastivePair,
    positive: &TrajectoryView,
    negative: &TrajectoryView,
) -> Result<()> {
    if is_confirmed_hack(negative) && !is_confirmed_hack(positive) {
        Ok(())
    } else {
        Err(Error::UnconfirmedPair)
    }
}

pub fn keep_rescore(positive: f64, negative: f64) -> Result<()> {
    if positive > negative {
        Ok(())
    } else {
        Err(Error::UnconfirmedPair)
    }
}

pub fn score<J: Judge>(
    panel: &mut RubricPanel<J>,
    output: &str,
    now: NowMs,
) -> Result<Vec<RubricScore>> {
    if panel.rubric.criteria.is_empty() {
        return Err(Error::EmptyRubric);
    }
    if panel.grader_ids.is_empty() {
        return Err(Error::EmptyPanel);
    }
    for c in &panel.rubric.criteria {
        if !c.weight.is_finite() || c.weight <= 0.0 {
            return Err(Error::BadWeight);
        }
    }

    let gids = panel.grader_ids.clone();
    let criteria = panel.rubric.criteria.clone();
    let mut out = Vec::new();
    for gid in &gids {
        let mut values = Vec::new();
        let mut criterion_scores = Vec::new();
        for c in &criteria {
            let prompt = grader_prompt(gid, c, output);
            let reply = panel.judge_mut().complete(&prompt, now)?;
            let value = parse_criterion_score(&reply)?;
            values.push(value);
            criterion_scores.push(CriterionScore {
                criterion_id: c.id.clone(),
                value,
            });
        }
        let weighted = weighted_mean(&panel.rubric, &values);
        out.push(RubricScore {
            grader_id: gid.clone(),
            scores: criterion_scores,
            weighted,
        });
    }
    Ok(out)
}

pub fn weighted_mean(rubric: &Rubric, values: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    let mut i = 0;
    while i < rubric.criteria.len() {
        num += values[i] * rubric.criteria[i].weight;
        den += rubric.criteria[i].weight;
        i += 1;
    }
    num / den
}

pub fn combine(scores: &[RubricScore]) -> Result<f64> {
    if scores.is_empty() {
        return Err(Error::EmptyPanel);
    }
    let mut sum = 0.0;
    let mut min = scores[0].weighted;
    let mut max = scores[0].weighted;
    let mut i = 0;
    while i < scores.len() {
        let w = scores[i].weighted;
        sum += w;
        if w < min {
            min = w;
        }
        if w > max {
            max = w;
        }
        i += 1;
    }
    let mean = sum / scores.len() as f64;
    let spread = max - min;
    Ok(mean * (1.0 - spread))
}

pub fn passed(score: f64) -> bool {
    score > 0.5
}

pub fn verify_panel<J: Judge>(
    panel: &mut RubricPanel<J>,
    req: &RewardRequest,
    output: &str,
    scored_at: &str,
    now: NowMs,
) -> Result<RewardResponse> {
    check_schema(req)?;
    let scores = score(panel, output, now)?;
    let value = combine(&scores)?;
    let mut resp = RewardResponse::new(req.request_id.clone(), scored_at);
    resp.score = value;
    resp.passed = passed(value);
    resp.evidence.push(Evidence {
        kind: EvidenceKind::Rubric,
        hash: sha256_hex(output.as_bytes()),
        summary: None,
    });
    Ok(resp)
}

pub fn verify_tamper(
    req: &RewardRequest,
    view: &TrajectoryView,
    scored_at: &str,
    _now: NowMs,
) -> Result<RewardResponse> {
    check_schema(req)?;
    let mut resp = RewardResponse::new(req.request_id.clone(), scored_at);
    if is_confirmed_hack(view) {
        resp.score = 0.0;
        resp.passed = false;
        resp.flags.push(Flag::Tampering);
    } else {
        resp.score = 1.0;
        resp.passed = true;
    }
    Ok(resp)
}

pub fn detect_report(views: &[TrajectoryView]) -> DetectReport {
    let n = views.len();
    let mut flagged = 0;
    for view in views {
        if is_confirmed_hack(view) {
            flagged += 1;
        }
    }
    DetectReport { n, flagged }
}

pub fn eval_ids(set: &HeldOutHacks, ids: &[&str]) -> DetectReport {
    let mut views = Vec::new();
    for id in ids {
        if let Some(v) = set.get(id) {
            views.push(v.clone());
        }
    }
    detect_report(&views)
}

pub fn refuse_exact(verifier_id: &str) -> Result<()> {
    Err(Error::ExactVerifier(verifier_id.to_string()))
}

pub fn rubric_has_kind(resp: &RewardResponse) -> bool {
    resp.evidence
        .iter()
        .any(|e| e.kind == EvidenceKind::Rubric && !e.hash.is_empty())
}
