//! Group: I6 `SolveRateFilter::probe` / `keep` on wave-2 payloads vs the reference.
//! Mint tests fail on the stub. Fixture probe tests fail until production
//! `score_attempt` scores grid / market / research / long-horizon / open-ended.

mod common;
mod reference;

use std::collections::VecDeque;

use common::*;
use prometheus_envs::NowMs;
use prometheus_rewards::ScriptedJudge;
use prometheus_tasks::{
    Error, LongHorizonTask, MintedTask, OpenEndedFactory, ResearchFactory, Result, SolveRate,
    SolveRateFilter, Solver, GRID_PASS_K,
};
use prometheus_verifiers::{Grid, GridTask};

/// Sequential replies for the same statement (D3 `SeqSolver`).
struct SeqSolver {
    replies: VecDeque<String>,
}

impl SeqSolver {
    fn from_replies(replies: &[&str]) -> Self {
        Self {
            replies: replies.iter().map(|s| (*s).to_string()).collect(),
        }
    }
}

impl Solver for SeqSolver {
    fn attempt(&mut self, _statement: &str, _now: NowMs) -> Result<String> {
        self.replies
            .pop_front()
            .ok_or_else(|| Error::Unverifiable("seq solver exhausted".into()))
    }
}

fn minted_arc() -> MintedTask {
    reference::RefArc::new(ARC_ID, vec![arc_row()])
        .mint(NOW, CREATED)
        .unwrap()
}

fn minted_forecast() -> MintedTask {
    reference::RefForecast::new(FORECAST_ID, vec![forecast_row()])
        .mint(NOW, CREATED)
        .unwrap()
}

fn minted_research() -> MintedTask {
    reference::RefResearch::new(RESEARCH_ID, vec![research_kaggle_row()])
        .mint(NOW, CREATED)
        .unwrap()
}

fn minted_lh() -> MintedTask {
    reference::RefLongHorizon::new(LONG_HORIZON_ID, vec![lh_row()])
        .mint(NOW, CREATED)
        .unwrap()
}

fn minted_oe() -> MintedTask {
    reference::RefOpenEnded::new(OPEN_ENDED_ID, vec![oe_row()])
        .mint(NOW, CREATED)
        .unwrap()
}

fn pin_probe(task: &MintedTask, replies: &[&str]) {
    let n = replies.len() as u32;
    let mut prod = SeqSolver::from_replies(replies);
    let mut refer = SeqSolver::from_replies(replies);
    assert_eq!(
        SolveRateFilter::probe(&mut prod, task, n, NOW),
        reference::probe(&mut refer, task, n, NOW)
    );
}

fn pin_probe_keep(task: &MintedTask, replies: &[&str]) {
    let n = replies.len() as u32;
    let mut prod = SeqSolver::from_replies(replies);
    let mut refer = SeqSolver::from_replies(replies);
    let got = SolveRateFilter::probe(&mut prod, task, n, NOW);
    let want = reference::probe(&mut refer, task, n, NOW);
    assert_eq!(got, want);
    if let Ok(rate) = got {
        assert_eq!(SolveRateFilter::keep(rate), reference::keep(rate));
    }
}

#[test]
fn grid_pass_at_k_second_match() {
    let task = minted_arc();
    let expected = &task.grid.as_ref().unwrap().expected.cells;
    let miss = vec![vec![9, 9], vec![9, 9]];
    let two = reference::encode_grids(&[miss, expected.clone()]);
    pin_probe(&task, &[&two]);
    let mut s = SeqSolver::from_replies(&[&two]);
    let rate = reference::probe(&mut s, &task, 1, NOW).unwrap();
    assert_eq!(rate, SolveRate { passed: 1, n: 1 });
    assert_eq!(reference::keep(rate), Err(Error::Trivial));
}

#[test]
fn grid_pass_at_k_first_match() {
    let task = minted_arc();
    let expected = &task.grid.as_ref().unwrap().expected.cells;
    let miss = vec![vec![0, 0], vec![0, 0]];
    let two = reference::encode_grids(&[expected.clone(), miss]);
    pin_probe(&task, &[&two]);
}

#[test]
fn grid_too_many_is_not_pass() {
    assert_eq!(GRID_PASS_K, 2);
    let task = minted_arc();
    let expected = task.grid.as_ref().unwrap().expected.cells.clone();
    let three = reference::encode_grids(&[expected.clone(), expected.clone(), expected]);
    pin_probe_keep(&task, &[&three]);
    let mut s = SeqSolver::from_replies(&[&three]);
    let rate = reference::probe(&mut s, &task, 1, NOW).unwrap();
    assert_eq!(rate.passed, 0);
    assert_eq!(reference::keep(rate), Err(Error::Impossible));
}

#[test]
fn grid_wrong_is_impossible() {
    let task = minted_arc();
    let wrong = reference::failing_attempt(&task);
    pin_probe_keep(&task, &[&wrong]);
}

#[test]
fn grid_one_grid_json_matches_expected() {
    let task = minted_arc();
    let pass = reference::passing_attempt(&task);
    pin_probe_keep(&task, &[&pass]);
    let mut s = SeqSolver::from_replies(&[&pass]);
    let rate = reference::probe(&mut s, &task, 1, NOW).unwrap();
    assert_eq!(rate.passed, 1);
    assert_eq!(reference::keep(rate), Err(Error::Trivial));
}

#[test]
fn market_strictly_beats_is_solved() {
    let task = minted_forecast();
    let market = task.market.as_ref().unwrap();
    assert!(reference::relative_log_score(1.0, market.market_p, market.outcome) > 0.0);
    let pass = reference::passing_attempt(&task);
    pin_probe_keep(&task, &[&pass]);
}

#[test]
fn market_tie_is_not_solved() {
    let task = minted_forecast();
    let p = task.market.as_ref().unwrap().market_p;
    let tie = reference::encode_market(p);
    pin_probe_keep(&task, &[&tie]);
    let mut s = SeqSolver::from_replies(&[&tie]);
    let rate = reference::probe(&mut s, &task, 1, NOW).unwrap();
    assert_eq!(rate.passed, 0);
    assert_eq!(reference::keep(rate), Err(Error::Impossible));
}

#[test]
fn market_worse_than_market_is_not_solved() {
    let task = minted_forecast();
    let wrong = reference::failing_attempt(&task);
    pin_probe(&task, &[&wrong]);
}

#[test]
fn market_outcome_false_zero_beats() {
    let row = forecast_src("m0", "Will the sun explode tomorrow?", 0.4, false);
    let task = reference::RefForecast::new(FORECAST_ID, vec![row])
        .mint(NOW, CREATED)
        .unwrap();
    let pass = reference::passing_attempt(&task);
    pin_probe_keep(&task, &[&pass]);
    let mut s = SeqSolver::from_replies(&[&pass]);
    let rate = reference::probe(&mut s, &task, 1, NOW).unwrap();
    assert_eq!(rate.passed, 1);
    assert_eq!(reference::keep(rate), Err(Error::Trivial));
}

#[test]
fn research_numeric_match_on_target() {
    let task = minted_research();
    let pass = reference::passing_attempt(&task);
    let fail = reference::failing_attempt(&task);
    pin_probe_keep(&task, &[&pass]);
    pin_probe_keep(&task, &[&fail]);
    let mut s = SeqSolver::from_replies(&["0.85"]);
    let rate = reference::probe(&mut s, &task, 1, NOW).unwrap();
    assert_eq!(rate.passed, 1);
    assert_eq!(reference::keep(rate), Err(Error::Trivial));
}

#[test]
fn research_kinds_all_numeric() {
    for row in [
        research_kaggle_row(),
        research_speedrun_row(),
        research_paper_row(),
    ] {
        let task = reference::RefResearch::new(RESEARCH_ID, vec![row])
            .mint(NOW, CREATED)
            .unwrap();
        let pass = reference::passing_attempt(&task);
        pin_probe(&task, &[&pass]);
    }
}

#[test]
fn long_horizon_all_checkpoints_and_final() {
    let task = minted_lh();
    let pass = reference::passing_attempt(&task);
    pin_probe_keep(&task, &[&pass]);
}

#[test]
fn long_horizon_failed_checkpoint_fails() {
    let task = minted_lh();
    let bad = reference::encode_long_horizon(
        &["WRONG".into(), "beta".into()],
        "omega",
    );
    pin_probe_keep(&task, &[&bad]);
    let mut s = SeqSolver::from_replies(&[&bad]);
    let rate = reference::probe(&mut s, &task, 1, NOW).unwrap();
    assert_eq!(rate.passed, 0);
    assert_eq!(reference::keep(rate), Err(Error::Impossible));
}

#[test]
fn long_horizon_failed_final_fails() {
    let task = minted_lh();
    let bad = reference::encode_long_horizon(&["alpha".into(), "beta".into()], "WRONG");
    pin_probe_keep(&task, &[&bad]);
}

#[test]
fn long_horizon_grid_checkpoint_uses_pass_k() {
    let row = lh_src(
        "hg",
        "Mixed long-horizon.",
        vec![grid_checkpoint(
            "cp-g",
            "draw",
            vec![vec![1, 2], vec![3, 4]],
        )],
        Some(prometheus_verifiers::MathTask::new("done")),
        None,
    );
    let mut task = reference::RefLongHorizon::new(LONG_HORIZON_ID, vec![row])
        .mint(NOW, CREATED)
        .unwrap();
    // Ensure the payload is the mixed one we minted.
    assert!(task
        .long_horizon
        .as_ref()
        .unwrap()
        .checkpoints[0]
        .grid
        .is_some());
    let pass = reference::passing_attempt(&task);
    pin_probe(&task, &[&pass]);
    let _ = &mut task;
}

#[test]
fn long_horizon_empty_checkpoints_final_only() {
    let row = lh_src(
        "he",
        "Final only.",
        vec![],
        Some(prometheus_verifiers::MathTask::new("omega")),
        None,
    );
    let task = reference::RefLongHorizon::new(LONG_HORIZON_ID, vec![row])
        .mint(NOW, CREATED)
        .unwrap();
    assert!(task.long_horizon.as_ref().unwrap().checkpoints.is_empty());
    let pass = reference::passing_attempt(&task);
    pin_probe_keep(&task, &[&pass]);
}

#[test]
fn open_ended_weighted_gt_half_is_solved() {
    let task = minted_oe();
    let pass = reference::passing_attempt(&task);
    pin_probe_keep(&task, &[&pass]);
}

#[test]
fn open_ended_weighted_eq_half_is_not_solved() {
    let task = minted_oe();
    let mut scores = std::collections::BTreeMap::new();
    scores.insert("clarity".into(), 0.5);
    scores.insert("accuracy".into(), 0.5);
    let half = reference::encode_open_ended(&scores);
    pin_probe_keep(&task, &[&half]);
    let mut s = SeqSolver::from_replies(&[&half]);
    let rate = reference::probe(&mut s, &task, 1, NOW).unwrap();
    assert_eq!(rate.passed, 0);
    assert_eq!(reference::keep(rate), Err(Error::Impossible));
}

#[test]
fn open_ended_weighted_unequal_criteria() {
    let row = oe_src("ow", "Write a short proof.", weighted_rubric());
    let task = reference::RefOpenEnded::new(OPEN_ENDED_ID, vec![row])
        .mint(NOW, CREATED)
        .unwrap();
    let mut high_b = std::collections::BTreeMap::new();
    high_b.insert("a".into(), 0.0);
    high_b.insert("b".into(), 1.0);
    // (1*0 + 3*1) / 4 = 0.75 > 0.5
    let pass = reference::encode_open_ended(&high_b);
    pin_probe_keep(&task, &[&pass]);

    let mut high_a = std::collections::BTreeMap::new();
    high_a.insert("a".into(), 1.0);
    high_a.insert("b".into(), 0.0);
    // (1*1 + 3*0) / 4 = 0.25
    let fail = reference::encode_open_ended(&high_a);
    pin_probe_keep(&task, &[&fail]);
}

#[test]
fn open_ended_scripted_judge_weighted_gt_half() {
    let task = minted_oe();
    let oe = task.open_ended.as_ref().unwrap();
    let output = "because two plus two is four";
    let mut judge = ScriptedJudge::new();
    for c in &oe.rubric.criteria {
        judge.insert(reference::grader_prompt(GRADER_ID, c, output), "0.9");
    }
    let scores = reference::scripted_scores(&oe.rubric, GRADER_ID, output, &mut judge, NOW).unwrap();
    let mut judge = ScriptedJudge::new();
    for c in &oe.rubric.criteria {
        judge.insert(reference::grader_prompt(GRADER_ID, c, output), "0.9");
    }
    let weighted =
        reference::scripted_weighted(&oe.rubric, GRADER_ID, output, &mut judge, NOW).unwrap();
    assert!(weighted > 0.5);
    let attempt = reference::encode_open_ended(&scores);
    pin_probe_keep(&task, &[&attempt]);
}

#[test]
fn open_ended_scripted_judge_le_half_not_solved() {
    let task = minted_oe();
    let oe = task.open_ended.as_ref().unwrap();
    let output = "idk";
    let mut judge = ScriptedJudge::new();
    for c in &oe.rubric.criteria {
        judge.insert(reference::grader_prompt(GRADER_ID, c, output), "0.4");
    }
    let scores = reference::scripted_scores(&oe.rubric, GRADER_ID, output, &mut judge, NOW).unwrap();
    let mut judge = ScriptedJudge::new();
    for c in &oe.rubric.criteria {
        judge.insert(reference::grader_prompt(GRADER_ID, c, output), "0.4");
    }
    let weighted = reference::scripted_weighted(&oe.rubric, GRADER_ID, output, &mut judge, NOW)
        .unwrap();
    assert!(weighted <= 0.5);
    pin_probe_keep(&task, &[&reference::encode_open_ended(&scores)]);
}

#[test]
fn interior_rate_is_kept_for_each_wave2_payload() {
    for task in [
        minted_arc(),
        minted_forecast(),
        minted_research(),
        minted_lh(),
        minted_oe(),
    ] {
        let pass = reference::passing_attempt(&task);
        let fail = reference::failing_attempt(&task);
        pin_probe_keep(&task, &[&pass, &fail]);
        let mut s = SeqSolver::from_replies(&[&pass, &fail]);
        let rate = reference::probe(&mut s, &task, 2, NOW).unwrap();
        assert_eq!(rate, SolveRate { passed: 1, n: 2 });
        assert_eq!(SolveRateFilter::keep(rate).unwrap(), rate);
    }
}

#[test]
fn zero_percent_impossible_hundred_percent_trivial() {
    for task in [
        minted_arc(),
        minted_forecast(),
        minted_research(),
        minted_lh(),
        minted_oe(),
    ] {
        let pass = reference::passing_attempt(&task);
        let fail = reference::failing_attempt(&task);
        pin_probe_keep(&task, &[&fail]);
        pin_probe_keep(&task, &[&pass]);
    }
}

#[test]
fn production_mint_then_probe_n_zero_is_badn() {
    let task = ResearchFactory::new(RESEARCH_ID, vec![research_kaggle_row()])
        .mint(NOW, CREATED)
        .unwrap();
    let mut s = SeqSolver::from_replies(&[]);
    assert_eq!(
        SolveRateFilter::probe(&mut s, &task, 0, NOW),
        Err(Error::BadN)
    );
}

#[test]
fn production_mint_then_probe_matches_reference() {
    let mut prod = OpenEndedFactory::new(OPEN_ENDED_ID, vec![oe_row()]);
    let mut refer = reference::RefOpenEnded::from_prod(&prod);
    let got = prod.mint(NOW, CREATED).unwrap();
    let want = refer.mint(NOW, CREATED).unwrap();
    assert_eq!(got, want);
    let pass = reference::passing_attempt(&got);
    pin_probe_keep(&got, &[&pass]);
}

#[test]
fn malformed_attempt_is_not_solved() {
    let task = minted_arc();
    pin_probe_keep(&task, &["not-json"]);
    let task = minted_forecast();
    pin_probe_keep(&task, &["nope"]);
    let task = minted_research();
    pin_probe_keep(&task, &["abc"]);
    let task = minted_lh();
    pin_probe_keep(&task, &["{}"]);
    let task = minted_oe();
    pin_probe_keep(&task, &["[]"]);
}

#[test]
fn grid_payload_on_minted_task_is_the_expected_grid() {
    let mut t = minted_arc();
    t.grid = Some(GridTask::new(Grid::new(vec![vec![1, 2], vec![3, 4]])));
    let one = "[[1,2],[3,4]]";
    pin_probe(&t, &[one]);
}

#[test]
fn long_horizon_task_struct_all_must_pass() {
    let mut task = minted_lh();
    task.long_horizon = Some(LongHorizonTask {
        checkpoints: vec![math_checkpoint("c0", "s", "alpha")],
        final_math: Some(prometheus_verifiers::MathTask::new("omega")),
        final_code: None,
    });
    let pass = reference::encode_long_horizon(&["alpha".into()], "omega");
    pin_probe(&task, &[&pass]);
}
