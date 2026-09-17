//! Group: RubricPanel score, combine, verify.

mod common;
mod reference;

use common::{
    assert_close, assert_meta, criterion, grader, one_criterion_rubric, panel_with_graders,
    reward_request, two_criterion_rubric, weighted_scores, NOW_MS, OUTPUT, SCORED_AT,
};
use prometheus_rewards::{
    CriterionScore, Error, EvidenceKind, GraderId, Judge, Result, Rubric, RubricPanel, RubricScore,
    ScriptedJudge, DISAGREE_EPS,
};

struct SeqJudge {
    replies: Vec<String>,
    i: usize,
    prompts: Vec<String>,
    nows: Vec<u64>,
}

impl SeqJudge {
    fn new(replies: &[&str]) -> Self {
        Self {
            replies: replies.iter().map(|s| (*s).to_string()).collect(),
            i: 0,
            prompts: Vec::new(),
            nows: Vec::new(),
        }
    }
}

impl Judge for SeqJudge {
    fn complete(&mut self, prompt: &str, now: prometheus_verifiers::NowMs) -> Result<String> {
        self.prompts.push(prompt.to_string());
        self.nows.push(now);
        if self.i >= self.replies.len() {
            return Err(Error::Unverifiable("seq judge exhausted".into()));
        }
        let r = self.replies[self.i].clone();
        self.i += 1;
        Ok(r)
    }
}

fn fill_scripted(panel: &mut RubricPanel<ScriptedJudge>, output: &str, reply: &str) {
    let gids = panel.grader_ids.clone();
    let criteria = panel.rubric.criteria.clone();
    for gid in &gids {
        for c in &criteria {
            let prompt = reference::grader_prompt(gid, c, output);
            panel.judge_mut().insert(prompt, reply);
        }
    }
}

fn two_grader_scripted(replies: &[(&str, &str, &str)]) -> RubricPanel<ScriptedJudge> {
    let rubric = two_criterion_rubric();
    let mut judge = ScriptedJudge::new();
    for gid in ["g1", "g2"] {
        for c in &rubric.criteria {
            let prompt = reference::grader_prompt(&grader(gid), c, OUTPUT);
            let reply = replies
                .iter()
                .find(|(g, cid, _)| *g == gid && *cid == c.id.0)
                .map(|(_, _, r)| *r)
                .unwrap_or("0");
            judge.insert(prompt, reply);
        }
    }
    RubricPanel::new("panel-2", rubric, vec![grader("g1"), grader("g2")], judge)
}

#[test]
fn empty_rubric_is_empty_rubric_error() {
    let mut panel = RubricPanel::new(
        "p",
        Rubric::new("empty", vec![]),
        vec![grader("g1")],
        ScriptedJudge::new(),
    );
    let err = panel.score(OUTPUT, NOW_MS).unwrap_err();
    assert_eq!(err, Error::EmptyRubric);
}

#[test]
fn empty_grader_ids_is_empty_panel_error() {
    let mut panel = RubricPanel::new("p", one_criterion_rubric(), vec![], ScriptedJudge::new());
    let err = panel.score(OUTPUT, NOW_MS).unwrap_err();
    assert_eq!(err, Error::EmptyPanel);
}

#[test]
fn both_empty_is_empty_rubric_or_empty_panel() {
    let mut panel = RubricPanel::new(
        "p",
        Rubric::new("empty", vec![]),
        vec![],
        ScriptedJudge::new(),
    );
    let err = panel.score(OUTPUT, NOW_MS).unwrap_err();
    assert!(
        matches!(err, Error::EmptyRubric | Error::EmptyPanel),
        "got {err:?}"
    );
}

#[test]
fn bad_weight_rejected_at_score_time() {
    for w in [0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY] {
        let mut panel = RubricPanel::new(
            "p",
            Rubric::new("r", vec![criterion("c", "p", w)]),
            vec![grader("g1")],
            ScriptedJudge::new(),
        );
        let err = panel.score(OUTPUT, NOW_MS).unwrap_err();
        assert_eq!(err, Error::BadWeight, "weight {w}");
    }
}

#[test]
fn nan_weight_rejected_at_score_time() {
    let mut panel = RubricPanel::new(
        "p",
        Rubric::new("r", vec![criterion("c", "p", f64::NAN)]),
        vec![grader("g1")],
        ScriptedJudge::new(),
    );
    assert_eq!(panel.score(OUTPUT, NOW_MS).unwrap_err(), Error::BadWeight);
}

#[test]
fn constructor_accepts_bad_weight_and_score_rejects() {
    let c = criterion("c", "p", 0.0);
    assert_eq!(c.weight, 0.0);
    let mut panel = RubricPanel::new(
        "p",
        Rubric::new("r", vec![c]),
        vec![grader("g1")],
        ScriptedJudge::new(),
    );
    assert_eq!(panel.score(OUTPUT, NOW_MS).unwrap_err(), Error::BadWeight);
}

fn assert_bad_score_reply(reply: &str) {
    let mut panel = RubricPanel::new(
        "p",
        one_criterion_rubric(),
        vec![grader("g1")],
        ScriptedJudge::new(),
    );
    fill_scripted(&mut panel, OUTPUT, reply);
    let err = panel.score(OUTPUT, NOW_MS).unwrap_err();
    match err {
        Error::BadScore(v) => {
            let parsed: f64 = reply.trim().parse().unwrap();
            if parsed.is_nan() {
                assert!(v.is_nan());
            } else {
                assert_eq!(v, parsed);
            }
        }
        other => panic!("reply {reply:?}: expected BadScore, got {other:?}"),
    }
}

#[test]
fn out_of_range_and_non_finite_replies_are_bad_score() {
    for reply in ["1.1", "-0.01", "2", "inf", "-inf", "nan"] {
        assert_bad_score_reply(reply);
    }
}

#[test]
fn non_numeric_reply_is_unverifiable() {
    let mut panel = RubricPanel::new(
        "p",
        one_criterion_rubric(),
        vec![grader("g1")],
        ScriptedJudge::new(),
    );
    fill_scripted(&mut panel, OUTPUT, "not-a-number");
    let err = panel.score(OUTPUT, NOW_MS).unwrap_err();
    assert!(
        matches!(err, Error::Unverifiable(_)),
        "non-numeric => Unverifiable, got {err:?}"
    );
}

#[test]
fn whitespace_only_reply_is_unverifiable() {
    let mut panel = RubricPanel::new(
        "p",
        one_criterion_rubric(),
        vec![grader("g1")],
        ScriptedJudge::new(),
    );
    fill_scripted(&mut panel, OUTPUT, "   \n");
    let err = panel.score(OUTPUT, NOW_MS).unwrap_err();
    assert!(matches!(err, Error::Unverifiable(_)), "got {err:?}");
}

#[test]
fn missing_scripted_prompt_is_unverifiable() {
    let mut panel = RubricPanel::new(
        "p",
        one_criterion_rubric(),
        vec![grader("g1")],
        ScriptedJudge::new(),
    );
    let err = panel.score(OUTPUT, NOW_MS).unwrap_err();
    assert!(matches!(err, Error::Unverifiable(_)), "got {err:?}");
}

#[test]
fn trimmed_in_range_decimal_is_accepted() {
    let mut panel = RubricPanel::new(
        "p",
        one_criterion_rubric(),
        vec![grader("g1")],
        ScriptedJudge::new(),
    );
    fill_scripted(&mut panel, OUTPUT, "  0.50\n");
    let scores = panel.score(OUTPUT, NOW_MS).unwrap();
    assert_eq!(scores.len(), 1);
    assert_eq!(scores[0].scores.len(), 1);
    assert_close(scores[0].scores[0].value, 0.5);
    assert_close(scores[0].weighted, 0.5);
}

#[test]
fn score_matches_reference_two_graders_two_criteria() {
    let mut panel = two_grader_scripted(&[
        ("g1", "clarity", "1.0"),
        ("g1", "truth", "0.5"),
        ("g2", "clarity", "0.8"),
        ("g2", "truth", "0.8"),
    ]);
    let got = panel.score(OUTPUT, NOW_MS).unwrap();
    let mut panel2 = two_grader_scripted(&[
        ("g1", "clarity", "1.0"),
        ("g1", "truth", "0.5"),
        ("g2", "clarity", "0.8"),
        ("g2", "truth", "0.8"),
    ]);
    let exp = reference::score(&mut panel2, OUTPUT, NOW_MS).unwrap();
    assert_eq!(got.len(), exp.len());
    assert_eq!(got.len(), 2);
    for (g, e) in got.iter().zip(exp.iter()) {
        assert_eq!(g.grader_id, e.grader_id);
        assert_eq!(g.scores.len(), e.scores.len());
        assert_eq!(g.scores.len(), 2);
        for (gs, es) in g.scores.iter().zip(e.scores.iter()) {
            assert_eq!(gs.criterion_id, es.criterion_id);
            assert_close(gs.value, es.value);
        }
        assert_close(g.weighted, e.weighted);
    }
    assert_close(got[0].weighted, 0.625);
    assert_close(got[1].weighted, 0.8);
}

#[test]
fn score_stores_criterion_scores_in_rubric_order() {
    let mut panel = two_grader_scripted(&[
        ("g1", "clarity", "0.1"),
        ("g1", "truth", "0.2"),
        ("g2", "clarity", "0.3"),
        ("g2", "truth", "0.4"),
    ]);
    let scores = panel.score(OUTPUT, NOW_MS).unwrap();
    assert_eq!(scores[0].scores[0].criterion_id.0, "clarity");
    assert_eq!(scores[0].scores[1].criterion_id.0, "truth");
    assert_eq!(scores[1].grader_id.0, "g2");
}

#[test]
fn grader_prompt_is_pinned_and_now_is_injected() {
    let rubric = one_criterion_rubric();
    let expected = reference::grader_prompt(&grader("alice"), &rubric.criteria[0], OUTPUT);
    assert_eq!(
        expected,
        "grader:alice\ncriterion:clarity\nIs it clear?\n\nthe answer is four"
    );
    let mut panel = RubricPanel::new("p", rubric, vec![grader("alice")], SeqJudge::new(&["0.25"]));
    let scores = panel.score(OUTPUT, NOW_MS).unwrap();
    assert_eq!(panel.judge().prompts, vec![expected]);
    assert_eq!(panel.judge().nows, vec![NOW_MS]);
    assert_close(scores[0].weighted, 0.25);
}

#[test]
fn combine_empty_is_empty_panel() {
    let panel = panel_with_graders(1);
    assert_eq!(panel.combine(&[]).unwrap_err(), Error::EmptyPanel);
}

#[test]
fn combine_one_grader_spread_is_zero() {
    let panel = panel_with_graders(1);
    let scores = weighted_scores(&[("g0", 0.8)]);
    assert_close(panel.combine(&scores).unwrap(), 0.8);
    assert_close(reference::combine(&scores).unwrap(), 0.8);
}

#[test]
fn combine_always_multiplies_by_one_minus_spread() {
    let panel = panel_with_graders(2);
    let scores = weighted_scores(&[("g0", 0.51), ("g1", 0.50)]);
    let got = panel.combine(&scores).unwrap();
    let exp = reference::combine(&scores).unwrap();
    assert_close(got, exp);
    let mean = 0.505;
    let spread = 0.01;
    assert_close(got, mean * (1.0 - spread));
    assert!((got - mean).abs() > 1e-6, "small spread must still bite");
}

#[test]
fn combine_matches_reference_when_spread_ge_disagree_eps() {
    let panel = panel_with_graders(2);
    let scores = weighted_scores(&[("g0", 1.0), ("g1", 0.7)]);
    let spread = 0.3;
    assert!(spread >= DISAGREE_EPS);
    let got = panel.combine(&scores).unwrap();
    let exp = reference::combine(&scores).unwrap();
    assert_close(got, exp);
    assert_close(got, 0.85 * (1.0 - 0.3));
    assert_ne!(got, 0.0);
}

#[test]
fn combine_three_graders_matches_reference() {
    let panel = panel_with_graders(3);
    let scores = weighted_scores(&[("g0", 0.2), ("g1", 0.5), ("g2", 0.9)]);
    let got = panel.combine(&scores).unwrap();
    assert_close(got, reference::combine(&scores).unwrap());
    let mean = (0.2 + 0.5 + 0.9) / 3.0;
    let spread = 0.7;
    assert_close(got, mean * (1.0 - spread));
}

#[test]
fn reference_combine_is_not_plain_mean_and_does_not_drop() {
    let scores = weighted_scores(&[("g0", 1.0), ("g1", 0.7)]);
    let got = reference::combine(&scores).unwrap();
    let mean = 0.85;
    let spread = 0.3;
    assert!(spread >= DISAGREE_EPS);
    assert!(
        (got - mean).abs() > 1e-6,
        "plain mean would be {mean}, spread must bite"
    );
    assert_close(got, mean * (1.0 - spread));
    let drop_on_disagree: Option<f64> = if spread >= DISAGREE_EPS {
        None
    } else {
        Some(mean)
    };
    assert!(
        drop_on_disagree.is_none(),
        "a drop-on-disagree policy would reject this panel"
    );
    assert!(got.is_finite() && got > 0.0, "must still return a value");
}

#[test]
fn combine_equal_graders_is_the_common_value() {
    let panel = panel_with_graders(2);
    let scores = weighted_scores(&[("g0", 0.4), ("g1", 0.4)]);
    assert_close(panel.combine(&scores).unwrap(), 0.4);
}

#[test]
fn verify_matches_reference_and_f1_meta() {
    let mut panel = two_grader_scripted(&[
        ("g1", "clarity", "1.0"),
        ("g1", "truth", "0.5"),
        ("g2", "clarity", "0.8"),
        ("g2", "truth", "0.8"),
    ]);
    let req = reward_request("panel-2");
    let got = panel.verify(&req, OUTPUT, SCORED_AT, NOW_MS).unwrap();
    let mut panel2 = two_grader_scripted(&[
        ("g1", "clarity", "1.0"),
        ("g1", "truth", "0.5"),
        ("g2", "clarity", "0.8"),
        ("g2", "truth", "0.8"),
    ]);
    let exp = reference::verify_panel(&mut panel2, &req, OUTPUT, SCORED_AT, NOW_MS).unwrap();
    assert_meta(&got, &req, SCORED_AT);
    assert_close(got.score, exp.score);
    assert_eq!(got.passed, exp.passed);
    assert_eq!(got.passed, got.score > 0.5);
    assert!(got.flags.is_empty());
    assert!(
        got.evidence
            .iter()
            .any(|e| e.kind == EvidenceKind::Rubric && !e.hash.is_empty()),
        "verify must include EvidenceKind::Rubric with a non-empty hash"
    );
    let mean = (0.625 + 0.8) / 2.0;
    let spread = 0.8 - 0.625;
    assert_close(got.score, mean * (1.0 - spread));
}

#[test]
fn passed_is_strictly_greater_than_half() {
    let panel = panel_with_graders(1);
    let at = weighted_scores(&[("g0", 0.5)]);
    let over = weighted_scores(&[("g0", 0.5000001)]);
    let under = weighted_scores(&[("g0", 0.4999999)]);
    assert!(!reference::passed(panel.combine(&at).unwrap()));
    assert!(reference::passed(panel.combine(&over).unwrap()));
    assert!(!reference::passed(panel.combine(&under).unwrap()));
    let mut p = RubricPanel::new(
        "p",
        one_criterion_rubric(),
        vec![grader("g1")],
        ScriptedJudge::new(),
    );
    fill_scripted(&mut p, OUTPUT, "0.5");
    let req = reward_request("p");
    let resp = p.verify(&req, OUTPUT, SCORED_AT, NOW_MS).unwrap();
    assert_close(resp.score, 0.5);
    assert!(!resp.passed);
}

#[test]
fn schema_mismatch_on_request() {
    let mut panel = RubricPanel::new(
        "p",
        one_criterion_rubric(),
        vec![grader("g1")],
        ScriptedJudge::new(),
    );
    fill_scripted(&mut panel, OUTPUT, "1");
    let mut req = reward_request("p");
    req.schema_id = "prometheus.not_a_request".into();
    assert!(matches!(
        panel.verify(&req, OUTPUT, SCORED_AT, NOW_MS).unwrap_err(),
        Error::Schema(_)
    ));
    let mut req = reward_request("p");
    req.schema_version = 99;
    assert!(matches!(
        panel.verify(&req, OUTPUT, SCORED_AT, NOW_MS).unwrap_err(),
        Error::Schema(_)
    ));
}

#[test]
fn scored_at_is_caller_string_not_now() {
    let mut panel = RubricPanel::new(
        "p",
        one_criterion_rubric(),
        vec![grader("g1")],
        ScriptedJudge::new(),
    );
    fill_scripted(&mut panel, OUTPUT, "1");
    let req = reward_request("p");
    let resp = panel
        .verify(&req, OUTPUT, "caller-supplied", NOW_MS)
        .unwrap();
    assert_eq!(resp.scored_at, "caller-supplied");
    assert_ne!(resp.scored_at, NOW_MS.to_string());
}

#[test]
fn endpoints_zero_and_one_are_valid_scores() {
    let mut panel = RubricPanel::new(
        "p",
        one_criterion_rubric(),
        vec![grader("g1")],
        ScriptedJudge::new(),
    );
    fill_scripted(&mut panel, OUTPUT, "0");
    let s = panel.score(OUTPUT, NOW_MS).unwrap();
    assert_close(s[0].scores[0].value, 0.0);
    let mut panel = RubricPanel::new(
        "p",
        one_criterion_rubric(),
        vec![grader("g1")],
        ScriptedJudge::new(),
    );
    fill_scripted(&mut panel, OUTPUT, "1");
    let s = panel.score(OUTPUT, NOW_MS).unwrap();
    assert_close(s[0].scores[0].value, 1.0);
}

#[test]
fn weighted_mean_uses_criterion_weights() {
    let rubric = two_criterion_rubric();
    assert_close(reference::weighted_mean(&rubric, &[1.0, 0.0]), 0.25);
    assert_close(reference::weighted_mean(&rubric, &[0.0, 1.0]), 0.75);
    let mut panel = two_grader_scripted(&[
        ("g1", "clarity", "1.0"),
        ("g1", "truth", "0.0"),
        ("g2", "clarity", "0.0"),
        ("g2", "truth", "1.0"),
    ]);
    let scores = panel.score(OUTPUT, NOW_MS).unwrap();
    assert_close(scores[0].weighted, 0.25);
    assert_close(scores[1].weighted, 0.75);
}

#[test]
fn combine_does_not_consult_criterion_values() {
    let panel = panel_with_graders(2);
    let scores = vec![
        RubricScore {
            grader_id: GraderId("g0".into()),
            scores: vec![CriterionScore {
                criterion_id: prometheus_rewards::CriterionId("c".into()),
                value: 0.0,
            }],
            weighted: 1.0,
        },
        RubricScore {
            grader_id: GraderId("g1".into()),
            scores: vec![CriterionScore {
                criterion_id: prometheus_rewards::CriterionId("c".into()),
                value: 1.0,
            }],
            weighted: 0.0,
        },
    ];
    assert_close(panel.combine(&scores).unwrap(), 0.0);
}
