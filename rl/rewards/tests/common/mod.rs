//! Shared fixtures for I3 rewards oracle tests.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use prometheus_rewards::{
    ContrastivePair, Criterion, CriterionId, GraderId, RewardRequest, RewardResponse, Rubric,
    RubricPanel, RubricScore, ScriptedJudge, TrajectoryView,
};
use prometheus_verifiers::{SCHEMA_REWARD_REQUEST, SCHEMA_VERSION};

pub const SCORED_AT: &str = "not-a-wall-clock";
pub const NOW_MS: u64 = 1_700_000_000_000;
pub const ABS_TOL: f64 = 1e-5;
pub const OUTPUT: &str = "the answer is four";

pub fn src_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

pub fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts/goldens/v1")
        .join(name)
}

pub fn walk_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    for ent in fs::read_dir(dir).expect("read src") {
        let ent = ent.expect("entry");
        let p = ent.path();
        if p.is_dir() {
            walk_rs(&p, out);
        } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}

pub fn src_does_not_import_tests() {
    let src = src_root();
    let mut files = Vec::new();
    walk_rs(&src, &mut files);
    for f in files {
        let text = fs::read_to_string(&f).unwrap_or_else(|e| panic!("read {f:?}: {e}"));
        for line in text.lines() {
            let t = line.trim();
            if t.starts_with("//") {
                continue;
            }
            assert!(
                !t.contains("tests::")
                    && !t.contains("tests/reference")
                    && !t.contains("crate::tests"),
                "{} must not import tests/: {t}",
                f.display()
            );
        }
    }
}

pub fn src_does_not_read_wall_clock() {
    let src = src_root();
    let mut files = Vec::new();
    walk_rs(&src, &mut files);
    let banned = [
        "SystemTime",
        "Instant::now",
        "Utc::now",
        "Local::now",
        "OffsetDateTime::now",
        "chrono::",
        "web_time",
        "unix_timestamp",
    ];
    for f in files {
        let text = fs::read_to_string(&f).unwrap_or_else(|e| panic!("read {f:?}: {e}"));
        for line in text.lines() {
            let t = line.trim();
            if t.starts_with("//") {
                continue;
            }
            for b in banned {
                assert!(
                    !t.contains(b),
                    "{} must not read wall clock ({b}): {t}",
                    f.display()
                );
            }
        }
    }
}

pub fn reward_request(verifier_id: &str) -> RewardRequest {
    RewardRequest::new(
        "rew-0001",
        "task-math-0001",
        "roll-0001",
        "df4d832092e8878123a97d334652a1b44bfc48e5c4035adcceb00b25487e7001",
        verifier_id,
        "snap-0001",
    )
}

pub fn assert_meta(resp: &RewardResponse, req: &RewardRequest, scored_at: &str) {
    assert_eq!(resp.schema_id, prometheus_verifiers::SCHEMA_REWARD_RESPONSE);
    assert_eq!(resp.schema_version, SCHEMA_VERSION);
    assert_eq!(resp.request_id, req.request_id);
    assert_eq!(resp.scored_at, scored_at);
    assert_ne!(resp.scored_at, NOW_MS.to_string());
}

pub fn assert_request_schema(req: &RewardRequest) {
    assert_eq!(req.schema_id, SCHEMA_REWARD_REQUEST);
    assert_eq!(req.schema_version, SCHEMA_VERSION);
}

pub fn assert_close(got: f64, exp: f64) {
    assert!(
        (got - exp).abs() <= ABS_TOL,
        "got {got} expected {exp} (tol {ABS_TOL})"
    );
}

pub fn view(
    wrote_test_file: bool,
    special_cased: bool,
    visible_passed: bool,
    hidden_passed: bool,
) -> TrajectoryView {
    TrajectoryView {
        wrote_test_file,
        test_write_paths: Vec::new(),
        special_cased,
        visible_passed,
        hidden_passed,
    }
}

pub fn clean_view() -> TrajectoryView {
    view(false, false, false, false)
}

pub fn test_write_view() -> TrajectoryView {
    TrajectoryView {
        wrote_test_file: true,
        test_write_paths: vec!["/tests/test_foo.py".into()],
        special_cased: false,
        visible_passed: false,
        hidden_passed: false,
    }
}

pub fn special_case_view() -> TrajectoryView {
    view(false, true, false, false)
}

pub fn visible_pass_hidden_fail_view() -> TrajectoryView {
    view(false, false, true, false)
}

pub fn hidden_pass_visible_fail_view() -> TrajectoryView {
    view(false, false, false, true)
}

pub fn all_hack_kinds_view() -> TrajectoryView {
    view(true, true, true, false)
}

pub fn pair(positive: &str, negative: &str) -> ContrastivePair {
    ContrastivePair::new("input", "pos-prompt", "neg-prompt", positive, negative)
}

pub fn detector() -> prometheus_rewards::TamperingDetector {
    prometheus_rewards::TamperingDetector::new("tamper")
}

pub fn criterion(id: &str, prompt: &str, weight: f64) -> Criterion {
    Criterion::new(id, prompt, weight)
}

pub fn one_criterion_rubric() -> Rubric {
    Rubric::new("rubric-1", vec![criterion("clarity", "Is it clear?", 1.0)])
}

pub fn two_criterion_rubric() -> Rubric {
    Rubric::new(
        "rubric-2",
        vec![
            criterion("clarity", "Is it clear?", 1.0),
            criterion("truth", "Is it true?", 3.0),
        ],
    )
}

pub fn grader(id: &str) -> GraderId {
    GraderId(id.to_string())
}

pub fn cid(id: &str) -> CriterionId {
    CriterionId(id.to_string())
}

pub fn weighted_scores(pairs: &[(&str, f64)]) -> Vec<RubricScore> {
    pairs
        .iter()
        .map(|(id, w)| RubricScore {
            grader_id: grader(id),
            scores: Vec::new(),
            weighted: *w,
        })
        .collect()
}

pub fn panel_with_graders(n: usize) -> RubricPanel<ScriptedJudge> {
    let ids: Vec<GraderId> = (0..n).map(|i| grader(&format!("g{i}"))).collect();
    RubricPanel::new("panel", one_criterion_rubric(), ids, ScriptedJudge::new())
}
