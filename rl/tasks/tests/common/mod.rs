//! Shared fixtures and hygiene checks for D3 oracle tests.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use prometheus_envs::NowMs;
use prometheus_tasks::{
    ArcSource, Checkpoint, Criterion, ForecastSource, Grid, LongHorizonSource, MintedTask,
    OpenEndedSource, Provenance, ResearchKind, ResearchSource, Rubric, Source, Split, TaskDomain,
    TaskSpec, VerifierId, SCHEMA_TASK_SPEC, SCHEMA_VERSION,
};
use prometheus_verifiers::{CodeTask, GridTask, MarketTask, MathTask, VerifierKind};

use crate::reference;

pub const NOW: NowMs = 1_700_000_000_000;
pub const CREATED: &str = "not-a-wall-clock";
pub const MATH_ID: &str = "math-1";
pub const CODE_ID: &str = "code-1";
pub const SWE_ID: &str = "swe-1";
pub const RESEARCH_ID: &str = "research-1";
pub const LONG_HORIZON_ID: &str = "long-horizon-1";
pub const ARC_ID: &str = "arc-1";
pub const FORECAST_ID: &str = "forecast-1";
pub const OPEN_ENDED_ID: &str = "open-ended-1";
pub const GRADER_ID: &str = "alice";

pub fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts/goldens/v1")
        .join(name)
}

pub fn src_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
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

pub fn src(id: &str, body: &str) -> Source {
    Source::new(id, format!("catalog-{id}"), body)
}

pub fn math_row() -> Source {
    src("gsm8k-1", "What is 2+2?")
}

pub fn math_row_b() -> Source {
    src("gsm8k-2", "What is 3+3?")
}

pub fn code_row() -> Source {
    src("humaneval-1", "Write f() that prints the answer.")
}

pub fn swe_row() -> Source {
    src("swebench-1", "repo@abc123")
}

pub fn sample_rubric() -> Rubric {
    Rubric::new(
        "rubric-1",
        vec![
            Criterion::new("clarity", "GRADER_ONLY: is it clear?", 1.0),
            Criterion::new("accuracy", "GRADER_ONLY: is it accurate?", 1.0),
        ],
    )
}

pub fn weighted_rubric() -> Rubric {
    Rubric::new(
        "rubric-w",
        vec![
            Criterion::new("a", "GRADER_ONLY: criterion a", 1.0),
            Criterion::new("b", "GRADER_ONLY: criterion b", 3.0),
        ],
    )
}

pub fn research_src(
    id: &str,
    body: &str,
    kind: ResearchKind,
    target: f64,
    budget_gpu_minutes: Option<u32>,
    rubric: Option<Rubric>,
) -> ResearchSource {
    ResearchSource {
        source: src(id, body),
        kind,
        target,
        budget_gpu_minutes,
        rubric,
    }
}

pub fn research_kaggle_row() -> ResearchSource {
    research_src(
        "k1",
        "What is the holdout accuracy?",
        ResearchKind::Kaggle,
        0.85,
        None,
        None,
    )
}

pub fn research_speedrun_row() -> ResearchSource {
    research_src(
        "sp1",
        "Reproduce the GPU-minute budget run.",
        ResearchKind::Speedrun,
        42.0,
        Some(8),
        None,
    )
}

pub fn research_paper_row() -> ResearchSource {
    research_src(
        "p1",
        "Reproduce table 2 numeric match.",
        ResearchKind::PaperRepro,
        3.25,
        None,
        Some(sample_rubric()),
    )
}

pub fn math_checkpoint(id: &str, statement: &str, expected: &str) -> Checkpoint {
    Checkpoint {
        id: id.into(),
        statement: statement.into(),
        math: Some(MathTask::new(expected)),
        code: None,
        grid: None,
    }
}

pub fn grid_checkpoint(id: &str, statement: &str, cells: Vec<Vec<u8>>) -> Checkpoint {
    Checkpoint {
        id: id.into(),
        statement: statement.into(),
        math: None,
        code: None,
        grid: Some(GridTask::new(Grid::new(cells))),
    }
}

pub fn lh_src(
    id: &str,
    body: &str,
    checkpoints: Vec<Checkpoint>,
    final_math: Option<MathTask>,
    final_code: Option<CodeTask>,
) -> LongHorizonSource {
    LongHorizonSource {
        source: src(id, body),
        checkpoints,
        final_math,
        final_code,
    }
}

pub fn lh_row() -> LongHorizonSource {
    lh_src(
        "h1",
        "Complete the multi-step mission.",
        vec![
            math_checkpoint("cp-0", "do step 0", "alpha"),
            math_checkpoint("cp-1", "do step 1", "beta"),
        ],
        Some(MathTask::new("omega")),
        None,
    )
}

pub fn arc_src(id: &str, body: &str, cells: Vec<Vec<u8>>) -> ArcSource {
    ArcSource {
        source: src(id, body),
        expected: Grid::new(cells),
    }
}

pub fn arc_row() -> ArcSource {
    arc_src(
        "g1",
        "Complete the output grid.",
        vec![vec![1, 2], vec![3, 4]],
    )
}

pub fn forecast_src(id: &str, body: &str, market_p: f64, outcome: bool) -> ForecastSource {
    ForecastSource {
        source: src(id, body),
        market: MarketTask::new(market_p, outcome),
    }
}

pub fn forecast_row() -> ForecastSource {
    forecast_src("m1", "Will it rain in Atlantis-XYZ?", 0.4, true)
}

pub fn oe_src(id: &str, body: &str, rubric: Rubric) -> OpenEndedSource {
    OpenEndedSource {
        source: src(id, body),
        rubric,
    }
}

pub fn oe_row() -> OpenEndedSource {
    oe_src("o1", "Write a short proof.", sample_rubric())
}

#[allow(clippy::too_many_arguments)]
pub fn spec_skeleton(
    task_id: &str,
    domain: TaskDomain,
    statement: &str,
    hidden_tests_hash: String,
    verifier_id: &str,
    env_image: Option<String>,
    horizon_s: u32,
    max_tool_calls: u32,
) -> TaskSpec {
    TaskSpec {
        schema_id: SCHEMA_TASK_SPEC.to_string(),
        schema_version: SCHEMA_VERSION,
        task_id: task_id.to_string(),
        domain,
        split: Split::Train,
        statement_hash: reference::statement_hash(statement),
        hidden_tests_hash,
        verifier_id: verifier_id.to_string(),
        env_image,
        horizon_s,
        max_tool_calls,
        provenance: Provenance::new("fixture"),
        created_at: CREATED.to_string(),
    }
}

pub fn math_task(statement: &str, expected: &str) -> MintedTask {
    let hidden = reference::sha256_hex(expected.as_bytes());
    MintedTask {
        spec: spec_skeleton(
            "math-fix:1",
            TaskDomain::Math,
            statement,
            hidden,
            reference::MATH_VERIFIER_ID,
            None,
            reference::MATH_HORIZON_S,
            reference::MATH_MAX_TOOL_CALLS,
        ),
        verifier_kind: VerifierKind::SymbolicMath,
        verifier_id: VerifierId(reference::MATH_VERIFIER_ID.to_string()),
        statement: statement.to_string(),
        code: None,
        math: Some(MathTask::new(expected)),
        grid: None,
        market: None,
        research: None,
        long_horizon: None,
        open_ended: None,
    }
}

pub fn code_task(statement: &str, expected_stdout: &[u8]) -> MintedTask {
    let mut image = prometheus_envs::Image::new("img:fixture");
    image.hidden_tests.insert(
        reference::hidden_test_path(),
        reference::HIDDEN_TEST_BODY.to_vec(),
    );
    image.hidden_tests_hash = reference::hidden_files_hash(&image.hidden_tests);
    let hidden = image.hidden_tests_hash.clone();
    let run = prometheus_verifiers::TestRun::python(
        reference::python_payload(),
        reference::CODE_TIMEOUT_S,
        expected_stdout.to_vec(),
    );
    MintedTask {
        spec: spec_skeleton(
            "code-fix:1",
            TaskDomain::Code,
            statement,
            hidden,
            reference::CODE_VERIFIER_ID,
            Some(image.id.0.clone()),
            reference::CODE_HORIZON_S,
            reference::CODE_MAX_TOOL_CALLS,
        ),
        verifier_kind: VerifierKind::SandboxedTests,
        verifier_id: VerifierId(reference::CODE_VERIFIER_ID.to_string()),
        statement: statement.to_string(),
        code: Some(CodeTask::new(image, run)),
        math: None,
        grid: None,
        market: None,
        research: None,
        long_horizon: None,
        open_ended: None,
    }
}

pub fn assert_hex(label: &str, s: &str) {
    assert!(
        reference::is_sha256_hex(s),
        "{label} is not lowercase hex SHA-256: {s:?}"
    );
}
