//! Shared fixtures and hygiene checks for D3 oracle tests.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use prometheus_envs::NowMs;
use prometheus_tasks::{
    MintedTask, Provenance, Source, Split, TaskDomain, TaskSpec, VerifierId, SCHEMA_TASK_SPEC,
    SCHEMA_VERSION,
};
use prometheus_verifiers::{CodeTask, MathTask, VerifierKind};

use crate::reference;

pub const NOW: NowMs = 1_700_000_000_000;
pub const CREATED: &str = "not-a-wall-clock";
pub const MATH_ID: &str = "math-1";
pub const CODE_ID: &str = "code-1";
pub const SWE_ID: &str = "swe-1";

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
    }
}

pub fn assert_hex(label: &str, s: &str) {
    assert!(
        reference::is_sha256_hex(s),
        "{label} is not lowercase hex SHA-256: {s:?}"
    );
}
