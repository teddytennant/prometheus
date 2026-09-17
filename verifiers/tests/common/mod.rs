//! Shared fixtures for D2 verifier oracle tests.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use prometheus_envs::{Image, Pool, PoolConfig};
use prometheus_verifiers::{
    Answer, CodeTask, Evidence, EvidenceKind, Grid, Mutant, RewardRequest, RewardResponse, TestRun,
    SCHEMA_REWARD_REQUEST, SCHEMA_VERSION,
};

pub const SCORED_AT: &str = "not-a-wall-clock";
pub const NOW_MS: u64 = 1_700_000_000_000;
pub const WORKSPACE_OUT: &str = "/workspace/out.txt";
pub const HIDDEN_TEST: &str = "/grader/hidden.txt";
pub const HIDDEN_BODY: &[u8] = b"secret-do-not-leak\n";
pub const EXPECTED_STDOUT: &[u8] = b"ok\n";

pub type CpuPool = Pool<prometheus_envs::InProcess>;
pub type CpuRegistry = prometheus_verifiers::Registry<prometheus_envs::InProcess>;

pub fn src_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

pub fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../contracts/goldens/v1")
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

pub fn evidence_kinds(evidence: &[Evidence]) -> Vec<EvidenceKind> {
    let mut ks: Vec<_> = evidence.iter().map(|e| e.kind).collect();
    ks.sort_by_key(|k| format!("{k:?}"));
    ks
}

pub fn has_kind(evidence: &[Evidence], kind: EvidenceKind) -> bool {
    evidence.iter().any(|e| e.kind == kind)
}

pub fn cpu_pool() -> CpuPool {
    Pool::in_process(PoolConfig {
        capacity: 32,
        ..Default::default()
    })
}

pub fn python_hidden_run() -> TestRun {
    TestRun::python(
        serde_json::json!({"code": "print(open(\"/workspace/out.txt\").read())"}),
        5,
        EXPECTED_STDOUT.to_vec(),
    )
}

pub fn code_image(id: &str) -> Image {
    let mut image = Image::new(id);
    image.agent_files.insert(
        "/workspace/README".into(),
        b"write ok to /workspace/out.txt".to_vec(),
    );
    image
        .hidden_tests
        .insert(HIDDEN_TEST.into(), HIDDEN_BODY.to_vec());
    image
}

pub fn code_task(id: &str, mutants: Vec<Mutant>) -> CodeTask {
    let mut task = CodeTask::new(code_image(id), python_hidden_run());
    task.mutants = mutants;
    task
}

pub fn strong_mutant() -> Mutant {
    Mutant::new(
        "flip-out",
        BTreeMap::from([(WORKSPACE_OUT.into(), b"mutant".to_vec())]),
    )
}

pub fn weak_mutant() -> Mutant {
    Mutant::new(
        "still-ok",
        BTreeMap::from([(WORKSPACE_OUT.into(), b"ok".to_vec())]),
    )
}

pub fn answer_ok() -> Answer {
    Answer::Code {
        files: BTreeMap::from([(WORKSPACE_OUT.into(), b"ok".to_vec())]),
    }
}

pub fn answer_wrong() -> Answer {
    Answer::Code {
        files: BTreeMap::from([(WORKSPACE_OUT.into(), b"nope".to_vec())]),
    }
}

pub fn grid(cells: Vec<Vec<u8>>) -> Grid {
    Grid::new(cells)
}

pub fn assert_request_schema(req: &RewardRequest) {
    assert_eq!(req.schema_id, SCHEMA_REWARD_REQUEST);
    assert_eq!(req.schema_version, SCHEMA_VERSION);
}
