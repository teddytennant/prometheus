//! Independent D3 reference: catalog mint, 0%/100% filter, horizon doubling.
//!
//! Slow and obvious. Production `src/` must not import this module.
//!
//! # Pins (exact numbers and schemes)
//!
//! ## Horizons
//! - [`CODE_HORIZON_S`] = 30, [`CODE_MAX_TOOL_CALLS`] = [`START_TOOL_CALLS`] (10).
//! - [`SWE_HORIZON_S`] = 60 (2× code), [`SWE_MAX_TOOL_CALLS`] = 20 (2× code).
//! - [`MATH_HORIZON_S`] = 30, [`MATH_MAX_TOOL_CALLS`] = [`START_TOOL_CALLS`] (10).
//!
//! ## Identifiers
//! - `task_id` = `{factory_id}:{source.id}` (stable given the same source id).
//! - `env_image` (code/SWE) = `img:{factory_id}:{source.id}`; math has none.
//! - Math verifier id `symbolic-math`; code/SWE `sandboxed-tests`.
//!
//! ## Hashes
//! Lowercase hex SHA-256.
//! - `statement_hash` = SHA-256(UTF-8 statement).
//! - Math `hidden_tests_hash` = SHA-256(UTF-8 `MathTask.expected`).
//! - Code/SWE `hidden_tests_hash` = SHA-256 of hidden-test bytes in sorted-path
//!   order (`path`, NUL, bytes) — same formula as D1 `Image`.
//!
//! ## Payloads
//! - Math: `statement = body`, `MathTask.expected` = SHA-256 hex of
//!   `math-expected:{body}` (non-empty, not the public statement).
//! - Code/SWE: one hidden test under `/grader/hidden.py`, one mutant, public
//!   statement is `body` and must not contain `TestRun.expected_stdout`.
//!   `expected_stdout` is SHA-256 hex of `code-stdout:{body}` or
//!   `swe-stdout:{body}`.
//!
//! ## Catalog
//! Round-robin with wrap-around. Cursor advances only after a successful mint.
//! `created_at` is the caller string; `now` is unused.
//!
//! ## Filter
//! Probe scores math by string equality with `expected`; code by byte equality
//! with `expected_stdout`. Solver errors propagate. `keep` drops n=0 / 0% / 100%.
//!
//! ## Horizon
//! `raise_if` doubles `tool_calls` iff `rate > 0.5` (strict). Equal does not raise.

#![allow(dead_code)]

use std::collections::BTreeMap;

use prometheus_envs::{Image, NowMs, GRADER_ROOT};
use prometheus_tasks::{
    CodeFactory, Error, FactoryId, MathFactory, MintedTask, Result, SolveRate, Solver, Source,
    Split, SweFactory, TaskDomain, TaskSpec, SCHEMA_TASK_SPEC, SCHEMA_VERSION, START_TOOL_CALLS,
};
use prometheus_verifiers::{CodeTask, MathTask, Mutant, TestRun, VerifierId, VerifierKind};
use sha2::{Digest, Sha256};

/// CodeFactory wall-clock horizon, seconds.
pub const CODE_HORIZON_S: u32 = 30;
/// SweFactory wall-clock horizon, seconds. Pin: 2× [`CODE_HORIZON_S`].
pub const SWE_HORIZON_S: u32 = 60;
/// MathFactory wall-clock horizon, seconds.
pub const MATH_HORIZON_S: u32 = 30;
/// CodeFactory `max_tool_calls`. Pin: [`START_TOOL_CALLS`].
pub const CODE_MAX_TOOL_CALLS: u32 = START_TOOL_CALLS;
/// SweFactory `max_tool_calls`. Pin: 2× CodeFactory default.
pub const SWE_MAX_TOOL_CALLS: u32 = START_TOOL_CALLS * 2;
/// MathFactory `max_tool_calls`. Pin: [`START_TOOL_CALLS`].
pub const MATH_MAX_TOOL_CALLS: u32 = START_TOOL_CALLS;

pub const CODE_TIMEOUT_S: u32 = 5;
pub const SWE_TIMEOUT_S: u32 = 10;

pub const MATH_VERIFIER_ID: &str = "symbolic-math";
pub const CODE_VERIFIER_ID: &str = "sandboxed-tests";
pub const SWE_VERIFIER_ID: &str = "sandboxed-tests";

pub const HIDDEN_TEST_NAME: &str = "hidden.py";
pub const HIDDEN_TEST_BODY: &[u8] = b"# hidden\nprint(open('/workspace/out.txt').read())\n";
pub const PYTHON_CODE: &str = "print(open('/workspace/out.txt').read())";
pub const CODE_MUTANT_ID: &str = "wrong-stdout";
pub const SWE_MUTANT_ID: &str = "wrong-patch";
pub const MUTANT_PATH: &str = "/workspace/out.txt";
pub const CODE_MUTANT_BODY: &[u8] = b"WRONG\n";
pub const SWE_MUTANT_BODY: &[u8] = b"WRONG_SWE\n";
pub const SWE_REPO_PATH: &str = "/workspace/repo";

const HEX: &[u8; 16] = b"0123456789abcdef";

pub fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

pub fn statement_hash(statement: &str) -> String {
    sha256_hex(statement.as_bytes())
}

/// D1 hidden-tests hash: sorted `(path, NUL, bytes)` concatenation.
pub fn hidden_files_hash(hidden: &BTreeMap<String, Vec<u8>>) -> String {
    let mut h = Sha256::new();
    for (path, bytes) in hidden {
        h.update(path.as_bytes());
        h.update([0u8]);
        h.update(bytes);
    }
    hex_lower(&h.finalize())
}

pub fn math_expected(body: &str) -> String {
    sha256_hex(format!("math-expected:{body}").as_bytes())
}

pub fn code_expected_stdout(body: &str) -> Vec<u8> {
    sha256_hex(format!("code-stdout:{body}").as_bytes()).into_bytes()
}

pub fn swe_expected_stdout(body: &str) -> Vec<u8> {
    sha256_hex(format!("swe-stdout:{body}").as_bytes()).into_bytes()
}

pub fn task_id(factory_id: &str, source_id: &str) -> String {
    format!("{factory_id}:{source_id}")
}

pub fn image_id(factory_id: &str, source_id: &str) -> String {
    format!("img:{factory_id}:{source_id}")
}

pub fn hidden_test_path() -> String {
    format!("{GRADER_ROOT}/{HIDDEN_TEST_NAME}")
}

pub fn python_payload() -> serde_json::Value {
    serde_json::json!({ "code": PYTHON_CODE })
}

#[derive(Debug, Clone, Copy)]
enum Kind {
    Math,
    Code,
    Swe,
}

/// Round-robin catalog used by the reference factories.
pub struct RefFactory {
    id: FactoryId,
    sources: Vec<Source>,
    cursor: usize,
    kind: Kind,
}

impl RefFactory {
    pub fn math(id: impl Into<String>, sources: Vec<Source>) -> Self {
        Self {
            id: FactoryId(id.into()),
            sources,
            cursor: 0,
            kind: Kind::Math,
        }
    }

    pub fn code(id: impl Into<String>, sources: Vec<Source>) -> Self {
        Self {
            id: FactoryId(id.into()),
            sources,
            cursor: 0,
            kind: Kind::Code,
        }
    }

    pub fn swe(id: impl Into<String>, sources: Vec<Source>) -> Self {
        Self {
            id: FactoryId(id.into()),
            sources,
            cursor: 0,
            kind: Kind::Swe,
        }
    }

    pub fn from_math(f: &MathFactory) -> Self {
        Self::math(f.id().0.clone(), f.sources().to_vec())
    }

    pub fn from_code(f: &CodeFactory) -> Self {
        Self::code(f.id().0.clone(), f.sources().to_vec())
    }

    pub fn from_swe(f: &SweFactory) -> Self {
        Self::swe(f.id().0.clone(), f.sources().to_vec())
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
        let minted = match self.kind {
            Kind::Math => mint_math(&self.id.0, source, created_at),
            Kind::Code => mint_code(&self.id.0, source, created_at),
            Kind::Swe => mint_swe(&self.id.0, source, created_at),
        };
        self.cursor = (i + 1) % self.sources.len();
        Ok(minted)
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

/// Math: attempt equals `MathTask.expected`.
/// Code: attempt bytes equal `TestRun.expected_stdout`.
/// Neither payload: [`Error::Unverifiable`]. Math wins if both are set.
pub fn score_attempt(task: &MintedTask, attempt: &str) -> Result<bool> {
    if let Some(math) = &task.math {
        return Ok(attempt == math.expected);
    }
    if let Some(code) = &task.code {
        return Ok(attempt.as_bytes() == code.run.expected_stdout.as_slice());
    }
    Err(Error::Unverifiable("no D2 payload".into()))
}

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

/// Double `tool_calls` iff `success_rate > RAISE_THRESHOLD` (strict).
pub fn raise_tool_calls(tool_calls: u32, success_rate: f64) -> u32 {
    if success_rate > prometheus_tasks::RAISE_THRESHOLD {
        tool_calls.saturating_mul(2)
    } else {
        tool_calls
    }
}
