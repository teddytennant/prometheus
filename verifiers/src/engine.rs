//! Sandboxed hidden tests plus mutation. Uses a D1 [`prometheus_envs::Pool`].

use std::collections::BTreeMap;

use prometheus_envs::{Backend, Image, Pool, ToolRequest, GRADER_ROOT};
use sha2::{Digest, Sha256};

use crate::{
    Answer, CodeTask, Error, Evidence, EvidenceKind, Mutant, NowMs, Result, RewardRequest,
    RewardResponse, TestRun,
};

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    hex_encode(&Sha256::digest(bytes))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

pub(crate) fn evidence(kind: EvidenceKind, payload: &str) -> Evidence {
    Evidence {
        kind,
        hash: sha256_hex(payload.as_bytes()),
        summary: None,
    }
}

pub(crate) fn base_response(
    req: &RewardRequest,
    scored_at: &str,
    score: f64,
    passed: bool,
) -> RewardResponse {
    let mut resp = RewardResponse::new(req.request_id.clone(), scored_at);
    resp.score = score;
    resp.passed = passed;
    resp
}

pub(crate) fn is_grader_path(path: &str) -> bool {
    path == GRADER_ROOT || path.starts_with("/grader/")
}

fn check_image_isolation(image: &Image) -> Result<()> {
    if image.agent_files.keys().any(|p| is_grader_path(p)) {
        return Err(Error::HiddenTestsVisible);
    }
    if image.hidden_tests.keys().any(|p| !is_grader_path(p)) {
        return Err(Error::HiddenTestsVisible);
    }
    Ok(())
}

fn reject_grader_writes(files: &BTreeMap<String, Vec<u8>>) -> Result<()> {
    if let Some(path) = files.keys().find(|p| is_grader_path(p)) {
        return Err(Error::TestFileWrite {
            path: (*path).clone(),
        });
    }
    Ok(())
}

fn overlay(image: &Image, files: &BTreeMap<String, Vec<u8>>, id: &str) -> Result<Image> {
    reject_grader_writes(files)?;
    let mut out = image.clone();
    out.id = prometheus_envs::ImageId(id.to_string());
    for (p, b) in files {
        out.agent_files.insert(p.clone(), b.clone());
    }
    check_image_isolation(&out)?;
    Ok(out)
}

fn run_hidden<B: Backend>(
    pool: &mut Pool<B>,
    image: Image,
    run: &TestRun,
    now: NowMs,
) -> Result<bool> {
    check_image_isolation(&image)?;
    let image_id = pool.register_image(image)?;
    let snap = pool.snapshot_from_image(&image_id, now)?;
    let fork = pool.fork_group(&snap, 1, now)?;
    let sb = fork
        .sandboxes
        .first()
        .ok_or_else(|| Error::Sandbox("empty fork".into()))?
        .clone();
    let mut treq = ToolRequest::new("grade", snap.0.clone(), run.tool);
    treq.payload = run.payload.clone();
    treq.timeout_s = Some(run.timeout_s);
    let call = pool.call(&sb, treq, now);
    let _ = pool.kill(&sb, now);
    let resp = call?;
    let want = sha256_hex(&run.expected_stdout);
    Ok(resp.ok && resp.stdout_artifact.content_hash == want)
}

fn mutant_passes<B: Backend>(
    pool: &mut Pool<B>,
    task: &CodeTask,
    mutant: &Mutant,
    now: NowMs,
) -> Result<bool> {
    let image = overlay(&task.image, &mutant.files, &format!("mut-{}", mutant.id))?;
    run_hidden(pool, image, &task.run, now)
}

pub(crate) fn verify_sandboxed<B: Backend>(
    pool: &mut Pool<B>,
    task: &CodeTask,
    req: &RewardRequest,
    answer: &Answer,
    scored_at: &str,
    now: NowMs,
) -> Result<RewardResponse> {
    let Answer::Code { files } = answer else {
        return Err(Error::WrongKind);
    };
    check_image_isolation(&task.image)?;
    reject_grader_writes(files)?;

    let mut any_mutant_failed = false;
    for m in &task.mutants {
        if mutant_passes(pool, task, m, now)? {
            return Err(Error::WeakTests { id: m.id.clone() });
        }
        any_mutant_failed = true;
    }

    let image = overlay(&task.image, files, "ans")?;
    let passed_tests = run_hidden(pool, image, &task.run, now)?;

    if passed_tests && !any_mutant_failed {
        return Err(Error::Unverifiable("no mutants".into()));
    }

    let score = if passed_tests { 1.0 } else { 0.0 };
    let mut resp = base_response(req, scored_at, score, passed_tests);
    resp.evidence
        .push(evidence(EvidenceKind::HiddenTests, "hidden_tests"));
    resp.evidence
        .push(evidence(EvidenceKind::Mutation, "mutation"));
    Ok(resp)
}
