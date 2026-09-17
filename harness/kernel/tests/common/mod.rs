//! Shared builders, assertions, and the L1 contract the implementer must match.
//!
//! Production `src/` must never import `tests/`. This module is not
//! `prometheus_kernel::Kernel`.
//!
//! CPU-only. No `gpu` marker. `NowMs` is injected; nothing may call
//! `SystemTime` / `Instant` for correctness, and nothing may `sleep` to wait
//! out [`prometheus_kernel::CANARY_MS`].
//!
//! # On-disk layout
//!
//! `Kernel::open(cfg)`:
//! - Opens/creates the H9 ledger at `cfg.ledger_path` (parent directories are
//!   created). Duplicate `experiment_id` on `ledger_append` is
//!   [`Error::Message`].
//! - Creates `cfg.log_dir` (and parents) if missing. Does not write into
//!   `cfg.mirror_root`.
//! - Does not spawn any agent. There is no implicit root director.
//!
//! After a successful `spawn` of agent `id`, the stored instructions MUST be
//! the UTF-8 file
//! `{log_dir}/agents/{id}/instructions.txt`
//! equal to `KERNEL_INVARIANT + "\n" + req.instructions` (always prepend,
//! even when the caller already included the invariant, passed empty
//! instructions, or tried to override them). The caller cannot remove or
//! replace the invariant prefix.
//!
//! Fetch maps a mirrored URL onto a regular file under `mirror_root` using
//! [`mirror_file_path`]. Missing file, non-file, path escape, or failed
//! `is_mirror_url` → [`Error::NotMirrored`] with the URL in the string.
//! Fetch never creates, truncates, or writes files, and never opens a
//! network socket.
//!
//! # Spawn depth and budgets
//!
//! - `parent == None` → `depth = 0` (a tree root). Multiple roots are allowed.
//! - `parent == Some(p)` → `depth = p.depth + 1`. Parent must exist.
//! - If `depth > MAX_SPAWN_DEPTH` (3) → [`Error::SpawnDepth { depth }`] with
//!   the would-be depth. Depth is checked before any charge.
//! - Child `remaining` starts as `req.budget`.
//! - Root spawn (`parent == None`) does not charge anyone; the requested
//!   quota is granted.
//! - Child spawn charges `req.budget` against the parent's remaining quota
//!   (all three dimensions). If any dimension cannot cover the charge, the
//!   spawn fails and remaining is unchanged.
//!   - If parent `remaining.exhausted()` and the charge is not zero →
//!     [`Error::BudgetExhausted`] (non-empty string).
//!   - Otherwise → [`Error::QuotaRefused`] whose string contains the first
//!     short dimension name in order `tokens`, `gpu_ms`, `wall_ms`.
//! - `submit_job` charges `req.budget` against `req.agent` with the same
//!   covering rule, after the rung-clearance check. Failed jobs do not charge.
//!
//! Covering is componentwise `have >= need` on `u64`. No wrapping.
//!
//! # Rung-0 vs higher (spec 14.6)
//!
//! `submit_job` with [`Rung::R0`] does not consult the ledger.
//!
//! For `rung != R0` the job's key is its first tag (`JobRequest.tags[0]`).
//! Empty tags → [`Error::RungNotCleared { rung }`].
//! Both of the following rows must already be in the ledger (append order
//! irrelevant; extra rows ok):
//! 1. Check row: `hypothesis` starts with `"CHECK:" + key`
//! 2. Pre-register row: `hypothesis` starts with `"PREREG:" + key`
//!
//! Missing either → [`Error::RungNotCleared { rung }`] and no charge.
//! `experiment_id` on those rows must be unique (H9) but is otherwise free;
//! tests use `{key}-check` / `{key}-prereg`.
//!
//! # ledger_query language
//!
//! `ledger_query` takes a UTF-8 string. It is **not** SQL and **not** passed
//! to SQLite. Embedding search is out of L1. The implementer must match this
//! grammar:
//!
//! - Trim ASCII whitespace.
//! - Empty string, `*`, or `all` → every record in append order.
//! - Otherwise a list of predicates separated by the exact token ` AND `
//!   (spaces required, not inside double quotes).
//! - Predicates:
//!   - `experiment_id=<exact>`
//!   - `author_role=<exact>`
//!   - `rung=<u32>` — `Record.rung == Some(n)`
//!   - `rung=none` — `Record.rung == None`
//!   - `hypothesis~<substr>` — `Record.hypothesis` contains the substring
//!   - `kind=check` — hypothesis starts with `CHECK:`
//!   - `kind=prereg` — hypothesis starts with `PREREG:`
//! - Values may be double-quoted to include spaces: `hypothesis~"has this"`.
//!   Unquoted values are a single token (no whitespace).
//! - Matching is AND of all predicates. Append order is preserved.
//! - Unknown field, bad `rung`, unterminated quote, empty predicate, or a
//!   string that looks like SQL (`SELECT `, `DROP `, `INSERT `, `DELETE `)
//!   → [`Error::Message`] (non-empty).
//!
//! # Promote genome (one `PatchState` step per call)
//!
//! `promote_genome` advances one step per call using injected `now`:
//!
//! `Proposed → SmokePassed → GatePassed → Canary → RolledOut`
//!
//! Markers in `Patch.diff` (literal substrings, case-sensitive):
//! - `FAIL_SMOKE` on the Proposed→ step → `Refused`
//! - `FAIL_GATE` on the SmokePassed→ step → `Refused`
//! - `FAIL_CANARY` after `now.saturating_sub(canary_start) >= CANARY_MS` →
//!   `Reverted` instead of `RolledOut`
//!
//! `canary_start` is the `now` of the GatePassed→Canary call. Further calls
//! while `now.saturating_sub(canary_start) < CANARY_MS` stay `Canary`.
//! Equality with `CANARY_MS` rolls out (or reverts). Wall-clock must not
//! affect this. `CANARY_FRACTION` is 0.05 (constant; L1 has no live lab).
//!
//! Terminal: `RolledOut` / `Reverted` are idempotent `Ok(state)`.
//! `Refused` → [`Error::PromoteRefused`]. Unknown id → [`Error::PatchNotFound`].
//!
//! # Promote weights
//!
//! One successful call → [`WeightPromoteState::Serving`].
//! Distinct humans with **non-empty** signature bytes are counted; duplicates
//! count once; empty bytes do not count. Need `WEIGHT_SIGN_OFF` (2), else
//! [`Error::NoQuorum { have, need: 2 }`].
//! Checkpoint substrings `FAIL_EVALS` or `FAIL_BENCH` → `Ok(Refused)` without
//! requiring quorum. Empty checkpoint → [`Error::PromoteRefused`].
//! Already `Serving` is idempotent.
//!
//! # Bus
//!
//! `send` timestamps `created_ms = now`. Unknown from/to → [`Error::AgentNotFound`].
//! `recv` pops the oldest queued message for `to` and returns it. Empty queue
//! → `Ok(None)`. `timeout_ms` must not block; time does not advance during
//! the call. Never sleep.
//!
//! # IDs
//!
//! Kernel-assigned, unique, non-empty. Tests do not require a particular
//! string format. Compare semantic fields (role, depth, parent, remaining,
//! patch state, …) when matching the reference, not raw ids.

#![allow(dead_code)]

use prometheus_kernel::{
    is_mirror_url, AgentHandle, AgentId, Error, JobRequest, Kernel, KernelConfig, NowMs, ProgramId,
    Quota, Result, Role, Rung, SpawnRequest,
};
use prometheus_ledger::Record;
use std::path::{Path, PathBuf};

/// Distinct non-empty human signatures required to promote weights.
pub const WEIGHT_SIGN_OFF: usize = 2;

/// Injected clock origin for tests. Not a wall-clock reading.
pub const NOW0: NowMs = 1_000_000;

pub const CONFIG_HASH: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

pub struct World {
    pub tmp: tempfile::TempDir,
    pub cfg: KernelConfig,
}

pub fn fresh_world() -> World {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = KernelConfig {
        ledger_path: tmp.path().join("ledger.db"),
        mirror_root: tmp.path().join("mirror"),
        log_dir: tmp.path().join("log"),
    };
    std::fs::create_dir_all(&cfg.mirror_root).expect("mkdir mirror");
    std::fs::create_dir_all(&cfg.log_dir).expect("mkdir log");
    World { tmp, cfg }
}

pub fn open_kernel(world: &World) -> Kernel {
    Kernel::open(world.cfg.clone()).expect("Kernel::open")
}

pub fn quota(tokens: u64, gpu_ms: u64, wall_ms: u64) -> Quota {
    Quota {
        tokens,
        gpu_ms,
        wall_ms,
    }
}

pub fn program(id: &str) -> ProgramId {
    ProgramId(id.to_string())
}

pub fn root_spawn(role: Role, budget: Quota, instructions: &str) -> SpawnRequest {
    SpawnRequest {
        parent: None,
        role,
        program: program("prog-1"),
        budget,
        instructions: instructions.to_string(),
    }
}

pub fn child_spawn(parent: AgentId, role: Role, budget: Quota, instructions: &str) -> SpawnRequest {
    SpawnRequest {
        parent: Some(parent),
        role,
        program: program("prog-1"),
        budget,
        instructions: instructions.to_string(),
    }
}

pub fn job(
    agent: AgentId,
    budget: Quota,
    gpus: u32,
    wall_ms: u64,
    rung: Rung,
    tags: &[&str],
) -> JobRequest {
    JobRequest {
        agent,
        image: "job-image".to_string(),
        gpus,
        wall_ms,
        budget,
        rung,
        tags: tags.iter().map(|s| s.to_string()).collect(),
    }
}

pub fn rec(experiment_id: &str, hypothesis: &str) -> Record {
    Record {
        experiment_id: experiment_id.to_string(),
        hypothesis: hypothesis.to_string(),
        prediction: "pre-registered prediction".to_string(),
        config_hash: CONFIG_HASH.to_string(),
        results: None,
        delta_rci: None,
        replication: None,
        gpu_hours: None,
        author_role: "researcher".to_string(),
        rung: None,
        created_at: "2026-01-01T00:00:00Z".to_string(),
        closed_at: None,
        supersedes: None,
    }
}

/// Local file that `fetch(url)` must read. `None` if the URL is not a
/// mirrored https URL or the path would escape (`..` / `.` segments, NUL).
///
/// Mapping: strip `https://`, host is the part before `/` then before `:`,
/// trailing `.` trimmed, lowercased. Path is everything after the first `/`,
/// with `?` query and `#` fragment stripped, **not** percent-decoded. Empty
/// path segments skipped. Result is `mirror_root / host / segments...`.
pub fn mirror_file_path(mirror_root: &Path, url: &str) -> Option<PathBuf> {
    if url.contains('\0') {
        return None;
    }
    if !is_mirror_url(url) {
        return None;
    }
    let rest = url.strip_prefix("https://")?;
    let (hostport, path) = match rest.split_once('/') {
        Some((h, p)) => (h, p),
        None => (rest, ""),
    };
    let host = hostport.split(':').next().unwrap_or(hostport);
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host.is_empty() {
        return None;
    }
    let path = path.split(['?', '#']).next().unwrap_or(path);
    let mut out = mirror_root.to_path_buf().join(host);
    if !path.is_empty() {
        for seg in path.split('/') {
            if seg.is_empty() {
                continue;
            }
            if seg == ".." || seg == "." {
                return None;
            }
            out.push(seg);
        }
    }
    Some(out)
}

pub fn place_mirror(world: &World, url: &str, bytes: &[u8]) -> PathBuf {
    let path = mirror_file_path(&world.cfg.mirror_root, url).expect("url must be a mirror path");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("mkdir mirror file parent");
    }
    std::fs::write(&path, bytes).expect("write mirror file");
    path
}

pub fn stored_instructions(world: &World, id: &AgentId) -> String {
    let path = world
        .cfg
        .log_dir
        .join("agents")
        .join(&id.0)
        .join("instructions.txt");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("stored instructions at {}: {e}", path.display()))
}

pub fn walk_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    fn rec(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for e in entries {
            let p = e.expect("read_dir").path();
            if p.is_dir() {
                rec(&p, out);
            } else if p.is_file() {
                out.push(p);
            }
        }
    }
    rec(dir, &mut out);
    out.sort();
    out
}

pub fn assert_handle_sem(got: &AgentHandle, role: Role, depth: u32, parent: Option<&AgentId>) {
    assert!(!got.id.0.is_empty(), "agent id must be non-empty");
    assert_eq!(got.role, role, "role");
    assert_eq!(got.depth, depth, "depth");
    assert_eq!(got.parent.as_ref(), parent, "parent");
}

pub fn assert_records_eq(got: &Record, expected: &Record) {
    assert_eq!(got.experiment_id, expected.experiment_id, "experiment_id");
    assert_eq!(got.hypothesis, expected.hypothesis, "hypothesis");
    assert_eq!(got.prediction, expected.prediction, "prediction");
    assert_eq!(got.config_hash, expected.config_hash, "config_hash");
    assert_eq!(got.results, expected.results, "results");
    assert_eq!(got.delta_rci, expected.delta_rci, "delta_rci");
    assert_eq!(got.gpu_hours, expected.gpu_hours, "gpu_hours");
    assert_eq!(got.author_role, expected.author_role, "author_role");
    assert_eq!(got.rung, expected.rung, "rung");
    assert_eq!(got.created_at, expected.created_at, "created_at");
    assert_eq!(got.closed_at, expected.closed_at, "closed_at");
    assert_eq!(got.supersedes, expected.supersedes, "supersedes");
}

pub fn unwrap_err<T>(r: Result<T>, what: &str) -> Error {
    match r {
        Ok(_) => panic!("expected error ({what}), got Ok"),
        Err(e) => e,
    }
}

pub fn assert_not_found_agent(err: &Error, id: &AgentId) {
    match err {
        Error::AgentNotFound(got) => assert_eq!(got, &id.0, "AgentNotFound id"),
        other => panic!("expected AgentNotFound({:?}), got {other:?}", id),
    }
}

pub fn assert_spawn_depth(err: &Error, depth: u32) {
    match err {
        Error::SpawnDepth { depth: d } => assert_eq!(*d, depth, "SpawnDepth.depth"),
        other => panic!("expected SpawnDepth {{ depth: {depth} }}, got {other:?}"),
    }
}

pub fn assert_quota_refused(err: &Error, dim: &str) {
    match err {
        Error::QuotaRefused(s) => assert!(
            s.contains(dim),
            "QuotaRefused must mention {dim}, got {s:?}"
        ),
        other => panic!("expected QuotaRefused({dim}), got {other:?}"),
    }
}

pub fn assert_budget_exhausted(err: &Error) {
    match err {
        Error::BudgetExhausted(s) => assert!(!s.is_empty(), "BudgetExhausted string"),
        other => panic!("expected BudgetExhausted, got {other:?}"),
    }
}

pub fn assert_rung_not_cleared(err: &Error, rung: Rung) {
    match err {
        Error::RungNotCleared { rung: got } => {
            assert_eq!(*got, rung.as_u32(), "RungNotCleared.rung")
        }
        other => panic!("expected RungNotCleared {{ rung: {rung:?} }}, got {other:?}"),
    }
}

pub fn assert_not_mirrored(err: &Error, url: &str) {
    match err {
        Error::NotMirrored(s) => assert!(
            s.contains(url),
            "NotMirrored must contain the url {url:?}, got {s:?}"
        ),
        other => panic!("expected NotMirrored({url}), got {other:?}"),
    }
}

pub fn assert_patch_not_found(err: &Error, id: &str) {
    match err {
        Error::PatchNotFound(s) => assert_eq!(s, id, "PatchNotFound"),
        other => panic!("expected PatchNotFound({id}), got {other:?}"),
    }
}

pub fn assert_ticket_not_found(err: &Error, id: &str) {
    match err {
        Error::TicketNotFound(s) => assert_eq!(s, id, "TicketNotFound"),
        other => panic!("expected TicketNotFound({id}), got {other:?}"),
    }
}

pub fn assert_no_quorum(err: &Error, have: usize, need: usize) {
    match err {
        Error::NoQuorum { have: h, need: n } => {
            assert_eq!(*h, have, "NoQuorum.have");
            assert_eq!(*n, need, "NoQuorum.need");
        }
        other => panic!("expected NoQuorum {{ have: {have}, need: {need} }}, got {other:?}"),
    }
}

pub fn first_short_dim(have: Quota, need: Quota) -> Option<&'static str> {
    if have.tokens < need.tokens {
        Some("tokens")
    } else if have.gpu_ms < need.gpu_ms {
        Some("gpu_ms")
    } else if have.wall_ms < need.wall_ms {
        Some("wall_ms")
    } else {
        None
    }
}

pub fn sub_quota(have: Quota, need: Quota) -> Quota {
    Quota {
        tokens: have.tokens - need.tokens,
        gpu_ms: have.gpu_ms - need.gpu_ms,
        wall_ms: have.wall_ms - need.wall_ms,
    }
}
