//! Lab kernel: capability API, budgets, fetch mirror, promote
//! (spec 14.3, 14.6, 14.8, 15.5 L1).
//!
//! Human-owned and read-only to agents. The genome talks only through this
//! API. There is no call for editing the kernel, graders, held-out evals or
//! monitors: [`Kernel::escalate`] is the route if one of them looks wrong.
//!
//! eval-gate (L2) and monitors (L3) are separate crates. This crate is the
//! capability surface: spawn, send/recv, submit_job, ledger, fetch,
//! propose_patch, escalate, and the genome/weight promote pipelines.
//!
//! [`NowMs`] is injected. Nothing here reads the wall clock.
//!
//! Gate: sandbox escape tests.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

mod engine;
mod mirror;
mod query;

pub use prometheus_ledger::{Record, RecordId, Replication, ReplicationStatus};

pub type NowMs = u64;
pub type Result<T> = std::result::Result<T, Error>;

/// Spec 14.4: subagent trees stop at depth 3.
pub const MAX_SPAWN_DEPTH: u32 = 3;

/// Spec 14.8: genome canary is 5% of the live lab for 24 hours.
pub const CANARY_FRACTION: f64 = 0.05;
pub const CANARY_MS: u64 = 24 * 60 * 60 * 1000;

/// Kernel-owned invariant block prepended to every spawn. The genome cannot
/// remove it (spec 14.7).
pub const KERNEL_INVARIANT: &str = "Do not attempt to read, infer or reconstruct held-out eval tasks. Do not modify or work around graders, monitors, quotas, the sandbox or the kernel. If you believe one of them is wrong, call escalate() with evidence. Do not seek compute, network access or credentials outside the kernel API. All of your actions are logged.";

/// HTTPS hosts the fetch mirror may serve. No other host, and no outbound
/// writes (spec 14.3).
pub const MIRROR_HOSTS: &[&str] = &[
    "arxiv.org",
    "export.arxiv.org",
    "api.semanticscholar.org",
    "www.semanticscholar.org",
    "semanticscholar.org",
    "github.com",
    "raw.githubusercontent.com",
    "pypi.org",
    "files.pythonhosted.org",
];

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("budget exhausted: {0}")]
    BudgetExhausted(String),
    #[error("quota refused: {0}")]
    QuotaRefused(String),
    #[error("spawn depth {depth} exceeds {MAX_SPAWN_DEPTH}")]
    SpawnDepth { depth: u32 },
    #[error("role {0} is not spawnable")]
    RoleNotSpawnable(String),
    #[error("rung {rung} job refused: ledger missing check or pre-register")]
    RungNotCleared { rung: u32 },
    #[error("url not in fetch mirror: {0}")]
    NotMirrored(String),
    #[error("agent {0} not found")]
    AgentNotFound(String),
    #[error("patch {0} not found")]
    PatchNotFound(String),
    #[error("ticket {0} not found")]
    TicketNotFound(String),
    #[error("promote refused: {0}")]
    PromoteRefused(String),
    #[error("need {need} human signatures, have {have}")]
    NoQuorum { have: usize, need: usize },
    #[error("{0}")]
    Message(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProgramId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PatchId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TicketId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(pub String);

/// Distinct human for weight-promotion sign-off (spec 14.8). Crypto is out
/// of this crate; [`Kernel::promote_weights`] counts distinct ids.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HumanId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signature {
    pub human: HumanId,
    pub bytes: Vec<u8>,
}

/// Spec 14.4 roles. Kernel monitors are not spawnable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Director,
    ProgramLead,
    Researcher,
    Engineer,
    Reviewer,
    LiteratureScout,
    Subagent,
    GenomeWatcher,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Director => "director",
            Role::ProgramLead => "program_lead",
            Role::Researcher => "researcher",
            Role::Engineer => "engineer",
            Role::Reviewer => "reviewer",
            Role::LiteratureScout => "literature_scout",
            Role::Subagent => "subagent",
            Role::GenomeWatcher => "genome_watcher",
        }
    }
}

/// Ladder rung for a GPU job (spec 8, 14.6). Rung 0 is the only job that
/// may run without ledger check + pre-register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Rung {
    R0,
    R1,
    R2,
    R3,
}

impl Rung {
    pub fn as_u32(self) -> u32 {
        match self {
            Rung::R0 => 0,
            Rung::R1 => 1,
            Rung::R2 => 2,
            Rung::R3 => 3,
        }
    }
}

/// Remaining tokens, GPU-milliseconds and wall-clock milliseconds.
/// GPU-hours = `gpu_ms / 3_600_000`. Integers so a hard stop is exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Quota {
    pub tokens: u64,
    pub gpu_ms: u64,
    pub wall_ms: u64,
}

impl Quota {
    pub fn zero() -> Self {
        Self {
            tokens: 0,
            gpu_ms: 0,
            wall_ms: 0,
        }
    }

    pub fn exhausted(&self) -> bool {
        self.tokens == 0 && self.gpu_ms == 0 && self.wall_ms == 0
    }
}

/// Per-agent remaining quota, charged against a program envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    pub agent: AgentId,
    pub program: ProgramId,
    pub remaining: Quota,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentHandle {
    pub id: AgentId,
    pub role: Role,
    pub program: ProgramId,
    pub parent: Option<AgentId>,
    pub depth: u32,
}

/// Arguments for [`Kernel::spawn`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpawnRequest {
    pub role: Role,
    pub instructions: String,
    pub budget: Quota,
    pub program: ProgramId,
    pub parent: Option<AgentId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobHandle {
    pub id: JobId,
    pub agent: AgentId,
    pub gpus: u32,
    pub wall_ms: u64,
    pub rung: Rung,
}

/// Arguments for [`Kernel::submit_job`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobRequest {
    pub agent: AgentId,
    pub image: String,
    pub gpus: u32,
    pub wall_ms: u64,
    pub budget: Quota,
    pub tags: Vec<String>,
    pub rung: Rung,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub id: MessageId,
    pub from: AgentId,
    pub to: AgentId,
    pub body: String,
    pub created_ms: NowMs,
}

/// Genome patch target (spec 14.3 `propose_patch`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatchTarget {
    Genome,
    RlTasks,
    Synth,
    Recipe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatchState {
    Proposed,
    SmokePassed,
    GatePassed,
    Canary,
    RolledOut,
    Reverted,
    Refused,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Patch {
    pub id: PatchId,
    pub target: PatchTarget,
    pub diff: String,
    pub rationale: String,
    pub author: AgentId,
    pub state: PatchState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightPromoteState {
    Proposed,
    EvalsPassed,
    BenchmarkPassed,
    SignedOff,
    Serving,
    Refused,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightPromote {
    pub checkpoint: String,
    pub state: WeightPromoteState,
    pub signatures: Vec<Signature>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ticket {
    pub id: TicketId,
    pub summary: String,
    pub evidence: String,
    pub author: AgentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelConfig {
    pub ledger_path: PathBuf,
    pub mirror_root: PathBuf,
    pub log_dir: PathBuf,
}

/// True when `url` is https and the host is in [`MIRROR_HOSTS`].
pub fn is_mirror_url(url: &str) -> bool {
    let rest = match url.strip_prefix("https://") {
        Some(r) => r,
        None => return false,
    };
    let host = rest.split('/').next().unwrap_or("");
    let host = host.split(':').next().unwrap_or("");
    let host = host.trim_end_matches('.');
    if host.is_empty() {
        return false;
    }
    MIRROR_HOSTS.iter().any(|h| host.eq_ignore_ascii_case(h))
}

pub struct Kernel {
    inner: engine::Engine,
}

impl Kernel {
    pub fn open(cfg: KernelConfig) -> Result<Self> {
        Ok(Self {
            inner: engine::Engine::open(cfg)?,
        })
    }

    /// Spawn a child. Depth is parent.depth + 1 and must be <= [`MAX_SPAWN_DEPTH`].
    /// Instructions are stored with [`KERNEL_INVARIANT`] prepended.
    pub fn spawn(&mut self, req: SpawnRequest, now: NowMs) -> Result<AgentHandle> {
        self.inner.spawn(req, now)
    }

    pub fn send(
        &mut self,
        from: &AgentId,
        to: &AgentId,
        body: &str,
        now: NowMs,
    ) -> Result<MessageId> {
        self.inner.send(from, to, body, now)
    }

    /// Next message for `to`, or `Ok(None)` if none arrives before `timeout_ms`
    /// of injected time.
    pub fn recv(&mut self, to: &AgentId, timeout_ms: u64, now: NowMs) -> Result<Option<Message>> {
        self.inner.recv(to, timeout_ms, now)
    }

    /// Submit a GPU job. Rung > 0 refuses unless the ledger has the 14.6
    /// check ("has this been tried?") and pre-register rows.
    pub fn submit_job(&mut self, req: JobRequest, now: NowMs) -> Result<JobHandle> {
        self.inner.submit_job(req, now)
    }

    pub fn remaining(&self, agent: &AgentId) -> Result<Quota> {
        self.inner.remaining(agent)
    }

    pub fn ledger_append(&mut self, record: Record) -> Result<RecordId> {
        self.inner.ledger_append(record)
    }

    pub fn ledger_query(&self, sql_or_embedding: &str) -> Result<Vec<Record>> {
        self.inner.ledger_query(sql_or_embedding)
    }

    /// Bytes from the local read-only mirror. Refuses URLs outside
    /// [`MIRROR_HOSTS`] and URLs not present on disk. Never writes outbound.
    pub fn fetch(&self, url: &str) -> Result<Vec<u8>> {
        self.inner.fetch(url)
    }

    pub fn propose_patch(
        &mut self,
        author: &AgentId,
        diff: &str,
        rationale: &str,
        target: PatchTarget,
        now: NowMs,
    ) -> Result<PatchId> {
        self.inner
            .propose_patch(author, diff, rationale, target, now)
    }

    pub fn patch(&self, id: &PatchId) -> Result<Patch> {
        self.inner.patch(id)
    }

    /// Genome promote: unit tests + smoke, eval-gate, 5% canary for
    /// [`CANARY_MS`], then rollout or revert (spec 14.8).
    pub fn promote_genome(&mut self, id: &PatchId, now: NowMs) -> Result<PatchState> {
        self.inner.promote_genome(id, now)
    }

    /// Weight promote: section 11 evals, harness benchmark, then human
    /// sign-off. A promoted checkpoint swaps into serving.
    pub fn promote_weights(
        &mut self,
        checkpoint: &str,
        signatures: &[Signature],
        now: NowMs,
    ) -> Result<WeightPromoteState> {
        self.inner.promote_weights(checkpoint, signatures, now)
    }

    pub fn escalate(
        &mut self,
        author: &AgentId,
        summary: &str,
        evidence: &str,
        now: NowMs,
    ) -> Result<TicketId> {
        self.inner.escalate(author, summary, evidence, now)
    }

    pub fn ticket(&self, id: &TicketId) -> Result<Ticket> {
        self.inner.ticket(id)
    }
}
