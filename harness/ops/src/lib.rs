//! Quorum-signed self-update, shadow node, harness CLI (spec 15.2, 15.5 H12).
//!
//! The harness builds new versions of itself like any other module, through
//! gates and the chaos suite on a shadow node, then rolls out one node at a
//! time with automatic rollback. Nodes refuse kernel updates that do not
//! carry a quorum of human signatures: the swarm is autonomous about the
//! work, not about its own guard rails.
//!
//! Artifact bytes live in the H8 CAS. Target kernel version lives in H4 Raft.
//! Rollout state is an H2 event log at `<dir>/log/`. `NowMs` is injected.

use prometheus_cas::{Digest, Store};
use prometheus_log::EventLog;
use prometheus_raft::{Cluster, NodeId};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub type NowMs = u64;

/// Distinct humans required on a kernel update. Spec 15.2: a quorum of
/// human signatures; two is the smallest quorum of three operators.
pub const DEFAULT_QUORUM: usize = 2;

pub const PIN_CURRENT: &str = "kernel.current";
pub const PIN_PREVIOUS: &str = "kernel.previous";
pub const PIN_SHADOW: &str = "kernel.shadow";

pub const EVENT_PROPOSE: &str = "ops.propose";
pub const EVENT_REFUSE: &str = "ops.refuse";
pub const EVENT_SHADOW: &str = "ops.shadow";
pub const EVENT_SHADOW_RESULT: &str = "ops.shadow_result";
pub const EVENT_APPLY: &str = "ops.apply";
pub const EVENT_PROMOTE: &str = "ops.promote";
pub const EVENT_ROLLBACK: &str = "ops.rollback";
pub const EVENT_UNHEALTHY: &str = "ops.unhealthy";

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HumanId(pub String);

/// Opaque signature bytes from one human. Crypto is out of this crate;
/// [`has_quorum`] counts distinct [`HumanId`]s with non-empty bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signature {
    pub human: HumanId,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Release {
    pub version: String,
    pub digest: Digest,
    pub commit: String,
    pub signatures: Vec<Signature>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpsConfig {
    pub quorum: usize,
}

impl Default for OpsConfig {
    fn default() -> Self {
        Self {
            quorum: DEFAULT_QUORUM,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeRole {
    Live,
    Shadow,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReleaseState {
    Idle,
    Proposed,
    Shadowing,
    Rolling { node: NodeId },
    Current,
    RolledBack { reason: String },
    Refused { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeStatus {
    pub id: NodeId,
    pub applied: String,
    pub healthy: bool,
    pub role: NodeRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Status {
    pub kernel_version: String,
    pub release: Option<Release>,
    pub state: ReleaseState,
    pub nodes: Vec<NodeStatus>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("need {need} human signatures, have {have}")]
    NoQuorum { have: usize, need: usize },
    #[error("refused kernel update: {0}")]
    Refused(String),
    #[error("bad release: {0}")]
    BadRelease(String),
    #[error("wrong state: {0}")]
    WrongState(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("digest mismatch: expected {expected} got {got}")]
    DigestMismatch { expected: String, got: String },
    #[error("{0}")]
    Cas(String),
    #[error("{0}")]
    Raft(String),
    #[error("{0}")]
    Log(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<prometheus_cas::Error> for Error {
    fn from(err: prometheus_cas::Error) -> Self {
        Error::Cas(err.to_string())
    }
}

impl From<prometheus_raft::Error> for Error {
    fn from(err: prometheus_raft::Error) -> Self {
        Error::Raft(err.to_string())
    }
}

impl From<prometheus_log::Error> for Error {
    fn from(err: prometheus_log::Error) -> Self {
        Error::Log(err.to_string())
    }
}

/// True iff at least `quorum` distinct humans signed with non-empty bytes.
pub fn has_quorum(_release: &Release, _quorum: usize) -> bool {
    unimplemented!("H12: has_quorum")
}

/// Rollout state machine. Does not own the CAS or Raft group; callers pass
/// them in. Create fails if `dir` exists.
pub struct Ops {
    dir: PathBuf,
    config: OpsConfig,
}

impl Ops {
    pub fn create(_dir: impl AsRef<Path>, _config: OpsConfig) -> Result<Self> {
        unimplemented!("H12: Ops::create")
    }

    pub fn open(_dir: impl AsRef<Path>, _config: OpsConfig) -> Result<Self> {
        unimplemented!("H12: Ops::open")
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config(&self) -> &OpsConfig {
        &self.config
    }

    pub fn log(&self) -> &EventLog {
        unimplemented!("H12: Ops::log")
    }

    /// Put `artifact` in `store`, pin as shadow, record Proposed.
    /// Refuses (and does not pin) if [`has_quorum`] is false.
    pub fn propose(
        &mut self,
        _release: Release,
        _artifact: &[u8],
        _store: &mut Store,
        _now: NowMs,
    ) -> Result<()> {
        unimplemented!("H12: Ops::propose")
    }

    /// Run the proposed release on the shadow node. Does not change live
    /// kernel_version. WrongState if not Proposed.
    pub fn start_shadow(&mut self, _store: &mut Store, _now: NowMs) -> Result<()> {
        unimplemented!("H12: Ops::start_shadow")
    }

    /// Shadow gates passed. Ready to roll out one node at a time.
    pub fn shadow_pass(&mut self, _now: NowMs) -> Result<()> {
        unimplemented!("H12: Ops::shadow_pass")
    }

    /// Shadow gates failed. Automatic rollback: unpin shadow, state RolledBack.
    /// Live kernel_version is unchanged.
    pub fn shadow_fail(&mut self, _reason: &str, _store: &mut Store, _now: NowMs) -> Result<()> {
        unimplemented!("H12: Ops::shadow_fail")
    }

    /// Apply the proposed version on one live node. Rolls out one node at a
    /// time. WrongState if shadow has not passed.
    pub fn apply_one(&mut self, _node: &NodeId, _now: NowMs) -> Result<()> {
        unimplemented!("H12: Ops::apply_one")
    }

    /// Mark a node unhealthy after apply. Triggers [`Ops::rollback`].
    pub fn mark_unhealthy(
        &mut self,
        _node: &NodeId,
        _reason: &str,
        _cluster: &mut Cluster,
        _store: &mut Store,
        _now: NowMs,
    ) -> Result<()> {
        unimplemented!("H12: Ops::mark_unhealthy")
    }

    /// Promote to current: pin previous=current, current=new, set Raft
    /// kernel_version. All live nodes must have applied and be healthy.
    pub fn promote(&mut self, _cluster: &mut Cluster, _store: &mut Store, _now: NowMs) -> Result<()> {
        unimplemented!("H12: Ops::promote")
    }

    /// Restore the previous pin and previous kernel_version. Gate: rollback
    /// on a bad release.
    pub fn rollback(
        &mut self,
        _reason: &str,
        _cluster: &mut Cluster,
        _store: &mut Store,
        _now: NowMs,
    ) -> Result<()> {
        unimplemented!("H12: Ops::rollback")
    }

    pub fn status(&self, _cluster: &Cluster) -> Status {
        unimplemented!("H12: Ops::status")
    }

    /// CLI `--json` payload. Same [`Status`], serialized.
    pub fn status_json(&self, _cluster: &Cluster) -> Result<String> {
        unimplemented!("H12: Ops::status_json")
    }

    /// Human-readable status for the CLI without `--json`.
    pub fn status_text(&self, _cluster: &Cluster) -> String {
        unimplemented!("H12: Ops::status_text")
    }
}

/// Parse `harness [--json] status`. Used by the `harness` binary.
pub fn cli(_args: &[String], _ops: &Ops, _cluster: &Cluster) -> Result<String> {
    unimplemented!("H12: cli")
}
