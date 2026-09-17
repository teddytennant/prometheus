//! Raft-replicated control state (spec 15.2, 15.5 H4, gate D2).
//!
//! Small control state lives in a Raft group (openraft) across at least three
//! always-on nodes in different places. NCShare nodes join only as ephemeral
//! workers; they are never voters.
//!
//! Replicated here, not in the H2 event log:
//! membership, coordinator role, token-broker role, kernel version, and the
//! refresh token (committed before it is used).
//!
//! CPU tests drive three in-process nodes through [`LocalGroup`]. Production
//! wires the same [`Cluster`] API to real RPC. `NowMs` is injected; nothing
//! in this crate reads the wall clock.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const DEFAULT_HEARTBEAT_PERIOD_MS: u64 = 15_000;
pub const MISSED_HEARTBEATS: u32 = 2;
pub const MIN_VOTERS: usize = 3;

pub type NowMs = u64;
pub type Attempt = u64;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Addr(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    AlwaysOn,
    Ephemeral,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: NodeId,
    pub addr: Addr,
    pub kind: NodeKind,
    pub trusted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Coordinator,
    TokenBroker,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleLease {
    pub role: Role,
    pub holder: NodeId,
    pub attempt: Attempt,
    pub expires_at: NowMs,
}

/// Opaque refresh-token blob. Tests use fake bytes. Generation 1 is the first commit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenRecord {
    pub generation: u64,
    pub blob: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlState {
    pub membership: Vec<NodeInfo>,
    pub coordinator: Option<RoleLease>,
    pub token_broker: Option<RoleLease>,
    pub kernel_version: String,
    pub token: Option<TokenRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub heartbeat_period_ms: u64,
    pub missed_heartbeats: u32,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            heartbeat_period_ms: DEFAULT_HEARTBEAT_PERIOD_MS,
            missed_heartbeats: MISSED_HEARTBEATS,
        }
    }
}

impl ClusterConfig {
    pub fn lease_ttl_ms(&self) -> u64 {
        self.heartbeat_period_ms
            .saturating_mul(u64::from(self.missed_heartbeats.max(1)))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("not leader")]
    NotLeader,
    #[error("not found: {0}")]
    NotFound(String),
    #[error("not holder: role={0:?} node={1}")]
    NotHolder(Role, String),
    #[error("role not claimable: {0:?}")]
    NotClaimable(Role),
    #[error("untrusted node {0}")]
    Untrusted(String),
    #[error("ephemeral node {0} cannot be a voter")]
    EphemeralVoter(String),
    #[error("need at least {MIN_VOTERS} voters, have {0}")]
    TooFewVoters(usize),
    #[error("duplicate node {0}")]
    Duplicate(String),
    #[error("token generation {got} is not successor of {current}")]
    TokenGeneration { current: u64, got: u64 },
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// One Raft member. Durable state is under `dir`. Bodies are unimplemented.
pub struct Cluster {
    dir: PathBuf,
    config: ClusterConfig,
    this: NodeId,
    state: ControlState,
}

impl Cluster {
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config(&self) -> &ClusterConfig {
        &self.config
    }

    pub fn this_id(&self) -> &NodeId {
        &self.this
    }

    pub fn state(&self) -> &ControlState {
        &self.state
    }

    pub fn is_leader(&self) -> bool {
        unimplemented!("H4: Cluster::is_leader")
    }

    /// Single-node group. Fails if `dir` exists. The node must be trusted AlwaysOn.
    pub fn bootstrap(dir: impl AsRef<Path>, this: NodeInfo, config: ClusterConfig) -> Result<Self> {
        let _ = (dir, this, config);
        unimplemented!("H4: Cluster::bootstrap")
    }

    /// Reopen a node from disk. Not leader until an election.
    pub fn open(dir: impl AsRef<Path>, this: NodeId, config: ClusterConfig) -> Result<Self> {
        let _ = (dir, this, config);
        unimplemented!("H4: Cluster::open")
    }

    /// Propose a trusted AlwaysOn voter. Leader only. Rejects Ephemeral.
    pub fn add_voter(&mut self, node: NodeInfo) -> Result<()> {
        let _ = node;
        unimplemented!("H4: Cluster::add_voter")
    }

    /// Propose an ephemeral worker (non-voter). Leader only.
    pub fn add_worker(&mut self, node: NodeInfo) -> Result<()> {
        let _ = node;
        unimplemented!("H4: Cluster::add_worker")
    }

    /// Remove a member. Leader only. Refuses to drop below [`MIN_VOTERS`] voters
    /// once the group has reached that size.
    pub fn remove_node(&mut self, id: &NodeId) -> Result<()> {
        let _ = id;
        unimplemented!("H4: Cluster::remove_node")
    }

    /// Claim coordinator or token-broker if vacant or expired as of `now`.
    /// First claim is attempt 1. Leader only. Holder must be a trusted member.
    pub fn claim_role(&mut self, role: Role, node: &NodeId, now: NowMs) -> Result<RoleLease> {
        let _ = (role, node, now);
        unimplemented!("H4: Cluster::claim_role")
    }

    /// Extend expiry to `now + ttl` if `node` holds `role`. Leader only.
    pub fn heartbeat_role(&mut self, role: Role, node: &NodeId, now: NowMs) -> Result<RoleLease> {
        let _ = (role, node, now);
        unimplemented!("H4: Cluster::heartbeat_role")
    }

    /// Drop role leases with `expires_at <= now`. Leader only.
    pub fn expire_roles(&mut self, now: NowMs) -> Result<Vec<Role>> {
        let _ = now;
        unimplemented!("H4: Cluster::expire_roles")
    }

    /// Commit a rotated refresh token before it is used. Generation is current+1
    /// (0 means none committed yet). Leader only.
    pub fn commit_token(&mut self, blob: Vec<u8>) -> Result<TokenRecord> {
        let _ = blob;
        unimplemented!("H4: Cluster::commit_token")
    }

    pub fn set_kernel_version(&mut self, version: String) -> Result<()> {
        let _ = version;
        unimplemented!("H4: Cluster::set_kernel_version")
    }
}

/// In-process n-node group for CPU tests and D2 probes. Production uses RPC.
pub struct LocalGroup {
    nodes: Vec<Cluster>,
}

impl LocalGroup {
    /// Start `n` AlwaysOn voters in `base_dir/<i>/`. `n` must be >= [`MIN_VOTERS`].
    pub fn start(n: usize, base_dir: impl AsRef<Path>, config: ClusterConfig) -> Result<Self> {
        let _ = (n, base_dir, config);
        unimplemented!("H4: LocalGroup::start")
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn get(&mut self, i: usize) -> Result<&mut Cluster> {
        self.nodes
            .get_mut(i)
            .ok_or_else(|| Error::NotFound(i.to_string()))
    }

    /// Drop Raft traffic to `isolated` indices. Isolated nodes still have disk.
    pub fn partition(&mut self, isolated: &[usize]) -> Result<()> {
        let _ = isolated;
        unimplemented!("H4: LocalGroup::partition")
    }

    pub fn heal(&mut self) -> Result<()> {
        unimplemented!("H4: LocalGroup::heal")
    }

    /// Kill process i (drop in-memory node; disk stays).
    pub fn crash(&mut self, i: usize) -> Result<()> {
        let _ = i;
        unimplemented!("H4: LocalGroup::crash")
    }

    /// Reopen a crashed node from disk and rejoin.
    pub fn restart(&mut self, i: usize) -> Result<()> {
        let _ = i;
        unimplemented!("H4: LocalGroup::restart")
    }

    /// Drive elections and apply committed entries. `now` is for role expiry.
    pub fn tick(&mut self, now: NowMs) -> Result<()> {
        let _ = now;
        unimplemented!("H4: LocalGroup::tick")
    }
}
