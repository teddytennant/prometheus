//! Raft-replicated control state (spec 15.2, 15.5 H4, gate D2).
//!
//! Small control state lives in a Raft group across at least three always-on
//! nodes in different places. NCShare nodes join only as ephemeral workers;
//! they are never voters.
//!
//! Replicated here, not in the H2 event log:
//! membership, coordinator role, token-broker role, kernel version, and the
//! refresh token (committed before it is used).
//!
//! CPU tests drive three in-process nodes through [`LocalGroup`]. Production
//! wires the same [`Cluster`] API to real RPC. `NowMs` is injected; nothing
//! in this crate reads the wall clock.

mod inner;

use inner::{Bus, NodeInner};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

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

/// One Raft member. Durable state (log + applied [`ControlState`]) lives under `dir`.
pub struct Cluster {
    dir: PathBuf,
    config: ClusterConfig,
    this: NodeId,
    state: ControlState,
    is_leader: bool,
    bus: Rc<RefCell<Bus>>,
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
        self.is_leader
    }

    fn key(&self) -> &str {
        &self.this.0
    }

    fn sync_from_bus(&mut self) {
        let bus = self.bus.borrow();
        if let Some(st) = bus.state_of(self.key()) {
            self.state = st;
        }
        self.is_leader = bus.is_leader(self.key());
    }

    fn from_inner(node: NodeInner, bus: Rc<RefCell<Bus>>) -> Self {
        let mut c = Self {
            dir: node.dir.clone(),
            config: node.config.clone(),
            this: NodeId(node.id.clone()),
            state: node.machine.state(),
            is_leader: node.role == inner::RaftRole::Leader,
            bus,
        };
        c.bus.borrow_mut().insert(node);
        c.sync_from_bus();
        c
    }

    /// Single-node group. Fails if `dir` exists. The node must be trusted AlwaysOn.
    /// The node is leader immediately.
    pub fn bootstrap(dir: impl AsRef<Path>, this: NodeInfo, config: ClusterConfig) -> Result<Self> {
        let node = inner::bootstrap_node(dir.as_ref().to_path_buf(), this, config)?;
        Ok(Self::from_inner(node, Rc::new(RefCell::new(Bus::new()))))
    }

    /// Reopen a node from disk. Not leader until an election.
    pub fn open(dir: impl AsRef<Path>, this: NodeId, config: ClusterConfig) -> Result<Self> {
        Self::open_on_bus(
            dir.as_ref().to_path_buf(),
            this,
            config,
            Rc::new(RefCell::new(Bus::new())),
        )
    }

    fn open_on_bus(
        dir: PathBuf,
        this: NodeId,
        config: ClusterConfig,
        bus: Rc<RefCell<Bus>>,
    ) -> Result<Self> {
        let node = inner::open_node(dir, this, config)?;
        Ok(Self::from_inner(node, bus))
    }

    /// Propose a trusted AlwaysOn voter. Leader only. Rejects Ephemeral.
    pub fn add_voter(&mut self, node: NodeInfo) -> Result<()> {
        self.bus.borrow_mut().add_voter(self.key(), node)?;
        self.sync_from_bus();
        Ok(())
    }

    /// Propose an ephemeral worker (non-voter). Leader only.
    pub fn add_worker(&mut self, node: NodeInfo) -> Result<()> {
        self.bus.borrow_mut().add_worker(self.key(), node)?;
        self.sync_from_bus();
        Ok(())
    }

    /// Remove a member. Leader only. Refuses to drop below [`MIN_VOTERS`] voters
    /// once the group has reached that size.
    pub fn remove_node(&mut self, id: &NodeId) -> Result<()> {
        self.bus.borrow_mut().remove_node(self.key(), id)?;
        self.sync_from_bus();
        Ok(())
    }

    /// Claim coordinator or token-broker if vacant or expired as of `now`.
    /// First claim is attempt 1. Leader only. Holder must be a trusted member.
    pub fn claim_role(&mut self, role: Role, node: &NodeId, now: NowMs) -> Result<RoleLease> {
        let lease = self
            .bus
            .borrow_mut()
            .claim_role(self.key(), role, node, now)?;
        self.sync_from_bus();
        Ok(lease)
    }

    /// Extend expiry to `now + ttl` if `node` holds `role`. Leader only.
    /// Succeeds for a still-recorded holder even if the lease is already due.
    pub fn heartbeat_role(&mut self, role: Role, node: &NodeId, now: NowMs) -> Result<RoleLease> {
        let lease = self
            .bus
            .borrow_mut()
            .heartbeat_role(self.key(), role, node, now)?;
        self.sync_from_bus();
        Ok(lease)
    }

    /// Drop role leases with `expires_at <= now`. Leader only. Does not bump attempt.
    pub fn expire_roles(&mut self, now: NowMs) -> Result<Vec<Role>> {
        let dropped = self.bus.borrow_mut().expire_roles(self.key(), now)?;
        self.sync_from_bus();
        Ok(dropped)
    }

    /// Commit a rotated refresh token before it is used. Generation is current+1
    /// (0 means none committed yet). Leader only.
    pub fn commit_token(&mut self, blob: Vec<u8>) -> Result<TokenRecord> {
        let rec = self.bus.borrow_mut().commit_token(self.key(), blob)?;
        self.sync_from_bus();
        Ok(rec)
    }

    pub fn set_kernel_version(&mut self, version: String) -> Result<()> {
        self.bus.borrow_mut().set_kernel_version(self.key(), version)?;
        self.sync_from_bus();
        Ok(())
    }
}

/// In-process n-node group for CPU tests and D2 probes. Production uses RPC.
///
/// Slots are stable: `crash(i)` drops the in-memory node so `get(i)` is
/// [`Error::NotFound`], while `len()` stays the started `n`.
pub struct LocalGroup {
    nodes: Vec<Option<Cluster>>,
    isolated: Vec<bool>,
    bus: Rc<RefCell<Bus>>,
    config: ClusterConfig,
    base_dir: PathBuf,
}

impl LocalGroup {
    /// Start `n` AlwaysOn voters in `base_dir/<i>/`. `n` must be >= [`MIN_VOTERS`].
    pub fn start(n: usize, base_dir: impl AsRef<Path>, config: ClusterConfig) -> Result<Self> {
        if n < MIN_VOTERS {
            return Err(Error::TooFewVoters(n));
        }
        let base_dir = base_dir.as_ref().to_path_buf();
        let members: Vec<NodeInfo> = (0..n).map(inner::slot_info).collect();
        let bus = Rc::new(RefCell::new(Bus::new()));
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            let dir = base_dir.join(i.to_string());
            let node = inner::initial_group_node(
                dir,
                members[i].clone(),
                &members,
                config.clone(),
            )?;
            nodes.push(Some(Cluster::from_inner(node, Rc::clone(&bus))));
        }
        Ok(Self {
            nodes,
            isolated: vec![false; n],
            bus,
            config,
            base_dir,
        })
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn get(&mut self, i: usize) -> Result<&mut Cluster> {
        match self.nodes.get_mut(i) {
            Some(Some(c)) => Ok(c),
            _ => Err(Error::NotFound(i.to_string())),
        }
    }

    fn sync_isolation(&self) {
        let mut ids = std::collections::HashSet::new();
        for (i, iso) in self.isolated.iter().enumerate() {
            if *iso {
                ids.insert(i.to_string());
            }
        }
        self.bus.borrow_mut().isolated = ids;
    }

    fn refresh_all(&mut self) {
        for slot in self.nodes.iter_mut().flatten() {
            slot.sync_from_bus();
        }
    }

    /// Drop Raft traffic to `isolated` indices. Isolated nodes still have disk.
    /// Replaces the previous isolated set.
    pub fn partition(&mut self, isolated: &[usize]) -> Result<()> {
        for &i in isolated {
            if i >= self.nodes.len() {
                return Err(Error::NotFound(i.to_string()));
            }
        }
        self.isolated = vec![false; self.nodes.len()];
        for &i in isolated {
            self.isolated[i] = true;
        }
        self.sync_isolation();
        Ok(())
    }

    pub fn heal(&mut self) -> Result<()> {
        self.isolated = vec![false; self.nodes.len()];
        self.sync_isolation();
        Ok(())
    }

    /// Kill process i (drop in-memory node; disk stays).
    pub fn crash(&mut self, i: usize) -> Result<()> {
        if i >= self.nodes.len() {
            return Err(Error::NotFound(i.to_string()));
        }
        let cluster = self.nodes[i]
            .take()
            .ok_or_else(|| Error::NotFound(i.to_string()))?;
        self.bus.borrow_mut().remove(&cluster.this.0);
        Ok(())
    }

    /// Reopen a crashed node from disk and rejoin.
    pub fn restart(&mut self, i: usize) -> Result<()> {
        if i >= self.nodes.len() {
            return Err(Error::NotFound(i.to_string()));
        }
        if self.nodes[i].is_some() {
            return Err(Error::Other(format!("node {i} is not crashed")));
        }
        let dir = self.base_dir.join(i.to_string());
        let this = NodeId(i.to_string());
        let c = Cluster::open_on_bus(dir, this, self.config.clone(), Rc::clone(&self.bus))?;
        self.nodes[i] = Some(c);
        self.sync_isolation();
        Ok(())
    }

    /// Drive elections and apply committed entries. `now` is for role expiry.
    /// When a majority exists, the leader expires due role leases as of `now`.
    pub fn tick(&mut self, now: NowMs) -> Result<()> {
        self.sync_isolation();
        self.bus.borrow_mut().tick(now)?;
        self.refresh_all();
        Ok(())
    }
}
