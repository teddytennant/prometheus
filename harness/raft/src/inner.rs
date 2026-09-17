//! In-process Raft: durable log, votes, apply, partitionable bus.
//!
//! Client mutations run on the leader and commit only with a reachable voter
//! majority. `tick` drives elections, log replication, apply, and due-role
//! expiry. Isolated ids get no AppendEntries or RequestVote traffic.

use crate::{
    ClusterConfig, ControlState, Error, NodeId, NodeInfo, NodeKind, NowMs, Result, Role, RoleLease,
    TokenRecord, MIN_VOTERS,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

const STATE_FILE: &str = "state.json";
const STATE_TMP: &str = "state.json.tmp";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RaftRole {
    Follower,
    Leader,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LogEntry {
    term: u64,
    cmd: Command,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum Command {
    AddVoter(NodeInfo),
    AddWorker(NodeInfo),
    RemoveNode(String),
    ClaimRole {
        role: Role,
        lease: RoleLease,
        coord_attempt: u64,
        broker_attempt: u64,
    },
    HeartbeatRole {
        role: Role,
        lease: RoleLease,
    },
    ExpireRoles {
        coordinator: Option<RoleLease>,
        token_broker: Option<RoleLease>,
    },
    CommitToken(TokenRecord),
    SetKernel(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Durable {
    current_term: u64,
    voted_for: Option<String>,
    log: Vec<LogEntry>,
    commit_index: u64,
    last_applied: u64,
    membership: Vec<NodeInfo>,
    coordinator: Option<RoleLease>,
    token_broker: Option<RoleLease>,
    kernel_version: String,
    token: Option<TokenRecord>,
    coord_attempt: u64,
    broker_attempt: u64,
    ever_reached_min_voters: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct Machine {
    pub membership: Vec<NodeInfo>,
    pub coordinator: Option<RoleLease>,
    pub token_broker: Option<RoleLease>,
    pub kernel_version: String,
    pub token: Option<TokenRecord>,
    pub coord_attempt: u64,
    pub broker_attempt: u64,
    pub ever_reached_min_voters: bool,
}

impl Machine {
    fn bootstrap(this: NodeInfo) -> Self {
        Self {
            membership: vec![this],
            coordinator: None,
            token_broker: None,
            kernel_version: "0".to_string(),
            token: None,
            coord_attempt: 0,
            broker_attempt: 0,
            ever_reached_min_voters: false,
        }
    }

    fn start_n(members: Vec<NodeInfo>) -> Self {
        let n = members.len();
        Self {
            membership: members,
            coordinator: None,
            token_broker: None,
            kernel_version: "0".to_string(),
            token: None,
            coord_attempt: 0,
            broker_attempt: 0,
            ever_reached_min_voters: n >= MIN_VOTERS,
        }
    }

    pub(crate) fn state(&self) -> ControlState {
        ControlState {
            membership: self.membership.clone(),
            coordinator: self.coordinator.clone(),
            token_broker: self.token_broker.clone(),
            kernel_version: self.kernel_version.clone(),
            token: self.token.clone(),
        }
    }

    fn voter_count(&self) -> usize {
        self.membership
            .iter()
            .filter(|n| n.kind == NodeKind::AlwaysOn)
            .count()
    }

    fn find(&self, id: &NodeId) -> Option<&NodeInfo> {
        self.membership.iter().find(|n| n.id == *id)
    }

    fn is_voter_id(&self, id: &str) -> bool {
        self.membership
            .iter()
            .any(|n| n.id.0 == id && n.kind == NodeKind::AlwaysOn)
    }

    fn drop_roles_held_by(&mut self, id: &NodeId) {
        if self
            .coordinator
            .as_ref()
            .map(|l| l.holder == *id)
            .unwrap_or(false)
        {
            self.coordinator = None;
        }
        if self
            .token_broker
            .as_ref()
            .map(|l| l.holder == *id)
            .unwrap_or(false)
        {
            self.token_broker = None;
        }
    }

    fn apply(&mut self, cmd: &Command) {
        match cmd {
            Command::AddVoter(node) => {
                if !self.membership.iter().any(|n| n.id == node.id) {
                    self.membership.push(node.clone());
                    if self.voter_count() >= MIN_VOTERS {
                        self.ever_reached_min_voters = true;
                    }
                }
            }
            Command::AddWorker(node) => {
                if !self.membership.iter().any(|n| n.id == node.id) {
                    self.membership.push(node.clone());
                }
            }
            Command::RemoveNode(id) => {
                let nid = NodeId(id.clone());
                self.membership.retain(|n| n.id != nid);
                self.drop_roles_held_by(&nid);
            }
            Command::ClaimRole {
                role,
                lease,
                coord_attempt,
                broker_attempt,
            } => {
                match role {
                    Role::Coordinator => self.coordinator = Some(lease.clone()),
                    Role::TokenBroker => self.token_broker = Some(lease.clone()),
                }
                self.coord_attempt = *coord_attempt;
                self.broker_attempt = *broker_attempt;
            }
            Command::HeartbeatRole { role, lease } => match role {
                Role::Coordinator => self.coordinator = Some(lease.clone()),
                Role::TokenBroker => self.token_broker = Some(lease.clone()),
            },
            Command::ExpireRoles {
                coordinator,
                token_broker,
            } => {
                self.coordinator = coordinator.clone();
                self.token_broker = token_broker.clone();
            }
            Command::CommitToken(rec) => self.token = Some(rec.clone()),
            Command::SetKernel(v) => self.kernel_version = v.clone(),
        }
    }

    fn validate_add_voter(&self, node: &NodeInfo) -> Result<()> {
        if self.membership.iter().any(|n| n.id == node.id) {
            return Err(Error::Duplicate(node.id.0.clone()));
        }
        if node.kind == NodeKind::Ephemeral {
            return Err(Error::EphemeralVoter(node.id.0.clone()));
        }
        if !node.trusted {
            return Err(Error::Untrusted(node.id.0.clone()));
        }
        Ok(())
    }

    fn validate_add_worker(&self, node: &NodeInfo) -> Result<()> {
        if self.membership.iter().any(|n| n.id == node.id) {
            return Err(Error::Duplicate(node.id.0.clone()));
        }
        if node.kind != NodeKind::Ephemeral {
            return Err(Error::Other(format!(
                "always-on node {} must use add_voter",
                node.id.0
            )));
        }
        Ok(())
    }

    fn validate_remove(&self, id: &NodeId) -> Result<()> {
        let idx = self
            .membership
            .iter()
            .position(|n| n.id == *id)
            .ok_or_else(|| Error::NotFound(id.0.clone()))?;
        let is_voter = self.membership[idx].kind == NodeKind::AlwaysOn;
        if is_voter {
            let would = self.voter_count() - 1;
            if self.ever_reached_min_voters && would < MIN_VOTERS {
                return Err(Error::TooFewVoters(would));
            }
            if would < 1 {
                return Err(Error::TooFewVoters(0));
            }
        }
        Ok(())
    }

    fn prepare_claim(
        &self,
        role: Role,
        node: &NodeId,
        now: NowMs,
        ttl: u64,
    ) -> Result<(RoleLease, u64, u64)> {
        let member = self
            .find(node)
            .ok_or_else(|| Error::NotFound(node.0.clone()))?;
        if !member.trusted {
            return Err(Error::Untrusted(node.0.clone()));
        }
        let current = match role {
            Role::Coordinator => &self.coordinator,
            Role::TokenBroker => &self.token_broker,
        };
        if let Some(lease) = current {
            if lease.expires_at > now {
                return Err(Error::NotClaimable(role));
            }
        }
        let (coord_attempt, broker_attempt) = match role {
            Role::Coordinator => (self.coord_attempt + 1, self.broker_attempt),
            Role::TokenBroker => (self.coord_attempt, self.broker_attempt + 1),
        };
        let attempt = match role {
            Role::Coordinator => coord_attempt,
            Role::TokenBroker => broker_attempt,
        };
        let lease = RoleLease {
            role,
            holder: node.clone(),
            attempt,
            expires_at: now.saturating_add(ttl),
        };
        Ok((lease, coord_attempt, broker_attempt))
    }

    fn prepare_heartbeat(
        &self,
        role: Role,
        node: &NodeId,
        now: NowMs,
        ttl: u64,
    ) -> Result<RoleLease> {
        let current = match role {
            Role::Coordinator => &self.coordinator,
            Role::TokenBroker => &self.token_broker,
        };
        let lease = current
            .as_ref()
            .ok_or_else(|| Error::NotHolder(role, node.0.clone()))?;
        if lease.holder != *node {
            return Err(Error::NotHolder(role, node.0.clone()));
        }
        Ok(RoleLease {
            role,
            holder: node.clone(),
            attempt: lease.attempt,
            expires_at: now.saturating_add(ttl),
        })
    }

    fn prepare_expire(&self, now: NowMs) -> (Vec<Role>, Option<RoleLease>, Option<RoleLease>) {
        let mut dropped = Vec::new();
        let mut coordinator = self.coordinator.clone();
        let mut token_broker = self.token_broker.clone();
        if coordinator
            .as_ref()
            .map(|l| l.expires_at <= now)
            .unwrap_or(false)
        {
            coordinator = None;
            dropped.push(Role::Coordinator);
        }
        if token_broker
            .as_ref()
            .map(|l| l.expires_at <= now)
            .unwrap_or(false)
        {
            token_broker = None;
            dropped.push(Role::TokenBroker);
        }
        (dropped, coordinator, token_broker)
    }
}

pub(crate) struct NodeInner {
    pub dir: PathBuf,
    pub config: ClusterConfig,
    pub id: String,
    pub current_term: u64,
    pub voted_for: Option<String>,
    pub log: Vec<LogEntry>,
    pub commit_index: u64,
    pub last_applied: u64,
    pub role: RaftRole,
    pub machine: Machine,
}

impl NodeInner {
    fn last_log_index(&self) -> u64 {
        self.log.len() as u64
    }

    fn last_log_term(&self) -> u64 {
        self.log.last().map(|e| e.term).unwrap_or(0)
    }

    fn to_durable(&self) -> Durable {
        Durable {
            current_term: self.current_term,
            voted_for: self.voted_for.clone(),
            log: self.log.clone(),
            commit_index: self.commit_index,
            last_applied: self.last_applied,
            membership: self.machine.membership.clone(),
            coordinator: self.machine.coordinator.clone(),
            token_broker: self.machine.token_broker.clone(),
            kernel_version: self.machine.kernel_version.clone(),
            token: self.machine.token.clone(),
            coord_attempt: self.machine.coord_attempt,
            broker_attempt: self.machine.broker_attempt,
            ever_reached_min_voters: self.machine.ever_reached_min_voters,
        }
    }

    fn from_durable(dir: PathBuf, config: ClusterConfig, id: String, d: Durable) -> Self {
        Self {
            dir,
            config,
            id,
            current_term: d.current_term,
            voted_for: d.voted_for,
            log: d.log,
            commit_index: d.commit_index.min(d.last_applied),
            last_applied: d.last_applied,
            role: RaftRole::Follower,
            machine: Machine {
                membership: d.membership,
                coordinator: d.coordinator,
                token_broker: d.token_broker,
                kernel_version: d.kernel_version,
                token: d.token,
                coord_attempt: d.coord_attempt,
                broker_attempt: d.broker_attempt,
                ever_reached_min_voters: d.ever_reached_min_voters,
            },
        }
    }
}

fn other(msg: impl Into<String>) -> Error {
    Error::Other(msg.into())
}

fn persist(node: &NodeInner) -> Result<()> {
    let path = node.dir.join(STATE_FILE);
    let tmp = node.dir.join(STATE_TMP);
    let data = serde_json::to_vec_pretty(&node.to_durable()).map_err(|e| other(e.to_string()))?;
    fs::write(&tmp, &data).map_err(|e| other(e.to_string()))?;
    fs::rename(&tmp, &path).map_err(|e| other(e.to_string()))?;
    Ok(())
}

fn load_durable(dir: &Path) -> Result<Durable> {
    let path = dir.join(STATE_FILE);
    let data = fs::read(&path).map_err(|e| other(e.to_string()))?;
    serde_json::from_slice(&data).map_err(|e| other(e.to_string()))
}

fn apply_committed(node: &mut NodeInner) {
    while node.last_applied < node.commit_index {
        let idx = node.last_applied as usize;
        let cmd = node.log[idx].cmd.clone();
        node.machine.apply(&cmd);
        node.last_applied += 1;
    }
}

pub(crate) struct Bus {
    pub nodes: HashMap<String, NodeInner>,
    pub isolated: HashSet<String>,
}

impl Bus {
    pub(crate) fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            isolated: HashSet::new(),
        }
    }

    fn is_isolated(&self, id: &str) -> bool {
        self.isolated.contains(id)
    }

    /// Nodes this id can send Raft RPCs to (including itself). Isolated nodes
    /// only see themselves.
    fn reachable_ids(&self, from: &str) -> Vec<String> {
        if !self.nodes.contains_key(from) {
            return Vec::new();
        }
        if self.is_isolated(from) {
            return vec![from.to_string()];
        }
        self.nodes
            .keys()
            .filter(|id| !self.is_isolated(id))
            .cloned()
            .collect()
    }

    fn has_majority(&self, from: &str) -> bool {
        let Some(node) = self.nodes.get(from) else {
            return false;
        };
        let total = node.machine.voter_count();
        if total == 0 {
            return false;
        }
        let reachable = self.reachable_ids(from);
        let acks = reachable
            .iter()
            .filter(|id| node.machine.is_voter_id(id))
            .count();
        acks * 2 > total
    }

    pub(crate) fn is_leader(&self, id: &str) -> bool {
        self.nodes
            .get(id)
            .map(|n| n.role == RaftRole::Leader)
            .unwrap_or(false)
    }

    pub(crate) fn state_of(&self, id: &str) -> Option<ControlState> {
        self.nodes.get(id).map(|n| n.machine.state())
    }

    fn leader_gate(&self, id: &str) -> Result<()> {
        if !self.is_leader(id) || !self.has_majority(id) {
            return Err(Error::NotLeader);
        }
        Ok(())
    }

    fn replicate_from(&mut self, leader_id: &str) -> Result<()> {
        let Some(leader) = self.nodes.get(leader_id) else {
            return Ok(());
        };
        if leader.role != RaftRole::Leader {
            return Ok(());
        }
        let leader_log = leader.log.clone();
        let leader_term = leader.current_term;
        let last = leader.last_log_index();
        let total = leader.machine.voter_count();
        let membership_voters: HashSet<String> = leader
            .machine
            .membership
            .iter()
            .filter(|n| n.kind == NodeKind::AlwaysOn)
            .map(|n| n.id.0.clone())
            .collect();

        let reachable = self.reachable_ids(leader_id);
        for pid in &reachable {
            if pid == leader_id {
                continue;
            }
            let Some(peer) = self.nodes.get_mut(pid) else {
                continue;
            };
            if leader_term < peer.current_term {
                continue;
            }
            peer.current_term = leader_term;
            peer.voted_for = Some(leader_id.to_string());
            peer.role = RaftRole::Follower;
            install_log(peer, &leader_log);
        }

        let mut voter_acks = 0usize;
        for pid in &reachable {
            let Some(n) = self.nodes.get(pid) else {
                continue;
            };
            if n.log.len() as u64 >= last && membership_voters.contains(pid) {
                voter_acks += 1;
            }
        }
        if total > 0 && voter_acks * 2 > total {
            if let Some(leader) = self.nodes.get_mut(leader_id) {
                leader.commit_index = last;
            }
        }
        let commit = self
            .nodes
            .get(leader_id)
            .map(|n| n.commit_index)
            .unwrap_or(0);
        for pid in &reachable {
            if let Some(n) = self.nodes.get_mut(pid) {
                let cap = n.log.len() as u64;
                n.commit_index = n.commit_index.max(commit.min(cap));
                apply_committed(n);
                persist(n)?;
            }
        }
        Ok(())
    }

    fn append_and_commit(&mut self, leader_id: &str, cmd: Command) -> Result<()> {
        self.leader_gate(leader_id)?;
        {
            let leader = self
                .nodes
                .get_mut(leader_id)
                .ok_or_else(|| Error::NotFound(leader_id.to_string()))?;
            let term = leader.current_term;
            leader.log.push(LogEntry { term, cmd });
        }
        self.replicate_from(leader_id)?;
        let leader = self
            .nodes
            .get(leader_id)
            .ok_or_else(|| Error::NotFound(leader_id.to_string()))?;
        if leader.last_applied < leader.last_log_index() {
            // Majority did not commit the new entry.
            return Err(Error::NotLeader);
        }
        Ok(())
    }

    pub(crate) fn add_voter(&mut self, from: &str, node: NodeInfo) -> Result<()> {
        self.leader_gate(from)?;
        self.nodes
            .get(from)
            .ok_or_else(|| Error::NotFound(from.to_string()))?
            .machine
            .validate_add_voter(&node)?;
        self.append_and_commit(from, Command::AddVoter(node))
    }

    pub(crate) fn add_worker(&mut self, from: &str, node: NodeInfo) -> Result<()> {
        self.leader_gate(from)?;
        self.nodes
            .get(from)
            .ok_or_else(|| Error::NotFound(from.to_string()))?
            .machine
            .validate_add_worker(&node)?;
        self.append_and_commit(from, Command::AddWorker(node))
    }

    pub(crate) fn remove_node(&mut self, from: &str, id: &NodeId) -> Result<()> {
        self.leader_gate(from)?;
        self.nodes
            .get(from)
            .ok_or_else(|| Error::NotFound(from.to_string()))?
            .machine
            .validate_remove(id)?;
        self.append_and_commit(from, Command::RemoveNode(id.0.clone()))
    }

    pub(crate) fn claim_role(
        &mut self,
        from: &str,
        role: Role,
        node: &NodeId,
        now: NowMs,
    ) -> Result<RoleLease> {
        self.leader_gate(from)?;
        let (lease, coord_attempt, broker_attempt) = {
            let n = self
                .nodes
                .get(from)
                .ok_or_else(|| Error::NotFound(from.to_string()))?;
            n.machine
                .prepare_claim(role, node, now, n.config.lease_ttl_ms())?
        };
        let out = lease.clone();
        self.append_and_commit(
            from,
            Command::ClaimRole {
                role,
                lease,
                coord_attempt,
                broker_attempt,
            },
        )?;
        Ok(out)
    }

    pub(crate) fn heartbeat_role(
        &mut self,
        from: &str,
        role: Role,
        node: &NodeId,
        now: NowMs,
    ) -> Result<RoleLease> {
        self.leader_gate(from)?;
        let lease = {
            let n = self
                .nodes
                .get(from)
                .ok_or_else(|| Error::NotFound(from.to_string()))?;
            n.machine
                .prepare_heartbeat(role, node, now, n.config.lease_ttl_ms())?
        };
        let out = lease.clone();
        self.append_and_commit(from, Command::HeartbeatRole { role, lease })?;
        Ok(out)
    }

    pub(crate) fn expire_roles(&mut self, from: &str, now: NowMs) -> Result<Vec<Role>> {
        self.leader_gate(from)?;
        let (dropped, coordinator, token_broker) = {
            let n = self
                .nodes
                .get(from)
                .ok_or_else(|| Error::NotFound(from.to_string()))?;
            n.machine.prepare_expire(now)
        };
        if dropped.is_empty() {
            return Ok(dropped);
        }
        self.append_and_commit(
            from,
            Command::ExpireRoles {
                coordinator,
                token_broker,
            },
        )?;
        Ok(dropped)
    }

    pub(crate) fn commit_token(&mut self, from: &str, blob: Vec<u8>) -> Result<TokenRecord> {
        self.leader_gate(from)?;
        let rec = {
            let n = self
                .nodes
                .get(from)
                .ok_or_else(|| Error::NotFound(from.to_string()))?;
            let current = n.machine.token.as_ref().map(|t| t.generation).unwrap_or(0);
            TokenRecord {
                generation: current + 1,
                blob,
            }
        };
        let out = rec.clone();
        self.append_and_commit(from, Command::CommitToken(rec))?;
        Ok(out)
    }

    pub(crate) fn set_kernel_version(&mut self, from: &str, version: String) -> Result<()> {
        self.leader_gate(from)?;
        self.append_and_commit(from, Command::SetKernel(version))
    }

    /// Drive elections, replicate, apply, and expire due roles when a majority
    /// of voters is reachable.
    pub(crate) fn tick(&mut self, now: NowMs) -> Result<()> {
        self.elect()?;
        let leaders: Vec<String> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.role == RaftRole::Leader)
            .map(|(id, _)| id.clone())
            .collect();
        for lid in leaders {
            if self.is_isolated(&lid) {
                continue;
            }
            self.replicate_from(&lid)?;
            if self.has_majority(&lid) {
                self.expire_roles(&lid, now)?;
            }
        }
        Ok(())
    }

    fn elect(&mut self) -> Result<()> {
        // Isolated nodes cannot take part in majority traffic; they step down.
        let isolated: Vec<String> = self
            .nodes
            .keys()
            .filter(|id| self.is_isolated(id))
            .cloned()
            .collect();
        for id in &isolated {
            if let Some(n) = self.nodes.get_mut(id) {
                n.role = RaftRole::Follower;
            }
        }

        let connected: Vec<String> = self
            .nodes
            .keys()
            .filter(|id| !self.is_isolated(id))
            .cloned()
            .collect();
        if connected.is_empty() {
            return Ok(());
        }

        // Use the most up-to-date connected log as the source of membership.
        let best = connected
            .iter()
            .max_by_key(|id| {
                let n = &self.nodes[*id];
                (
                    n.last_log_term(),
                    n.last_log_index(),
                    std::cmp::Reverse(id.as_str()),
                )
            })
            .cloned()
            .expect("connected non-empty");
        let total = self.nodes[&best].machine.voter_count();
        let connected_voters: Vec<String> = connected
            .iter()
            .filter(|id| self.nodes[&best].machine.is_voter_id(id))
            .cloned()
            .collect();

        if total == 0 || connected_voters.len() * 2 <= total {
            for id in &connected {
                if let Some(n) = self.nodes.get_mut(id) {
                    n.role = RaftRole::Follower;
                }
            }
            return Ok(());
        }

        if let Some(cur) = connected_voters
            .iter()
            .find(|id| self.nodes[*id].role == RaftRole::Leader)
            .cloned()
        {
            // Incumbent still reachable with a majority. Others follow.
            for id in &connected {
                if id != &cur {
                    if let Some(n) = self.nodes.get_mut(id) {
                        n.role = RaftRole::Follower;
                    }
                }
            }
            return Ok(());
        }

        // RequestVote: pick the most up-to-date connected voter (term, index,
        // then lowest id). Every reachable voter grants; that is a majority.
        let winner = connected_voters
            .iter()
            .max_by_key(|id| {
                let n = &self.nodes[*id];
                (
                    n.last_log_term(),
                    n.last_log_index(),
                    std::cmp::Reverse(id.as_str()),
                )
            })
            .cloned()
            .expect("connected voters non-empty");

        let new_term = connected
            .iter()
            .map(|id| self.nodes[id].current_term)
            .max()
            .unwrap_or(0)
            .saturating_add(1);

        for id in &connected {
            if let Some(n) = self.nodes.get_mut(id) {
                n.current_term = new_term;
                n.voted_for = Some(winner.clone());
                n.role = if *id == winner {
                    RaftRole::Leader
                } else {
                    RaftRole::Follower
                };
                persist(n)?;
            }
        }
        Ok(())
    }

    pub(crate) fn insert(&mut self, node: NodeInner) {
        self.nodes.insert(node.id.clone(), node);
    }

    pub(crate) fn remove(&mut self, id: &str) {
        self.nodes.remove(id);
    }
}

fn install_log(peer: &mut NodeInner, leader_log: &[LogEntry]) {
    let mut i = 0usize;
    while i < peer.log.len() && i < leader_log.len() && peer.log[i].term == leader_log[i].term {
        i += 1;
    }
    if i < peer.log.len() {
        // Conflict: only uncommitted suffix can diverge; applied prefix is a
        // snapshot and must remain a prefix of the leader log.
        peer.log.truncate(i);
        if peer.commit_index > i as u64 {
            peer.commit_index = i as u64;
        }
        if peer.last_applied > i as u64 {
            peer.last_applied = i as u64;
        }
    }
    if i < leader_log.len() {
        peer.log.extend_from_slice(&leader_log[i..]);
    }
}

pub(crate) fn check_bootstrap_node(this: &NodeInfo) -> Result<()> {
    if this.kind == NodeKind::Ephemeral {
        return Err(Error::EphemeralVoter(this.id.0.clone()));
    }
    if !this.trusted {
        return Err(Error::Untrusted(this.id.0.clone()));
    }
    Ok(())
}

pub(crate) fn bootstrap_node(
    dir: PathBuf,
    this: NodeInfo,
    config: ClusterConfig,
) -> Result<NodeInner> {
    check_bootstrap_node(&this)?;
    if dir.exists() {
        return Err(other(format!("bootstrap dir exists: {}", dir.display())));
    }
    fs::create_dir_all(&dir).map_err(|e| other(e.to_string()))?;
    let id = this.id.0.clone();
    let node = NodeInner {
        dir,
        config,
        id,
        current_term: 1,
        voted_for: Some(this.id.0.clone()),
        log: Vec::new(),
        commit_index: 0,
        last_applied: 0,
        role: RaftRole::Leader,
        machine: Machine::bootstrap(this),
    };
    persist(&node)?;
    Ok(node)
}

pub(crate) fn open_node(dir: PathBuf, this: NodeId, config: ClusterConfig) -> Result<NodeInner> {
    if !dir.exists() {
        return Err(other(format!("open missing dir: {}", dir.display())));
    }
    let durable = load_durable(&dir)?;
    Ok(NodeInner::from_durable(dir, config, this.0, durable))
}

pub(crate) fn initial_group_node(
    dir: PathBuf,
    this: NodeInfo,
    members: &[NodeInfo],
    config: ClusterConfig,
) -> Result<NodeInner> {
    if dir.exists() {
        return Err(other(format!("start dir exists: {}", dir.display())));
    }
    fs::create_dir_all(&dir).map_err(|e| other(e.to_string()))?;
    let id = this.id.0.clone();
    let node = NodeInner {
        dir,
        config,
        id,
        current_term: 0,
        voted_for: None,
        log: Vec::new(),
        commit_index: 0,
        last_applied: 0,
        role: RaftRole::Follower,
        machine: Machine::start_n(members.to_vec()),
    };
    persist(&node)?;
    Ok(node)
}

pub(crate) fn slot_info(i: usize) -> NodeInfo {
    NodeInfo {
        id: NodeId(i.to_string()),
        addr: crate::Addr(format!("local://{i}")),
        kind: NodeKind::AlwaysOn,
        trusted: true,
    }
}
