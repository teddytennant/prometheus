//! Independent in-memory H4 reference: membership, role leases, token, kernel.
//!
//! Slow and obvious. No openraft. Compiled only as a submodule of the
//! integration tests. Production `src/lib.rs` must never import this module.
//!
//! Quorum is `alive_voters * 2 > total_voters`. A successful client mutation
//! on the leader updates committed state, the leader's memory, and the disk of
//! every alive (not crashed, not isolated) slot. Followers' in-memory `state`
//! catches up on `tick`. `tick(now)` also expires due role leases when a
//! majority exists.

#![allow(dead_code)]

use prometheus_raft::{
    Addr, ClusterConfig, ControlState, Error, NodeId, NodeInfo, NodeKind, NowMs, Result, Role,
    RoleLease, TokenRecord, MIN_VOTERS,
};

#[derive(Clone, Debug)]
pub struct RefMachine {
    pub config: ClusterConfig,
    pub membership: Vec<NodeInfo>,
    pub coordinator: Option<RoleLease>,
    pub token_broker: Option<RoleLease>,
    pub kernel_version: String,
    pub token: Option<TokenRecord>,
    pub coord_attempt: u64,
    pub broker_attempt: u64,
    pub ever_reached_min_voters: bool,
}

pub fn slot_info(i: usize) -> NodeInfo {
    NodeInfo {
        id: NodeId(i.to_string()),
        addr: Addr(format!("local:{i}")),
        kind: NodeKind::AlwaysOn,
        trusted: true,
    }
}

impl RefMachine {
    pub fn bootstrap(config: ClusterConfig, this: NodeInfo) -> Result<Self> {
        if this.kind == NodeKind::Ephemeral {
            return Err(Error::EphemeralVoter(this.id.0));
        }
        if !this.trusted {
            return Err(Error::Untrusted(this.id.0));
        }
        if this.kind != NodeKind::AlwaysOn {
            return Err(Error::EphemeralVoter(this.id.0));
        }
        Ok(Self {
            config,
            membership: vec![this],
            coordinator: None,
            token_broker: None,
            kernel_version: "0".to_string(),
            token: None,
            coord_attempt: 0,
            broker_attempt: 0,
            ever_reached_min_voters: false,
        })
    }

    pub fn start_n(config: ClusterConfig, n: usize) -> Result<Self> {
        if n < MIN_VOTERS {
            return Err(Error::TooFewVoters(n));
        }
        let membership: Vec<NodeInfo> = (0..n).map(slot_info).collect();
        Ok(Self {
            config,
            membership,
            coordinator: None,
            token_broker: None,
            kernel_version: "0".to_string(),
            token: None,
            coord_attempt: 0,
            broker_attempt: 0,
            ever_reached_min_voters: true,
        })
    }

    pub fn state(&self) -> ControlState {
        ControlState {
            membership: self.membership.clone(),
            coordinator: self.coordinator.clone(),
            token_broker: self.token_broker.clone(),
            kernel_version: self.kernel_version.clone(),
            token: self.token.clone(),
        }
    }

    pub fn voter_count(&self) -> usize {
        self.membership
            .iter()
            .filter(|n| n.kind == NodeKind::AlwaysOn)
            .count()
    }

    pub fn find(&self, id: &NodeId) -> Option<&NodeInfo> {
        self.membership.iter().find(|n| n.id == *id)
    }

    pub fn add_voter(&mut self, node: NodeInfo) -> Result<()> {
        if self.membership.iter().any(|n| n.id == node.id) {
            return Err(Error::Duplicate(node.id.0));
        }
        if node.kind == NodeKind::Ephemeral {
            return Err(Error::EphemeralVoter(node.id.0));
        }
        if !node.trusted {
            return Err(Error::Untrusted(node.id.0));
        }
        if node.kind != NodeKind::AlwaysOn {
            return Err(Error::EphemeralVoter(node.id.0));
        }
        self.membership.push(node);
        if self.voter_count() >= MIN_VOTERS {
            self.ever_reached_min_voters = true;
        }
        Ok(())
    }

    pub fn add_worker(&mut self, node: NodeInfo) -> Result<()> {
        if self.membership.iter().any(|n| n.id == node.id) {
            return Err(Error::Duplicate(node.id.0));
        }
        if node.kind != NodeKind::Ephemeral {
            return Err(Error::Other(format!(
                "always-on node {} must use add_voter",
                node.id.0
            )));
        }
        self.membership.push(node);
        Ok(())
    }

    pub fn remove_node(&mut self, id: &NodeId) -> Result<()> {
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
        let removed = self.membership.remove(idx);
        self.drop_roles_held_by(&removed.id);
        Ok(())
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

    pub fn claim_role(&mut self, role: Role, node: &NodeId, now: NowMs) -> Result<RoleLease> {
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
        let attempt = match role {
            Role::Coordinator => {
                self.coord_attempt += 1;
                self.coord_attempt
            }
            Role::TokenBroker => {
                self.broker_attempt += 1;
                self.broker_attempt
            }
        };
        let lease = RoleLease {
            role,
            holder: node.clone(),
            attempt,
            expires_at: now.saturating_add(self.config.lease_ttl_ms()),
        };
        match role {
            Role::Coordinator => self.coordinator = Some(lease.clone()),
            Role::TokenBroker => self.token_broker = Some(lease.clone()),
        }
        Ok(lease)
    }

    pub fn heartbeat_role(&mut self, role: Role, node: &NodeId, now: NowMs) -> Result<RoleLease> {
        let lease_slot = match role {
            Role::Coordinator => &mut self.coordinator,
            Role::TokenBroker => &mut self.token_broker,
        };
        let lease = lease_slot
            .as_mut()
            .ok_or_else(|| Error::NotHolder(role, node.0.clone()))?;
        if lease.holder != *node {
            return Err(Error::NotHolder(role, node.0.clone()));
        }
        lease.expires_at = now.saturating_add(self.config.lease_ttl_ms());
        Ok(lease.clone())
    }

    pub fn expire_roles(&mut self, now: NowMs) -> Result<Vec<Role>> {
        let mut dropped = Vec::new();
        if self
            .coordinator
            .as_ref()
            .map(|l| l.expires_at <= now)
            .unwrap_or(false)
        {
            self.coordinator = None;
            dropped.push(Role::Coordinator);
        }
        if self
            .token_broker
            .as_ref()
            .map(|l| l.expires_at <= now)
            .unwrap_or(false)
        {
            self.token_broker = None;
            dropped.push(Role::TokenBroker);
        }
        Ok(dropped)
    }

    pub fn commit_token(&mut self, blob: Vec<u8>) -> Result<TokenRecord> {
        let current = self.token.as_ref().map(|t| t.generation).unwrap_or(0);
        let rec = TokenRecord {
            generation: current + 1,
            blob,
        };
        self.token = Some(rec.clone());
        Ok(rec)
    }

    pub fn set_kernel_version(&mut self, version: String) -> Result<()> {
        self.kernel_version = version;
        Ok(())
    }
}

/// Single-node smoke oracle. Always leader after a successful bootstrap.
#[derive(Clone, Debug)]
pub struct RefCluster {
    pub this: NodeId,
    pub is_leader: bool,
    pub machine: RefMachine,
}

impl RefCluster {
    pub fn bootstrap(this: NodeInfo, config: ClusterConfig) -> Result<Self> {
        let id = this.id.clone();
        let machine = RefMachine::bootstrap(config, this)?;
        Ok(Self {
            this: id,
            is_leader: true,
            machine,
        })
    }

    pub fn state(&self) -> ControlState {
        self.machine.state()
    }

    fn leader_gate(&self) -> Result<()> {
        if !self.is_leader {
            return Err(Error::NotLeader);
        }
        Ok(())
    }

    pub fn add_voter(&mut self, node: NodeInfo) -> Result<()> {
        self.leader_gate()?;
        self.machine.add_voter(node)
    }

    pub fn add_worker(&mut self, node: NodeInfo) -> Result<()> {
        self.leader_gate()?;
        self.machine.add_worker(node)
    }

    pub fn remove_node(&mut self, id: &NodeId) -> Result<()> {
        self.leader_gate()?;
        self.machine.remove_node(id)
    }

    pub fn claim_role(&mut self, role: Role, node: &NodeId, now: NowMs) -> Result<RoleLease> {
        self.leader_gate()?;
        self.machine.claim_role(role, node, now)
    }

    pub fn heartbeat_role(&mut self, role: Role, node: &NodeId, now: NowMs) -> Result<RoleLease> {
        self.leader_gate()?;
        self.machine.heartbeat_role(role, node, now)
    }

    pub fn expire_roles(&mut self, now: NowMs) -> Result<Vec<Role>> {
        self.leader_gate()?;
        self.machine.expire_roles(now)
    }

    pub fn commit_token(&mut self, blob: Vec<u8>) -> Result<TokenRecord> {
        self.leader_gate()?;
        self.machine.commit_token(blob)
    }

    pub fn set_kernel_version(&mut self, version: String) -> Result<()> {
        self.leader_gate()?;
        self.machine.set_kernel_version(version)
    }
}

#[derive(Clone, Debug)]
pub struct RefNode {
    pub info: NodeInfo,
    pub is_leader: bool,
    pub state: ControlState,
}

/// In-process n-node group. Slots stay stable across crash/restart.
#[derive(Clone, Debug)]
pub struct RefGroup {
    n: usize,
    nodes: Vec<Option<RefNode>>,
    isolated: Vec<bool>,
    committed: RefMachine,
    disk: Vec<RefMachine>,
    leader_slot: Option<usize>,
}

impl RefGroup {
    pub fn start(n: usize, config: ClusterConfig) -> Result<Self> {
        let committed = RefMachine::start_n(config, n)?;
        let nodes = (0..n)
            .map(|i| {
                Some(RefNode {
                    info: slot_info(i),
                    is_leader: false,
                    state: committed.state(),
                })
            })
            .collect();
        let disk = vec![committed.clone(); n];
        Ok(Self {
            n,
            nodes,
            isolated: vec![false; n],
            committed,
            disk,
            leader_slot: None,
        })
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn committed_state(&self) -> ControlState {
        self.committed.state()
    }

    pub fn get(&self, i: usize) -> Result<&RefNode> {
        self.nodes
            .get(i)
            .and_then(|n| n.as_ref())
            .ok_or_else(|| Error::NotFound(i.to_string()))
    }

    pub fn leader_index(&self) -> Option<usize> {
        self.leader_slot.filter(|&i| {
            self.nodes
                .get(i)
                .and_then(|n| n.as_ref())
                .map(|n| n.is_leader)
                .unwrap_or(false)
        })
    }

    fn is_crashed(&self, i: usize) -> bool {
        i >= self.n || self.nodes[i].is_none()
    }

    fn is_alive(&self, i: usize) -> bool {
        i < self.n && self.nodes[i].is_some() && !self.isolated[i]
    }

    fn slot_is_voter(&self, i: usize) -> bool {
        let id = NodeId(i.to_string());
        self.committed
            .find(&id)
            .map(|n| n.kind == NodeKind::AlwaysOn)
            .unwrap_or(false)
    }

    fn alive_voter_count(&self) -> usize {
        (0..self.n)
            .filter(|&i| self.is_alive(i) && self.slot_is_voter(i))
            .count()
    }

    pub fn has_majority(&self) -> bool {
        let total = self.committed.voter_count();
        if total == 0 {
            return false;
        }
        self.alive_voter_count() * 2 > total
    }

    fn persist_alive(&mut self) {
        for i in 0..self.n {
            if self.is_alive(i) {
                self.disk[i] = self.committed.clone();
            }
        }
    }

    fn apply_leader<T>(
        &mut self,
        via: usize,
        f: impl FnOnce(&mut RefMachine) -> Result<T>,
    ) -> Result<T> {
        if self.is_crashed(via) {
            return Err(Error::NotFound(via.to_string()));
        }
        let is_leader = self
            .nodes
            .get(via)
            .and_then(|n| n.as_ref())
            .map(|n| n.is_leader)
            .unwrap_or(false);
        if !is_leader || !self.has_majority() {
            return Err(Error::NotLeader);
        }
        let out = f(&mut self.committed)?;
        if let Some(node) = self.nodes[via].as_mut() {
            node.state = self.committed.state();
        }
        self.persist_alive();
        Ok(out)
    }

    fn elect(&mut self) {
        if !self.has_majority() {
            self.leader_slot = None;
            return;
        }
        if let Some(cur) = self.leader_slot {
            if self.is_alive(cur) && self.slot_is_voter(cur) {
                return;
            }
        }
        self.leader_slot = (0..self.n).find(|&i| self.is_alive(i) && self.slot_is_voter(i));
    }

    pub fn tick(&mut self, now: NowMs) -> Result<()> {
        if self.has_majority() {
            let _ = self.committed.expire_roles(now)?;
            self.elect();
            let leader = self.leader_slot;
            for i in 0..self.n {
                if self.is_alive(i) {
                    if let Some(node) = self.nodes[i].as_mut() {
                        node.state = self.committed.state();
                        node.is_leader = leader == Some(i);
                    }
                    self.disk[i] = self.committed.clone();
                } else if let Some(node) = self.nodes[i].as_mut() {
                    node.is_leader = false;
                }
            }
        } else {
            self.leader_slot = None;
            for i in 0..self.n {
                if let Some(node) = self.nodes[i].as_mut() {
                    node.is_leader = false;
                }
            }
        }
        Ok(())
    }

    pub fn partition(&mut self, isolated: &[usize]) -> Result<()> {
        for &i in isolated {
            if i >= self.n {
                return Err(Error::NotFound(i.to_string()));
            }
        }
        self.isolated = vec![false; self.n];
        for &i in isolated {
            self.isolated[i] = true;
        }
        Ok(())
    }

    pub fn heal(&mut self) -> Result<()> {
        self.isolated = vec![false; self.n];
        Ok(())
    }

    pub fn crash(&mut self, i: usize) -> Result<()> {
        if i >= self.n || self.nodes[i].is_none() {
            return Err(Error::NotFound(i.to_string()));
        }
        self.nodes[i] = None;
        if self.leader_slot == Some(i) {
            self.leader_slot = None;
        }
        Ok(())
    }

    pub fn restart(&mut self, i: usize) -> Result<()> {
        if i >= self.n {
            return Err(Error::NotFound(i.to_string()));
        }
        if self.nodes[i].is_some() {
            return Err(Error::Other(format!("node {i} is not crashed")));
        }
        self.nodes[i] = Some(RefNode {
            info: slot_info(i),
            is_leader: false,
            state: self.disk[i].state(),
        });
        Ok(())
    }

    pub fn add_voter(&mut self, via: usize, node: NodeInfo) -> Result<()> {
        self.apply_leader(via, |m| m.add_voter(node))
    }

    pub fn add_worker(&mut self, via: usize, node: NodeInfo) -> Result<()> {
        self.apply_leader(via, |m| m.add_worker(node))
    }

    pub fn remove_node(&mut self, via: usize, id: &NodeId) -> Result<()> {
        self.apply_leader(via, |m| m.remove_node(id))
    }

    pub fn claim_role(
        &mut self,
        via: usize,
        role: Role,
        node: &NodeId,
        now: NowMs,
    ) -> Result<RoleLease> {
        self.apply_leader(via, |m| m.claim_role(role, node, now))
    }

    pub fn heartbeat_role(
        &mut self,
        via: usize,
        role: Role,
        node: &NodeId,
        now: NowMs,
    ) -> Result<RoleLease> {
        self.apply_leader(via, |m| m.heartbeat_role(role, node, now))
    }

    pub fn expire_roles(&mut self, via: usize, now: NowMs) -> Result<Vec<Role>> {
        self.apply_leader(via, |m| m.expire_roles(now))
    }

    pub fn commit_token(&mut self, via: usize, blob: Vec<u8>) -> Result<TokenRecord> {
        self.apply_leader(via, |m| m.commit_token(blob))
    }

    pub fn set_kernel_version(&mut self, via: usize, version: String) -> Result<()> {
        self.apply_leader(via, |m| m.set_kernel_version(version))
    }

    pub fn tick_until_leader(&mut self, now: NowMs) -> usize {
        self.tick(now).expect("ref tick");
        self.leader_index()
            .expect("reference elects a leader in one tick when a majority exists")
    }
}
