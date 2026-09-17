//! Independent slow reference for H12 quorum-signed self-update.
//!
//! Plain SHA-256 (`sha2`) plus an in-memory state machine. Production
//! (`harness/ops/src`) must never import this module.

#![allow(dead_code)]

use prometheus_cas::Digest;
use prometheus_ops::{
    Error, NodeRole, NodeStatus, OpsConfig, Release, ReleaseState, Result, Status, EVENT_APPLY,
    EVENT_PROMOTE, EVENT_PROPOSE, EVENT_REFUSE, EVENT_ROLLBACK, EVENT_SHADOW, EVENT_SHADOW_RESULT,
    EVENT_UNHEALTHY, PIN_CURRENT, PIN_PREVIOUS, PIN_SHADOW,
};
use prometheus_raft::NodeId;
use sha2::{Digest as ShaDigest, Sha256};
use std::collections::{HashMap, HashSet};

pub fn digest_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex(&h.finalize())
}

fn hex(bytes: &[u8]) -> String {
    const T: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(T[(b >> 4) as usize] as char);
        s.push(T[(b & 0xf) as usize] as char);
    }
    s
}

pub fn digest_of(bytes: &[u8]) -> Digest {
    Digest(digest_hex(bytes))
}

/// Distinct humans with non-empty signature bytes, compared to `quorum`.
pub fn ref_has_quorum(release: &Release, quorum: usize) -> bool {
    distinct_signers(release) >= quorum
}

pub fn distinct_signers(release: &Release) -> usize {
    let mut seen: HashSet<&str> = HashSet::new();
    for sig in &release.signatures {
        if sig.bytes.is_empty() {
            continue;
        }
        seen.insert(&sig.human.0);
    }
    seen.len()
}

pub struct RefOps {
    pub config: OpsConfig,
    pub state: ReleaseState,
    pub release: Option<Release>,
    pub nodes: Vec<NodeStatus>,
    pub kernel_version: String,
    pub previous_version: String,
    pub pins: HashMap<String, String>,
    pub shadow_passed: bool,
    pub events: Vec<String>,
}

impl RefOps {
    pub fn new(config: OpsConfig, membership: Vec<NodeId>, kernel_version: String) -> Self {
        let nodes = membership
            .iter()
            .map(|id| NodeStatus {
                id: id.clone(),
                applied: kernel_version.clone(),
                healthy: true,
                role: NodeRole::Live,
            })
            .collect();
        Self {
            config,
            state: ReleaseState::Idle,
            release: None,
            nodes,
            previous_version: kernel_version.clone(),
            kernel_version,
            pins: HashMap::new(),
            shadow_passed: false,
            events: Vec::new(),
        }
    }

    pub fn status(&self) -> Status {
        Status {
            kernel_version: self.kernel_version.clone(),
            release: self.release.clone(),
            state: self.state.clone(),
            nodes: self.nodes.clone(),
        }
    }

    pub fn status_json(&self) -> Result<String> {
        serde_json::to_string(&self.status()).map_err(|e| Error::Other(e.to_string()))
    }

    pub fn status_text(&self) -> String {
        ref_status_text(&self.status())
    }

    fn live_mut(&mut self) -> impl Iterator<Item = &mut NodeStatus> {
        self.nodes.iter_mut().filter(|n| n.role == NodeRole::Live)
    }

    fn live(&self) -> impl Iterator<Item = &NodeStatus> {
        self.nodes.iter().filter(|n| n.role == NodeRole::Live)
    }

    fn find_live_mut(&mut self, node: &NodeId) -> Result<&mut NodeStatus> {
        if self
            .nodes
            .iter()
            .any(|n| n.id == *node && n.role == NodeRole::Shadow)
        {
            return Err(Error::WrongState(format!(
                "node {} is the shadow node, not live",
                node.0
            )));
        }
        self.nodes
            .iter_mut()
            .find(|n| n.id == *node && n.role == NodeRole::Live)
            .ok_or_else(|| Error::NotFound(node.0.clone()))
    }

    fn add_shadow(&mut self) {
        if self.nodes.iter().any(|n| n.role == NodeRole::Shadow) {
            return;
        }
        let applied = self
            .release
            .as_ref()
            .map(|r| r.version.clone())
            .unwrap_or_else(|| self.kernel_version.clone());
        self.nodes.push(NodeStatus {
            id: NodeId("shadow".to_string()),
            applied,
            healthy: true,
            role: NodeRole::Shadow,
        });
    }

    fn drop_shadow(&mut self) {
        self.nodes.retain(|n| n.role != NodeRole::Shadow);
    }

    fn reset_live_to_kernel(&mut self) {
        let k = self.kernel_version.clone();
        for n in self.live_mut() {
            n.applied = k.clone();
            n.healthy = true;
            n.role = NodeRole::Live;
        }
    }

    fn pin_shadow(&mut self, digest: &str) {
        self.pins.insert(PIN_SHADOW.to_string(), digest.to_string());
    }

    fn unpin_shadow(&mut self) {
        self.pins.remove(PIN_SHADOW);
    }

    fn propose_allowed(&self) -> bool {
        matches!(
            self.state,
            ReleaseState::Idle
                | ReleaseState::Refused { .. }
                | ReleaseState::RolledBack { .. }
                | ReleaseState::Current
        )
    }

    pub fn propose(&mut self, release: Release, artifact: &[u8]) -> Result<()> {
        if !self.propose_allowed() {
            return Err(Error::WrongState(format!("propose from {:?}", self.state)));
        }
        let got = digest_hex(artifact);
        if got != release.digest.0 {
            return Err(Error::DigestMismatch {
                expected: release.digest.0,
                got,
            });
        }
        let have = distinct_signers(&release);
        let need = self.config.quorum;
        if have < need {
            self.release = Some(release);
            let reason = format!("need {need} human signatures, have {have}");
            self.state = ReleaseState::Refused {
                reason: reason.clone(),
            };
            self.events.push(EVENT_REFUSE.to_string());
            return Err(Error::NoQuorum { have, need });
        }
        self.pin_shadow(&release.digest.0);
        self.release = Some(release);
        self.state = ReleaseState::Proposed;
        self.shadow_passed = false;
        self.events.push(EVENT_PROPOSE.to_string());
        Ok(())
    }

    pub fn start_shadow(&mut self) -> Result<()> {
        if !matches!(self.state, ReleaseState::Proposed) {
            return Err(Error::WrongState(format!(
                "start_shadow from {:?}",
                self.state
            )));
        }
        let digest = self.release.as_ref().map(|r| r.digest.0.clone());
        if let Some(d) = digest {
            self.pin_shadow(&d);
        }
        self.add_shadow();
        self.state = ReleaseState::Shadowing;
        self.events.push(EVENT_SHADOW.to_string());
        Ok(())
    }

    pub fn shadow_pass(&mut self) -> Result<()> {
        if !matches!(self.state, ReleaseState::Shadowing) {
            return Err(Error::WrongState(format!(
                "shadow_pass from {:?}",
                self.state
            )));
        }
        self.shadow_passed = true;
        self.events.push(EVENT_SHADOW_RESULT.to_string());
        Ok(())
    }

    pub fn shadow_fail(&mut self, reason: &str) -> Result<()> {
        if !matches!(self.state, ReleaseState::Shadowing) {
            return Err(Error::WrongState(format!(
                "shadow_fail from {:?}",
                self.state
            )));
        }
        self.events.push(EVENT_SHADOW_RESULT.to_string());
        self.unpin_shadow();
        self.shadow_passed = false;
        self.drop_shadow();
        self.reset_live_to_kernel();
        self.state = ReleaseState::RolledBack {
            reason: reason.to_string(),
        };
        Ok(())
    }

    fn apply_ready(&self) -> bool {
        self.shadow_passed
            && matches!(
                self.state,
                ReleaseState::Shadowing | ReleaseState::Rolling { .. }
            )
    }

    pub fn apply_one(&mut self, node: &NodeId) -> Result<()> {
        if !self.apply_ready() {
            return Err(Error::WrongState(
                "shadow has not passed; cannot apply".into(),
            ));
        }
        let version = self
            .release
            .as_ref()
            .ok_or_else(|| Error::WrongState("no proposed release".into()))?
            .version
            .clone();
        {
            let slot = self.find_live_mut(node)?;
            slot.applied = version;
            slot.healthy = true;
            slot.role = NodeRole::Live;
        }
        self.state = ReleaseState::Rolling { node: node.clone() };
        self.events.push(EVENT_APPLY.to_string());
        Ok(())
    }

    fn all_live_applied_healthy(&self) -> bool {
        let Some(rel) = &self.release else {
            return false;
        };
        let live: Vec<_> = self.live().collect();
        !live.is_empty()
            && live
                .iter()
                .all(|n| n.applied == rel.version && n.healthy && n.role == NodeRole::Live)
    }

    pub fn promote(&mut self) -> Result<()> {
        if !self.shadow_passed {
            return Err(Error::WrongState("shadow has not passed".into()));
        }
        if !self.all_live_applied_healthy() {
            return Err(Error::WrongState(
                "all live nodes must have applied and be healthy".into(),
            ));
        }
        let rel = self
            .release
            .as_ref()
            .ok_or_else(|| Error::WrongState("no proposed release".into()))?;
        let new_d = rel.digest.0.clone();
        let new_v = rel.version.clone();
        if let Some(cur) = self.pins.remove(PIN_CURRENT) {
            self.pins.insert(PIN_PREVIOUS.to_string(), cur);
        }
        self.pins.remove(PIN_SHADOW);
        self.pins.insert(PIN_CURRENT.to_string(), new_d);
        self.previous_version = self.kernel_version.clone();
        self.kernel_version = new_v;
        self.drop_shadow();
        self.state = ReleaseState::Current;
        self.shadow_passed = false;
        self.events.push(EVENT_PROMOTE.to_string());
        Ok(())
    }

    fn rollback_allowed(&self) -> bool {
        matches!(
            self.state,
            ReleaseState::Proposed
                | ReleaseState::Shadowing
                | ReleaseState::Rolling { .. }
                | ReleaseState::Current
        )
    }

    pub fn rollback(&mut self, reason: &str) -> Result<()> {
        if !self.rollback_allowed() {
            return Err(Error::WrongState(format!("rollback from {:?}", self.state)));
        }
        if matches!(self.state, ReleaseState::Current) {
            self.pins.remove(PIN_SHADOW);
            self.pins.remove(PIN_CURRENT);
            if let Some(prev) = self.pins.remove(PIN_PREVIOUS) {
                self.pins.insert(PIN_CURRENT.to_string(), prev);
            }
            self.kernel_version = self.previous_version.clone();
        } else {
            self.unpin_shadow();
        }
        self.shadow_passed = false;
        self.drop_shadow();
        self.reset_live_to_kernel();
        self.state = ReleaseState::RolledBack {
            reason: reason.to_string(),
        };
        self.events.push(EVENT_ROLLBACK.to_string());
        Ok(())
    }

    pub fn mark_unhealthy(&mut self, node: &NodeId, reason: &str) -> Result<()> {
        if !self.rollback_allowed() {
            return Err(Error::WrongState(format!(
                "mark_unhealthy from {:?}",
                self.state
            )));
        }
        {
            let slot = self.find_live_mut(node)?;
            slot.healthy = false;
        }
        self.events.push(EVENT_UNHEALTHY.to_string());
        self.rollback(reason)
    }
}

pub fn ref_status_text(status: &Status) -> String {
    let mut s = String::new();
    s.push_str("kernel_version: ");
    s.push_str(&status.kernel_version);
    s.push('\n');
    s.push_str("state: ");
    s.push_str(&state_label(&status.state));
    s.push('\n');
    match &status.release {
        Some(r) => {
            s.push_str("release: ");
            s.push_str(&r.version);
            s.push('\n');
        }
        None => s.push_str("release: none\n"),
    }
    s.push_str("nodes:\n");
    for n in &status.nodes {
        s.push_str(&format!(
            "  {} applied={} healthy={} role={}\n",
            n.id.0,
            n.applied,
            n.healthy,
            role_label(n.role)
        ));
    }
    s
}

fn state_label(st: &ReleaseState) -> String {
    match st {
        ReleaseState::Idle => "idle".into(),
        ReleaseState::Proposed => "proposed".into(),
        ReleaseState::Shadowing => "shadowing".into(),
        ReleaseState::Rolling { node } => format!("rolling:{}", node.0),
        ReleaseState::Current => "current".into(),
        ReleaseState::RolledBack { reason } => format!("rolled_back:{reason}"),
        ReleaseState::Refused { reason } => format!("refused:{reason}"),
    }
}

fn role_label(role: NodeRole) -> &'static str {
    match role {
        NodeRole::Live => "live",
        NodeRole::Shadow => "shadow",
    }
}

/// Parse `harness [--json] status` args (without argv0). `--json` may sit before
/// or after `status`.
pub fn ref_cli(args: &[String], refer: &RefOps) -> Result<String> {
    let mut json = false;
    let mut cmd: Option<&str> = None;
    for a in args {
        if a == "--json" {
            json = true;
            continue;
        }
        if cmd.is_some() {
            return Err(Error::Other(format!("unexpected argument {a}")));
        }
        cmd = Some(a.as_str());
    }
    match cmd {
        Some("status") => {
            if json {
                refer.status_json()
            } else {
                Ok(refer.status_text())
            }
        }
        Some(other) => Err(Error::Other(format!("unknown command {other}"))),
        None => Err(Error::Other("missing command".into())),
    }
}
