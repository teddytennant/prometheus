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

use prometheus_cas::{Digest, PinName, Store};
use prometheus_log::{Append, Event, EventLog};
use prometheus_raft::{Cluster, NodeId, MIN_VOTERS};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest as _, Sha256};
use std::collections::HashSet;
use std::fs;
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

/// Identity of the shadow node. `apply_one` has no `Cluster`, so live
/// membership is the H4 always-on set (`"0"`..`MIN_VOTERS-1`); this id is
/// reserved and never a live member.
const SHADOW_NODE_ID: &str = "shadow";
const BOOTSTRAP_KERNEL: &str = "0";

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
#[serde(rename_all = "snake_case")]
pub struct Status {
    pub kernel_version: String,
    pub nodes: Vec<NodeStatus>,
    pub release: Option<Release>,
    pub state: ReleaseState,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("wrong state: {0}")]
    WrongState(String),
    #[error("no quorum: have {have}, need {need}")]
    NoQuorum { have: usize, need: usize },
    #[error("digest mismatch: expected {expected} got {got}")]
    DigestMismatch { expected: String, got: String },
    #[error("refused: {0}")]
    Refused(String),
    #[error("bad release: {0}")]
    BadRelease(String),
    #[error("cas: {0}")]
    Cas(String),
    #[error("raft: {0}")]
    Raft(String),
    #[error("{0}")]
    Log(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<prometheus_log::Error> for Error {
    fn from(err: prometheus_log::Error) -> Self {
        Error::Log(err.to_string())
    }
}

impl From<prometheus_cas::Error> for Error {
    fn from(err: prometheus_cas::Error) -> Self {
        match err {
            prometheus_cas::Error::NotFound(s) => Error::NotFound(s),
            prometheus_cas::Error::DigestMismatch { expected, got } => {
                Error::DigestMismatch { expected, got }
            }
            other => Error::Cas(other.to_string()),
        }
    }
}

impl From<prometheus_raft::Error> for Error {
    fn from(err: prometheus_raft::Error) -> Self {
        Error::Raft(err.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Other(err.to_string())
    }
}

/// Distinct humans with non-empty signature bytes. Duplicates count once.
pub fn has_quorum(release: &Release, quorum: usize) -> bool {
    quorum_count(release) >= quorum
}

fn quorum_count(release: &Release) -> usize {
    let mut seen = HashSet::new();
    for sig in &release.signatures {
        if !sig.bytes.is_empty() {
            seen.insert(&sig.human);
        }
    }
    seen.len()
}

/// Rollout state machine. Does not own the CAS or Raft group; callers pass
/// them in. Create fails if `dir` exists.
pub struct Ops {
    dir: PathBuf,
    config: OpsConfig,
    log: EventLog,
    state: ReleaseState,
    release: Option<Release>,
    nodes: Vec<NodeStatus>,
    shadow_passed: bool,
    kernel_version: String,
    previous_version: String,
    current_digest: Option<String>,
    previous_digest: Option<String>,
}

impl Ops {
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn log(&self) -> &EventLog {
        &self.log
    }

    pub fn config(&self) -> &OpsConfig {
        &self.config
    }

    /// Fails if `dir` exists. Creates `dir/log/` as an H2 EventLog.
    pub fn create(dir: impl AsRef<Path>, config: OpsConfig) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        if dir.exists() {
            return Err(Error::Other(format!("already exists: {}", dir.display())));
        }
        fs::create_dir(&dir)?;
        let log = match EventLog::create(dir.join("log")) {
            Ok(log) => log,
            Err(err) => {
                let _ = fs::remove_dir_all(&dir);
                return Err(err.into());
            }
        };
        Ok(Self::fresh(dir, config, log))
    }

    /// Replay rollout state from `dir/log/`.
    pub fn open(dir: impl AsRef<Path>, config: OpsConfig) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let log = EventLog::open(dir.join("log"))?;
        let events: Vec<Event> = log.iter().cloned().collect();
        let mut ops = Self::fresh(dir, config, log);
        for event in events {
            ops.apply_event(&event)?;
        }
        Ok(ops)
    }

    fn fresh(dir: PathBuf, config: OpsConfig, log: EventLog) -> Self {
        Self {
            dir,
            config,
            log,
            state: ReleaseState::Idle,
            release: None,
            nodes: bootstrap_nodes(),
            shadow_passed: false,
            kernel_version: BOOTSTRAP_KERNEL.to_string(),
            previous_version: BOOTSTRAP_KERNEL.to_string(),
            current_digest: None,
            previous_digest: None,
        }
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

    fn apply_ready(&self) -> bool {
        self.shadow_passed
            && matches!(
                self.state,
                ReleaseState::Shadowing | ReleaseState::Rolling { .. }
            )
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

    /// Put + pin `kernel.shadow` if signatures meet quorum and digest matches.
    /// No quorum → refuse event, no put. Digest mismatch → error, no event.
    pub fn propose(
        &mut self,
        release: Release,
        artifact: &[u8],
        store: &mut Store,
        now: NowMs,
    ) -> Result<()> {
        if !self.propose_allowed() {
            return Err(self.wrong_state("idle|current|refused|rolled_back"));
        }
        let got = sha256_hex(artifact);
        if got != release.digest.0 {
            return Err(Error::DigestMismatch {
                expected: release.digest.0.clone(),
                got,
            });
        }
        let have = quorum_count(&release);
        let need = self.config.quorum;
        if have < need {
            let reason = format!("need {need} human signatures, have {have}");
            let payload = json!({
                "release": release,
                "reason": reason,
                "have": have,
                "need": need,
            });
            let event = self.append(EVENT_REFUSE, payload, now)?;
            self.apply_event(&event)?;
            return Err(Error::NoQuorum { have, need });
        }
        store.put(artifact)?;
        store.pin(&release.digest, PinName(PIN_SHADOW.to_string()))?;
        let payload =
            serde_json::to_value(&release).map_err(|err| Error::Other(err.to_string()))?;
        let event = self.append(EVENT_PROPOSE, payload, now)?;
        self.apply_event(&event)?;
        Ok(())
    }

    /// Pin the proposed artifact as shadow. Does not change live kernel_version.
    pub fn start_shadow(&mut self, store: &mut Store, now: NowMs) -> Result<()> {
        if !matches!(self.state, ReleaseState::Proposed) {
            return Err(self.wrong_state("proposed"));
        }
        if let Some(release) = &self.release {
            pin_ignore_duplicate(store, &release.digest, PIN_SHADOW)?;
        }
        let event = self.append(EVENT_SHADOW, json!({ "node": SHADOW_NODE_ID }), now)?;
        self.apply_event(&event)?;
        Ok(())
    }

    pub fn shadow_pass(&mut self, now: NowMs) -> Result<()> {
        if !matches!(self.state, ReleaseState::Shadowing) {
            return Err(self.wrong_state("shadowing"));
        }
        let event = self.append(EVENT_SHADOW_RESULT, json!({ "pass": true }), now)?;
        self.apply_event(&event)?;
        Ok(())
    }

    /// Automatic rollback of the shadow pin.
    pub fn shadow_fail(&mut self, reason: &str, store: &mut Store, now: NowMs) -> Result<()> {
        if !matches!(self.state, ReleaseState::Shadowing) {
            return Err(self.wrong_state("shadowing"));
        }
        unpin_ignore_missing(store, PIN_SHADOW)?;
        let event = self.append(
            EVENT_SHADOW_RESULT,
            json!({ "pass": false, "reason": reason }),
            now,
        )?;
        self.apply_event(&event)?;
        Ok(())
    }

    /// Apply the proposed version to one live node. WrongState until shadow_pass.
    pub fn apply_one(&mut self, node: &NodeId, now: NowMs) -> Result<()> {
        if !self.apply_ready() {
            return Err(self.wrong_state("shadow_pass"));
        }
        if self
            .nodes
            .iter()
            .any(|n| n.id == *node && n.role == NodeRole::Shadow)
        {
            return Err(self.wrong_state("live"));
        }
        if !self
            .nodes
            .iter()
            .any(|n| n.id == *node && n.role == NodeRole::Live)
        {
            return Err(Error::NotFound(node.0.clone()));
        }
        let event = self.append(EVENT_APPLY, json!({ "node": &node.0 }), now)?;
        self.apply_event(&event)?;
        Ok(())
    }

    /// Pin previous=current, current=new. All live nodes must have applied.
    pub fn promote(&mut self, cluster: &mut Cluster, store: &mut Store, now: NowMs) -> Result<()> {
        if !self.shadow_passed {
            return Err(self.wrong_state("shadow_pass"));
        }
        if !self.all_live_applied_healthy() {
            return Err(self.wrong_state("all live applied and healthy"));
        }
        let Some(release) = self.release.clone() else {
            return Err(self.wrong_state("release"));
        };
        self.promote_store(store, &release)?;
        cluster.set_kernel_version(release.version.clone())?;
        let event = self.append(
            EVENT_PROMOTE,
            json!({ "version": release.version }),
            now,
        )?;
        self.apply_event(&event)?;
        Ok(())
    }

    /// Restore previous pin and previous kernel_version (only if already Current).
    /// In-progress rollouts only drop the shadow pin; live kernel is unchanged.
    pub fn rollback(
        &mut self,
        reason: &str,
        cluster: &mut Cluster,
        store: &mut Store,
        now: NowMs,
    ) -> Result<()> {
        if !self.rollback_allowed() {
            return Err(self.wrong_state("proposed|shadowing|rolling|current"));
        }
        let from_current = matches!(self.state, ReleaseState::Current);
        self.rollback_store(store, from_current)?;
        if from_current {
            cluster.set_kernel_version(self.previous_version.clone())?;
        }
        let event = self.append(
            EVENT_ROLLBACK,
            json!({ "reason": reason, "from_current": from_current }),
            now,
        )?;
        self.apply_event(&event)?;
        Ok(())
    }

    pub fn mark_unhealthy(
        &mut self,
        node: &NodeId,
        reason: &str,
        cluster: &mut Cluster,
        store: &mut Store,
        now: NowMs,
    ) -> Result<()> {
        if !self.rollback_allowed() {
            return Err(self.wrong_state("proposed|shadowing|rolling|current"));
        }
        if self
            .nodes
            .iter()
            .any(|n| n.id == *node && n.role == NodeRole::Shadow)
        {
            return Err(self.wrong_state("live"));
        }
        let Some(slot) = self
            .nodes
            .iter_mut()
            .find(|n| n.id == *node && n.role == NodeRole::Live)
        else {
            return Err(Error::NotFound(node.0.clone()));
        };
        slot.healthy = false;
        let event = self.append(
            EVENT_UNHEALTHY,
            json!({ "node": &node.0, "reason": reason }),
            now,
        )?;
        self.apply_event(&event)?;
        self.rollback(reason, cluster, store, now)
    }

    pub fn status(&self, cluster: &Cluster) -> Status {
        let kernel_version = cluster.state().kernel_version.clone();
        let mut nodes: Vec<NodeStatus> = cluster
            .state()
            .membership
            .iter()
            .map(|member| {
                let found = self
                    .nodes
                    .iter()
                    .find(|n| n.id == member.id && n.role == NodeRole::Live);
                NodeStatus {
                    id: member.id.clone(),
                    applied: found
                        .map(|n| n.applied.clone())
                        .unwrap_or_else(|| kernel_version.clone()),
                    healthy: found.map(|n| n.healthy).unwrap_or(true),
                    role: NodeRole::Live,
                }
            })
            .collect();
        for node in &self.nodes {
            if node.role == NodeRole::Shadow {
                nodes.push(node.clone());
            }
        }
        Status {
            kernel_version,
            nodes,
            release: self.release.clone(),
            state: self.state.clone(),
        }
    }

    pub fn status_json(&self, cluster: &Cluster) -> Result<String> {
        serde_json::to_string(&self.status(cluster)).map_err(|err| Error::Other(err.to_string()))
    }

    pub fn status_text(&self, cluster: &Cluster) -> String {
        status_text(&self.status(cluster))
    }

    fn append(&mut self, event_type: &str, payload: Value, now: NowMs) -> Result<Event> {
        self.log
            .append(Append {
                event_type: event_type.to_string(),
                payload,
                timestamp: timestamp(now),
                task_id: None,
                attempt: None,
                node_id: None,
            })
            .map_err(Into::into)
    }

    fn apply_event(&mut self, event: &Event) -> Result<()> {
        match event.event_type.as_str() {
            EVENT_PROPOSE => {
                let release: Release = serde_json::from_value(event.payload.clone())
                    .map_err(|err| Error::Other(err.to_string()))?;
                self.release = Some(release);
                self.shadow_passed = false;
                self.state = ReleaseState::Proposed;
            }
            EVENT_REFUSE => {
                if let Some(value) = event.payload.get("release") {
                    self.release = Some(
                        serde_json::from_value(value.clone())
                            .map_err(|err| Error::Other(err.to_string()))?,
                    );
                }
                let reason = payload_str(&event.payload, "reason")
                    .unwrap_or("no quorum")
                    .to_string();
                self.state = ReleaseState::Refused { reason };
            }
            EVENT_SHADOW => {
                self.push_shadow();
                self.state = ReleaseState::Shadowing;
            }
            EVENT_SHADOW_RESULT => {
                let pass = event
                    .payload
                    .get("pass")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if pass {
                    self.shadow_passed = true;
                } else {
                    let reason = payload_str(&event.payload, "reason").unwrap_or("shadow failed");
                    self.shadow_fail_memory(reason);
                }
            }
            EVENT_APPLY => {
                let id = payload_str(&event.payload, "node").unwrap_or("");
                let node = NodeId(id.to_string());
                let version = self
                    .release
                    .as_ref()
                    .map(|r| r.version.clone())
                    .unwrap_or_default();
                if let Some(found) = self
                    .nodes
                    .iter_mut()
                    .find(|n| n.id == node && n.role == NodeRole::Live)
                {
                    found.applied = version;
                    found.healthy = true;
                }
                self.state = ReleaseState::Rolling { node };
            }
            EVENT_PROMOTE => {
                if let Some(release) = &self.release {
                    self.previous_version = self.kernel_version.clone();
                    self.kernel_version = release.version.clone();
                    if let Some(current) = self.current_digest.take() {
                        self.previous_digest = Some(current);
                    }
                    self.current_digest = Some(release.digest.0.clone());
                }
                self.drop_shadow();
                self.shadow_passed = false;
                self.state = ReleaseState::Current;
            }
            EVENT_ROLLBACK => {
                let reason = payload_str(&event.payload, "reason").unwrap_or("rollback");
                let from_current = event
                    .payload
                    .get("from_current")
                    .and_then(Value::as_bool)
                    .unwrap_or(matches!(self.state, ReleaseState::Current));
                self.rollback_memory(reason, from_current);
            }
            EVENT_UNHEALTHY => {
                if let Some(id) = payload_str(&event.payload, "node") {
                    let node = NodeId(id.to_string());
                    if let Some(found) = self
                        .nodes
                        .iter_mut()
                        .find(|n| n.id == node && n.role == NodeRole::Live)
                    {
                        found.healthy = false;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn promote_store(&self, store: &mut Store, release: &Release) -> Result<()> {
        unpin_ignore_missing(store, PIN_SHADOW)?;
        if let Some(current) = &self.current_digest {
            unpin_ignore_missing(store, PIN_CURRENT)?;
            unpin_ignore_missing(store, PIN_PREVIOUS)?;
            store.pin(
                &Digest(current.clone()),
                PinName(PIN_PREVIOUS.to_string()),
            )?;
        }
        store.pin(&release.digest, PinName(PIN_CURRENT.to_string()))?;
        Ok(())
    }

    fn rollback_store(&self, store: &mut Store, from_current: bool) -> Result<()> {
        if from_current {
            unpin_ignore_missing(store, PIN_SHADOW)?;
            unpin_ignore_missing(store, PIN_CURRENT)?;
            if let Some(previous) = &self.previous_digest {
                unpin_ignore_missing(store, PIN_PREVIOUS)?;
                store.pin(
                    &Digest(previous.clone()),
                    PinName(PIN_CURRENT.to_string()),
                )?;
            }
        } else {
            unpin_ignore_missing(store, PIN_SHADOW)?;
        }
        Ok(())
    }

    fn rollback_memory(&mut self, reason: &str, from_current: bool) {
        if from_current {
            if let Some(previous) = self.previous_digest.take() {
                self.current_digest = Some(previous);
            } else {
                self.current_digest = None;
            }
            self.kernel_version = self.previous_version.clone();
        }
        self.shadow_passed = false;
        self.drop_shadow();
        self.reset_live_to_kernel();
        self.state = ReleaseState::RolledBack {
            reason: reason.to_string(),
        };
    }

    fn shadow_fail_memory(&mut self, reason: &str) {
        self.shadow_passed = false;
        self.drop_shadow();
        self.reset_live_to_kernel();
        self.state = ReleaseState::RolledBack {
            reason: reason.to_string(),
        };
    }

    fn reset_live_to_kernel(&mut self) {
        let kernel = self.kernel_version.clone();
        for node in &mut self.nodes {
            if node.role == NodeRole::Live {
                node.applied = kernel.clone();
                node.healthy = true;
            }
        }
    }

    fn push_shadow(&mut self) {
        if self.nodes.iter().any(|n| n.role == NodeRole::Shadow) {
            return;
        }
        let applied = self
            .release
            .as_ref()
            .map(|r| r.version.clone())
            .unwrap_or_else(|| self.kernel_version.clone());
        self.nodes.push(NodeStatus {
            id: NodeId(SHADOW_NODE_ID.to_string()),
            applied,
            healthy: true,
            role: NodeRole::Shadow,
        });
    }

    fn drop_shadow(&mut self) {
        self.nodes.retain(|n| n.role != NodeRole::Shadow);
    }

    fn all_live_applied_healthy(&self) -> bool {
        let Some(release) = &self.release else {
            return false;
        };
        let mut any = false;
        for node in &self.nodes {
            if node.role == NodeRole::Live {
                any = true;
                if node.applied != release.version || !node.healthy {
                    return false;
                }
            }
        }
        any
    }

    fn wrong_state(&self, needed: &str) -> Error {
        Error::WrongState(format!(
            "current {}, needed {needed}",
            state_name(&self.state)
        ))
    }
}

/// `--json` may appear before or after the subcommand.
pub fn cli(args: &[String], ops: &Ops, cluster: &Cluster) -> Result<String> {
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
                ops.status_json(cluster)
            } else {
                Ok(ops.status_text(cluster))
            }
        }
        Some(other) => Err(Error::Other(format!("unknown command {other}"))),
        None => Err(Error::Other("missing command".into())),
    }
}

fn bootstrap_nodes() -> Vec<NodeStatus> {
    (0..MIN_VOTERS)
        .map(|i| NodeStatus {
            id: NodeId(i.to_string()),
            applied: BOOTSTRAP_KERNEL.to_string(),
            healthy: true,
            role: NodeRole::Live,
        })
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn timestamp(now: NowMs) -> String {
    let secs = now / 1000;
    let millis = now % 1000;
    format!("{secs}.{millis:03}Z")
}

fn payload_str<'a>(payload: &'a Value, key: &str) -> Option<&'a str> {
    payload.get(key).and_then(Value::as_str)
}

fn state_name(state: &ReleaseState) -> String {
    match state {
        ReleaseState::Idle => "idle".into(),
        ReleaseState::Proposed => "proposed".into(),
        ReleaseState::Shadowing => "shadowing".into(),
        ReleaseState::Rolling { node } => format!("rolling:{}", node.0),
        ReleaseState::Current => "current".into(),
        ReleaseState::RolledBack { reason } => format!("rolled_back:{reason}"),
        ReleaseState::Refused { reason } => format!("refused:{reason}"),
    }
}

fn status_text(status: &Status) -> String {
    let mut s = String::new();
    s.push_str("kernel_version: ");
    s.push_str(&status.kernel_version);
    s.push('\n');
    s.push_str("state: ");
    s.push_str(&state_name(&status.state));
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
        let role = match n.role {
            NodeRole::Live => "live",
            NodeRole::Shadow => "shadow",
        };
        s.push_str(&format!(
            "  {} applied={} healthy={} role={}\n",
            n.id.0, n.applied, n.healthy, role
        ));
    }
    s
}

fn unpin_ignore_missing(store: &mut Store, name: &str) -> Result<()> {
    match store.unpin(&PinName(name.to_string())) {
        Ok(()) => Ok(()),
        Err(prometheus_cas::Error::NotFound(_)) => Ok(()),
        Err(err) => Err(err.into()),
    }
}

fn pin_ignore_duplicate(store: &mut Store, digest: &Digest, name: &str) -> Result<()> {
    match store.pin(digest, PinName(name.to_string())) {
        Ok(()) => Ok(()),
        Err(prometheus_cas::Error::DuplicatePin(_)) => Ok(()),
        Err(err) => Err(err.into()),
    }
}
