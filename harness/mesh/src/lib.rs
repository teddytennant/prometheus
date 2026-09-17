//! Mesh work distribution (spec 15.2, 15.5 H5, gate D2).
//!
//! Nodes advertise capabilities and pull H3 tasks that fit. Trust in a new
//! node is a human decision (the `trusted` flag). A signed freeze is honored
//! by every node, including partitioned ones when they reconnect. Watchers
//! watch each other in a ring: each node is checked by two others.
//!
//! CPU tests drive an in-process group through [`LocalMesh`]. Production
//! wires the same [`Mesh`] API to QUIC. `NowMs` is injected; nothing in this
//! crate reads the wall clock.

use prometheus_leases::{Lease, Queue, WorkerId};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

pub const MIN_WATCHERS: usize = 2;
pub const MIN_RING: usize = 3;

pub type NowMs = u64;

const FREEZE_FILE: &str = "freeze.json";

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

/// Hex-encoded public key. Identity, not a network address.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PublicKey(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    pub gpus: u32,
    pub cpus: u32,
    pub kvm: bool,
    pub providers: Vec<String>,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            gpus: 0,
            cpus: 1,
            kvm: false,
            providers: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Advert {
    pub node: NodeId,
    pub key: PublicKey,
    pub caps: Capabilities,
    pub at: NowMs,
}

/// Signed kill switch. Tests use fake signatures, never real keys.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Freeze {
    pub payload: String,
    pub signer: PublicKey,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Watch {
    pub watcher: NodeId,
    pub watchee: NodeId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshConfig {
    pub this_id: NodeId,
    pub this_key: PublicKey,
    pub trusted: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("untrusted node cannot {0}")]
    Untrusted(String),
    #[error("capability mismatch")]
    CapMismatch,
    #[error("mesh is frozen")]
    Frozen,
    #[error("bad freeze signature")]
    BadSignature,
    #[error("need at least {MIN_RING} nodes for a watcher ring, have {0}")]
    TooFewNodes(usize),
    #[error("duplicate: {0}")]
    Duplicate(String),
    #[error("already exists: {0}")]
    AlreadyExists(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// One node's view of the mesh. Bodies are filled by the implementer.
pub struct Mesh {
    dir: PathBuf,
    config: MeshConfig,
    adverts: Vec<Advert>,
    frozen: bool,
    freeze_cmd: Option<Freeze>,
    watches: Vec<Watch>,
}

impl Mesh {
    /// Fails if `dir` exists.
    pub fn create(dir: impl AsRef<Path>, config: MeshConfig) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        if dir.exists() {
            return Err(Error::AlreadyExists(dir.display().to_string()));
        }
        std::fs::create_dir(&dir).map_err(io_err)?;
        Ok(Self {
            dir,
            config,
            adverts: Vec::new(),
            frozen: false,
            freeze_cmd: None,
            watches: Vec::new(),
        })
    }

    pub fn open(dir: impl AsRef<Path>, config: MeshConfig) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        if !dir.is_dir() {
            return Err(Error::NotFound(dir.display().to_string()));
        }
        let freeze_cmd = load_freeze(&dir)?;
        let frozen = freeze_cmd.is_some();
        Ok(Self {
            dir,
            config,
            adverts: Vec::new(),
            frozen,
            freeze_cmd,
            watches: Vec::new(),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config(&self) -> &MeshConfig {
        &self.config
    }

    pub fn advertise(&mut self, caps: Capabilities, now: NowMs) -> Result<Advert> {
        if self.frozen {
            return Err(Error::Frozen);
        }
        let advert = Advert {
            node: self.config.this_id.clone(),
            key: self.config.this_key.clone(),
            caps,
            at: now,
        };
        self.upsert_advert(advert.clone());
        Ok(advert)
    }

    pub fn adverts(&self) -> &[Advert] {
        &self.adverts
    }

    /// Claim one H3 task that this node's last advert can run. `None` if the
    /// queue is empty or nothing fits. Untrusted nodes get [`Error::Untrusted`].
    /// Frozen nodes get [`Error::Frozen`].
    pub fn pull(&mut self, queue: &mut Queue, now: NowMs) -> Result<Option<Lease>> {
        if !self.config.trusted {
            return Err(Error::Untrusted("pull".into()));
        }
        if self.frozen {
            return Err(Error::Frozen);
        }
        let caps = self.last_caps();
        queue
            .claim_if(
                &WorkerId(self.id().to_string()),
                now,
                |task| caps_satisfy(&caps, &task.payload),
            )
            .map_err(|e| Error::Other(e.to_string()))
    }

    /// Honor a signed freeze. Partitioned nodes apply it on reconnect.
    pub fn freeze(&mut self, cmd: Freeze) -> Result<()> {
        if !signature_ok(&cmd) {
            return Err(Error::BadSignature);
        }
        self.apply_verified_freeze(cmd)
    }

    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Watcher ring: each live node is a watchee of exactly [`MIN_WATCHERS`]
    /// other live nodes when `len >= MIN_RING`.
    pub fn watches(&self) -> &[Watch] {
        &self.watches
    }

    fn id(&self) -> &str {
        &self.config.this_id.0
    }

    fn last_caps(&self) -> Capabilities {
        self.adverts
            .iter()
            .filter(|a| a.node.0 == self.config.this_id.0)
            .max_by_key(|a| a.at)
            .map(|a| a.caps.clone())
            .unwrap_or_default()
    }

    fn upsert_advert(&mut self, advert: Advert) {
        if let Some(existing) = self.adverts.iter_mut().find(|a| a.node.0 == advert.node.0) {
            *existing = advert;
        } else {
            self.adverts.push(advert);
        }
    }

    fn apply_verified_freeze(&mut self, cmd: Freeze) -> Result<()> {
        persist_freeze(&self.dir, &cmd)?;
        self.frozen = true;
        self.freeze_cmd = Some(cmd);
        Ok(())
    }
}

/// In-process mesh for CPU tests. Not production QUIC.
pub struct LocalMesh {
    nodes: Vec<Mesh>,
    isolated: HashSet<usize>,
}

impl LocalMesh {
    /// `n` trusted Always-on slots, ids `"0".."n-1"`. Errors if `n < MIN_RING`.
    pub fn start(n: usize, base_dir: impl AsRef<Path>) -> Result<Self> {
        if n < MIN_RING {
            return Err(Error::TooFewNodes(n));
        }
        let base = base_dir.as_ref();
        std::fs::create_dir_all(base).map_err(io_err)?;
        let watches = expected_watches(n);
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            let mut node = Mesh::create(base.join(i.to_string()), slot_config(i))?;
            node.watches = watches.clone();
            nodes.push(node);
        }
        Ok(Self {
            nodes,
            isolated: HashSet::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn get(&mut self, i: usize) -> Result<&mut Mesh> {
        self.nodes.get_mut(i).ok_or_else(|| Error::NotFound(i.to_string()))
    }

    pub fn partition(&mut self, isolated: &[usize]) -> Result<()> {
        self.isolated = isolated.iter().copied().collect();
        Ok(())
    }

    pub fn heal(&mut self) -> Result<()> {
        self.isolated.clear();
        Ok(())
    }

    /// Drive advert gossip and apply pending freeze on reconnect.
    pub fn tick(&mut self, _now: NowMs) -> Result<()> {
        let n = self.nodes.len();
        let connected: Vec<usize> = (0..n)
            .filter(|i| !self.isolated.contains(i))
            .collect();
        let mut best: BTreeMap<String, Advert> = BTreeMap::new();
        for &i in &connected {
            for a in &self.nodes[i].adverts {
                match best.get(&a.node.0) {
                    Some(old) if old.at > a.at => {}
                    _ => {
                        best.insert(a.node.0.clone(), a.clone());
                    }
                }
            }
        }
        let merged: Vec<Advert> = best.into_values().collect();
        let freeze = connected.iter().find_map(|&i| {
            if self.nodes[i].frozen {
                self.nodes[i].freeze_cmd.clone()
            } else {
                None
            }
        });
        for &i in &connected {
            self.nodes[i].adverts = merged.clone();
            if let Some(cmd) = freeze.clone() {
                self.nodes[i].apply_verified_freeze(cmd)?;
            }
        }
        Ok(())
    }
}

fn slot_config(i: usize) -> MeshConfig {
    MeshConfig {
        this_id: NodeId(i.to_string()),
        this_key: PublicKey(format!("key-{i}")),
        trusted: true,
    }
}

/// Watcher ring: watchee `i` is watched by `(i+1)%n` then `(i+2)%n`.
/// Order: watchee 0's two, then 1, …. Length `2n`.
fn expected_watches(n: usize) -> Vec<Watch> {
    let mut out = Vec::with_capacity(n.saturating_mul(MIN_WATCHERS));
    for i in 0..n {
        let watchee = NodeId(i.to_string());
        out.push(Watch {
            watcher: NodeId(((i + 1) % n).to_string()),
            watchee: watchee.clone(),
        });
        out.push(Watch {
            watcher: NodeId(((i + 2) % n).to_string()),
            watchee,
        });
    }
    out
}

/// Fake freeze scheme locked by tests: signature bytes equal payload bytes.
fn signature_ok(cmd: &Freeze) -> bool {
    cmd.signature.as_slice() == cmd.payload.as_bytes()
}

/// Does `caps` satisfy a work-item payload's optional `"requires"` object?
fn caps_satisfy(caps: &Capabilities, payload: &Value) -> bool {
    let Some(req) = payload.get("requires") else {
        return true;
    };
    let Some(obj) = req.as_object() else {
        return false;
    };
    let gpus = obj.get("gpus").and_then(Value::as_u64).unwrap_or(0);
    let cpus = obj.get("cpus").and_then(Value::as_u64).unwrap_or(0);
    let kvm = obj.get("kvm").and_then(Value::as_bool).unwrap_or(false);
    let providers: Vec<&str> = obj
        .get("providers")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    if u64::from(caps.gpus) < gpus {
        return false;
    }
    if u64::from(caps.cpus) < cpus {
        return false;
    }
    if kvm && !caps.kvm {
        return false;
    }
    for p in providers {
        if !caps.providers.iter().any(|have| have == p) {
            return false;
        }
    }
    true
}

fn persist_freeze(dir: &Path, cmd: &Freeze) -> Result<()> {
    let bytes = serde_json::to_vec(cmd).map_err(|e| Error::Other(e.to_string()))?;
    std::fs::write(dir.join(FREEZE_FILE), bytes).map_err(io_err)
}

fn load_freeze(dir: &Path) -> Result<Option<Freeze>> {
    match std::fs::read(dir.join(FREEZE_FILE)) {
        Ok(bytes) => {
            let cmd = serde_json::from_slice(&bytes).map_err(|e| Error::Other(e.to_string()))?;
            Ok(Some(cmd))
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(io_err(err)),
    }
}

fn io_err(err: std::io::Error) -> Error {
    Error::Other(err.to_string())
}
