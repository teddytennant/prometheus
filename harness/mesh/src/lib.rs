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

use prometheus_leases::{Lease, Queue};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const MIN_WATCHERS: usize = 2;
pub const MIN_RING: usize = 3;

pub type NowMs = u64;

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
}

impl Mesh {
    /// Fails if `dir` exists.
    pub fn create(_dir: impl AsRef<Path>, _config: MeshConfig) -> Result<Self> {
        unimplemented!("H5: Mesh::create")
    }

    pub fn open(_dir: impl AsRef<Path>, _config: MeshConfig) -> Result<Self> {
        unimplemented!("H5: Mesh::open")
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config(&self) -> &MeshConfig {
        &self.config
    }

    pub fn advertise(&mut self, _caps: Capabilities, _now: NowMs) -> Result<Advert> {
        unimplemented!("H5: Mesh::advertise")
    }

    pub fn adverts(&self) -> &[Advert] {
        unimplemented!("H5: Mesh::adverts")
    }

    /// Claim one H3 task that this node's last advert can run. `None` if the
    /// queue is empty or nothing fits. Untrusted nodes get [`Error::Untrusted`].
    /// Frozen nodes get [`Error::Frozen`].
    pub fn pull(&mut self, _queue: &mut Queue, _now: NowMs) -> Result<Option<Lease>> {
        unimplemented!("H5: Mesh::pull")
    }

    /// Honor a signed freeze. Partitioned nodes apply it on reconnect.
    pub fn freeze(&mut self, _cmd: Freeze) -> Result<()> {
        unimplemented!("H5: Mesh::freeze")
    }

    pub fn is_frozen(&self) -> bool {
        unimplemented!("H5: Mesh::is_frozen")
    }

    /// Watcher ring: each live node is a watchee of exactly [`MIN_WATCHERS`]
    /// other live nodes when `len >= MIN_RING`.
    pub fn watches(&self) -> &[Watch] {
        unimplemented!("H5: Mesh::watches")
    }
}

/// In-process mesh for CPU tests. Not production QUIC.
pub struct LocalMesh {
    nodes: Vec<Mesh>,
}

impl LocalMesh {
    /// `n` trusted Always-on slots, ids `"0".."n-1"`. Errors if `n < MIN_RING`.
    pub fn start(_n: usize, _base_dir: impl AsRef<Path>) -> Result<Self> {
        unimplemented!("H5: LocalMesh::start")
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

    pub fn partition(&mut self, _isolated: &[usize]) -> Result<()> {
        unimplemented!("H5: LocalMesh::partition")
    }

    pub fn heal(&mut self) -> Result<()> {
        unimplemented!("H5: LocalMesh::heal")
    }

    /// Drive advert gossip and apply pending freeze on reconnect.
    pub fn tick(&mut self, _now: NowMs) -> Result<()> {
        unimplemented!("H5: LocalMesh::tick")
    }
}
