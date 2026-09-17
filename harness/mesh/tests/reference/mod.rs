//! In-memory reference for H5 mesh work distribution.
//!
//! Slow and obvious. Production must never import this module. The matching
//! rules below are the oracle for `pull` capability checks and the watcher ring.

#![allow(dead_code)]

use prometheus_leases::{Lease, Queue, TaskId, TaskState, WorkerId, EVENT_ENQUEUED};
use prometheus_mesh::{
    Advert, Capabilities, Error, Freeze, MeshConfig, NodeId, NowMs, PublicKey, Result, Watch,
    MIN_RING, MIN_WATCHERS,
};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

/// Does `caps` satisfy a work-item payload's optional `"requires"` object?
///
/// Missing `requires` → any node. Non-object `requires` → unsatisfiable.
/// `gpus`/`cpus`: node value must be >= required (missing required field → 0).
/// `kvm`: required true means node.kvm must be true; required false is a no-op.
/// `providers`: node must include every listed string; extras on the node are ok.
pub fn caps_satisfy(caps: &Capabilities, payload: &Value) -> bool {
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

/// Fake freeze scheme locked by tests: signature bytes equal payload bytes.
pub fn signature_ok(cmd: &Freeze) -> bool {
    cmd.signature.as_slice() == cmd.payload.as_bytes()
}

/// Watcher ring: watchee i is watched by (i+1)%n then (i+2)%n.
/// Order: watchee 0's two, then 1, .... Length 2n for n >= MIN_RING.
pub fn expected_watches(n: usize) -> Vec<Watch> {
    assert!(
        n >= MIN_RING,
        "ring is only defined for n >= MIN_RING ({MIN_RING})"
    );
    assert_eq!(MIN_WATCHERS, 2, "ring construction uses two watchers");
    let mut out = Vec::with_capacity(n.saturating_mul(MIN_WATCHERS));
    for i in 0..n {
        let watchee = NodeId(i.to_string());
        let w1 = NodeId(((i + 1) % n).to_string());
        let w2 = NodeId(((i + 2) % n).to_string());
        out.push(Watch {
            watcher: w1,
            watchee: watchee.clone(),
        });
        out.push(Watch {
            watcher: w2,
            watchee,
        });
    }
    out
}

/// Queued task ids in enqueue order (H3 log). Expired-requeue order is not
/// modeled; H5 tests do not expire.
pub fn queued_in_order(queue: &Queue) -> Vec<TaskId> {
    let mut ids = Vec::new();
    for event in queue.log().iter() {
        if event.event_type != EVENT_ENQUEUED {
            continue;
        }
        if let Some(id) = event.task_id.clone() {
            ids.push(TaskId(id));
        }
    }
    ids.into_iter()
        .filter(|id| {
            queue
                .get(id)
                .map(|t| t.state == TaskState::Queued)
                .unwrap_or(false)
        })
        .collect()
}

/// First queued task `caps` can run, skipping mismatches. None if nothing fits.
pub fn first_fitting(queue: &Queue, caps: &Capabilities) -> Option<TaskId> {
    for id in queued_in_order(queue) {
        let task = queue.get(&id).expect("queued id");
        if caps_satisfy(caps, &task.payload) {
            return Some(id);
        }
    }
    None
}

fn slot_config(i: usize) -> MeshConfig {
    MeshConfig {
        this_id: NodeId(i.to_string()),
        this_key: PublicKey(format!("key-{i}")),
        trusted: true,
    }
}

/// In-memory mesh node. Disk is used only by [`RefMesh::create`] so
/// create-if-exists can be compared; freeze durability is tested on production.
#[derive(Debug, Clone)]
pub struct RefMesh {
    pub dir: PathBuf,
    pub config: MeshConfig,
    pub adverts: Vec<Advert>,
    pub frozen: bool,
    pub freeze_cmd: Option<Freeze>,
    pub watches: Vec<Watch>,
}

impl RefMesh {
    pub fn new(config: MeshConfig) -> Self {
        Self {
            dir: PathBuf::new(),
            config,
            adverts: Vec::new(),
            frozen: false,
            freeze_cmd: None,
            watches: Vec::new(),
        }
    }

    pub fn create(dir: impl AsRef<Path>, config: MeshConfig) -> Result<Self> {
        let dir = dir.as_ref();
        if dir.exists() {
            return Err(Error::AlreadyExists(dir.display().to_string()));
        }
        std::fs::create_dir(dir).map_err(|e| Error::Other(e.to_string()))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            config,
            adverts: Vec::new(),
            frozen: false,
            freeze_cmd: None,
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

    fn upsert_advert(&mut self, advert: Advert) {
        if let Some(existing) = self.adverts.iter_mut().find(|a| a.node.0 == advert.node.0) {
            *existing = advert;
        } else {
            self.adverts.push(advert);
        }
    }

    fn own_caps(&self) -> Capabilities {
        self.adverts
            .iter()
            .filter(|a| a.node.0 == self.config.this_id.0)
            .max_by_key(|a| a.at)
            .map(|a| a.caps.clone())
            .unwrap_or_default()
    }

    /// Scan queued tasks; skip cap mismatches; claim the first fit via H3.
    ///
    /// If the fitting task is not the FIFO head, `Queue::claim` cannot skip.
    /// The reference still returns the lease production must produce (attempt 1,
    /// worker = node id, expires_at = now + ttl) without claiming a mismatch.
    /// Scan tests assert production queue state directly.
    pub fn pull(&mut self, queue: &mut Queue, now: NowMs) -> Result<Option<Lease>> {
        if !self.config.trusted {
            return Err(Error::Untrusted("pull".into()));
        }
        if self.frozen {
            return Err(Error::Frozen);
        }
        let caps = self.own_caps();
        let queued = queued_in_order(queue);
        let Some(fit) = queued.iter().find(|id| {
            queue
                .get(id)
                .map(|t| caps_satisfy(&caps, &t.payload))
                .unwrap_or(false)
        }) else {
            return Ok(None);
        };
        let fit = fit.clone();
        let head = queued.first();
        if head == Some(&fit) {
            return queue
                .claim(&WorkerId(self.config.this_id.0.clone()), now)
                .map_err(|e| Error::Other(e.to_string()));
        }
        let ttl = queue.config().lease_ttl_ms();
        Ok(Some(Lease {
            task_id: fit,
            worker_id: WorkerId(self.config.this_id.0.clone()),
            attempt: 1,
            expires_at: now.saturating_add(ttl),
        }))
    }

    pub fn freeze(&mut self, cmd: Freeze) -> Result<()> {
        if !signature_ok(&cmd) {
            return Err(Error::BadSignature);
        }
        self.frozen = true;
        self.freeze_cmd = Some(cmd);
        Ok(())
    }

    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    pub fn watches(&self) -> &[Watch] {
        &self.watches
    }
}

/// In-process cluster. Isolated indices get no gossip until `heal` then `tick`.
#[derive(Debug, Clone)]
pub struct RefLocalMesh {
    nodes: Vec<RefMesh>,
    isolated: HashSet<usize>,
}

impl RefLocalMesh {
    pub fn start(n: usize) -> Result<Self> {
        if n < MIN_RING {
            return Err(Error::TooFewNodes(n));
        }
        let watches = expected_watches(n);
        let nodes = (0..n)
            .map(|i| RefMesh {
                dir: PathBuf::from(i.to_string()),
                config: slot_config(i),
                adverts: Vec::new(),
                frozen: false,
                freeze_cmd: None,
                watches: watches.clone(),
            })
            .collect();
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

    pub fn get(&mut self, i: usize) -> Result<&mut RefMesh> {
        self.nodes
            .get_mut(i)
            .ok_or_else(|| Error::NotFound(i.to_string()))
    }

    pub fn partition(&mut self, isolated: &[usize]) -> Result<()> {
        self.isolated = isolated.iter().copied().collect();
        Ok(())
    }

    pub fn heal(&mut self) -> Result<()> {
        self.isolated.clear();
        Ok(())
    }

    /// Gossip adverts and freeze among non-isolated nodes. Isolated nodes keep
    /// their local state until a later tick after heal.
    pub fn tick(&mut self, _now: NowMs) -> Result<()> {
        let n = self.nodes.len();
        let connected: Vec<usize> = (0..n).filter(|i| !self.isolated.contains(i)).collect();
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
                self.nodes[i].frozen = true;
                self.nodes[i].freeze_cmd = Some(cmd);
            }
        }
        Ok(())
    }
}
