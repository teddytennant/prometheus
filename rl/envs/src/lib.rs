//! Firecracker environment fleet (spec 9.3, 9.4, 12, 15.5 D1).
//!
//! Pool of sandboxes, snapshot, fork, and the F1 tool API. CPU tests drive
//! an in-process [`InProcess`] backend: no KVM, no live network. Production
//! talks Firecracker. [`NowMs`] is injected; nothing here reads the wall clock.
//!
//! Isolation every backend must hold: no live internet, no credentials in
//! sandboxes, egress blocked, hidden tests outside the agent filesystem
//! view, writes to test files blocked and flagged.
//!
//! The 1M-sandbox fleet is S3 (spec 15.6) and out of scope. The CPU gate is
//! correctness of fork-from-snapshot and the tool API. The hardware gate is
//! sub-200ms fork on one node (`kvm` feature).

mod engine;

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use engine::Engine;

/// Injected clock. Milliseconds since an arbitrary origin. Never wall time.
pub type NowMs = u64;

/// Fork budget on one node (spec 9.4, 15.5 D1 gate).
pub const FORK_BUDGET_MS: u64 = 200;

/// GRPO group size (spec 9.2). All 16 samples fork from one snapshot.
pub const GRPO_GROUP: usize = 16;

/// Reserved absolute prefix for hidden tests and graders. Not in the agent view.
pub const GRADER_ROOT: &str = "/grader";

pub const SCHEMA_TOOL_REQUEST: &str = "prometheus.env_tool_request";
pub const SCHEMA_TOOL_RESPONSE: &str = "prometheus.env_tool_response";
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImageId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SnapshotId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SandboxId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CallId(pub String);

/// F1 `prometheus.env_tool_request.tool` enum (spec 9.4, 9.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tool {
    Shell,
    Editor,
    Python,
    Browser,
    Lean,
    Notes,
    Subagent,
}

/// Filesystem view. Agent never sees [`View::Grader`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum View {
    Agent,
    Grader,
}

/// Training egress: deny all live traffic. Browser reads [`OfflineWeb`] only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EgressPolicy {
    DenyAll,
}

/// F1 artifact (content-addressed blob metadata).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Artifact {
    pub content_hash: String,
    pub bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

/// F1 `prometheus.env_tool_request` v1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolRequest {
    pub schema_id: String,
    pub schema_version: u32,
    pub call_id: String,
    pub env_snapshot_id: String,
    pub tool: Tool,
    pub payload: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_s: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_call_id: Option<String>,
}

impl ToolRequest {
    pub fn new(call_id: impl Into<String>, env_snapshot_id: impl Into<String>, tool: Tool) -> Self {
        Self {
            schema_id: SCHEMA_TOOL_REQUEST.to_string(),
            schema_version: SCHEMA_VERSION,
            call_id: call_id.into(),
            env_snapshot_id: env_snapshot_id.into(),
            tool,
            payload: serde_json::json!({}),
            timeout_s: None,
            parent_call_id: None,
        }
    }
}

/// F1 `prometheus.env_tool_response` v1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResponse {
    pub schema_id: String,
    pub schema_version: u32,
    pub call_id: String,
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    pub stdout_artifact: Artifact,
    pub stderr_artifact: Artifact,
    pub truncated: bool,
    pub duration_ms: u64,
    pub snapshot_id_after: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Host-side offline web snapshot. Browser tool serves only these pages.
///
/// `cutoff_unix_s` is the time-sliced retrieval cut (spec 9.4): a page with
/// `snapshot_unix_s` after the cutoff is not served.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OfflineWeb {
    pub pages: BTreeMap<String, OfflinePage>,
    pub cutoff_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfflinePage {
    pub body: Vec<u8>,
    pub snapshot_unix_s: u64,
    pub media_type: String,
}

/// GPU sandbox pool (spec 9.4). Separate from the CPU fleet. Hard walltime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuConfig {
    pub gpus: u32,
    pub walltime_s: u32,
    pub mig: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolConfig {
    pub capacity: usize,
    pub default_timeout_s: u32,
    pub egress: EgressPolicy,
    pub gpu: Option<GpuConfig>,
    pub offline_web: OfflineWeb,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            capacity: GRPO_GROUP,
            default_timeout_s: 30,
            egress: EgressPolicy::DenyAll,
            gpu: None,
            offline_web: OfflineWeb::default(),
        }
    }
}

impl PoolConfig {
    pub fn hold(&self) -> bool {
        self.capacity > 0 && self.default_timeout_s > 0 && self.egress == EgressPolicy::DenyAll
    }
}

/// Rootfs plus hidden tests. Hidden tests live under [`GRADER_ROOT`] and are
/// never in the agent view. `hidden_tests_hash` is lowercase hex SHA-256 of
/// the hidden-test bytes in sorted-path order (path, NUL, bytes, per file).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub id: ImageId,
    pub agent_files: BTreeMap<String, Vec<u8>>,
    pub hidden_tests: BTreeMap<String, Vec<u8>>,
    pub hidden_tests_hash: String,
    pub env: BTreeMap<String, String>,
}

impl Image {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: ImageId(id.into()),
            agent_files: BTreeMap::new(),
            hidden_tests: BTreeMap::new(),
            hidden_tests_hash: String::new(),
            env: BTreeMap::new(),
        }
    }
}

/// Flagged write to a grader/test path (spec 9.3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TamperFlag {
    pub sandbox: SandboxId,
    pub path: String,
    pub call_id: String,
    pub now_ms: NowMs,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForkResult {
    pub sandboxes: Vec<SandboxId>,
    /// Per-sandbox fork duration, milliseconds of injected [`NowMs`] or the
    /// backend's measured boot. Compared to [`FORK_BUDGET_MS`] at the kvm gate.
    pub durations_ms: Vec<u64>,
}

impl ForkResult {
    pub fn all_under_budget(&self) -> bool {
        self.durations_ms.iter().all(|&d| d <= FORK_BUDGET_MS)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("pool exhausted (capacity {capacity})")]
    PoolExhausted { capacity: usize },
    #[error("snapshot not found")]
    SnapshotNotFound,
    #[error("sandbox not found")]
    SandboxNotFound,
    #[error("image not found")]
    ImageNotFound,
    #[error("egress denied")]
    EgressDenied,
    #[error("live internet is not available in training sandboxes")]
    LiveInternet,
    #[error("credential {name} is not allowed in a sandbox")]
    CredentialInSandbox { name: String },
    #[error("write to test file {path} blocked")]
    TestFileWrite { path: String },
    #[error("hidden tests leaked into the agent view")]
    HiddenTestsVisible,
    #[error("tool call timed out")]
    Timeout,
    #[error("gpu walltime exceeded")]
    GpuTimeLimit,
    #[error("bad tool request: {0}")]
    BadToolRequest(String),
    #[error("backend: {0}")]
    Backend(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Sandbox lifecycle. CPU tests use [`InProcess`]. Production uses [`Firecracker`].
pub trait Backend {
    fn boot(&mut self, image: &Image, now: NowMs) -> Result<SandboxId>;
    fn snapshot(&mut self, sandbox: &SandboxId, now: NowMs) -> Result<SnapshotId>;
    /// Fork one sandbox from `snapshot`. Returns the new id and duration_ms.
    fn fork(&mut self, snapshot: &SnapshotId, now: NowMs) -> Result<(SandboxId, u64)>;
    fn call(&mut self, sandbox: &SandboxId, req: &ToolRequest, now: NowMs) -> Result<ToolResponse>;
    fn kill(&mut self, sandbox: &SandboxId, now: NowMs) -> Result<()>;
    fn list(&self, sandbox: &SandboxId, view: View) -> Result<BTreeMap<String, Vec<u8>>>;
}

/// In-process backend for CPU tests. No KVM, no sockets, no live network.
pub struct InProcess {
    cfg: PoolConfig,
    inner: Engine,
}

impl InProcess {
    pub fn new(cfg: PoolConfig) -> Self {
        Self {
            inner: Engine::new(cfg.clone()),
            cfg,
        }
    }

    pub fn config(&self) -> &PoolConfig {
        &self.cfg
    }
}

impl Backend for InProcess {
    fn boot(&mut self, image: &Image, now: NowMs) -> Result<SandboxId> {
        self.inner.boot(image, now)
    }

    fn snapshot(&mut self, sandbox: &SandboxId, now: NowMs) -> Result<SnapshotId> {
        self.inner.snapshot(sandbox, now)
    }

    fn fork(&mut self, snapshot: &SnapshotId, now: NowMs) -> Result<(SandboxId, u64)> {
        self.inner.fork(snapshot, now)
    }

    fn call(&mut self, sandbox: &SandboxId, req: &ToolRequest, now: NowMs) -> Result<ToolResponse> {
        self.inner.call(sandbox, req.clone(), now)
    }

    fn kill(&mut self, sandbox: &SandboxId, now: NowMs) -> Result<()> {
        self.inner.kill(sandbox, now)
    }

    fn list(&self, sandbox: &SandboxId, view: View) -> Result<BTreeMap<String, Vec<u8>>> {
        self.inner.list(sandbox, view)
    }
}

/// Firecracker backend. Constructed only on a node with `/dev/kvm`.
pub struct Firecracker {
    socket: PathBuf,
    kernel: PathBuf,
    rootfs: PathBuf,
}

impl Firecracker {
    pub fn new(socket: PathBuf, kernel: PathBuf, rootfs: PathBuf) -> Result<Self> {
        Ok(Self {
            socket,
            kernel,
            rootfs,
        })
    }

    pub fn socket(&self) -> &PathBuf {
        &self.socket
    }

    pub fn kernel(&self) -> &PathBuf {
        &self.kernel
    }

    pub fn rootfs(&self) -> &PathBuf {
        &self.rootfs
    }
}

impl Backend for Firecracker {
    fn boot(&mut self, _image: &Image, _now: NowMs) -> Result<SandboxId> {
        unimplemented!("D1: Firecracker::boot")
    }

    fn snapshot(&mut self, _sandbox: &SandboxId, _now: NowMs) -> Result<SnapshotId> {
        unimplemented!("D1: Firecracker::snapshot")
    }

    fn fork(&mut self, _snapshot: &SnapshotId, _now: NowMs) -> Result<(SandboxId, u64)> {
        unimplemented!("D1: Firecracker::fork")
    }

    fn call(
        &mut self,
        _sandbox: &SandboxId,
        _req: &ToolRequest,
        _now: NowMs,
    ) -> Result<ToolResponse> {
        unimplemented!("D1: Firecracker::call")
    }

    fn kill(&mut self, _sandbox: &SandboxId, _now: NowMs) -> Result<()> {
        unimplemented!("D1: Firecracker::kill")
    }

    fn list(&self, _sandbox: &SandboxId, _view: View) -> Result<BTreeMap<String, Vec<u8>>> {
        unimplemented!("D1: Firecracker::list")
    }
}

/// Sandbox pool. Capacity is live sandboxes, not snapshots.
pub struct Pool<B: Backend> {
    backend: B,
    cfg: PoolConfig,
    inner: Engine,
}

pub type CpuPool = Pool<InProcess>;

impl Pool<InProcess> {
    pub fn in_process(cfg: PoolConfig) -> Self {
        Self {
            backend: InProcess::new(cfg.clone()),
            inner: Engine::new(cfg.clone()),
            cfg,
        }
    }
}

impl Pool<Firecracker> {
    pub fn firecracker(backend: Firecracker, cfg: PoolConfig) -> Self {
        Self {
            backend,
            inner: Engine::new(cfg.clone()),
            cfg,
        }
    }
}

impl<B: Backend> Pool<B> {
    pub fn config(&self) -> &PoolConfig {
        &self.cfg
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn live_count(&self) -> usize {
        self.inner.live_count()
    }

    pub fn tamper_flags(&self) -> &[TamperFlag] {
        self.inner.tamper_flags()
    }

    /// Reject images that carry credentials or put hidden tests in the agent tree.
    pub fn register_image(&mut self, image: Image) -> Result<ImageId> {
        self.inner.register_image(image)
    }

    pub fn snapshot_from_image(&mut self, image: &ImageId, now: NowMs) -> Result<SnapshotId> {
        self.inner.snapshot_from_image(image, now)
    }

    /// Fork `n` sandboxes from an identical snapshot. `n` is typically [`GRPO_GROUP`].
    /// Fails with [`Error::PoolExhausted`] when `live + n > capacity`.
    pub fn fork_group(
        &mut self,
        snapshot: &SnapshotId,
        n: usize,
        now: NowMs,
    ) -> Result<ForkResult> {
        self.inner.fork_group(snapshot, n, now)
    }

    /// Fork the snapshot twice (spec 9.3 contrastive pair).
    pub fn fork_pair(
        &mut self,
        snapshot: &SnapshotId,
        now: NowMs,
    ) -> Result<(SandboxId, SandboxId)> {
        self.inner.fork_pair(snapshot, now)
    }

    pub fn call(
        &mut self,
        sandbox: &SandboxId,
        req: ToolRequest,
        now: NowMs,
    ) -> Result<ToolResponse> {
        self.inner.call(sandbox, req, now)
    }

    pub fn snapshot(&mut self, sandbox: &SandboxId, now: NowMs) -> Result<SnapshotId> {
        self.inner.snapshot(sandbox, now)
    }

    pub fn kill(&mut self, sandbox: &SandboxId, now: NowMs) -> Result<()> {
        self.inner.kill(sandbox, now)
    }

    pub fn agent_view(&self, sandbox: &SandboxId) -> Result<BTreeMap<String, Vec<u8>>> {
        self.inner.agent_view(sandbox)
    }

    pub fn grader_view(&self, sandbox: &SandboxId) -> Result<BTreeMap<String, Vec<u8>>> {
        self.inner.grader_view(sandbox)
    }
}
