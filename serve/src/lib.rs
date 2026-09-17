//! Stock SGLang serving (spec 15.2, 15.5 F5, gate D5).
//!
//! Deploys stock SGLang on NCShare through H7, exposes a batch generation
//! API, and registers as the H6 provider of last resort. The SGLang fork
//! (latent decode, recurrence, MTP) is C1/C2/I4, not this crate.
//!
//! CPU tests drive an in-process [`LocalEngine`]. Production talks HTTP to
//! a process H7 started. `NowMs` is injected; nothing here reads the wall
//! clock.

use prometheus_providers::{Aimd, AimdConfig, Failover, ProviderId};
use prometheus_slurm::{Client, JobId};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub type NowMs = u64;

/// H6 provider id for the local stock engine. Last in the failover list.
pub const LAST_RESORT_ID: &str = "sglang";

/// Default open-weights model id. A label, not a download.
pub const DEFAULT_MODEL: &str = "open-weights";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineConfig {
    pub model: String,
    /// Must be one of [`prometheus_slurm::ALLOWED_GPUS`].
    pub gpus: u32,
    pub max_batch: u32,
    pub port: u16,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            gpus: 1,
            max_batch: 8,
            port: 30000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompts: Vec<String>,
    pub max_tokens: u32,
    pub temperature: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Completion {
    pub prompt_index: usize,
    pub text: String,
    pub finish_reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub model: String,
    pub completions: Vec<Completion>,
}

impl GenerateResponse {
    /// OpenAI-style JSON body for [`prometheus_providers::validate_response`].
    pub fn openai_body(&self) -> Vec<u8> {
        unimplemented!("F5: GenerateResponse::openai_body")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Deployment {
    pub job: JobId,
    pub endpoint: String,
    pub model: String,
    pub gpus: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("engine is not running")]
    NotRunning,
    #[error("last-resort engine is down")]
    EngineDown,
    #[error("gpus must be one of prometheus_slurm::ALLOWED_GPUS, got {0}")]
    BadGpuCount(u32),
    #[error("empty batch")]
    EmptyBatch,
    #[error("batch larger than max_batch ({0})")]
    BatchTooLarge(u32),
    #[error("hosted provider {0} has no local engine")]
    Hosted(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("{0}")]
    Provider(String),
    #[error("{0}")]
    Slurm(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<prometheus_slurm::Error> for Error {
    fn from(err: prometheus_slurm::Error) -> Self {
        match err {
            prometheus_slurm::Error::BadGpuCount(n) => Error::BadGpuCount(n),
            other => Error::Slurm(other.to_string()),
        }
    }
}

impl From<prometheus_providers::Error> for Error {
    fn from(err: prometheus_providers::Error) -> Self {
        Error::Provider(err.to_string())
    }
}

/// In-process stock engine. Deterministic completions. Not SGLang.
pub struct LocalEngine {
    cfg: EngineConfig,
    up: bool,
}

impl LocalEngine {
    pub fn start(cfg: EngineConfig) -> Result<Self> {
        let _ = cfg;
        unimplemented!("F5: LocalEngine::start")
    }

    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    pub fn is_up(&self) -> bool {
        self.up
    }

    pub fn provider_id(&self) -> ProviderId {
        ProviderId(LAST_RESORT_ID.to_string())
    }

    pub fn mark_down(&mut self) {
        unimplemented!("F5: LocalEngine::mark_down")
    }

    pub fn mark_up(&mut self) {
        unimplemented!("F5: LocalEngine::mark_up")
    }

    pub fn generate(&mut self, _req: GenerateRequest, _now: NowMs) -> Result<GenerateResponse> {
        unimplemented!("F5: LocalEngine::generate")
    }
}

/// Submit stock SGLang through H7. CPU tests use [`LocalEngine`] instead.
pub struct Deployer {
    client: Client,
    script: PathBuf,
}

impl Deployer {
    pub fn new(client: Client, script: impl AsRef<Path>) -> Self {
        Self {
            client,
            script: script.as_ref().to_path_buf(),
        }
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn script(&self) -> &Path {
        &self.script
    }

    /// `sbatch` via H7. Refuses a gpu count H7 would refuse.
    pub fn submit(
        &self,
        _cfg: &EngineConfig,
        _run_id: &str,
        _suffix: &str,
        _walltime: &str,
    ) -> Result<Deployment> {
        unimplemented!("F5: Deployer::submit")
    }

    pub fn teardown(&self, _dep: &Deployment) -> Result<()> {
        unimplemented!("F5: Deployer::teardown")
    }
}

/// Routes generation through H6 failover. [`LAST_RESORT_ID`] is always last.
pub struct Router {
    failover: Failover,
    aimd: Aimd,
    engine: LocalEngine,
}

impl Router {
    /// `hosted` are tried first. The engine is appended as last resort.
    pub fn new(_hosted: Vec<ProviderId>, _engine: LocalEngine, _aimd: AimdConfig) -> Result<Self> {
        unimplemented!("F5: Router::new")
    }

    pub fn failover(&self) -> &Failover {
        &self.failover
    }

    pub fn aimd(&self) -> &Aimd {
        &self.aimd
    }

    pub fn engine(&self) -> &LocalEngine {
        &self.engine
    }

    pub fn last_resort(&self) -> ProviderId {
        ProviderId(LAST_RESORT_ID.to_string())
    }

    /// Mark a hosted provider down. Last resort stays up unless the engine is.
    pub fn mark_down(&mut self, _id: &ProviderId) -> Result<()> {
        unimplemented!("F5: Router::mark_down")
    }

    pub fn mark_up(&mut self, _id: &ProviderId) -> Result<()> {
        unimplemented!("F5: Router::mark_up")
    }

    /// Generate on the current provider. Hosted ids without a local backend
    /// return [`Error::Hosted`]. When every hosted provider is down, this
    /// uses the last-resort engine (D5) and the AIMD window is the reduced
    /// swarm size.
    pub fn generate(&mut self, _req: GenerateRequest, _now: NowMs) -> Result<GenerateResponse> {
        unimplemented!("F5: Router::generate")
    }
}
