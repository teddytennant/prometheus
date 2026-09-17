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
use prometheus_slurm::{Client, JobId, SubmitRequest, ALLOWED_GPUS};
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
        let choices: Vec<serde_json::Value> = self
            .completions
            .iter()
            .map(|c| {
                serde_json::json!({
                    "index": c.prompt_index,
                    "message": { "role": "assistant", "content": c.text },
                    "finish_reason": c.finish_reason,
                })
            })
            .collect();
        serde_json::to_vec(&serde_json::json!({
            "model": self.model,
            "choices": choices,
        }))
        .expect("serialize openai_body")
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

fn fnv1a32(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for &b in bytes {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

fn mix32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn completion_text(prompt: &str, max_tokens: u32, temperature: f32) -> String {
    if max_tokens == 0 {
        return String::new();
    }
    let seed = fnv1a32(prompt.as_bytes())
        ^ max_tokens.wrapping_mul(0x9E37_79B9)
        ^ temperature.to_bits();
    let mut parts = Vec::with_capacity(max_tokens as usize);
    for j in 0..max_tokens {
        let token = mix32(seed.wrapping_add((j.wrapping_add(1)).wrapping_mul(0x85EB_CA6B)));
        parts.push(format!("{token:08x}"));
    }
    parts.join(" ")
}

/// In-process stock engine. Deterministic completions. Not SGLang.
pub struct LocalEngine {
    cfg: EngineConfig,
    up: bool,
}

impl LocalEngine {
    pub fn start(cfg: EngineConfig) -> Result<Self> {
        if !ALLOWED_GPUS.contains(&cfg.gpus) {
            return Err(Error::BadGpuCount(cfg.gpus));
        }
        Ok(Self { cfg, up: true })
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
        self.up = false;
    }

    pub fn mark_up(&mut self) {
        self.up = true;
    }

    pub fn generate(&mut self, req: GenerateRequest, _now: NowMs) -> Result<GenerateResponse> {
        if !self.is_up() {
            return Err(Error::EngineDown);
        }
        if req.prompts.is_empty() {
            return Err(Error::EmptyBatch);
        }
        if req.prompts.len() > self.cfg.max_batch as usize {
            return Err(Error::BatchTooLarge(self.cfg.max_batch));
        }
        let completions = req
            .prompts
            .iter()
            .enumerate()
            .map(|(i, p)| Completion {
                prompt_index: i,
                text: completion_text(p, req.max_tokens, req.temperature),
                finish_reason: "stop".to_string(),
            })
            .collect();
        Ok(GenerateResponse {
            model: self.cfg.model.clone(),
            completions,
        })
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
        cfg: &EngineConfig,
        run_id: &str,
        suffix: &str,
        walltime: &str,
    ) -> Result<Deployment> {
        if !ALLOWED_GPUS.contains(&cfg.gpus) {
            return Err(Error::BadGpuCount(cfg.gpus));
        }
        let job = self.client.submit(SubmitRequest {
            run_id: run_id.to_string(),
            suffix: suffix.to_string(),
            script: self.script.clone(),
            walltime: walltime.to_string(),
            gpus: cfg.gpus,
        })?;
        Ok(Deployment {
            job,
            endpoint: format!("http://127.0.0.1:{}", cfg.port),
            model: cfg.model.clone(),
            gpus: cfg.gpus,
        })
    }

    pub fn teardown(&self, dep: &Deployment) -> Result<()> {
        self.client.cancel(&dep.job)?;
        Ok(())
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
    pub fn new(hosted: Vec<ProviderId>, engine: LocalEngine, aimd: AimdConfig) -> Result<Self> {
        let mut providers = hosted;
        let already_last = providers
            .last()
            .map(|p| p.0.as_str() == LAST_RESORT_ID)
            .unwrap_or(false);
        if !already_last {
            providers.push(ProviderId(LAST_RESORT_ID.to_string()));
        }
        let failover = Failover::new(providers);
        let mut aimd = Aimd::new(aimd);
        let max = aimd.config().max_window;
        while aimd.window() < max {
            let before = aimd.window();
            aimd.on_success(0);
            if aimd.window() <= before {
                break;
            }
        }
        Ok(Self {
            failover,
            aimd,
            engine,
        })
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
    pub fn mark_down(&mut self, id: &ProviderId) -> Result<()> {
        let hosted_up = id.0 != LAST_RESORT_ID && !self.failover.down().contains(id);
        self.failover.mark_down(id)?;
        if hosted_up {
            self.aimd.on_outage();
        }
        Ok(())
    }

    pub fn mark_up(&mut self, id: &ProviderId) -> Result<()> {
        self.failover.mark_up(id)?;
        Ok(())
    }

    /// Generate on the current provider. Hosted ids without a local backend
    /// return [`Error::Hosted`]. When every hosted provider is down, this
    /// uses the last-resort engine (D5) and the AIMD window is the reduced
    /// swarm size.
    pub fn generate(&mut self, req: GenerateRequest, now: NowMs) -> Result<GenerateResponse> {
        let current = match self.failover.current() {
            Ok(id) => id.clone(),
            Err(prometheus_providers::Error::NoProvider) => return Err(Error::EngineDown),
            Err(err) => return Err(err.into()),
        };
        if current.0 != LAST_RESORT_ID {
            return Err(Error::Hosted(current.0));
        }
        self.engine.generate(req, now)
    }
}
