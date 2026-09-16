//! Slurm backend for NCShare H200 verification (spec 15.5 H7, gate D3).
//!
//! Built on the patterns in `prometheus-build/ncshare.sh`:
//! - every `squeue` / `sacct` call is client-side timeout-wrapped (a hung
//!   slurmctld has been observed to block a bare `squeue` for hours)
//! - a job-name record is written **before** `sbatch`, then the numeric id
//!   is filled in, so a crash between submit and record cannot orphan a job
//! - reap cancels only job ids this run wrote; never `scancel -u`
//!
//! Job-name prefix is `fv-`. Allowed GPU counts are 1, 2, or 4. At most two
//! `fv-` jobs may be live at once (shared 8-GPU cap with other users of the
//! account).

use std::path::{Path, PathBuf};
use std::time::Duration;

pub const JOB_PREFIX: &str = "fv";
pub const ALLOWED_GPUS: [u32; 3] = [1, 2, 4];
pub const MAX_CONCURRENT_JOBS: u32 = 2;
pub const QUERY_TIMEOUT: Duration = Duration::from_secs(45);
pub const DEFAULT_PARTITION: &str = "gpu";
pub const DEFAULT_REMOTE_ROOT: &str = "/work/ttennant1/prometheus";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("gpus must be one of {ALLOWED_GPUS:?}, got {0}")]
    BadGpuCount(u32),
    #[error("already {MAX_CONCURRENT_JOBS} live fv- jobs; refusing submit")]
    ConcurrentCap,
    #[error("squeue/sacct timed out after {QUERY_TIMEOUT:?}")]
    QueryTimeout,
    #[error("sbatch did not return a job id: {0}")]
    Submit(String),
    #[error("unknown job {0}")]
    UnknownJob(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Numeric Slurm job id once `sbatch` returns.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JobId(pub String);

/// `fv-<suffix>` used to look up a job if the numeric id never landed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JobName(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    Timeout,
    Preempted,
    NodeFail,
    OutOfMemory,
    /// Query itself failed or timed out. Not a Slurm state.
    Unknown,
}

impl JobState {
    pub fn is_terminal(self) -> bool {
        !matches!(
            self,
            JobState::Pending | JobState::Running | JobState::Unknown
        )
    }
}

#[derive(Debug, Clone)]
pub struct SubmitRequest {
    pub run_id: String,
    pub suffix: String,
    pub script: PathBuf,
    pub walltime: String,
    pub gpus: u32,
}

#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub cluster: String,
    pub remote_root: PathBuf,
    pub partition: String,
    pub state_dir: PathBuf,
    pub query_timeout: Duration,
    pub max_concurrent: u32,
}

impl Default for ClientConfig {
    fn default() -> Self {
        unimplemented!("ClientConfig::default")
    }
}

/// Talks to Slurm on the NCShare login node over SSH.
pub struct Client {
    _cfg: ClientConfig,
}

impl Client {
    pub fn new(_cfg: ClientConfig) -> Result<Self> {
        unimplemented!("Client::new")
    }

    /// Record the job name under `state_dir/runs/<run_id>/` first, then
    /// `sbatch`. Refuse if `gpus` is not in `ALLOWED_GPUS` or if two `fv-`
    /// jobs are already live.
    pub fn submit(&self, _req: SubmitRequest) -> Result<JobId> {
        unimplemented!("Client::submit")
    }

    /// Timeout-wrapped `sacct`. Returns `JobState::Unknown` if the query
    /// itself fails, never hangs.
    pub fn state(&self, _id: &JobId) -> Result<JobState> {
        unimplemented!("Client::state")
    }

    /// Poll `state` until terminal or `budget` elapses. Cancel the job on
    /// local budget expiry.
    pub fn wait(&self, _id: &JobId, _budget: Duration) -> Result<JobState> {
        unimplemented!("Client::wait")
    }

    /// `scancel` this id only.
    pub fn cancel(&self, _id: &JobId) -> Result<()> {
        unimplemented!("Client::cancel")
    }

    /// Copy `remote_root/<run_id>/` to `dest`.
    pub fn fetch(&self, _run_id: &str, _dest: &Path) -> Result<()> {
        unimplemented!("Client::fetch")
    }

    /// Cancel every job id recorded for `run_id`. Never `scancel -u`.
    pub fn reap_tracked(&self, _run_id: &str) -> Result<()> {
        unimplemented!("Client::reap_tracked")
    }

    /// Path of the job-ids file for `run_id`. Created before `sbatch`.
    pub fn job_ids_path(&self, _run_id: &str) -> PathBuf {
        unimplemented!("Client::job_ids_path")
    }
}

/// Parse a `sacct`/`squeue` state token. `None` if empty or unrecognised.
pub fn parse_state(_raw: &str) -> Option<JobState> {
    unimplemented!("parse_state")
}
