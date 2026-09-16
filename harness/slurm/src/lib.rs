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

use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

pub const JOB_PREFIX: &str = "fv";
pub const ALLOWED_GPUS: [u32; 3] = [1, 2, 4];
pub const MAX_CONCURRENT_JOBS: u32 = 2;
pub const QUERY_TIMEOUT: Duration = Duration::from_secs(45);
pub const DEFAULT_PARTITION: &str = "gpu";
pub const DEFAULT_REMOTE_ROOT: &str = "/work/ttennant1/prometheus";
/// NCShare login host (`CLUSTER` default in prometheus-build/ncshare.sh).
pub const DEFAULT_CLUSTER: &str = "ncshare";

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
        Self {
            cluster: DEFAULT_CLUSTER.to_string(),
            remote_root: PathBuf::from(DEFAULT_REMOTE_ROOT),
            partition: DEFAULT_PARTITION.to_string(),
            state_dir: default_state_dir(),
            query_timeout: QUERY_TIMEOUT,
            max_concurrent: MAX_CONCURRENT_JOBS,
        }
    }
}

fn default_state_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".local/state/prometheus-slurm")
}

/// Talks to Slurm on the NCShare login node over SSH.
pub struct Client {
    cfg: ClientConfig,
}

impl Client {
    pub fn new(cfg: ClientConfig) -> Result<Self> {
        Ok(Self { cfg })
    }

    /// Record the job name under `state_dir/runs/<run_id>/` first, then
    /// `sbatch`. Refuse if `gpus` is not in `ALLOWED_GPUS` or if two `fv-`
    /// jobs are already live.
    pub fn submit(&self, req: SubmitRequest) -> Result<JobId> {
        if !ALLOWED_GPUS.contains(&req.gpus) {
            return Err(Error::BadGpuCount(req.gpus));
        }

        let live = self.count_live_fv_jobs()?;
        if live >= self.cfg.max_concurrent {
            return Err(Error::ConcurrentCap);
        }

        let name = JobName(format!("{JOB_PREFIX}-{}", req.suffix));
        self.record_job_name(&req.run_id, &name)?;

        let script = req.script.to_string_lossy().into_owned();
        let gres = format!("--gres=gpu:h200:{}", req.gpus);
        let stdout = self.run_slurm(&[
            "sbatch",
            "-J",
            &name.0,
            "-t",
            &req.walltime,
            "-p",
            &self.cfg.partition,
            &gres,
            &script,
        ])?;
        let id = parse_job_id(&stdout).ok_or_else(|| Error::Submit(stdout.clone()))?;
        self.record_job_id(&req.run_id, &id)?;
        Ok(JobId(id))
    }

    /// Timeout-wrapped `sacct`. Returns `JobState::Unknown` if the query
    /// itself fails, never hangs.
    pub fn state(&self, id: &JobId) -> Result<JobState> {
        match self.run_slurm(&["sacct", "-j", &id.0, "-n", "-X", "-o", "State"]) {
            Ok(stdout) => {
                let token = stdout
                    .lines()
                    .map(str::trim)
                    .find(|l| !l.is_empty())
                    .unwrap_or("");
                Ok(parse_state(token).unwrap_or(JobState::Unknown))
            }
            Err(Error::QueryTimeout) => Err(Error::QueryTimeout),
            Err(_) => Ok(JobState::Unknown),
        }
    }

    /// Poll `state` until terminal or `budget` elapses. Cancel the job on
    /// local budget expiry.
    pub fn wait(&self, id: &JobId, budget: Duration) -> Result<JobState> {
        let start = Instant::now();
        loop {
            let st = match self.state(id) {
                Ok(s) => s,
                Err(Error::QueryTimeout) => JobState::Unknown,
                Err(e) => return Err(e),
            };
            if st.is_terminal() {
                return Ok(st);
            }
            if start.elapsed() >= budget {
                let _ = self.cancel(id);
                return Ok(JobState::Cancelled);
            }
            let remaining = budget.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                let _ = self.cancel(id);
                return Ok(JobState::Cancelled);
            }
            std::thread::sleep(remaining.min(Duration::from_millis(50)));
        }
    }

    /// `scancel` this id only.
    pub fn cancel(&self, id: &JobId) -> Result<()> {
        self.run_slurm(&["scancel", &id.0])?;
        Ok(())
    }

    /// Copy `remote_root/<run_id>/` to `dest`.
    pub fn fetch(&self, run_id: &str, dest: &Path) -> Result<()> {
        fs::create_dir_all(dest)
            .map_err(|e| Error::Other(format!("mkdir {}: {e}", dest.display())))?;
        let remote = self
            .cfg
            .remote_root
            .join(run_id)
            .to_string_lossy()
            .into_owned();
        let src = format!("{}:{remote}/", self.cfg.cluster);
        let dest_s = dest.to_string_lossy().into_owned();
        let status = Command::new("rsync")
            .args(["-a", &src, &dest_s])
            .status()
            .map_err(|e| Error::Other(format!("spawn rsync: {e}")))?;
        if !status.success() {
            return Err(Error::Other(format!("rsync failed: {status}")));
        }
        Ok(())
    }

    /// Cancel every job id recorded for `run_id`. Never `scancel -u`.
    pub fn reap_tracked(&self, run_id: &str) -> Result<()> {
        let path = self.job_ids_path(run_id);
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(Error::Other(format!("read {}: {e}", path.display()))),
        };
        for line in text.lines() {
            let tok = line.trim();
            if !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit()) {
                self.cancel(&JobId(tok.to_string()))?;
            }
        }
        Ok(())
    }

    /// Path of the job-ids file for `run_id`. Created before `sbatch`.
    pub fn job_ids_path(&self, run_id: &str) -> PathBuf {
        self.cfg.state_dir.join("runs").join(run_id).join("job-ids")
    }

    fn run_dir(&self, run_id: &str) -> PathBuf {
        self.cfg.state_dir.join("runs").join(run_id)
    }

    fn record_job_name(&self, run_id: &str, name: &JobName) -> Result<()> {
        let dir = self.run_dir(run_id);
        fs::create_dir_all(&dir)
            .map_err(|e| Error::Other(format!("mkdir {}: {e}", dir.display())))?;
        let name_path = dir.join("job-name");
        fs::write(&name_path, format!("{}\n", name.0))
            .map_err(|e| Error::Other(format!("write {}: {e}", name_path.display())))?;
        let ids_path = self.job_ids_path(run_id);
        if !ids_path.exists() {
            fs::write(&ids_path, "")
                .map_err(|e| Error::Other(format!("create {}: {e}", ids_path.display())))?;
        }
        Ok(())
    }

    fn record_job_id(&self, run_id: &str, id: &str) -> Result<()> {
        let path = self.job_ids_path(run_id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| Error::Other(format!("mkdir {}: {e}", parent.display())))?;
        }
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| Error::Other(format!("open {}: {e}", path.display())))?;
        writeln!(f, "{id}").map_err(|e| Error::Other(format!("write {}: {e}", path.display())))?;
        Ok(())
    }

    fn count_live_fv_jobs(&self) -> Result<u32> {
        let out = self.run_slurm(&["squeue", "--noheader"])?;
        Ok(count_live_fv(&out))
    }

    fn run_slurm(&self, remote: &[&str]) -> Result<String> {
        let ssh = std::env::var("PROMETHEUS_SLURM_SSH").unwrap_or_else(|_| "ssh".into());
        let mut child = Command::new(&ssh)
            .arg(&self.cfg.cluster)
            .args(remote)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| Error::Other(format!("spawn {ssh}: {e}")))?;
        match wait_with_timeout(&mut child, self.cfg.query_timeout) {
            Ok(Some(status)) => {
                let stdout = read_pipe(child.stdout.take());
                let stderr = read_pipe(child.stderr.take());
                if !status.success() {
                    return Err(classify_remote_failure(remote, status, &stdout, &stderr));
                }
                Ok(stdout)
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(Error::QueryTimeout)
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(Error::Other(format!("wait ssh: {e}")))
            }
        }
    }
}

fn classify_remote_failure(
    remote: &[&str],
    status: ExitStatus,
    stdout: &str,
    stderr: &str,
) -> Error {
    let detail = format!("status={status} stdout={stdout} stderr={stderr}");
    if remote.first().copied() == Some("sbatch") {
        Error::Submit(detail)
    } else {
        Error::Other(detail)
    }
}

fn read_pipe(pipe: Option<impl Read>) -> String {
    let mut buf = String::new();
    if let Some(mut r) = pipe {
        let _ = r.read_to_string(&mut buf);
    }
    buf
}

fn wait_with_timeout(child: &mut Child, timeout: Duration) -> io::Result<Option<ExitStatus>> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        let now = Instant::now();
        if now >= deadline {
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(5).min(deadline - now));
    }
}

fn parse_job_id(stdout: &str) -> Option<String> {
    stdout.split_whitespace().find_map(|tok| {
        if !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit()) {
            Some(tok.to_string())
        } else {
            None
        }
    })
}

fn count_live_fv(squeue_out: &str) -> u32 {
    let prefix = format!("{JOB_PREFIX}-");
    squeue_out
        .lines()
        .filter(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let named = fields
                .iter()
                .any(|f| *f == JOB_PREFIX || f.starts_with(&prefix));
            named && fields.iter().any(|f| is_live_queue_state(f))
        })
        .count() as u32
}

fn is_live_queue_state(token: &str) -> bool {
    matches!(
        token.to_ascii_uppercase().as_str(),
        "R" | "PD" | "RUNNING" | "PENDING"
    )
}

/// Parse a `sacct`/`squeue` state token. `None` if empty or unrecognised.
pub fn parse_state(raw: &str) -> Option<JobState> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    let token = s.split_whitespace().next()?.trim_end_matches('+');
    match token.to_ascii_uppercase().as_str() {
        "PENDING" => Some(JobState::Pending),
        "RUNNING" => Some(JobState::Running),
        "COMPLETED" => Some(JobState::Completed),
        "FAILED" => Some(JobState::Failed),
        "CANCELLED" | "CANCELED" => Some(JobState::Cancelled),
        "TIMEOUT" => Some(JobState::Timeout),
        "PREEMPTED" => Some(JobState::Preempted),
        "NODE_FAIL" => Some(JobState::NodeFail),
        "OUT_OF_MEMORY" => Some(JobState::OutOfMemory),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_config_default_does_not_panic() {
        let cfg = ClientConfig::default();
        assert_eq!(cfg.partition, DEFAULT_PARTITION);
        assert_eq!(cfg.remote_root, PathBuf::from(DEFAULT_REMOTE_ROOT));
        assert_eq!(cfg.query_timeout, QUERY_TIMEOUT);
        assert_eq!(cfg.max_concurrent, MAX_CONCURRENT_JOBS);
        assert_eq!(cfg.cluster, DEFAULT_CLUSTER);
        assert!(!cfg.cluster.is_empty());
        assert!(!cfg.state_dir.as_os_str().is_empty());
    }
}
