//! In-process-controlled mock cluster for H7 Client tests.
//!
//! Production code talks to Slurm by spawning `ssh` (and optionally `scp` /
//! `rsync`). Tests drop shim binaries on `PATH` and set `PROMETHEUS_SLURM_SSH`
//! to the shim's absolute path. Shim behaviour is scripted via files under
//! `mock_dir` — no network, no real Slurm.
//!
//! Tests in this binary take `LOCK` so process-global env mutations do not
//! race.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use prometheus_slurm::{Client, ClientConfig, SubmitRequest};
use tempfile::TempDir;

pub static LOCK: Mutex<()> = Mutex::new(());

pub const FAKE_SSH: &str = include_str!("fake_ssh.sh");
pub const FAKE_RSYNC: &str = include_str!("fake_rsync.py");

pub struct EnvGuard {
    saved: Vec<(String, Option<String>)>,
}

impl EnvGuard {
    fn set(keys_vals: &[(&str, String)]) -> Self {
        let mut saved = Vec::new();
        for (k, v) in keys_vals {
            saved.push((k.to_string(), std::env::var(k).ok()));
            std::env::set_var(k, v);
        }
        Self { saved }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (k, v) in self.saved.iter().rev() {
            match v {
                Some(val) => std::env::set_var(k, val),
                None => std::env::remove_var(k),
            }
        }
    }
}

pub struct Mock {
    _tmp: TempDir,
    pub mock_dir: PathBuf,
    pub state_dir: PathBuf,
    pub script: PathBuf,
    pub query_timeout: Duration,
    pub max_concurrent: u32,
    pub remote_root: PathBuf,
    _env: EnvGuard,
    _lock: MutexGuard<'static, ()>,
}

impl Mock {
    pub fn new() -> Self {
        let lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().expect("tempdir");
        let mock_dir = tmp.path().join("mock");
        let state_dir = tmp.path().join("state");
        let bin_dir = tmp.path().join("bin");
        fs::create_dir_all(&mock_dir).unwrap();
        fs::create_dir_all(&state_dir).unwrap();
        fs::create_dir_all(&bin_dir).unwrap();
        fs::create_dir_all(mock_dir.join("states")).unwrap();
        fs::create_dir_all(mock_dir.join("remote")).unwrap();

        let ssh = bin_dir.join("ssh");
        fs::write(&ssh, FAKE_SSH).unwrap();
        fs::set_permissions(&ssh, fs::Permissions::from_mode(0o755)).unwrap();
        let scp = bin_dir.join("scp");
        fs::write(&scp, FAKE_SSH).unwrap();
        fs::set_permissions(&scp, fs::Permissions::from_mode(0o755)).unwrap();
        let rsync = bin_dir.join("rsync");
        fs::write(&rsync, FAKE_RSYNC).unwrap();
        fs::set_permissions(&rsync, fs::Permissions::from_mode(0o755)).unwrap();

        let script = tmp.path().join("job.sh");
        fs::write(&script, "#!/bin/bash\necho mock-job\n").unwrap();

        let remote_root = PathBuf::from("/work/ttennant1/prometheus");
        let old_path = std::env::var("PATH").unwrap_or_default();
        let new_path = format!("{}:{old_path}", bin_dir.display());
        let env = EnvGuard::set(&[
            ("PATH", new_path),
            ("PROMETHEUS_SLURM_SSH", ssh.to_string_lossy().into_owned()),
            (
                "PROMETHEUS_SLURM_MOCK_DIR",
                mock_dir.to_string_lossy().into_owned(),
            ),
            (
                "PROMETHEUS_SLURM_STATE_DIR",
                state_dir.to_string_lossy().into_owned(),
            ),
            (
                "PROMETHEUS_SLURM_REMOTE_ROOT",
                remote_root.to_string_lossy().into_owned(),
            ),
        ]);

        Self {
            _tmp: tmp,
            mock_dir,
            state_dir,
            script,
            query_timeout: Duration::from_millis(200),
            max_concurrent: 2,
            remote_root,
            _env: env,
            _lock: lock,
        }
    }

    pub fn config(&self) -> ClientConfig {
        ClientConfig {
            cluster: "mock-cluster".into(),
            remote_root: self.remote_root.clone(),
            partition: "gpu".into(),
            state_dir: self.state_dir.clone(),
            query_timeout: self.query_timeout,
            max_concurrent: self.max_concurrent,
        }
    }

    pub fn client(&self) -> Client {
        Client::new(self.config()).expect("Client::new")
    }

    pub fn submit_req(&self, run_id: &str, suffix: &str, gpus: u32) -> SubmitRequest {
        SubmitRequest {
            run_id: run_id.into(),
            suffix: suffix.into(),
            script: self.script.clone(),
            walltime: "00:30:00".into(),
            gpus,
        }
    }

    pub fn set_sbatch_mode(&self, mode: &str) {
        fs::write(self.mock_dir.join("sbatch_mode"), mode).unwrap();
    }

    pub fn set_sbatch_jobid(&self, id: &str) {
        fs::write(self.mock_dir.join("sbatch_jobid"), id).unwrap();
    }

    pub fn set_squeue_out(&self, text: &str) {
        fs::write(self.mock_dir.join("squeue_out"), text).unwrap();
    }

    pub fn hang_queries(&self) {
        fs::write(self.mock_dir.join("hang_queries"), "squeue,sacct").unwrap();
    }

    pub fn set_job_state(&self, jobid: &str, state: &str) {
        fs::write(self.mock_dir.join("states").join(jobid), state).unwrap();
    }

    pub fn set_default_state(&self, state: &str) {
        fs::write(self.mock_dir.join("default_state"), state).unwrap();
    }

    pub fn remote_log(&self) -> String {
        fs::read_to_string(self.mock_dir.join("remote.log")).unwrap_or_default()
    }

    pub fn remotes(&self) -> Vec<String> {
        self.remote_log()
            .lines()
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    pub fn sbatch_lines(&self) -> Vec<String> {
        self.remotes()
            .into_iter()
            .filter(|l| l.to_ascii_lowercase().contains("sbatch"))
            .collect()
    }

    pub fn scancel_lines(&self) -> Vec<String> {
        let from_log = fs::read_to_string(self.mock_dir.join("scancel.log")).unwrap_or_default();
        if !from_log.trim().is_empty() {
            return from_log
                .lines()
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
        self.remotes()
            .into_iter()
            .filter(|l| l.to_ascii_lowercase().contains("scancel"))
            .collect()
    }

    pub fn cancelled_ids(&self) -> Vec<String> {
        fs::read_to_string(self.mock_dir.join("cancelled_ids"))
            .unwrap_or_default()
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    pub fn used_scancel_u(&self) -> bool {
        self.mock_dir.join("scancel_u").exists()
            || self
                .scancel_lines()
                .iter()
                .any(|l| scancel_has_user_flag(l))
    }

    pub fn state_at_sbatch(&self) -> PathBuf {
        self.mock_dir.join("state_at_sbatch")
    }
}

pub fn scancel_has_user_flag(line: &str) -> bool {
    line.split_whitespace().any(|t| {
        t == "-u"
            || t == "--user"
            || t.starts_with("--user=")
            || t.starts_with("-u") && t != "-u" && t.chars().nth(2).is_some_and(|c| c != '-')
    })
}

pub fn dir_contains_needle(dir: &Path, needle: &str) -> bool {
    let Ok(rd) = fs::read_dir(dir) else {
        return false;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            if dir_contains_needle(&p, needle) {
                return true;
            }
        } else {
            if p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.contains(needle))
            {
                return true;
            }
            if let Ok(txt) = fs::read_to_string(&p) {
                if txt.contains(needle) {
                    return true;
                }
            }
        }
    }
    false
}

pub fn job_name_on_disk(root: &Path, run_id: &str, job_name: &str) -> bool {
    let run_dir = root.join("runs").join(run_id);
    dir_contains_needle(&run_dir, job_name) || dir_contains_needle(root, job_name)
}
