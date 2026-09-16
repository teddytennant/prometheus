//! Implementer-facing tests for `prometheus_slurm::Client`.
//!
//! Every test here calls production APIs that are `unimplemented!` today, so
//! `cargo test -p prometheus-slurm` is red on the stub and must go green
//! once `Client` is real. No GPU: the cluster is a file-backed mock.
//!
//! ## Mock / SSH contract
//!
//! Tests install a fake `ssh` (and `scp`/`rsync`) on `PATH` and set
//! `PROMETHEUS_SLURM_SSH` to that absolute path. The Client **must**:
//!
//! 1. Spawn `std::env::var("PROMETHEUS_SLURM_SSH").unwrap_or_else(|_| "ssh".into())`
//!    (or an `ssh` resolved from `PATH`). Do not hard-code `/usr/bin/ssh`.
//! 2. Wrap every `sbatch` / `squeue` / `sacct` spawn in `cfg.query_timeout`.
//!    A child that overruns returns `Error::QueryTimeout` or
//!    `JobState::Unknown` and must not block the caller. Use the config
//!    field, not the `QUERY_TIMEOUT` constant, so tests can inject 200ms.
//! 3. Write a job-name record under `state_dir/runs/<run_id>/` **before**
//!    invoking `sbatch`, then fill in the numeric id. `job_ids_path(run_id)`
//!    is `state_dir/runs/<run_id>/job-ids`.
//! 4. Name jobs `fv-<suffix>` (`JOB_PREFIX`).
//! 5. Never `scancel -u` / `scancel --user`. Reap only ids recorded for that
//!    `run_id`.
//! 6. Refuse `gpus` outside `{1,2,4}` and a third live `fv-` job
//!    (`cfg.max_concurrent`, default 2). Live = Pending or Running on the
//!    cluster (query `squeue`; do not trust only in-process counters).
//! 7. Pass a hard walltime to every `sbatch` (`-t` / `--time`).
//!
//! No real network. No real Slurm.

mod common;

use std::fs;
use std::time::{Duration, Instant};

use prometheus_slurm::{Error, JobId, JobState, JOB_PREFIX};

use common::{job_name_on_disk, scancel_has_user_flag, Mock};

fn expected_job_name(suffix: &str) -> String {
    format!("{JOB_PREFIX}-{suffix}")
}

#[test]
fn submit_rejects_gpus_3() {
    let mock = Mock::new();
    let client = mock.client();
    let req = mock.submit_req("run-gpu3", "g3", 3);
    match client.submit(req) {
        Err(Error::BadGpuCount(3)) => {}
        other => panic!("expected Error::BadGpuCount(3), got {other:?}"),
    }
    assert!(
        mock.sbatch_lines().is_empty(),
        "gpus=3 must be refused before sbatch, log={:?}",
        mock.sbatch_lines()
    );
}

#[test]
fn submit_rejects_gpus_8() {
    let mock = Mock::new();
    let client = mock.client();
    let req = mock.submit_req("run-gpu8", "g8", 8);
    match client.submit(req) {
        Err(Error::BadGpuCount(8)) => {}
        other => panic!("expected Error::BadGpuCount(8), got {other:?}"),
    }
    assert!(
        mock.sbatch_lines().is_empty(),
        "gpus=8 must be refused before sbatch, log={:?}",
        mock.sbatch_lines()
    );
}

#[test]
fn submit_rejects_third_concurrent_live_fv_job() {
    let mock = Mock::new();
    // Two live fv- jobs plus an unrelated job. Only fv- counts toward the cap.
    mock.set_squeue_out(
        "111 gpu fv-one alice R 1:00:00 1 n1\n\
         222 gpu fv-two alice PD 0:10:00 1 n1\n\
         333 gpu opp-other alice R 2:00:00 1 n1\n",
    );
    let client = mock.client();
    let req = mock.submit_req("run-cap", "third", 1);
    match client.submit(req) {
        Err(Error::ConcurrentCap) => {}
        other => panic!("expected Error::ConcurrentCap, got {other:?}"),
    }
    assert!(
        mock.sbatch_lines().is_empty(),
        "third live fv- job must not reach sbatch, log={:?}",
        mock.sbatch_lines()
    );
}

#[test]
fn submit_allows_when_only_one_fv_job_is_live() {
    let mock = Mock::new();
    mock.set_squeue_out(
        "111 gpu fv-one alice R 1:00:00 1 n1\n\
         333 gpu opp-other alice R 2:00:00 1 n1\n",
    );
    mock.set_sbatch_jobid("5555");
    let client = mock.client();
    let req = mock.submit_req("run-ok", "second", 1);
    let id = client.submit(req).expect("one live fv- job must not cap");
    assert_eq!(id.0, "5555");
}

#[test]
fn submit_writes_job_name_before_sbatch_even_if_sbatch_fails() {
    let mock = Mock::new();
    mock.set_sbatch_mode("fail");
    let client = mock.client();
    let suffix = "pre-fail";
    let run_id = "run-pre-fail";
    let name = expected_job_name(suffix);
    let err = client
        .submit(mock.submit_req(run_id, suffix, 1))
        .expect_err("sbatch fail must surface");
    assert!(
        matches!(err, Error::Submit(_) | Error::Other(_)),
        "unexpected error variant {err:?}"
    );
    assert!(
        job_name_on_disk(&mock.state_at_sbatch(), run_id, &name),
        "job name {name} must already be on disk under state_dir/runs/{run_id}/ \
         at the moment sbatch is invoked (snapshot missing or name absent). \
         snapshot={:?} live={}",
        mock.state_at_sbatch(),
        job_name_on_disk(&mock.state_dir, run_id, &name)
    );
    assert!(
        job_name_on_disk(&mock.state_dir, run_id, &name),
        "job name {name} must remain on disk after sbatch fails"
    );
}

#[test]
fn submit_writes_job_name_before_sbatch_even_if_sbatch_panics() {
    let mock = Mock::new();
    mock.set_sbatch_mode("panic");
    let client = mock.client();
    let suffix = "pre-panic";
    let run_id = "run-pre-panic";
    let name = expected_job_name(suffix);
    let _ = client.submit(mock.submit_req(run_id, suffix, 1));
    assert!(
        job_name_on_disk(&mock.state_at_sbatch(), run_id, &name),
        "job name {name} must be recorded before mocked sbatch aborts"
    );
    assert!(
        job_name_on_disk(&mock.state_dir, run_id, &name),
        "job name {name} must survive a crashing sbatch so lookup by name remains possible"
    );
}

#[test]
fn state_on_hung_squeue_returns_timeout_or_unknown_quickly() {
    let mock = Mock::new();
    mock.hang_queries();
    let client = mock.client();
    let start = Instant::now();
    let result = client.state(&JobId("999".into()));
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(2),
        "hung squeue/sacct must be bounded by cfg.query_timeout (200ms), elapsed={elapsed:?}"
    );
    match result {
        Err(Error::QueryTimeout) | Ok(JobState::Unknown) => {}
        other => panic!("stuck query must return QueryTimeout or JobState::Unknown, got {other:?}"),
    }
}

#[test]
fn cancel_sends_scancel_for_that_id_never_minus_u() {
    let mock = Mock::new();
    let client = mock.client();
    client
        .cancel(&JobId("777".into()))
        .expect("cancel of a single id");
    let lines = mock.scancel_lines();
    assert!(
        !lines.is_empty(),
        "cancel must spawn scancel, remote log:\n{}",
        mock.remote_log()
    );
    assert!(
        lines.iter().any(|l| l.contains("777")),
        "scancel must mention job 777, got {lines:?}"
    );
    assert!(
        !mock.used_scancel_u(),
        "never scancel -u / --user; got {lines:?}"
    );
    for l in &lines {
        assert!(
            !scancel_has_user_flag(l),
            "scancel line must not contain -u/--user: {l}"
        );
    }
}

#[test]
fn reap_tracked_cancels_only_ids_listed_for_that_run() {
    let mock = Mock::new();
    let run_a = mock.state_dir.join("runs").join("run-a");
    let run_b = mock.state_dir.join("runs").join("run-b");
    fs::create_dir_all(&run_a).unwrap();
    fs::create_dir_all(&run_b).unwrap();
    fs::write(run_a.join("job-ids"), "101\n102\n").unwrap();
    fs::write(run_b.join("job-ids"), "201\n").unwrap();
    let client = mock.client();
    client.reap_tracked("run-a").expect("reap_tracked(run-a)");
    let cancelled = mock.cancelled_ids();
    let joined = mock.scancel_lines().join(" ");
    assert!(
        cancelled.iter().any(|id| id == "101") || joined.contains("101"),
        "must scancel 101; cancelled={cancelled:?} lines={joined:?}"
    );
    assert!(
        cancelled.iter().any(|id| id == "102") || joined.contains("102"),
        "must scancel 102; cancelled={cancelled:?} lines={joined:?}"
    );
    assert!(
        !cancelled.iter().any(|id| id == "201") && !joined.contains("201"),
        "must not scancel run-b's 201; cancelled={cancelled:?} lines={joined:?}"
    );
    assert!(
        !mock.used_scancel_u(),
        "reap must never scancel -u; lines={joined:?}"
    );
}

#[test]
fn wait_returns_terminal_state() {
    let mock = Mock::new();
    mock.set_job_state("42", "COMPLETED");
    let client = mock.client();
    let st = client
        .wait(&JobId("42".into()), Duration::from_secs(2))
        .expect("wait on COMPLETED");
    assert_eq!(st, JobState::Completed);
}

#[test]
fn wait_on_timeout_cancels_the_job() {
    let mock = Mock::new();
    mock.set_job_state("88", "RUNNING");
    mock.set_default_state("RUNNING");
    let client = mock.client();
    let start = Instant::now();
    let result = client.wait(&JobId("88".into()), Duration::from_millis(250));
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(2),
        "wait budget is 250ms; elapsed={elapsed:?}"
    );
    let lines = mock.scancel_lines();
    let cancelled = mock.cancelled_ids();
    assert!(
        cancelled.iter().any(|id| id == "88") || lines.iter().any(|l| l.contains("88")),
        "wait must cancel the job on local budget expiry; cancelled={cancelled:?} lines={lines:?} result={result:?}"
    );
    match result {
        Ok(st) => assert!(
            st.is_terminal() || st == JobState::Unknown,
            "wait after budget should not return a live state, got {st:?}"
        ),
        Err(_) => {}
    }
}

#[test]
fn job_name_is_fv_dash_suffix() {
    let mock = Mock::new();
    mock.set_sbatch_jobid("9001");
    let client = mock.client();
    let suffix = "train-v0";
    let name = expected_job_name(suffix);
    assert_eq!(name, "fv-train-v0");
    let id = client
        .submit(mock.submit_req("run-name", suffix, 1))
        .expect("submit");
    assert_eq!(id.0, "9001");
    let sbatch = mock.sbatch_lines().join(" ");
    assert!(
        sbatch.contains(&name),
        "sbatch must use job name {name}, got {sbatch:?}"
    );
    assert!(
        job_name_on_disk(&mock.state_dir, "run-name", &name),
        "job-name record under state_dir/runs/run-name/ must contain {name}"
    );
}

#[test]
fn job_ids_path_is_state_dir_runs_run_id_job_ids() {
    let mock = Mock::new();
    let client = mock.client();
    let p = client.job_ids_path("run-xyz");
    assert_eq!(
        p,
        mock.state_dir.join("runs").join("run-xyz").join("job-ids")
    );
}

#[test]
fn submit_fills_numeric_id_in_job_ids_file() {
    let mock = Mock::new();
    mock.set_sbatch_jobid("4242");
    let client = mock.client();
    let run_id = "run-fill";
    let id = client
        .submit(mock.submit_req(run_id, "fill", 2))
        .expect("submit gpus=2");
    assert_eq!(id.0, "4242");
    let path = client.job_ids_path(run_id);
    let text = fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "job-ids file must exist at {} after successful submit",
            path.display()
        )
    });
    assert!(
        text.contains("4242"),
        "job-ids must contain the numeric id, got {text:?}"
    );
}

#[test]
fn submit_includes_hard_walltime_and_accepts_allowed_gpus() {
    let mock = Mock::new();
    mock.set_sbatch_jobid("11");
    let client = mock.client();
    let mut req = mock.submit_req("run-wall", "wall", 4);
    req.walltime = "01:15:00".into();
    client.submit(req).expect("gpus=4 allowed");
    let sbatch = mock.sbatch_lines().join(" ");
    assert!(
        sbatch.contains("01:15:00"),
        "sbatch must include the request walltime, got {sbatch:?}"
    );
    let has_time_flag = sbatch
        .split_whitespace()
        .any(|t| t == "-t" || t == "--time" || t.starts_with("-t") || t.starts_with("--time="));
    assert!(
        has_time_flag,
        "sbatch must pass -t/--time (hard walltime), got {sbatch:?}"
    );
}

#[test]
fn fetch_copies_run_artifacts_to_dest() {
    let mock = Mock::new();
    let run_id = "run-fetch";
    let remote = mock.mock_dir.join("remote").join(run_id);
    fs::create_dir_all(&remote).unwrap();
    fs::write(remote.join("slurm.out"), "hello-from-cluster\n").unwrap();
    let dest = mock.state_dir.join("fetch-dest");
    let client = mock.client();
    client.fetch(run_id, &dest).expect("fetch");
    let found = dest.join("slurm.out");
    let nested = dest.join(run_id).join("slurm.out");
    let body = fs::read_to_string(&found)
        .or_else(|_| fs::read_to_string(&nested))
        .unwrap_or_default();
    assert!(
        body.contains("hello-from-cluster"),
        "fetch must copy remote {run_id}/slurm.out into dest (tried {} and {})",
        found.display(),
        nested.display()
    );
}
