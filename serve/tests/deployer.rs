//! Group: Deployer constructors/getters and BadGpuCount without slurm.

mod common;
mod reference;

use common::{assert_bad_gpu, disallowed_gpus, engine_config};
use prometheus_serve::Deployer;
use prometheus_slurm::{Client, ClientConfig};
use std::path::Path;
use std::time::Duration;

fn test_client() -> (tempfile::TempDir, Client) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let client = Client::new(ClientConfig {
        cluster: "mock-cluster".into(),
        remote_root: tmp.path().join("remote"),
        partition: "gpu".into(),
        state_dir: tmp.path().join("state"),
        query_timeout: Duration::from_secs(1),
        max_concurrent: 2,
    })
    .expect("Client::new");
    (tmp, client)
}

#[test]
fn new_stores_client_and_script() {
    let (_tmp, client) = test_client();
    let script = Path::new("/tmp/oracle-sglang.sh");
    let d = Deployer::new(client, script);
    assert_eq!(d.script(), script);
    let _ = d.client();
}

#[test]
fn submit_rejects_illegal_gpu_count_without_slurm() {
    // Must not look up sbatch/ssh on PATH. A BadGpuCount unit test is enough;
    // live submit-success waits for a mock. Live teardown is skipped.
    let (_tmp, client) = test_client();
    let d = Deployer::new(client, "/no/such/sglang.sh");
    for gpus in disallowed_gpus() {
        let cfg = engine_config(gpus, 8);
        let err = d
            .submit(&cfg, "run", "sfx", "00:30:00")
            .expect_err("illegal gpus");
        assert_bad_gpu(&err, gpus);
        assert!(reference::start_rejects_gpus(gpus));
    }
}
