#![cfg(feature = "kvm")]
//! Oracle tests for leftover D1 Firecracker Backend methods.
//!
//! Production `boot` / `snapshot` / `fork` / `call` / `kill` / `list` are
//! `unimplemented!` stubs. Tests prefixed `prod_` call those methods and must
//! fail until the Backend is implemented. Tests prefixed `ref_` exercise the
//! in-memory simulator in `tests/reference/firecracker.rs` and must pass.
//!
//! Pool<Firecracker> uses Engine, not Backend. These tests call Backend
//! methods on Firecracker / RefFirecracker directly.
//!
//! KvmMissing: when `/dev/kvm` is absent, production `Firecracker::new` or the
//! first `boot` must fail. Prefer `Error::Backend` text that names kvm until a
//! dedicated `Error::KvmMissing` variant exists. This host has `/dev/kvm`, so
//! that branch is not exercised.
//!
//! Real Firecracker VMs are skipped unless both `PROMETHEUS_FC_KERNEL` and
//! `PROMETHEUS_FC_ROOTFS` are set to existing files.

mod common;
mod reference;

use std::path::{Path, PathBuf};

use prometheus_envs::{
    Backend, Error, Firecracker, PoolConfig, SandboxId, SnapshotId, View, FORK_BUDGET_MS,
    GRADER_ROOT, GRPO_GROUP,
};

use common::{
    editor_write, is_egress_or_live, python, sample_image, shell, src_does_not_import_tests,
    HIDDEN_TEST, README_BODY, WORKSPACE_README,
};
use reference::artifact_for;
use reference::firecracker::RefFirecracker;

struct FcFiles {
    _dir: tempfile::TempDir,
    socket: PathBuf,
    kernel: PathBuf,
    rootfs: PathBuf,
}

impl FcFiles {
    fn present() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let kernel = dir.path().join("vmlinux.bin");
        let rootfs = dir.path().join("rootfs.ext4");
        std::fs::write(&kernel, b"oracle-kernel").unwrap();
        std::fs::write(&rootfs, b"oracle-rootfs").unwrap();
        Self {
            socket: dir.path().join("firecracker.sock"),
            kernel,
            rootfs,
            _dir: dir,
        }
    }

    fn missing_kernel() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let rootfs = dir.path().join("rootfs.ext4");
        std::fs::write(&rootfs, b"oracle-rootfs").unwrap();
        Self {
            socket: dir.path().join("firecracker.sock"),
            kernel: dir.path().join("vmlinux.bin"),
            rootfs,
            _dir: dir,
        }
    }

    fn missing_rootfs() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let kernel = dir.path().join("vmlinux.bin");
        std::fs::write(&kernel, b"oracle-kernel").unwrap();
        Self {
            socket: dir.path().join("firecracker.sock"),
            kernel,
            rootfs: dir.path().join("rootfs.ext4"),
            _dir: dir,
        }
    }
}

fn prod_fc(files: &FcFiles) -> Firecracker {
    Firecracker::new(
        files.socket.clone(),
        files.kernel.clone(),
        files.rootfs.clone(),
    )
    .expect("Firecracker::new")
}

fn ref_fc(files: &FcFiles) -> RefFirecracker {
    RefFirecracker::new(
        files.socket.clone(),
        files.kernel.clone(),
        files.rootfs.clone(),
    )
    .expect("RefFirecracker::new")
}

fn ref_fc_cfg(files: &FcFiles, cfg: PoolConfig) -> RefFirecracker {
    RefFirecracker::with_config(
        files.socket.clone(),
        files.kernel.clone(),
        files.rootfs.clone(),
        cfg,
    )
    .expect("RefFirecracker::with_config")
}

fn backend_mentions(err: &Error, needle: &str) -> bool {
    match err {
        Error::Backend(s) => s.to_ascii_lowercase().contains(needle),
        _ => false,
    }
}

fn real_vm_kernel_rootfs() -> Option<(PathBuf, PathBuf)> {
    let kernel = std::env::var("PROMETHEUS_FC_KERNEL")
        .ok()
        .filter(|s| !s.is_empty())?;
    let rootfs = std::env::var("PROMETHEUS_FC_ROOTFS")
        .ok()
        .filter(|s| !s.is_empty())?;
    let k = PathBuf::from(kernel);
    let r = PathBuf::from(rootfs);
    if k.is_file() && r.is_file() {
        Some((k, r))
    } else {
        None
    }
}

fn assert_grpo_forks<B: Backend>(b: &mut B) {
    let img = sample_image("grpo");
    let sb = b.boot(&img, 0).expect("boot");
    let snap = b.snapshot(&sb, 1).expect("snapshot");
    b.kill(&sb, 2)
        .expect("kill parent so 16 forks fit capacity");
    let mut ids = Vec::with_capacity(GRPO_GROUP);
    for i in 0..GRPO_GROUP {
        let (id, ms) = b.fork(&snap, 10 + i as u64).expect("fork");
        assert!(
            ms <= FORK_BUDGET_MS,
            "fork duration_ms {ms} exceeds FORK_BUDGET_MS {FORK_BUDGET_MS}"
        );
        ids.push(id);
    }
    assert_eq!(ids.len(), GRPO_GROUP);
    let overflow = b.fork(&snap, 100);
    assert_eq!(
        overflow,
        Err(Error::PoolExhausted {
            capacity: PoolConfig::default().capacity
        })
    );
}

fn assert_agent_grader_hidden<B: Backend>(b: &mut B) {
    let sb = b.boot(&sample_image("views"), 0).expect("boot");
    let agent = b.list(&sb, View::Agent).expect("agent list");
    let grader = b.list(&sb, View::Grader).expect("grader list");
    assert_eq!(
        agent.get(WORKSPACE_README).map(Vec::as_slice),
        Some(README_BODY)
    );
    assert!(
        !agent.contains_key(HIDDEN_TEST),
        "Agent view must hide {HIDDEN_TEST}"
    );
    assert!(
        !agent
            .keys()
            .any(|p| p == GRADER_ROOT || p.starts_with("/grader/")),
        "Agent view leaked grader paths: {:?}",
        agent.keys().collect::<Vec<_>>()
    );
    assert_eq!(
        grader.get(HIDDEN_TEST).map(Vec::as_slice),
        Some(b"assert True\n".as_ref())
    );
}

fn assert_tamper<B: Backend>(b: &mut B) {
    let sb = b.boot(&sample_image("tamper"), 0).expect("boot");
    let req = editor_write("tamper", "snap", HIDDEN_TEST, "hacked");
    assert_eq!(
        b.call(&sb, &req, 1),
        Err(Error::TestFileWrite {
            path: HIDDEN_TEST.into()
        })
    );
}

fn assert_egress<B: Backend>(b: &mut B) {
    let sb = b.boot(&sample_image("egress"), 0).expect("boot");
    let curl = shell("curl", "snap", "curl https://example.com");
    assert!(is_egress_or_live(
        &b.call(&sb, &curl, 1).expect_err("shell egress")
    ));
    let py = python(
        "py-net",
        "snap",
        "import urllib.request; urllib.request.urlopen('https://example.com')",
    );
    assert!(is_egress_or_live(
        &b.call(&sb, &py, 2).expect_err("python egress")
    ));
}

fn assert_credentials<B: Backend>(b: &mut B) {
    let mut img = sample_image("creds");
    img.env.insert("API_SECRET".into(), "x".into());
    assert_eq!(
        b.boot(&img, 0),
        Err(Error::CredentialInSandbox {
            name: "API_SECRET".into()
        })
    );
}

fn assert_not_found<B: Backend>(b: &mut B) {
    let missing_sb = SandboxId("no-such-sandbox".into());
    let missing_snap = SnapshotId("no-such-snapshot".into());
    assert_eq!(b.snapshot(&missing_sb, 0), Err(Error::SandboxNotFound));
    assert_eq!(b.kill(&missing_sb, 0), Err(Error::SandboxNotFound));
    assert_eq!(
        b.list(&missing_sb, View::Agent),
        Err(Error::SandboxNotFound)
    );
    let req = shell("ghost", "snap", "echo hi");
    assert_eq!(b.call(&missing_sb, &req, 0), Err(Error::SandboxNotFound));
    assert_eq!(b.fork(&missing_snap, 0), Err(Error::SnapshotNotFound));
}

fn assert_kill<B: Backend>(b: &mut B) {
    let sb = b.boot(&sample_image("kill"), 0).expect("boot");
    b.kill(&sb, 1).expect("kill");
    assert_eq!(b.list(&sb, View::Agent), Err(Error::SandboxNotFound));
    assert_eq!(b.kill(&sb, 2), Err(Error::SandboxNotFound));
}

// ---------------------------------------------------------------------------
// Production Firecracker Backend stubs. Must panic unimplemented until D1.
// Do not #[ignore]. Do not #[should_panic].
// ---------------------------------------------------------------------------

#[test]
fn prod_boot_errors_if_kernel_missing() {
    let files = FcFiles::missing_kernel();
    let mut fc = prod_fc(&files);
    let err = fc.boot(&sample_image("k"), 0).expect_err("missing kernel");
    assert!(
        backend_mentions(&err, "kernel"),
        "expected Backend error naming kernel, got {err:?}"
    );
}

#[test]
fn prod_boot_errors_if_rootfs_missing() {
    let files = FcFiles::missing_rootfs();
    let mut fc = prod_fc(&files);
    let err = fc.boot(&sample_image("r"), 0).expect_err("missing rootfs");
    assert!(
        backend_mentions(&err, "rootfs"),
        "expected Backend error naming rootfs, got {err:?}"
    );
}

#[test]
fn prod_grpo_group_16_forks_under_budget_then_pool_exhausted() {
    let files = FcFiles::present();
    let mut fc = prod_fc(&files);
    assert_grpo_forks(&mut fc);
}

#[test]
fn prod_kill() {
    let files = FcFiles::present();
    let mut fc = prod_fc(&files);
    assert_kill(&mut fc);
}

#[test]
fn prod_agent_vs_grader_list_hides_tests() {
    let files = FcFiles::present();
    let mut fc = prod_fc(&files);
    assert_agent_grader_hidden(&mut fc);
}

#[test]
fn prod_tamper_on_test_file_write() {
    let files = FcFiles::present();
    let mut fc = prod_fc(&files);
    assert_tamper(&mut fc);
}

#[test]
fn prod_egress_denied() {
    let files = FcFiles::present();
    let mut fc = prod_fc(&files);
    assert_egress(&mut fc);
}

#[test]
fn prod_credentials_rejected_on_boot() {
    let files = FcFiles::present();
    let mut fc = prod_fc(&files);
    assert_credentials(&mut fc);
}

#[test]
fn prod_snapshot_sandbox_not_found() {
    let files = FcFiles::present();
    let mut fc = prod_fc(&files);
    assert_eq!(
        fc.snapshot(&SandboxId("no-such-sandbox".into()), 0),
        Err(Error::SandboxNotFound)
    );
}

#[test]
fn prod_fork_snapshot_not_found() {
    let files = FcFiles::present();
    let mut fc = prod_fc(&files);
    assert_eq!(
        fc.fork(&SnapshotId("no-such-snapshot".into()), 0),
        Err(Error::SnapshotNotFound)
    );
}

#[test]
fn prod_call_sandbox_not_found() {
    let files = FcFiles::present();
    let mut fc = prod_fc(&files);
    let req = shell("ghost", "snap", "echo hi");
    assert_eq!(
        fc.call(&SandboxId("no-such-sandbox".into()), &req, 0),
        Err(Error::SandboxNotFound)
    );
}

#[test]
fn prod_kill_sandbox_not_found() {
    let files = FcFiles::present();
    let mut fc = prod_fc(&files);
    assert_eq!(
        fc.kill(&SandboxId("no-such-sandbox".into()), 0),
        Err(Error::SandboxNotFound)
    );
}

#[test]
fn prod_list_sandbox_not_found() {
    let files = FcFiles::present();
    let fc = prod_fc(&files);
    assert_eq!(
        fc.list(&SandboxId("no-such-sandbox".into()), View::Agent),
        Err(Error::SandboxNotFound)
    );
}

#[test]
fn prod_list() {
    let files = FcFiles::present();
    let mut fc = prod_fc(&files);
    let sb = fc.boot(&sample_image("list"), 0).expect("boot");
    let agent = fc.list(&sb, View::Agent).expect("list");
    assert!(agent.contains_key(WORKSPACE_README));
}

#[test]
fn prod_call() {
    let files = FcFiles::present();
    let mut fc = prod_fc(&files);
    let sb = fc.boot(&sample_image("call"), 0).expect("boot");
    let out = fc
        .call(&sb, &shell("echo", "snap", "echo hello"), 1)
        .expect("call");
    let art = artifact_for(b"hello\n");
    assert!(out.ok, "call not ok: {:?}", out.error);
    assert_eq!(out.stdout_artifact.content_hash, art.content_hash);
}

#[test]
fn prod_snapshot() {
    let files = FcFiles::present();
    let mut fc = prod_fc(&files);
    let sb = fc.boot(&sample_image("snap"), 0).expect("boot");
    let _ = fc.snapshot(&sb, 1).expect("snapshot");
}

#[test]
fn prod_fork() {
    let files = FcFiles::present();
    let mut fc = prod_fc(&files);
    let sb = fc.boot(&sample_image("fork"), 0).expect("boot");
    let snap = fc.snapshot(&sb, 1).expect("snapshot");
    let (child, ms) = fc.fork(&snap, 2).expect("fork");
    assert!(ms <= FORK_BUDGET_MS);
    assert_ne!(child, sb);
}

// ---------------------------------------------------------------------------
// In-memory reference. Must pass without production Firecracker / Engine.
// ---------------------------------------------------------------------------

#[test]
fn ref_src_does_not_import_tests() {
    src_does_not_import_tests();
}

#[test]
fn ref_records_socket_kernel_rootfs() {
    let files = FcFiles::present();
    let fc = ref_fc(&files);
    assert_eq!(fc.socket(), files.socket.as_path());
    assert_eq!(fc.kernel(), files.kernel.as_path());
    assert_eq!(fc.rootfs(), files.rootfs.as_path());
}

#[test]
fn ref_new_ok_when_images_missing_boot_fails() {
    let files = FcFiles::missing_kernel();
    let fc = ref_fc(&files);
    assert!(!fc.kernel().is_file());
}

#[test]
fn ref_boot_errors_if_kernel_missing() {
    let files = FcFiles::missing_kernel();
    let mut fc = ref_fc(&files);
    let err = fc.boot(&sample_image("k"), 0).expect_err("missing kernel");
    assert!(
        backend_mentions(&err, "kernel"),
        "expected Backend error naming kernel, got {err:?}"
    );
}

#[test]
fn ref_boot_errors_if_rootfs_missing() {
    let files = FcFiles::missing_rootfs();
    let mut fc = ref_fc(&files);
    let err = fc.boot(&sample_image("r"), 0).expect_err("missing rootfs");
    assert!(
        backend_mentions(&err, "rootfs"),
        "expected Backend error naming rootfs, got {err:?}"
    );
}

#[test]
fn ref_grpo_group_16_forks_under_budget_then_pool_exhausted() {
    let files = FcFiles::present();
    let mut fc = ref_fc(&files);
    assert_eq!(fc.config().capacity, 16);
    assert_grpo_forks(&mut fc);
    assert_eq!(fc.live_count(), GRPO_GROUP);
}

#[test]
fn ref_kill() {
    let files = FcFiles::present();
    let mut fc = ref_fc(&files);
    assert_kill(&mut fc);
    assert_eq!(fc.live_count(), 0);
}

#[test]
fn ref_agent_vs_grader_list_hides_tests() {
    let files = FcFiles::present();
    let mut fc = ref_fc(&files);
    assert_agent_grader_hidden(&mut fc);
}

#[test]
fn ref_tamper_on_test_file_write() {
    let files = FcFiles::present();
    let mut fc = ref_fc(&files);
    assert_tamper(&mut fc);
    assert!(
        !fc.tamper_flags().is_empty(),
        "test-file write must set a tamper flag"
    );
}

#[test]
fn ref_egress_denied() {
    let files = FcFiles::present();
    let mut fc = ref_fc(&files);
    assert_egress(&mut fc);
}

#[test]
fn ref_credentials_rejected_on_boot() {
    let files = FcFiles::present();
    let mut fc = ref_fc(&files);
    assert_credentials(&mut fc);
}

#[test]
fn ref_snapshot_not_found_and_sandbox_not_found() {
    let files = FcFiles::present();
    let mut fc = ref_fc(&files);
    assert_not_found(&mut fc);
}

#[test]
fn ref_pool_exhausted_small_capacity() {
    let files = FcFiles::present();
    let mut fc = ref_fc_cfg(
        &files,
        PoolConfig {
            capacity: 2,
            ..PoolConfig::default()
        },
    );
    fc.boot(&sample_image("a"), 0).unwrap();
    fc.boot(&sample_image("b"), 1).unwrap();
    assert_eq!(
        fc.boot(&sample_image("c"), 2),
        Err(Error::PoolExhausted { capacity: 2 })
    );
}

#[test]
fn ref_hidden_tests_visible_if_not_under_grader() {
    let files = FcFiles::present();
    let mut fc = ref_fc(&files);
    let mut img = sample_image("leak");
    img.hidden_tests
        .insert("/workspace/secret_test.py".into(), b"assert 0".to_vec());
    assert_eq!(fc.boot(&img, 0), Err(Error::HiddenTestsVisible));
}

#[test]
fn ref_forks_are_isolated() {
    let files = FcFiles::present();
    let mut fc = ref_fc(&files);
    let sb = fc.boot(&sample_image("iso"), 0).unwrap();
    let snap = fc.snapshot(&sb, 1).unwrap();
    let (a, ms_a) = fc.fork(&snap, 2).unwrap();
    let (b, ms_b) = fc.fork(&snap, 3).unwrap();
    assert!(ms_a <= FORK_BUDGET_MS && ms_b <= FORK_BUDGET_MS);
    fc.call(
        &a,
        &editor_write("w", "snap", WORKSPACE_README, "from-a"),
        4,
    )
    .unwrap();
    let listed_b = fc.list(&b, View::Agent).unwrap();
    assert_eq!(
        listed_b.get(WORKSPACE_README).map(Vec::as_slice),
        Some(README_BODY)
    );
}

#[test]
fn ref_call_shell_echo() {
    let files = FcFiles::present();
    let mut fc = ref_fc(&files);
    let sb = fc.boot(&sample_image("echo"), 0).unwrap();
    let out = fc
        .call(&sb, &shell("echo", "snap", "echo hello"), 1)
        .unwrap();
    let art = artifact_for(b"hello\n");
    assert!(out.ok, "call not ok: {:?}", out.error);
    assert_eq!(out.stdout_artifact.content_hash, art.content_hash);
}

#[test]
fn ref_kvm_device_present_on_this_host() {
    // Documentation for KvmMissing: this box has /dev/kvm. When it does not,
    // production Firecracker construction or boot must fail. The in-memory
    // simulator never opens /dev/kvm, so this assertion is about the host,
    // not the stub Backend methods.
    assert!(
        Path::new("/dev/kvm").exists(),
        "expected /dev/kvm on this host (KvmMissing is the production failure when absent)"
    );
}

// ---------------------------------------------------------------------------
// Real VM. Skip unless PROMETHEUS_FC_KERNEL and PROMETHEUS_FC_ROOTFS are set.
// ---------------------------------------------------------------------------

#[test]
fn real_vm_boot_skipped_unless_kernel_rootfs_env() {
    let Some((kernel, rootfs)) = real_vm_kernel_rootfs() else {
        eprintln!(
            "skip real-VM: set PROMETHEUS_FC_KERNEL and PROMETHEUS_FC_ROOTFS to existing files"
        );
        return;
    };
    let socket = std::env::temp_dir().join("prometheus-fc-oracle.sock");
    let mut fc = Firecracker::new(socket, kernel, rootfs).expect("Firecracker::new");
    let sb = fc.boot(&sample_image("real-vm"), 0).expect("real-VM boot");
    let agent = fc.list(&sb, View::Agent).expect("list");
    assert!(
        !agent
            .keys()
            .any(|p| p == GRADER_ROOT || p.starts_with("/grader/")),
        "Agent view leaked grader paths"
    );
    fc.kill(&sb, 1).expect("kill");
}
