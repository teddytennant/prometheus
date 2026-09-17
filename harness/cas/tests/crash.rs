//! Group: crash / replay. Child put+pin then `_exit` / SIGKILL; parent open+get.
//! Linux only. Few kills (not the H11 1,000).

mod common;
mod reference;

use common::{cfg, pin};
use prometheus_cas::Store;
use reference::digest_hex;
use std::path::{Path, PathBuf};

const CHILD_KIND: &str = "PROMETHEUS_CAS_H8_CRASH_CHILD";
const CHILD_DIR: &str = "PROMETHEUS_CAS_H8_CRASH_DIR";
const CHILD_B0: &str = "PROMETHEUS_CAS_H8_CRASH_B0";
const CHILD_B1: &str = "PROMETHEUS_CAS_H8_CRASH_B1";
const CHILD_SENTINEL: &str = "PROMETHEUS_CAS_H8_CRASH_SENTINEL";

const BLOB: &[u8] = b"crash-blob-bytes";
const PIN_NAME: &str = "crash-pin";

fn child_mode() -> Option<String> {
    std::env::var(CHILD_KIND).ok()
}

fn child_dir() -> PathBuf {
    PathBuf::from(std::env::var(CHILD_DIR).expect("child dir"))
}

fn child_backends() -> Vec<PathBuf> {
    vec![
        PathBuf::from(std::env::var(CHILD_B0).expect("b0")),
        PathBuf::from(std::env::var(CHILD_B1).expect("b1")),
    ]
}

#[cfg(unix)]
fn exit_hard() -> ! {
    extern "C" {
        fn _exit(code: i32) -> !;
    }
    unsafe { _exit(0) }
}

#[cfg(unix)]
fn spawn_child(
    test_name: &str,
    kind: &str,
    dir: &Path,
    backends: &[PathBuf],
    sentinel: Option<&Path>,
) -> std::process::ExitStatus {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = std::process::Command::new(exe);
    cmd.args(["--exact", "--nocapture", "--test-threads=1", test_name])
        .env(CHILD_KIND, kind)
        .env(CHILD_DIR, dir)
        .env(CHILD_B0, &backends[0])
        .env(CHILD_B1, &backends[1])
        .env("RUST_TEST_THREADS", "1");
    if let Some(path) = sentinel {
        cmd.env(CHILD_SENTINEL, path);
    }
    cmd.status().expect("spawn crash child")
}

#[cfg(unix)]
fn child_put_pin() {
    let dir = child_dir();
    let backends = child_backends();
    let mut store = Store::create(&dir, backends, cfg()).expect("child create");
    let d = store.put(BLOB).expect("child put");
    store.pin(&d, pin(PIN_NAME)).expect("child pin");
}

#[cfg(unix)]
fn parent_open_and_get(dir: &Path, backends: Vec<PathBuf>) {
    let mut store = Store::open(dir, backends, cfg()).expect("parent open");
    let digest = prometheus_cas::Digest(digest_hex(BLOB));
    assert_eq!(store.get(&digest).expect("parent get"), BLOB);
    let report = store.gc().expect("parent gc");
    assert_eq!(
        report.blobs_removed, 0,
        "pin must survive crash so gc keeps the blob"
    );
    assert_eq!(store.get(&digest).expect("get after gc"), BLOB);
}

#[cfg(unix)]
#[test]
fn crash_exit_after_put_and_pin_is_durable() {
    if child_mode().as_deref() == Some("exit") {
        child_put_pin();
        exit_hard();
    }
    // Parent hits create first so a stub run fails without spawning.
    let probe = common::fresh_store();
    let _probe = Store::create(&probe.dir, probe.backends.clone(), cfg()).expect("create");
    let h = common::fresh_store();
    let status = spawn_child(
        "crash_exit_after_put_and_pin_is_durable",
        "exit",
        &h.dir,
        &h.backends,
        None,
    );
    assert!(status.success(), "crash child _exit: {status}");
    parent_open_and_get(&h.dir, h.backends.clone());
}

#[cfg(unix)]
#[test]
fn kill9_after_put_and_pin_is_durable() {
    if child_mode().as_deref() == Some("kill9") {
        child_put_pin();
        let sentinel = PathBuf::from(std::env::var(CHILD_SENTINEL).expect("sentinel"));
        std::fs::write(&sentinel, b"ready").expect("write sentinel");
        loop {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
    let probe = common::fresh_store();
    let _probe = Store::create(&probe.dir, probe.backends.clone(), cfg()).expect("create");
    let h = common::fresh_store();
    let sentinel = h.parent.path().join("sentinel");
    let exe = std::env::current_exe().expect("current_exe");
    let mut child = std::process::Command::new(&exe)
        .args([
            "--exact",
            "--nocapture",
            "--test-threads=1",
            "kill9_after_put_and_pin_is_durable",
        ])
        .env(CHILD_KIND, "kill9")
        .env(CHILD_DIR, &h.dir)
        .env(CHILD_B0, &h.backends[0])
        .env(CHILD_B1, &h.backends[1])
        .env(CHILD_SENTINEL, &sentinel)
        .env("RUST_TEST_THREADS", "1")
        .spawn()
        .expect("spawn kill9 child");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !sentinel.exists() {
        if std::time::Instant::now() > deadline {
            let _ = child.kill();
            panic!("kill9 child did not write sentinel");
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!("kill9 child exited before sentinel: {status}");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    #[cfg(unix)]
    {
        let _ = std::process::Command::new("kill")
            .args(["-9", &child.id().to_string()])
            .status();
    }
    let _ = child.wait();
    parent_open_and_get(&h.dir, h.backends.clone());
}

#[cfg(not(unix))]
#[test]
fn crash_tests_require_unix_but_still_hit_create() {
    let h = common::fresh_store();
    let _ = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
}
