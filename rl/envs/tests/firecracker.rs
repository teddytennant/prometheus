//! Group: Firecracker backend. Skipped unless `--features kvm`.
#![cfg(feature = "kvm")]

mod common;
mod reference;

use std::path::PathBuf;

use common::{sample_image, src_does_not_import_tests};
use prometheus_envs::{Firecracker, Pool, PoolConfig, GRPO_GROUP};

#[test]
fn firecracker_pool_snapshot_and_fork_group() {
    src_does_not_import_tests();
    let dir = tempfile::tempdir().expect("tmpdir");
    let backend = Firecracker::new(
        dir.path().join("fc.sock"),
        PathBuf::from("/nonexistent/vmlinux"),
        PathBuf::from("/nonexistent/rootfs.ext4"),
    )
    .expect("Firecracker::new");
    let _ = (backend.socket(), backend.kernel(), backend.rootfs());
    let mut pool = Pool::firecracker(backend, PoolConfig::default());
    let img = sample_image("img-fc");
    let id = pool.register_image(img).unwrap();
    let snap = pool.snapshot_from_image(&id, 0).unwrap();
    let g = pool.fork_group(&snap, GRPO_GROUP, 1).unwrap();
    assert_eq!(g.sandboxes.len(), GRPO_GROUP);
    assert!(g.all_under_budget());
    assert_eq!(pool.live_count(), GRPO_GROUP);
}
