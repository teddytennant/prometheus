//! In-memory Firecracker Backend simulator (D1 oracle).
//!
//! Slow and obvious. Production (`rl/envs/src`) must never import this module
//! and this module must never call production `Firecracker` or `Engine`.
//!
//! Contract (same isolation as InProcess, plus image-file checks):
//! - `boot` errors if `kernel` or `rootfs` is not a regular file
//! - hidden tests live under [`GRADER_ROOT`]; Agent `list` never sees them
//! - deny-all egress; credential-like env keys rejected on boot
//! - test-file writes are `TestFileWrite` and raise a tamper flag
//! - default live capacity is [`PoolConfig::default`] (16, matching GRPO)
//!
//! KvmMissing: production `Firecracker::new` or first `boot` must fail when
//! `/dev/kvm` is absent (until a dedicated `Error::KvmMissing` variant exists,
//! `Error::Backend` text that names kvm). This simulator never talks to KVM.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use prometheus_envs::{
    Backend, Error, Image, NowMs, PoolConfig, Result, SandboxId, SnapshotId, TamperFlag,
    ToolRequest, ToolResponse, View,
};

use super::RefPool;

/// Independent Firecracker Backend: in-memory isolation plus kernel/rootfs checks.
pub struct RefFirecracker {
    socket: PathBuf,
    kernel: PathBuf,
    rootfs: PathBuf,
    inner: RefPool,
}

impl RefFirecracker {
    pub fn new(socket: PathBuf, kernel: PathBuf, rootfs: PathBuf) -> Result<Self> {
        Self::with_config(socket, kernel, rootfs, PoolConfig::default())
    }

    pub fn with_config(
        socket: PathBuf,
        kernel: PathBuf,
        rootfs: PathBuf,
        cfg: PoolConfig,
    ) -> Result<Self> {
        Ok(Self {
            socket,
            kernel,
            rootfs,
            inner: RefPool::new(cfg),
        })
    }

    pub fn socket(&self) -> &Path {
        &self.socket
    }

    pub fn kernel(&self) -> &Path {
        &self.kernel
    }

    pub fn rootfs(&self) -> &Path {
        &self.rootfs
    }

    pub fn live_count(&self) -> usize {
        self.inner.live_count()
    }

    pub fn tamper_flags(&self) -> &[TamperFlag] {
        self.inner.tamper_flags()
    }

    pub fn config(&self) -> &PoolConfig {
        self.inner.config()
    }

    fn require_boot_images(&self) -> Result<()> {
        if !self.kernel.is_file() {
            return Err(Error::Backend(format!(
                "kernel image missing: {}",
                self.kernel.display()
            )));
        }
        if !self.rootfs.is_file() {
            return Err(Error::Backend(format!(
                "rootfs image missing: {}",
                self.rootfs.display()
            )));
        }
        Ok(())
    }
}

impl Backend for RefFirecracker {
    fn boot(&mut self, image: &Image, now: NowMs) -> Result<SandboxId> {
        self.require_boot_images()?;
        let iid = self.inner.register_image(image.clone())?;
        let snap = self.inner.snapshot_from_image(&iid, now)?;
        let g = self.inner.fork_group(&snap, 1, now)?;
        Ok(g.sandboxes
            .into_iter()
            .next()
            .expect("fork_group(1) yields one sandbox"))
    }

    fn snapshot(&mut self, sandbox: &SandboxId, now: NowMs) -> Result<SnapshotId> {
        self.inner.snapshot(sandbox, now)
    }

    fn fork(&mut self, snapshot: &SnapshotId, now: NowMs) -> Result<(SandboxId, u64)> {
        let g = self.inner.fork_group(snapshot, 1, now)?;
        Ok((g.sandboxes[0].clone(), g.durations_ms[0]))
    }

    fn call(&mut self, sandbox: &SandboxId, req: &ToolRequest, now: NowMs) -> Result<ToolResponse> {
        self.inner.call(sandbox, req.clone(), now)
    }

    fn kill(&mut self, sandbox: &SandboxId, now: NowMs) -> Result<()> {
        self.inner.kill(sandbox, now)
    }

    fn list(&self, sandbox: &SandboxId, view: View) -> Result<BTreeMap<String, Vec<u8>>> {
        match view {
            View::Agent => self.inner.agent_view(sandbox),
            View::Grader => self.inner.grader_view(sandbox),
        }
    }
}
