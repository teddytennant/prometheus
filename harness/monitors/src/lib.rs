//! Lab monitors and kill switch (spec 14.10, 15.5 L3).
//!
//! Four boundaries: the kernel, the graders, the held-out eval-gate, and
//! the monitors themselves. A planted violation on any of them trips the
//! kill switch. Kill freezes the kernel, graders, eval-gate, and monitors.
//!
//! [`NowMs`] is injected. Nothing here reads the wall clock.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub type NowMs = u64;
pub type Result<T> = std::result::Result<T, Error>;

/// Boundaries spec 14.10 names. The genome cannot disable these.
pub const BOUNDARIES: &[&str] = &["kernel", "graders", "held_out", "monitors"];

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("unknown boundary {0}")]
    UnknownBoundary(String),
    #[error("kill switch already frozen")]
    AlreadyFrozen,
    #[error("grader hash mismatch")]
    GraderHashMismatch,
    #[error("held-out access blocked")]
    HeldOutAccess,
    #[error("monitor self-check failed")]
    MonitorSelfCheck,
    #[error("{0}")]
    Message(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Boundary {
    Kernel,
    Graders,
    HeldOut,
    Monitors,
}

impl Boundary {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Kernel => "kernel",
            Self::Graders => "graders",
            Self::HeldOut => "held_out",
            Self::Monitors => "monitors",
        }
    }

    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "kernel" => Ok(Self::Kernel),
            "graders" => Ok(Self::Graders),
            "held_out" => Ok(Self::HeldOut),
            "monitors" => Ok(Self::Monitors),
            other => Err(Error::UnknownBoundary(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Violation {
    pub boundary: Boundary,
    pub at_ms: NowMs,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KillSnapshot {
    pub frozen: bool,
    pub frozen_at: Option<NowMs>,
    pub reason: Option<String>,
    pub violations: Vec<Violation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonitorConfig {
    /// Durable freeze file. Absent means not frozen.
    pub freeze_path: PathBuf,
    /// SHA-256 hex of grader sources, keyed by grader id.
    pub grader_hashes: BTreeMap<String, String>,
}

/// Kill switch plus the four boundary watches.
#[allow(dead_code)]
pub struct Monitors {
    cfg: MonitorConfig,
    snapshot: KillSnapshot,
}

impl Monitors {
    /// Open, loading a previous freeze from ``cfg.freeze_path`` if present.
    pub fn open(cfg: MonitorConfig) -> Result<Self> {
        let _ = cfg;
        unimplemented!("L3 Monitors::open")
    }

    pub fn frozen(&self) -> bool {
        unimplemented!("L3 Monitors::frozen")
    }

    pub fn snapshot(&self) -> KillSnapshot {
        unimplemented!("L3 Monitors::snapshot")
    }

    /// Record a grader-source hash. Mismatch against the config trips kill.
    pub fn note_grader_hash(
        &mut self,
        grader_id: &str,
        sha256_hex: &str,
        now: NowMs,
    ) -> Result<()> {
        let _ = (grader_id, sha256_hex, now);
        unimplemented!("L3 Monitors::note_grader_hash")
    }

    /// Record an attempt to read held-out task text. Any attempt trips kill.
    pub fn note_held_out_access(&mut self, who: &str, now: NowMs) -> Result<()> {
        let _ = (who, now);
        unimplemented!("L3 Monitors::note_held_out_access")
    }

    /// Record a kernel capability the genome should not have.
    pub fn note_kernel_escape(&mut self, capability: &str, now: NowMs) -> Result<()> {
        let _ = (capability, now);
        unimplemented!("L3 Monitors::note_kernel_escape")
    }

    /// Record a monitor self-check failure (tamper with this crate).
    pub fn note_self_check(&mut self, detail: &str, now: NowMs) -> Result<()> {
        let _ = (detail, now);
        unimplemented!("L3 Monitors::note_self_check")
    }

    /// Plant a violation for the L3 gate. Same path as a real trip.
    pub fn plant(&mut self, boundary: Boundary, detail: &str, now: NowMs) -> Result<()> {
        let _ = (boundary, detail, now);
        unimplemented!("L3 Monitors::plant")
    }

    /// Freeze kernel, graders, eval-gate, and monitors. Idempotent.
    pub fn kill(&mut self, reason: &str, now: NowMs) -> Result<KillSnapshot> {
        let _ = (reason, now);
        unimplemented!("L3 Monitors::kill")
    }
}
