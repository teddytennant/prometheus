//! Lab monitors and kill switch (spec 14.10, 15.5 L3).
//!
//! Four boundaries: the kernel, the graders, the held-out eval-gate, and
//! the monitors themselves. A planted violation on any of them trips the
//! kill switch. Kill freezes the kernel, graders, eval-gate, and monitors.
//!
//! [`NowMs`] is injected. Nothing here reads the wall clock.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

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
        let snapshot = load_freeze(&cfg.freeze_path)?;
        Ok(Self { cfg, snapshot })
    }

    pub fn frozen(&self) -> bool {
        self.snapshot.frozen
    }

    pub fn snapshot(&self) -> KillSnapshot {
        self.snapshot.clone()
    }

    /// Record a grader-source hash. Mismatch against the config trips kill.
    pub fn note_grader_hash(
        &mut self,
        grader_id: &str,
        sha256_hex: &str,
        now: NowMs,
    ) -> Result<()> {
        match self.cfg.grader_hashes.get(grader_id) {
            Some(expected) if expected == sha256_hex => Ok(()),
            _ => self.trip(Boundary::Graders, grader_id, now),
        }
    }

    /// Record an attempt to read held-out task text. Any attempt trips kill.
    pub fn note_held_out_access(&mut self, who: &str, now: NowMs) -> Result<()> {
        self.trip(Boundary::HeldOut, who, now)
    }

    /// Record a kernel capability the genome should not have.
    pub fn note_kernel_escape(&mut self, capability: &str, now: NowMs) -> Result<()> {
        self.trip(Boundary::Kernel, capability, now)
    }

    /// Record a monitor self-check failure (tamper with this crate).
    pub fn note_self_check(&mut self, detail: &str, now: NowMs) -> Result<()> {
        self.trip(Boundary::Monitors, detail, now)
    }

    /// Plant a violation for the L3 gate. Same path as a real trip.
    pub fn plant(&mut self, boundary: Boundary, detail: &str, now: NowMs) -> Result<()> {
        self.trip(boundary, detail, now)
    }

    /// Freeze kernel, graders, eval-gate, and monitors. Idempotent.
    pub fn kill(&mut self, reason: &str, now: NowMs) -> Result<KillSnapshot> {
        if self.snapshot.frozen {
            return Err(Error::AlreadyFrozen);
        }
        self.apply_kill(reason, now)?;
        Ok(self.snapshot.clone())
    }

    fn trip(&mut self, boundary: Boundary, detail: &str, now: NowMs) -> Result<()> {
        self.snapshot.violations.push(Violation {
            boundary,
            at_ms: now,
            detail: detail.to_string(),
        });
        if !self.snapshot.frozen {
            self.apply_kill(freeze_reason(boundary), now)?;
        } else {
            self.persist()?;
        }
        Err(trip_err(boundary))
    }

    fn apply_kill(&mut self, reason: &str, now: NowMs) -> Result<()> {
        self.snapshot.frozen = true;
        self.snapshot.frozen_at = Some(now);
        self.snapshot.reason = Some(reason.to_string());
        self.persist()
    }

    fn persist(&self) -> Result<()> {
        write_freeze(&self.cfg.freeze_path, &self.snapshot)
    }
}

fn freeze_reason(boundary: Boundary) -> &'static str {
    match boundary {
        Boundary::Kernel => "kernel escape",
        Boundary::Graders => "grader hash mismatch",
        Boundary::HeldOut => "held-out access blocked",
        Boundary::Monitors => "monitor self-check failed",
    }
}

fn trip_err(boundary: Boundary) -> Error {
    match boundary {
        Boundary::Kernel => Error::Message("kernel escape".into()),
        Boundary::Graders => Error::GraderHashMismatch,
        Boundary::HeldOut => Error::HeldOutAccess,
        Boundary::Monitors => Error::MonitorSelfCheck,
    }
}

fn empty_snapshot() -> KillSnapshot {
    KillSnapshot {
        frozen: false,
        frozen_at: None,
        reason: None,
        violations: Vec::new(),
    }
}

fn load_freeze(path: &Path) -> Result<KillSnapshot> {
    if !path.exists() {
        return Ok(empty_snapshot());
    }
    if path.is_dir() {
        return Err(Error::Message(format!(
            "freeze path is a directory: {}",
            path.display()
        )));
    }
    let bytes = fs::read(path).map_err(|e| Error::Message(e.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|e| Error::Message(e.to_string()))
}

fn write_freeze(path: &Path, snapshot: &KillSnapshot) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| Error::Message(e.to_string()))?;
        }
    }
    let bytes = serde_json::to_vec(snapshot).map_err(|e| Error::Message(e.to_string()))?;
    fs::write(path, bytes).map_err(|e| Error::Message(e.to_string()))
}
