//! Independent L3 monitors / kill-switch reference.
//! **Not** `prometheus_monitors::Monitors`. Production never imports `tests/`.
//!
//! Slow and obvious: one snapshot in memory, one JSON file on disk, string
//! equality for grader hashes. No wall clock. `NowMs` is always an argument.
//!
//! # Freeze file (`MonitorConfig.freeze_path`)
//!
//! Absent file means not frozen: empty snapshot
//! `{frozen: false, frozen_at: null, reason: null, violations: []}`.
//!
//! Present file is serde JSON of [`KillSnapshot`] (default serde: unit enum
//! variants as `"Kernel"` / `"Graders"` / `"HeldOut"` / `"Monitors"`). Load
//! that snapshot as the in-memory state. Corrupt / truncated / non-object /
//! wrong types / directory at the path -> [`Error::Message`] (fail closed).
//! Extra JSON keys are ignored (default serde).
//!
//! `kill` / any trip writes the current snapshot with `serde_json` (compact).
//! Parent directories are created. Byte-for-byte JSON is not a contract;
//! `serde_json` round-trip equality with [`KillSnapshot`] is.
//!
//! # Kill
//!
//! First `kill(reason, now)`:
//! 1. `frozen = true`, `frozen_at = Some(now)`, `reason = Some(reason)`.
//! 2. Does not append a violation (trips append first, then kill).
//! 3. Persist. Return `Ok(snapshot.clone())`.
//!
//! Second `kill`: snapshot unchanged (first `frozen_at` / `reason` win),
//! persist is skipped, return [`Error::AlreadyFrozen`]. Still frozen.
//!
//! # Trips (same path for `plant` and the `note_*` methods)
//!
//! 1. Append `Violation { boundary, at_ms: now, detail }`.
//! 2. If not yet frozen, apply `kill(freeze_reason(boundary), now)` in place
//!    (do not return `AlreadyFrozen` to the caller).
//! 3. If already frozen, keep `frozen_at` / `reason`, persist the extra
//!    violation.
//! 4. Return `trip_err(boundary)`.
//!
//! `detail` is the caller string: capability / grader id / who / self-check
//! text / `plant`'s `detail`.
//!
//! | boundary | `freeze_reason`              | `trip_err`                         |
//! |----------|------------------------------|------------------------------------|
//! | Kernel   | `kernel escape`              | `Error::Message("kernel escape")`  |
//! | Graders  | `grader hash mismatch`       | `Error::GraderHashMismatch`        |
//! | HeldOut  | `held-out access blocked`    | `Error::HeldOutAccess`             |
//! | Monitors | `monitor self-check failed`  | `Error::MonitorSelfCheck`          |
//!
//! `plant(b, detail, now)` is exactly that trip. So:
//! - `plant(Kernel, cap, now)` == `note_kernel_escape(cap, now)`
//! - `plant(HeldOut, who, now)` == `note_held_out_access(who, now)`
//! - `plant(Monitors, d, now)` == `note_self_check(d, now)`
//! - `plant(Graders, id, now)` == `note_grader_hash(id, <mismatch>, now)`
//!   when `detail` is the grader id (mismatch does not put the hash in
//!   `detail`).
//!
//! # `note_grader_hash`
//!
//! Exact string equality against `cfg.grader_hashes[grader_id]`.
//! Match (id present and strings equal): `Ok(())`, no freeze, no violation.
//! Any other case (wrong hash, unknown id, empty config): Graders trip
//! with `detail = grader_id`.
//!
//! # Four names
//!
//! `BOUNDARIES` is `kernel`, `graders`, `held_out`, `monitors`. The genome
//! cannot disable them; this reference has no disable / unfreeze API.

#![allow(dead_code)]

use std::fs;
use std::path::Path;

use prometheus_monitors::{Boundary, Error, KillSnapshot, MonitorConfig, NowMs, Result, Violation};

/// Same four names as [`prometheus_monitors::BOUNDARIES`], duplicated on
/// purpose so the oracle does not treat the production constant as gospel
/// without a test.
pub const BOUNDARIES: &[&str] = &["kernel", "graders", "held_out", "monitors"];

pub fn freeze_reason(boundary: Boundary) -> &'static str {
    match boundary {
        Boundary::Kernel => "kernel escape",
        Boundary::Graders => "grader hash mismatch",
        Boundary::HeldOut => "held-out access blocked",
        Boundary::Monitors => "monitor self-check failed",
    }
}

pub fn trip_err(boundary: Boundary) -> Error {
    match boundary {
        Boundary::Kernel => Error::Message("kernel escape".into()),
        Boundary::Graders => Error::GraderHashMismatch,
        Boundary::HeldOut => Error::HeldOutAccess,
        Boundary::Monitors => Error::MonitorSelfCheck,
    }
}

pub fn parse_boundary(name: &str) -> Result<Boundary> {
    match name {
        "kernel" => Ok(Boundary::Kernel),
        "graders" => Ok(Boundary::Graders),
        "held_out" => Ok(Boundary::HeldOut),
        "monitors" => Ok(Boundary::Monitors),
        other => Err(Error::UnknownBoundary(other.to_string())),
    }
}

pub struct RefMonitors {
    cfg: MonitorConfig,
    snapshot: KillSnapshot,
}

impl RefMonitors {
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

    pub fn note_held_out_access(&mut self, who: &str, now: NowMs) -> Result<()> {
        self.trip(Boundary::HeldOut, who, now)
    }

    pub fn note_kernel_escape(&mut self, capability: &str, now: NowMs) -> Result<()> {
        self.trip(Boundary::Kernel, capability, now)
    }

    pub fn note_self_check(&mut self, detail: &str, now: NowMs) -> Result<()> {
        self.trip(Boundary::Monitors, detail, now)
    }

    pub fn plant(&mut self, boundary: Boundary, detail: &str, now: NowMs) -> Result<()> {
        self.trip(boundary, detail, now)
    }

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

pub fn empty_snapshot() -> KillSnapshot {
    KillSnapshot {
        frozen: false,
        frozen_at: None,
        reason: None,
        violations: Vec::new(),
    }
}

pub fn load_freeze(path: &Path) -> Result<KillSnapshot> {
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

pub fn write_freeze(path: &Path, snapshot: &KillSnapshot) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| Error::Message(e.to_string()))?;
        }
    }
    let bytes = serde_json::to_vec(snapshot).map_err(|e| Error::Message(e.to_string()))?;
    fs::write(path, bytes).map_err(|e| Error::Message(e.to_string()))
}
