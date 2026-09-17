//! Held-out eval-gate (spec 14.3, 14.8, 14.9, 15.5 L2).
//!
//! Runs the harness benchmark on request and returns scores only. Task
//! text never leaves this crate: there is no method that yields prompts,
//! answers, or item bodies. Gate: tasks never leave the gate.
//!
//! Public F3 suites (spec 11) are run here for the lab's held-out split.
//! The genome talks to this through the kernel capability API, not by
//! importing this crate.
//!
//! [`NowMs`] is injected. Nothing here reads the wall clock.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

mod store;

pub type NowMs = u64;
pub type Result<T> = std::result::Result<T, Error>;

/// Held-out suites rotate on this period (spec 14.9: every quarter).
pub const ROTATION_MS: u64 = 90 * 24 * 60 * 60 * 1000;

/// Harness-benchmark suite slugs (spec 14.8, 14.9). Subset of F3 public
/// suites plus internal speedruns. All run as held-out.
pub const HELD_OUT_SUITES: &[&str] = &[
    "re_bench",
    "mle_bench",
    "paperbench",
    "speedrun",
    "metr_time_horizon",
];

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("unknown suite {0}")]
    UnknownSuite(String),
    #[error("suite {0} not held-out")]
    NotHeldOut(String),
    #[error("subject {0} not found")]
    SubjectNotFound(String),
    #[error("task leak blocked")]
    TaskLeak,
    #[error("rotation refused: {0}")]
    RotationRefused(String),
    #[error("{0}")]
    Message(String),
}

/// What to score. Genome rev or a weight checkpoint (spec 14.3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Subject {
    GenomeRev(String),
    Checkpoint(String),
}

impl Subject {
    pub fn as_str(&self) -> &str {
        match self {
            Subject::GenomeRev(s) | Subject::Checkpoint(s) => s,
        }
    }
}

/// Fixed-compute scores. No task text, no prompts, no answers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scores {
    pub suite: String,
    pub subject: Subject,
    /// Research capability index in millipoints (spec 14.9).
    pub rci_milli: i64,
    pub n_items: u32,
    pub compute_ms: u64,
    pub harness_version: String,
    /// Hash of the held-out items, not the items.
    pub suite_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateConfig {
    pub suite_root: PathBuf,
    pub harness_version: String,
}

/// True when `slug` is a held-out harness-benchmark suite.
pub fn is_held_out_suite(slug: &str) -> bool {
    HELD_OUT_SUITES.contains(&slug)
}

pub struct EvalGate {
    store: store::Store,
}

impl EvalGate {
    pub fn open(cfg: GateConfig) -> Result<Self> {
        Ok(Self {
            store: store::Store::open(cfg)?,
        })
    }

    /// Run `suite` on `subject` at fixed compute. Returns [`Scores`] only.
    pub fn request(&self, subject: Subject, suite: &str, now: NowMs) -> Result<Scores> {
        let _ = now;
        self.store.request(subject, suite)
    }

    /// Install a new held-out suite written after `cutoff_ms`. Previous
    /// suite hashes stay queryable; item bodies do not.
    pub fn rotate(&mut self, suite: &str, cutoff_ms: NowMs, now: NowMs) -> Result<String> {
        self.store.rotate(suite, cutoff_ms, now)
    }

    pub fn suite_hash(&self, suite: &str) -> Result<String> {
        self.store.suite_hash(suite)
    }
}
