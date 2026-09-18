//! Rung 0 end to end (spec 6, 15.5 A7, 16.2 V5).
//!
//! CPU analog of V5: frozen ladder numbers, a step loop that records a
//! loss curve, and checkpoint/resume across job boundaries. Does not
//! train a 0.1B model, does not touch GPUs, and does not import
//! `prometheus-tokenizer` or the data loader. The tokenizer freeze is a
//! hash string the caller supplies (F6). Token counts are injected.
//!
//! Rungs 1 to 3 specs live here so I1 can fit them. [`RungRun`] only
//! accepts rung 0.

use serde::{Deserialize, Serialize};

/// Spec 6 rung 0: 0.1B active.
pub const RUNG_0_ACTIVE_PARAMS: u64 = 100_000_000;
/// Spec 6 rung 0: 1B total.
pub const RUNG_0_TOTAL_PARAMS: u64 = 1_000_000_000;
/// Spec 6 rung 0: 20B tokens.
pub const RUNG_0_TOKENS: u64 = 20_000_000_000;
/// Spec 6 rung 0: 64 GPUs (the ladder). V5 on NCShare uses [`V5_H200_GPUS`].
pub const RUNG_0_GPUS: u64 = 64;
/// Spec 16.2 V5: 8 H200s.
pub const V5_H200_GPUS: u64 = 8;

/// Spec 6 rung 1: 1B active.
pub const RUNG_1_ACTIVE_PARAMS: u64 = 1_000_000_000;
/// Spec 6 rung 1: 15B total.
pub const RUNG_1_TOTAL_PARAMS: u64 = 15_000_000_000;
/// Spec 6 rung 1: 200B tokens.
pub const RUNG_1_TOKENS: u64 = 200_000_000_000;
/// Spec 6 rung 1: 1k GPUs.
pub const RUNG_1_GPUS: u64 = 1_000;

/// Spec 6 rung 2: 8B active.
pub const RUNG_2_ACTIVE_PARAMS: u64 = 8_000_000_000;
/// Spec 6 rung 2: 120B total.
pub const RUNG_2_TOTAL_PARAMS: u64 = 120_000_000_000;
/// Spec 6 rung 2: 1.5T tokens.
pub const RUNG_2_TOKENS: u64 = 1_500_000_000_000;
/// Spec 6 rung 2: 5k GPUs.
pub const RUNG_2_GPUS: u64 = 5_000;

/// Spec 6 rung 3: 40B active.
pub const RUNG_3_ACTIVE_PARAMS: u64 = 40_000_000_000;
/// Spec 6 rung 3: 700B total.
pub const RUNG_3_TOTAL_PARAMS: u64 = 700_000_000_000;
/// Spec 6 rung 3: 6T tokens.
pub const RUNG_3_TOKENS: u64 = 6_000_000_000_000;
/// Spec 6 rung 3: 15k GPUs.
pub const RUNG_3_GPUS: u64 = 15_000;

/// Ladder id. [`RungRun`] accepts only [`RungId::Zero`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RungId {
    Zero,
    One,
    Two,
    Three,
}

/// Frozen ladder row (spec 6).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RungSpec {
    pub id: RungId,
    pub active_params: u64,
    pub total_params: u64,
    pub tokens: u64,
    pub gpus: u64,
}

/// CPU-analog run config. `token_budget` is the test/CPU ceiling, not
/// the ladder's 20B, and must be `<= spec.tokens`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RungConfig {
    pub spec: RungSpec,
    pub token_budget: u64,
    pub tokens_per_step: u64,
    /// Frozen tokenizer artifact hash (F6). Must be non-empty.
    pub tokenizer_hash: String,
    pub seed: u64,
}

/// One optimizer step's bookkeeping. `step` is 1-based after the call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RungStepReport {
    pub step: u64,
    pub tokens_seen: u64,
    pub loss: f64,
    pub done: bool,
}

/// Job-boundary snapshot (spec 16.2 V5: checkpoint/resume works).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RungCheckpoint {
    pub step: u64,
    pub tokens_seen: u64,
    pub loss_curve: Vec<f64>,
    pub tokenizer_hash: String,
    pub seed: u64,
    pub token_budget: u64,
    pub tokens_per_step: u64,
    pub rung: RungId,
}

/// A7 errors. Distinct from [`crate::ControlError`] (A6).
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RungError {
    #[error("RungRun is rung 0 only (I1 owns rungs 1-3)")]
    NotRung0,
    #[error("token_budget must be > 0 and <= spec.tokens")]
    InvalidBudget,
    #[error("tokens_per_step must be > 0")]
    InvalidTokensPerStep,
    #[error("tokenizer_hash must be non-empty")]
    EmptyTokenizerHash,
    #[error("tokenizer hash changed; the tokenizer is frozen before rung 0")]
    TokenizerChanged,
    #[error("rung already consumed its token_budget")]
    AlreadyDone,
    #[error("loss must be finite")]
    NonFiniteLoss,
    #[error("checkpoint does not match this config")]
    CheckpointMismatch,
    #[error("checkpoint tokens_seen exceeds token_budget")]
    ResumePastBudget,
}

/// Ladder row for `id`. Numbers are the spec 6 table, not V5's 8 GPUs.
pub fn rung_spec(id: RungId) -> RungSpec {
    match id {
        RungId::Zero => RungSpec {
            id: RungId::Zero,
            active_params: RUNG_0_ACTIVE_PARAMS,
            total_params: RUNG_0_TOTAL_PARAMS,
            tokens: RUNG_0_TOKENS,
            gpus: RUNG_0_GPUS,
        },
        RungId::One => RungSpec {
            id: RungId::One,
            active_params: RUNG_1_ACTIVE_PARAMS,
            total_params: RUNG_1_TOTAL_PARAMS,
            tokens: RUNG_1_TOKENS,
            gpus: RUNG_1_GPUS,
        },
        RungId::Two => RungSpec {
            id: RungId::Two,
            active_params: RUNG_2_ACTIVE_PARAMS,
            total_params: RUNG_2_TOTAL_PARAMS,
            tokens: RUNG_2_TOKENS,
            gpus: RUNG_2_GPUS,
        },
        RungId::Three => RungSpec {
            id: RungId::Three,
            active_params: RUNG_3_ACTIVE_PARAMS,
            total_params: RUNG_3_TOTAL_PARAMS,
            tokens: RUNG_3_TOKENS,
            gpus: RUNG_3_GPUS,
        },
    }
}

/// Spec 6 rung 0 row.
pub fn rung_0_spec() -> RungSpec {
    rung_spec(RungId::Zero)
}

/// Check order (must match tests):
/// 1. `spec.id != Zero` → [`RungError::NotRung0`]
/// 2. `token_budget == 0` or `token_budget > spec.tokens` → [`RungError::InvalidBudget`]
/// 3. `tokens_per_step == 0` → [`RungError::InvalidTokensPerStep`]
/// 4. empty `tokenizer_hash` → [`RungError::EmptyTokenizerHash`]
pub fn validate_rung_config(config: &RungConfig) -> Result<(), RungError> {
    if config.spec.id != RungId::Zero {
        return Err(RungError::NotRung0);
    }
    if config.token_budget == 0 || config.token_budget > config.spec.tokens {
        return Err(RungError::InvalidBudget);
    }
    if config.tokens_per_step == 0 {
        return Err(RungError::InvalidTokensPerStep);
    }
    if config.tokenizer_hash.is_empty() {
        return Err(RungError::EmptyTokenizerHash);
    }
    Ok(())
}

/// Rung 0 run. CPU analog: the caller injects each step's loss.
#[derive(Debug, Clone)]
pub struct RungRun {
    config: RungConfig,
    step: u64,
    tokens_seen: u64,
    loss_curve: Vec<f64>,
}

impl RungRun {
    /// New run at step 0, zero tokens, empty curve. Validates config.
    pub fn new(config: RungConfig) -> Result<Self, RungError> {
        validate_rung_config(&config)?;
        Ok(Self {
            config,
            step: 0,
            tokens_seen: 0,
            loss_curve: Vec::new(),
        })
    }

    /// Consume up to `tokens_per_step` tokens (last step may be short)
    /// and append `loss`. Non-finite loss → [`RungError::NonFiniteLoss`].
    /// Already done → [`RungError::AlreadyDone`].
    pub fn step(&mut self, loss: f64) -> Result<RungStepReport, RungError> {
        if !loss.is_finite() {
            return Err(RungError::NonFiniteLoss);
        }
        if self.done() {
            return Err(RungError::AlreadyDone);
        }
        let remaining = self.config.token_budget - self.tokens_seen;
        let added = self.config.tokens_per_step.min(remaining);
        self.tokens_seen += added;
        self.step += 1;
        self.loss_curve.push(loss);
        Ok(RungStepReport {
            step: self.step,
            tokens_seen: self.tokens_seen,
            loss,
            done: self.done(),
        })
    }

    /// Snapshot for a job boundary. Does not consume tokens.
    pub fn checkpoint(&self) -> RungCheckpoint {
        RungCheckpoint {
            step: self.step,
            tokens_seen: self.tokens_seen,
            loss_curve: self.loss_curve.clone(),
            tokenizer_hash: self.config.tokenizer_hash.clone(),
            seed: self.config.seed,
            token_budget: self.config.token_budget,
            tokens_per_step: self.config.tokens_per_step,
            rung: self.config.spec.id,
        }
    }

    /// Resume across a job boundary. Validates `config` first, then
    /// requires matching tokenizer hash, seed, budget, tokens_per_step,
    /// and rung. `tokens_seen > token_budget` → [`RungError::ResumePastBudget`].
    /// A finished run may be resumed; the next [`Self::step`] is
    /// [`RungError::AlreadyDone`].
    pub fn resume(config: RungConfig, ckpt: &RungCheckpoint) -> Result<Self, RungError> {
        validate_rung_config(&config)?;
        if ckpt.tokenizer_hash != config.tokenizer_hash {
            return Err(RungError::TokenizerChanged);
        }
        if ckpt.seed != config.seed
            || ckpt.token_budget != config.token_budget
            || ckpt.tokens_per_step != config.tokens_per_step
            || ckpt.rung != config.spec.id
        {
            return Err(RungError::CheckpointMismatch);
        }
        if ckpt.tokens_seen > config.token_budget {
            return Err(RungError::ResumePastBudget);
        }
        Ok(Self {
            config,
            step: ckpt.step,
            tokens_seen: ckpt.tokens_seen,
            loss_curve: ckpt.loss_curve.clone(),
        })
    }

    pub fn done(&self) -> bool {
        self.tokens_seen >= self.config.token_budget
    }

    pub fn step_index(&self) -> u64 {
        self.step
    }

    pub fn tokens_seen(&self) -> u64 {
        self.tokens_seen
    }

    pub fn remaining_tokens(&self) -> u64 {
        self.config.token_budget.saturating_sub(self.tokens_seen)
    }

    pub fn loss_curve(&self) -> &[f64] {
        &self.loss_curve
    }

    pub fn config(&self) -> &RungConfig {
        &self.config
    }
}
