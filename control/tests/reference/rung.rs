//! Independent A7 reference: slow, obvious rung-0 bookkeeping.
//!
//! Production (`prometheus_control`) must never import this module. Tests
//! compare `rung_spec` / `validate_rung_config` / `RungRun` against these
//! functions. No production calls. Integer policy: u64 counts, f64 loss.

#![allow(dead_code)]

use prometheus_control::rung::{
    RungCheckpoint, RungConfig, RungError, RungId, RungSpec, RungStepReport,
};

/// Spec 6 table, written as literals so a wrong production constant is visible.
pub fn rung_spec(id: RungId) -> RungSpec {
    match id {
        RungId::Zero => RungSpec {
            id: RungId::Zero,
            active_params: 100_000_000,
            total_params: 1_000_000_000,
            tokens: 20_000_000_000,
            gpus: 64,
        },
        RungId::One => RungSpec {
            id: RungId::One,
            active_params: 1_000_000_000,
            total_params: 15_000_000_000,
            tokens: 200_000_000_000,
            gpus: 1_000,
        },
        RungId::Two => RungSpec {
            id: RungId::Two,
            active_params: 8_000_000_000,
            total_params: 120_000_000_000,
            tokens: 1_500_000_000_000,
            gpus: 5_000,
        },
        RungId::Three => RungSpec {
            id: RungId::Three,
            active_params: 40_000_000_000,
            total_params: 700_000_000_000,
            tokens: 6_000_000_000_000,
            gpus: 15_000,
        },
    }
}

pub fn rung_0_spec() -> RungSpec {
    rung_spec(RungId::Zero)
}

/// Tiny CPU-analog config. `token_budget` is a test ceiling, not 20B.
pub fn tiny_rung0_config(token_budget: u64, tokens_per_step: u64) -> RungConfig {
    RungConfig {
        spec: rung_0_spec(),
        token_budget,
        tokens_per_step,
        tokenizer_hash: "sha256:test-tokenizer".to_string(),
        seed: 7,
    }
}

/// Check order matches the iface (first hit wins).
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

/// Independent rung-0 run. Same public behavior as `RungRun`.
#[derive(Debug, Clone)]
pub struct RefRungRun {
    config: RungConfig,
    step: u64,
    tokens_seen: u64,
    loss_curve: Vec<f64>,
}

impl RefRungRun {
    pub fn new(config: RungConfig) -> Result<Self, RungError> {
        validate_rung_config(&config)?;
        Ok(Self {
            config,
            step: 0,
            tokens_seen: 0,
            loss_curve: Vec::new(),
        })
    }

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
