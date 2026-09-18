//! Scaling-law fit on rungs 1 to 3 (spec 6, 15.5 I1).
//!
//! CPU analog: injected final losses, no training, no GPUs. Predicts a
//! held-out rung and the flagship loss. Spec 6: a flagship 5% or more
//! above prediction at 10% of training is stopped.
//!
//! The law is `L(C) = A * C^{-alpha}` with `C = 6 * active_params * tokens`
//! (spec 6 quotes 6ND FLOP). Two distinct rungs determine A and alpha.
//! Three rungs use the same log-linear least squares. Rung 0 is kernel
//! correctness, not a fit point.
//!
//! LR and batch transfer wait for muP numbers rung 1 decides. Token count
//! for the flagship is [`FLAGSHIP_TOKENS`]. This module's gate is loss
//! prediction, not a 180T-token run.

use crate::rung::{RungId, RungSpec};
use serde::{Deserialize, Serialize};

/// Spec 6 flagship: 0.4T active.
pub const FLAGSHIP_ACTIVE_PARAMS: u64 = 400_000_000_000;
/// Spec 6 flagship: 7.4T total.
pub const FLAGSHIP_TOTAL_PARAMS: u64 = 7_400_000_000_000;
/// Spec 6 flagship: ~180T tokens.
pub const FLAGSHIP_TOKENS: u64 = 180_000_000_000_000;
/// Spec 6 flagship: ~100k GPUs.
pub const FLAGSHIP_GPUS: u64 = 100_000;

/// Spec 6 FLOP: `6 * N * D`.
pub const FLOP_COEFF: f64 = 6.0;

/// Stop if observed loss is this fraction or more above prediction.
pub const STOP_RELATIVE_EXCESS: f64 = 0.05;
/// Spec 6: the check is at this fraction of the token budget.
pub const STOP_FRACTION: f64 = 0.10;

/// Spec 6 flagship row. Not a [`crate::rung::RungId`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlagshipSpec {
    pub active_params: u64,
    pub total_params: u64,
    pub tokens: u64,
    pub gpus: u64,
}

/// Injected final loss for one rung. `spec.id` must be 1, 2, or 3.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RungObservation {
    pub spec: RungSpec,
    pub loss: f64,
}

/// Fitted `L(C) = A * C^{-alpha}`. Fields are private.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalingFit {
    a: f64,
    alpha: f64,
}

/// I1 errors. Distinct from [`crate::ControlError`] and [`crate::rung::RungError`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ScaleError {
    #[error("rung 0 is not a scaling-law point")]
    Rung0NotUsed,
    #[error("need at least two rungs from 1 to 3")]
    NeedTwoRungs,
    #[error("duplicate rung in the fit")]
    DuplicateRung,
    #[error("loss must be finite")]
    NonFiniteLoss,
    #[error("loss must be > 0")]
    NonPositiveLoss,
    #[error("active_params and tokens must be > 0")]
    InvalidCompute,
    #[error("rungs do not determine a unique (A, alpha)")]
    DegenerateFit,
    #[error("held-out rung was in the train set")]
    HeldOutInTrain,
    #[error("prediction must be finite")]
    NonFinitePrediction,
    #[error("predicted loss must be finite and > 0")]
    InvalidPredicted,
    #[error("observed loss must be finite")]
    InvalidObserved,
}

fn compute_c(active_params: u64, tokens: u64) -> f64 {
    FLOP_COEFF * (active_params as f64) * (tokens as f64)
}

/// `6 * active_params * tokens` as f64. Overflows u64 at flagship scale.
pub fn compute_flops(active_params: u64, tokens: u64) -> Result<f64, ScaleError> {
    if active_params == 0 || tokens == 0 {
        return Err(ScaleError::InvalidCompute);
    }
    Ok(compute_c(active_params, tokens))
}

/// Flagship ladder row. Numbers are the spec 6 table.
pub fn flagship_spec() -> FlagshipSpec {
    FlagshipSpec {
        active_params: FLAGSHIP_ACTIVE_PARAMS,
        total_params: FLAGSHIP_TOTAL_PARAMS,
        tokens: FLAGSHIP_TOKENS,
        gpus: FLAGSHIP_GPUS,
    }
}

/// Fit `L(C) = A * C^{-alpha}` on rungs 1 to 3.
///
/// Needs at least two observations, unique ids, finite loss > 0,
/// positive compute, distinct C. Rung 0 is [`ScaleError::Rung0NotUsed`].
pub fn fit(observations: &[RungObservation]) -> Result<ScalingFit, ScaleError> {
    for obs in observations {
        validate_observation(obs)?;
    }
    for i in 0..observations.len() {
        for j in 0..i {
            if observations[j].spec.id == observations[i].spec.id {
                return Err(ScaleError::DuplicateRung);
            }
        }
    }
    if observations.len() < 2 {
        return Err(ScaleError::NeedTwoRungs);
    }

    let mut cs = Vec::with_capacity(observations.len());
    let mut ln_cs = Vec::with_capacity(observations.len());
    let mut ln_ls = Vec::with_capacity(observations.len());
    for obs in observations {
        let c = compute_c(obs.spec.active_params, obs.spec.tokens);
        if !c.is_finite() {
            return Err(ScaleError::DegenerateFit);
        }
        let ln_c = c.ln();
        let ln_l = obs.loss.ln();
        if !ln_c.is_finite() || !ln_l.is_finite() {
            return Err(ScaleError::DegenerateFit);
        }
        cs.push(c);
        ln_cs.push(ln_c);
        ln_ls.push(ln_l);
    }
    for i in 0..cs.len() {
        for j in 0..i {
            if cs[i] == cs[j] {
                return Err(ScaleError::DegenerateFit);
            }
        }
    }

    // ln L = ln A - alpha * ln C, ordinary least squares (exact for two points).
    let n = ln_cs.len() as f64;
    let mean_x = ln_cs.iter().sum::<f64>() / n;
    let mean_y = ln_ls.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..ln_cs.len() {
        let dx = ln_cs[i] - mean_x;
        let dy = ln_ls[i] - mean_y;
        num += dx * dy;
        den += dx * dx;
    }
    if den == 0.0 || !den.is_finite() {
        return Err(ScaleError::DegenerateFit);
    }
    let slope = num / den;
    let intercept = mean_y - slope * mean_x;
    Ok(ScalingFit {
        a: intercept.exp(),
        alpha: -slope,
    })
}

impl ScalingFit {
    /// `A` in `L = A * C^{-alpha}`.
    pub fn a(&self) -> f64 {
        self.a
    }

    /// `alpha` in `L = A * C^{-alpha}`.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Predict loss at `C = 6 * active_params * tokens`.
    pub fn predict(&self, active_params: u64, tokens: u64) -> Result<f64, ScaleError> {
        if active_params == 0 || tokens == 0 {
            return Err(ScaleError::InvalidCompute);
        }
        let c = compute_c(active_params, tokens);
        let pred = self.a * c.powf(-self.alpha);
        if !pred.is_finite() {
            return Err(ScaleError::NonFinitePrediction);
        }
        Ok(pred)
    }

    /// Predict loss for a ladder row. Rejects rung 0.
    pub fn predict_spec(&self, spec: &RungSpec) -> Result<f64, ScaleError> {
        if spec.id == RungId::Zero {
            return Err(ScaleError::Rung0NotUsed);
        }
        self.predict(spec.active_params, spec.tokens)
    }

    /// Predict flagship loss at [`FLAGSHIP_ACTIVE_PARAMS`] * [`FLAGSHIP_TOKENS`].
    pub fn predict_flagship(&self) -> Result<f64, ScaleError> {
        self.predict(FLAGSHIP_ACTIVE_PARAMS, FLAGSHIP_TOKENS)
    }
}

/// Fit on `train`, predict `held_out`. `held_out.id` must not appear in `train`.
pub fn predict_held_out(
    train: &[RungObservation],
    held_out: &RungSpec,
) -> Result<f64, ScaleError> {
    for obs in train {
        if obs.spec.id == held_out.id {
            return Err(ScaleError::HeldOutInTrain);
        }
    }
    fit(train)?.predict_spec(held_out)
}

/// Spec 6: stop if `observed >= predicted * (1 + STOP_RELATIVE_EXCESS)`.
///
/// The caller passes the loss at [`STOP_FRACTION`] of the token budget.
/// `predicted` must be finite and > 0. `observed` must be finite.
pub fn should_stop(predicted: f64, observed: f64) -> Result<bool, ScaleError> {
    if !predicted.is_finite() || predicted <= 0.0 {
        return Err(ScaleError::InvalidPredicted);
    }
    if !observed.is_finite() {
        return Err(ScaleError::InvalidObserved);
    }
    Ok(observed >= predicted * (1.0 + STOP_RELATIVE_EXCESS))
}

/// Rung 1 to 3 only. Rung 0 is [`ScaleError::Rung0NotUsed`].
pub fn validate_observation(obs: &RungObservation) -> Result<(), ScaleError> {
    if obs.spec.id == RungId::Zero {
        return Err(ScaleError::Rung0NotUsed);
    }
    if obs.spec.active_params == 0 || obs.spec.tokens == 0 {
        return Err(ScaleError::InvalidCompute);
    }
    if !obs.loss.is_finite() {
        return Err(ScaleError::NonFiniteLoss);
    }
    if obs.loss <= 0.0 {
        return Err(ScaleError::NonPositiveLoss);
    }
    Ok(())
}
