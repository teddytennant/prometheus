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

use crate::rung::RungSpec;
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
#[allow(dead_code)]
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

/// `6 * active_params * tokens` as f64. Overflows u64 at flagship scale.
pub fn compute_flops(active_params: u64, tokens: u64) -> Result<f64, ScaleError> {
    let _ = (active_params, tokens);
    unimplemented!("I1: compute_flops")
}

/// Flagship ladder row. Numbers are the spec 6 table.
pub fn flagship_spec() -> FlagshipSpec {
    unimplemented!("I1: flagship_spec")
}

/// Fit `L(C) = A * C^{-alpha}` on rungs 1 to 3.
///
/// Needs at least two observations, unique ids, finite loss > 0,
/// positive compute, distinct C. Rung 0 is [`ScaleError::Rung0NotUsed`].
pub fn fit(observations: &[RungObservation]) -> Result<ScalingFit, ScaleError> {
    let _ = observations;
    unimplemented!("I1: fit")
}

impl ScalingFit {
    /// `A` in `L = A * C^{-alpha}`.
    pub fn a(&self) -> f64 {
        let _ = self;
        unimplemented!("I1: ScalingFit::a")
    }

    /// `alpha` in `L = A * C^{-alpha}`.
    pub fn alpha(&self) -> f64 {
        let _ = self;
        unimplemented!("I1: ScalingFit::alpha")
    }

    /// Predict loss at `C = 6 * active_params * tokens`.
    pub fn predict(&self, active_params: u64, tokens: u64) -> Result<f64, ScaleError> {
        let _ = (self, active_params, tokens);
        unimplemented!("I1: ScalingFit::predict")
    }

    /// Predict loss for a ladder row. Rejects rung 0.
    pub fn predict_spec(&self, spec: &RungSpec) -> Result<f64, ScaleError> {
        let _ = (self, spec);
        unimplemented!("I1: ScalingFit::predict_spec")
    }

    /// Predict flagship loss at [`FLAGSHIP_ACTIVE_PARAMS`] * [`FLAGSHIP_TOKENS`].
    pub fn predict_flagship(&self) -> Result<f64, ScaleError> {
        let _ = self;
        unimplemented!("I1: ScalingFit::predict_flagship")
    }
}

/// Fit on `train`, predict `held_out`. `held_out.id` must not appear in `train`.
pub fn predict_held_out(
    train: &[RungObservation],
    held_out: &RungSpec,
) -> Result<f64, ScaleError> {
    let _ = (train, held_out);
    unimplemented!("I1: predict_held_out")
}

/// Spec 6: stop if `observed >= predicted * (1 + STOP_RELATIVE_EXCESS)`.
///
/// The caller passes the loss at [`STOP_FRACTION`] of the token budget.
/// `predicted` must be finite and > 0. `observed` must be finite.
pub fn should_stop(predicted: f64, observed: f64) -> Result<bool, ScaleError> {
    let _ = (predicted, observed);
    unimplemented!("I1: should_stop")
}

/// Rung 1 to 3 only. Rung 0 is [`ScaleError::Rung0NotUsed`].
pub fn validate_observation(obs: &RungObservation) -> Result<(), ScaleError> {
    let _ = obs;
    unimplemented!("I1: validate_observation")
}
