//! Independent I1 reference: slow, obvious scaling-law arithmetic.
//!
//! Production (`prometheus_control`) must never import this module. Tests
//! compare `fit` / `predict*` / `should_stop` / `compute_flops` against these
//! functions. Plain `f64`, natural log, no extra crates.
//!
//! Law: `L(C) = A * C^{-alpha}` with `C = 6 * N * D` (cast N and D to f64
//! before multiplying). Two distinct C values determine A and alpha. Three
//! rungs: ordinary least squares of `ln L = ln A - alpha * ln C`. Rung 0 is
//! never a fit point.

#![allow(dead_code)]

use prometheus_control::rung::{RungId, RungSpec};
use prometheus_control::scale::{FlagshipSpec, RungObservation, ScaleError};

/// Spec-6 flagship row as literals so a wrong production constant is visible.
pub const FLAGSHIP_ACTIVE_PARAMS: u64 = 400_000_000_000;
pub const FLAGSHIP_TOTAL_PARAMS: u64 = 7_400_000_000_000;
pub const FLAGSHIP_TOKENS: u64 = 180_000_000_000_000;
pub const FLAGSHIP_GPUS: u64 = 100_000;
pub const FLOP_COEFF: f64 = 6.0;
pub const STOP_RELATIVE_EXCESS: f64 = 0.05;
pub const STOP_FRACTION: f64 = 0.10;

/// Tiny synthetic observation. `total_params` / `gpus` are unused by the law
/// and deliberately not equal to `active_params` / a real cluster size.
pub fn observation(id: RungId, active_params: u64, tokens: u64, loss: f64) -> RungObservation {
    RungObservation {
        spec: RungSpec {
            id,
            active_params,
            total_params: 999_999,
            tokens,
            gpus: 42,
        },
        loss,
    }
}

pub fn law_loss(a: f64, alpha: f64, active_params: u64, tokens: u64) -> f64 {
    let c = compute_c(active_params, tokens);
    a * c.powf(-alpha)
}

pub fn compute_c(active_params: u64, tokens: u64) -> f64 {
    FLOP_COEFF * (active_params as f64) * (tokens as f64)
}

pub fn flagship_spec() -> FlagshipSpec {
    FlagshipSpec {
        active_params: FLAGSHIP_ACTIVE_PARAMS,
        total_params: FLAGSHIP_TOTAL_PARAMS,
        tokens: FLAGSHIP_TOKENS,
        gpus: FLAGSHIP_GPUS,
    }
}

pub fn compute_flops(active_params: u64, tokens: u64) -> Result<f64, ScaleError> {
    if active_params == 0 || tokens == 0 {
        return Err(ScaleError::InvalidCompute);
    }
    Ok(compute_c(active_params, tokens))
}

/// First-hit order matches the iface / oracle lock.
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

/// Independent fit. Same public behavior as `scale::fit`.
#[derive(Debug, Clone, PartialEq)]
pub struct RefScalingFit {
    pub a: f64,
    pub alpha: f64,
}

impl RefScalingFit {
    pub fn a(&self) -> f64 {
        self.a
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn predict(&self, active_params: u64, tokens: u64) -> Result<f64, ScaleError> {
        predict_with(self.a, self.alpha, active_params, tokens)
    }

    pub fn predict_spec(&self, spec: &RungSpec) -> Result<f64, ScaleError> {
        if spec.id == RungId::Zero {
            return Err(ScaleError::Rung0NotUsed);
        }
        self.predict(spec.active_params, spec.tokens)
    }

    pub fn predict_flagship(&self) -> Result<f64, ScaleError> {
        self.predict(FLAGSHIP_ACTIVE_PARAMS, FLAGSHIP_TOKENS)
    }
}

fn predict_with(a: f64, alpha: f64, active_params: u64, tokens: u64) -> Result<f64, ScaleError> {
    if active_params == 0 || tokens == 0 {
        return Err(ScaleError::InvalidCompute);
    }
    let c = compute_c(active_params, tokens);
    let pred = a * c.powf(-alpha);
    if !pred.is_finite() {
        return Err(ScaleError::NonFinitePrediction);
    }
    Ok(pred)
}

pub fn fit(observations: &[RungObservation]) -> Result<RefScalingFit, ScaleError> {
    for obs in observations {
        validate_observation(obs)?;
    }
    if first_duplicate_id(observations).is_some() {
        return Err(ScaleError::DuplicateRung);
    }
    if observations.len() < 2 {
        return Err(ScaleError::NeedTwoRungs);
    }

    let mut cs: Vec<f64> = Vec::with_capacity(observations.len());
    let mut ln_cs: Vec<f64> = Vec::with_capacity(observations.len());
    let mut ln_ls: Vec<f64> = Vec::with_capacity(observations.len());
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

    let n = ln_cs.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for i in 0..ln_cs.len() {
        sum_x += ln_cs[i];
        sum_y += ln_ls[i];
    }
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..ln_cs.len() {
        let dx = ln_cs[i] - mean_x;
        let dy = ln_ls[i] - mean_y;
        num += dx * dy;
        den += dx * dx;
    }
    // Distinct finite C already implies den > 0; keep the guard obvious.
    if den == 0.0 || !den.is_finite() {
        return Err(ScaleError::DegenerateFit);
    }
    let slope = num / den; // ln L = ln A + slope * ln C, slope = -alpha
    let intercept = mean_y - slope * mean_x;
    let alpha = -slope;
    let a = intercept.exp();
    Ok(RefScalingFit { a, alpha })
}

fn first_duplicate_id(observations: &[RungObservation]) -> Option<RungId> {
    for i in 0..observations.len() {
        for j in 0..i {
            if observations[j].spec.id == observations[i].spec.id {
                return Some(observations[i].spec.id);
            }
        }
    }
    None
}

pub fn predict_held_out(train: &[RungObservation], held_out: &RungSpec) -> Result<f64, ScaleError> {
    for obs in train {
        if obs.spec.id == held_out.id {
            return Err(ScaleError::HeldOutInTrain);
        }
    }
    fit(train)?.predict_spec(held_out)
}

pub fn should_stop(predicted: f64, observed: f64) -> Result<bool, ScaleError> {
    if !predicted.is_finite() || predicted <= 0.0 {
        return Err(ScaleError::InvalidPredicted);
    }
    if !observed.is_finite() {
        return Err(ScaleError::InvalidObserved);
    }
    Ok(observed >= predicted * (1.0 + STOP_RELATIVE_EXCESS))
}

pub fn rel_err(got: f64, expect: f64) -> f64 {
    if got == expect {
        return 0.0;
    }
    let scale = got.abs().max(expect.abs()).max(1e-30);
    (got - expect).abs() / scale
}

pub fn assert_rel_close(got: f64, expect: f64, tol: f64, what: &str) {
    assert!(
        got.is_finite() && expect.is_finite(),
        "{what}: non-finite got={got} expect={expect}"
    );
    let rel = rel_err(got, expect);
    assert!(
        rel <= tol,
        "{what}: got={got} expect={expect} rel={rel} tol={tol}"
    );
}

pub fn log_residual(a: f64, alpha: f64, active_params: u64, tokens: u64, loss: f64) -> f64 {
    let ln_l = loss.ln();
    let ln_c = compute_c(active_params, tokens).ln();
    ln_l - (a.ln() - alpha * ln_c)
}
