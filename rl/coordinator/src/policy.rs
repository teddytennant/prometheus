//! Truncated IS and batch types for off-policy RL (spec 9.2).

#[derive(Debug, Clone)]
pub struct Batch {
    pub policy_version: u64,
    pub tokens: Vec<u32>,
    pub logprobs: Vec<f32>,
    pub routing_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParityHalt {
    pub max_abs: f32,
    pub threshold: f32,
}

/// Truncated importance sampling weight: min(c, exp(logp_target - logp_behavior)).
pub fn truncated_is(logp_behavior: f32, logp_target: f32, c: f32) -> f32 {
    let ratio = (logp_target - logp_behavior).exp();
    if !ratio.is_finite() {
        return 0.0;
    }
    ratio.min(c).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tis_clips() {
        let w = truncated_is(-1.0, -1.0, 2.0);
        assert!((w - 1.0).abs() < 1e-5);
        let w = truncated_is(-4.0, 0.0, 2.0);
        assert!((w - 2.0).abs() < 1e-5);
        assert!(truncated_is(0.0, -50.0, 2.0) < 1e-10);
    }
}
