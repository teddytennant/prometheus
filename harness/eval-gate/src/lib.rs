//! Held-out eval gate (spec 14.3, 15.5 L2).

mod gate;

pub use gate::{EvalGate, HiddenTask, HonestSolver, Solver, StubSolver};

#[derive(Debug, Clone)]
pub struct EvalResult {
    pub passed: u32,
    pub total: u32,
}

impl EvalResult {
    pub fn frac(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.passed as f64 / self.total as f64
        }
    }

    pub fn pass(&self) -> bool {
        self.frac() >= 0.5
    }
}

pub fn promote(r: &EvalResult) -> Result<(), String> {
    if r.pass() {
        Ok(())
    } else {
        Err("gate".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_fail() {
        let r = EvalResult {
            passed: 1,
            total: 4,
        };
        assert!(promote(&r).is_err());
        let r = EvalResult {
            passed: 3,
            total: 4,
        };
        assert!(promote(&r).is_ok());
    }
}
