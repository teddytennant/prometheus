//! Held-out harness benchmark as a promotion gate (spec 14.3).

use crate::EvalResult;

pub trait Solver {
    fn solve(&self, id: &str, prompt: &str) -> String;
}

#[derive(Debug, Clone)]
pub struct HiddenTask {
    pub id: String,
    pub prompt: String,
    pub expected: String,
}

#[derive(Debug, Clone)]
pub struct EvalGate {
    pub tasks: Vec<HiddenTask>,
    pub pass_threshold: f64,
}

impl EvalGate {
    pub fn seed() -> Self {
        Self {
            tasks: vec![
                HiddenTask {
                    id: "h1".into(),
                    prompt: "echo ok".into(),
                    expected: "ok".into(),
                },
                HiddenTask {
                    id: "h2".into(),
                    prompt: "add 2 3".into(),
                    expected: "5".into(),
                },
                HiddenTask {
                    id: "h3".into(),
                    prompt: "not stub".into(),
                    expected: "real".into(),
                },
            ],
            pass_threshold: 0.67,
        }
    }

    pub fn score(&self, solver: &dyn Solver) -> EvalResult {
        let mut passed = 0u32;
        for t in &self.tasks {
            if solver.solve(&t.id, &t.prompt) == t.expected {
                passed += 1;
            }
        }
        EvalResult {
            passed,
            total: self.tasks.len() as u32,
        }
    }

    pub fn pass_fail(&self, r: &EvalResult) -> bool {
        r.pass() && r.frac() + f64::EPSILON >= self.pass_threshold
    }
}

#[derive(Debug, Default)]
pub struct StubSolver;

impl Solver for StubSolver {
    fn solve(&self, _id: &str, _prompt: &str) -> String {
        String::new()
    }
}

#[derive(Debug, Default)]
pub struct HonestSolver;

impl Solver for HonestSolver {
    fn solve(&self, _id: &str, prompt: &str) -> String {
        match prompt {
            "echo ok" => "ok".into(),
            "add 2 3" => "5".into(),
            "not stub" => "real".into(),
            _ => String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_fails_honest_passes() {
        let g = EvalGate::seed();
        let bad = g.score(&StubSolver);
        assert!(!g.pass_fail(&bad));
        let good = g.score(&HonestSolver);
        assert!(g.pass_fail(&good));
        assert_eq!(good.passed, 3);
    }
}
