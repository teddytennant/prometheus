//! Interface -> oracle -> stub-fail -> best-of-3 -> gates -> reviewer veto -> ledger.

use crate::{Pipeline, Record};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Interface,
    Oracle,
    StubFail,
    Impls,
    Gates,
    Reviewer,
    Ledger,
}

#[derive(Debug, Clone)]
pub struct Case {
    pub name: String,
    pub stub_passes: bool,
    pub impl_pass: [bool; 3],
}

#[derive(Debug, Clone)]
pub struct Suite {
    pub cases: Vec<Case>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    StubPassed(String),
    ReviewerVeto,
    NoImplPassed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunResult {
    pub chosen: usize,
    pub scores: [usize; 3],
}

impl Pipeline {
    /// Reject suites that pass a stub, pick best of 3 impls, honor reviewer veto.
    pub fn run(&mut self, suite: &Suite, reviewer_veto: bool) -> Result<RunResult, RunError> {
        self.push(Record {
            role: "interface".into(),
            ok: true,
        });
        self.push(Record {
            role: "oracle".into(),
            ok: true,
        });
        for c in &suite.cases {
            if c.stub_passes {
                self.push(Record {
                    role: "stub".into(),
                    ok: false,
                });
                return Err(RunError::StubPassed(c.name.clone()));
            }
        }
        self.push(Record {
            role: "stub".into(),
            ok: true,
        });
        let mut scores = [0usize; 3];
        for c in &suite.cases {
            for i in 0..3 {
                if c.impl_pass[i] {
                    scores[i] += 1;
                }
            }
        }
        self.push(Record {
            role: "impls".into(),
            ok: scores.iter().any(|&s| s > 0),
        });
        let chosen = scores
            .iter()
            .enumerate()
            .max_by_key(|(i, s)| (*s, -(*i as isize)))
            .map(|(i, _)| i)
            .unwrap_or(0);
        if scores[chosen] == 0 {
            self.push(Record {
                role: "gates".into(),
                ok: false,
            });
            return Err(RunError::NoImplPassed);
        }
        self.push(Record {
            role: "gates".into(),
            ok: true,
        });
        if reviewer_veto {
            self.push(Record {
                role: "reviewer".into(),
                ok: false,
            });
            return Err(RunError::ReviewerVeto);
        }
        self.push(Record {
            role: "reviewer".into(),
            ok: true,
        });
        self.push(Record {
            role: "ledger".into(),
            ok: true,
        });
        Ok(RunResult { chosen, scores })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pipeline;

    fn suite(stub: bool, impls: [[bool; 3]; 2]) -> Suite {
        Suite {
            cases: vec![
                Case {
                    name: "a".into(),
                    stub_passes: stub,
                    impl_pass: impls[0],
                },
                Case {
                    name: "b".into(),
                    stub_passes: false,
                    impl_pass: impls[1],
                },
            ],
        }
    }

    #[test]
    fn rejects_stub_and_veto_picks_best() {
        let mut p = Pipeline::default();
        assert!(matches!(
            p.run(&suite(true, [[true, true, true], [true, true, true]]), false),
            Err(RunError::StubPassed(_))
        ));
        let mut p = Pipeline::default();
        let r = p
            .run(&suite(false, [[true, false, true], [true, false, false]]), false)
            .unwrap();
        assert_eq!(r.chosen, 0);
        assert_eq!(r.scores, [2, 0, 1]);
        let mut p = Pipeline::default();
        assert_eq!(
            p.run(&suite(false, [[true, true, true], [true, true, true]]), true),
            Err(RunError::ReviewerVeto)
        );
    }
}
