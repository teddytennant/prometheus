//! Group: SolveRateFilter probe/keep vs the reference. Must fail on the stub.
//!
//! Probe scores the attached D2 payload with the reference (string / stdout
//! equality). No live sandbox.

mod common;
mod reference;

use common::{code_task, math_task, NOW};
use prometheus_envs::NowMs;
use prometheus_tasks::{Error, MintedTask, Result, SolveRate, SolveRateFilter, Solver};

struct SeqSolver {
    replies: Vec<String>,
    i: usize,
    seen: Vec<(String, NowMs)>,
}

impl SeqSolver {
    fn new(replies: &[&str]) -> Self {
        Self {
            replies: replies.iter().map(|s| (*s).to_string()).collect(),
            i: 0,
            seen: Vec::new(),
        }
    }
}

impl Solver for SeqSolver {
    fn attempt(&mut self, statement: &str, now: NowMs) -> Result<String> {
        self.seen.push((statement.to_string(), now));
        let r = self
            .replies
            .get(self.i)
            .ok_or_else(|| Error::Unverifiable("scripted solver exhausted".into()))?;
        self.i += 1;
        Ok(r.clone())
    }
}

struct PanicSolver;

impl Solver for PanicSolver {
    fn attempt(&mut self, _statement: &str, _now: NowMs) -> Result<String> {
        panic!("probe n==0 must not call Solver::attempt");
    }
}

fn pin_keep(rate: SolveRate) {
    assert_eq!(SolveRateFilter::keep(rate), reference::keep(rate));
}

fn pin_probe(task: &MintedTask, replies: &[&str], n: u32) {
    let mut prod_solver = SeqSolver::new(replies);
    let mut ref_solver = SeqSolver::new(replies);
    let got = SolveRateFilter::probe(&mut prod_solver, task, n, NOW);
    let want = reference::probe(&mut ref_solver, task, n, NOW);
    assert_eq!(got, want);
    if got.is_ok() {
        assert_eq!(prod_solver.seen.len(), n as usize);
        for (stmt, now) in &prod_solver.seen {
            assert_eq!(stmt, task.statement());
            assert_eq!(*now, NOW);
        }
    }
}

#[test]
fn probe_n_zero_is_bad_n_and_does_not_call_solver() {
    let task = math_task("q", "1");
    assert_eq!(
        SolveRateFilter::probe(&mut PanicSolver, &task, 0, NOW),
        Err(Error::BadN)
    );
    assert_eq!(
        reference::probe(&mut PanicSolver, &task, 0, NOW),
        Err(Error::BadN)
    );
}

#[test]
fn keep_n_zero_is_bad_n() {
    pin_keep(SolveRate::new(0, 0));
    pin_keep(SolveRate::new(1, 0));
}

#[test]
fn keep_rate_zero_is_impossible() {
    pin_keep(SolveRate::new(0, 1));
    pin_keep(SolveRate::new(0, 4));
}

#[test]
fn keep_rate_one_is_trivial() {
    pin_keep(SolveRate::new(1, 1));
    pin_keep(SolveRate::new(4, 4));
}

#[test]
fn keep_half_is_ok() {
    let rate = SolveRate::new(1, 2);
    pin_keep(rate);
    let kept = SolveRateFilter::keep(rate).expect("1/2 kept");
    assert!(kept.in_open_interval());
    assert_eq!(kept.rate(), 0.5);
    assert_eq!(kept, rate);
}

#[test]
fn keep_open_interval_gate() {
    // 0 dropped, 1 dropped, 1/2 kept — D3 gate.
    assert_eq!(
        SolveRateFilter::keep(SolveRate::new(0, 2)),
        Err(Error::Impossible)
    );
    assert_eq!(
        SolveRateFilter::keep(SolveRate::new(2, 2)),
        Err(Error::Trivial)
    );
    assert_eq!(
        SolveRateFilter::keep(SolveRate::new(1, 2)),
        Ok(SolveRate::new(1, 2))
    );
}

#[test]
fn keep_passed_greater_than_n_is_trivial() {
    pin_keep(SolveRate::new(3, 2));
}

#[test]
fn probe_math_all_pass() {
    let task = math_task("2+2?", "4");
    pin_probe(&task, &["4", "4"], 2);
}

#[test]
fn probe_math_all_fail() {
    let task = math_task("2+2?", "4");
    pin_probe(&task, &["5", "0"], 2);
}

#[test]
fn probe_math_half() {
    let task = math_task("2+2?", "4");
    pin_probe(&task, &["4", "nope"], 2);
    let mut s = SeqSolver::new(&["4", "nope"]);
    let rate = SolveRateFilter::probe(&mut s, &task, 2, NOW).unwrap();
    assert_eq!(rate, SolveRate::new(1, 2));
    assert!(SolveRateFilter::keep(rate).unwrap().in_open_interval());
}

#[test]
fn probe_code_equals_expected_stdout() {
    let task = code_task("write f", b"HIDDEN_OK");
    pin_probe(&task, &["HIDDEN_OK", "nope"], 2);
    let mut s = SeqSolver::new(&["HIDDEN_OK", "HIDDEN_OK"]);
    let rate = SolveRateFilter::probe(&mut s, &task, 2, NOW).unwrap();
    assert_eq!(rate, SolveRate::new(2, 2));
}

#[test]
fn probe_code_byte_equality_not_utf8_prefix() {
    let task = code_task("write f", b"OK");
    // "OK\n" must not pass when expected is "OK".
    pin_probe(&task, &["OK\n", "OK"], 2);
}

#[test]
fn probe_propagates_solver_error() {
    let task = math_task("q", "1");
    let mut s = SeqSolver::new(&["1"]);
    assert!(matches!(
        SolveRateFilter::probe(&mut s, &task, 2, NOW),
        Err(Error::Unverifiable(_))
    ));
}

#[test]
fn probe_unverifiable_without_payload() {
    let mut task = math_task("q", "1");
    task.math = None;
    task.code = None;
    let mut s = SeqSolver::new(&["1"]);
    assert_eq!(
        SolveRateFilter::probe(&mut s, &task, 1, NOW),
        Err(Error::Unverifiable("no D2 payload".into()))
    );
}

#[test]
fn probe_math_wins_when_both_payloads_set() {
    let mut task = math_task("q", "math-ans");
    task.code = code_task("q", b"code-ans").code;
    pin_probe(&task, &["math-ans", "code-ans"], 2);
}

#[test]
fn probe_uses_statement_not_hidden_expected() {
    let task = math_task("public problem", "SECRET");
    let mut s = SeqSolver::new(&["SECRET"]);
    let _ = SolveRateFilter::probe(&mut s, &task, 1, NOW).unwrap();
    assert_eq!(s.seen[0].0, "public problem");
    assert_ne!(s.seen[0].0, "SECRET");
}
