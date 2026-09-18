//! E4 agentic-trajectory oracle tests (spec 8, 15.5).
//!
//! Production `generate_one` / `generate_batch` / `outcome_verified_rate` /
//! `keep_verified` must match `common::agentic_ref`. Every test below calls a
//! production function, so `cargo test -p prometheus-synth --offline --test agentic`
//! is red on the `unimplemented!` stubs.

mod common;

use std::cell::RefCell;
use std::rc::Rc;

use prometheus_synth::{
    generate_batch, generate_one, keep_verified, outcome_verified_rate, AgenticTask, Error,
    Generator, Outcome, Sandbox, Step, Trajectory,
};

use common::agentic_ref;

const GOLDEN_N: u32 = 4;
const GOLDEN_MAX_STEPS: u32 = 2;
const GOLDEN_MAX_TOKENS: u32 = 32;
const GOLDEN_TEMPERATURE: f32 = 0.5;
const GOLDEN_VERIFIED_RATE: f64 = 0.25;

fn task(id: &str, prompt: &str) -> AgenticTask {
    AgenticTask {
        task_id: id.into(),
        prompt: prompt.into(),
    }
}

fn fix_task() -> AgenticTask {
    task("t1", "fix the bug")
}

fn traj(id: &str, outcome: Outcome, steps: Vec<Step>) -> Trajectory {
    Trajectory {
        task_id: id.into(),
        steps,
        outcome,
    }
}

fn step(tool: &str, args: &str, result: &str) -> Step {
    Step {
        tool: tool.into(),
        args: args.into(),
        result: result.into(),
    }
}

#[derive(Clone, Default)]
struct CallLog {
    calls: Rc<RefCell<Vec<(Vec<String>, u32, f32)>>>,
}

impl CallLog {
    fn snapshot(&self) -> Vec<(Vec<String>, u32, f32)> {
        self.calls.borrow().clone()
    }
}

struct QueueGenerator {
    completions: Vec<String>,
    pos: usize,
    log: CallLog,
}

impl QueueGenerator {
    fn new(completions: &[&str]) -> Self {
        Self {
            completions: completions.iter().map(|s| (*s).to_string()).collect(),
            pos: 0,
            log: CallLog::default(),
        }
    }

    fn with_log(completions: &[&str], log: CallLog) -> Self {
        Self {
            completions: completions.iter().map(|s| (*s).to_string()).collect(),
            pos: 0,
            log,
        }
    }
}

impl Generator for QueueGenerator {
    fn generate(
        &mut self,
        prompts: &[String],
        max_tokens: u32,
        temperature: f32,
    ) -> prometheus_synth::Result<Vec<String>> {
        self.log
            .calls
            .borrow_mut()
            .push((prompts.to_vec(), max_tokens, temperature));
        let n = prompts.len();
        if self.pos + n > self.completions.len() {
            panic!("not enough scripted completions");
        }
        let out = self.completions[self.pos..self.pos + n].to_vec();
        self.pos += n;
        Ok(out)
    }
}

struct ConstGenerator {
    completion: String,
    log: CallLog,
}

impl ConstGenerator {
    fn new(completion: &str) -> Self {
        Self {
            completion: completion.into(),
            log: CallLog::default(),
        }
    }

    fn with_log(completion: &str, log: CallLog) -> Self {
        Self {
            completion: completion.into(),
            log,
        }
    }
}

impl Generator for ConstGenerator {
    fn generate(
        &mut self,
        prompts: &[String],
        max_tokens: u32,
        temperature: f32,
    ) -> prometheus_synth::Result<Vec<String>> {
        self.log
            .calls
            .borrow_mut()
            .push((prompts.to_vec(), max_tokens, temperature));
        Ok(vec![self.completion.clone(); prompts.len()])
    }
}

struct MismatchGenerator {
    n: usize,
}

impl Generator for MismatchGenerator {
    fn generate(
        &mut self,
        _prompts: &[String],
        _max_tokens: u32,
        _temperature: f32,
    ) -> prometheus_synth::Result<Vec<String>> {
        Ok(vec!["submit pass".into(); self.n])
    }
}

struct BoomGenerator;

impl Generator for BoomGenerator {
    fn generate(
        &mut self,
        _prompts: &[String],
        _max_tokens: u32,
        _temperature: f32,
    ) -> prometheus_synth::Result<Vec<String>> {
        panic!("generator must not be called");
    }
}

struct ErrGenerator {
    err: Error,
}

impl Generator for ErrGenerator {
    fn generate(
        &mut self,
        _prompts: &[String],
        _max_tokens: u32,
        _temperature: f32,
    ) -> prometheus_synth::Result<Vec<String>> {
        Err(self.err.clone())
    }
}

struct ScriptSandbox {
    submit: String,
    other: String,
}

impl ScriptSandbox {
    fn new(submit: &str, other: &str) -> Self {
        Self {
            submit: submit.into(),
            other: other.into(),
        }
    }
}

impl Sandbox for ScriptSandbox {
    fn call(&mut self, tool: &str, _args: &str) -> prometheus_synth::Result<String> {
        if tool == agentic_ref::SUBMIT_TOOL {
            Ok(self.submit.clone())
        } else {
            Ok(self.other.clone())
        }
    }
}

struct BoomSandbox;

impl Sandbox for BoomSandbox {
    fn call(&mut self, _tool: &str, _args: &str) -> prometheus_synth::Result<String> {
        panic!("sandbox must not be called");
    }
}

struct ErrSandbox {
    err: Error,
}

impl Sandbox for ErrSandbox {
    fn call(&mut self, _tool: &str, _args: &str) -> prometheus_synth::Result<String> {
        Err(self.err.clone())
    }
}

struct QueueSandbox {
    results: Vec<prometheus_synth::Result<String>>,
    pos: usize,
}

impl QueueSandbox {
    fn new(results: Vec<prometheus_synth::Result<String>>) -> Self {
        Self { results, pos: 0 }
    }
}

impl Sandbox for QueueSandbox {
    fn call(&mut self, _tool: &str, _args: &str) -> prometheus_synth::Result<String> {
        if self.pos >= self.results.len() {
            panic!("not enough scripted sandbox results");
        }
        let out = self.results[self.pos].clone();
        self.pos += 1;
        out
    }
}

fn golden_completions() -> &'static [&'static str] {
    // n=4, max_steps=2:
    // 0: shell then submit pass -> Verified
    // 1: shell then submit fail -> Failed (submit result is not "verified")
    // 2: empty completion -> Failed, no sandbox call
    // 3: two non-terminal shells -> Truncated
    &[
        "shell ls",
        "submit pass",
        "shell ls",
        "submit fail",
        "",
        "shell ls",
        "shell pwd",
    ]
}

fn golden_expected() -> Vec<Trajectory> {
    vec![
        traj(
            "t1",
            Outcome::Verified,
            vec![
                step("shell", "ls", "ok"),
                step("submit", "pass", "verified"),
            ],
        ),
        traj(
            "t1",
            Outcome::Failed,
            vec![step("shell", "ls", "ok"), step("submit", "fail", "no")],
        ),
        traj("t1", Outcome::Failed, vec![]),
        traj(
            "t1",
            Outcome::Truncated,
            vec![step("shell", "ls", "ok"), step("shell", "pwd", "ok")],
        ),
    ]
}

// ---------------------------------------------------------------------------
// generate_one: errors
// ---------------------------------------------------------------------------

#[test]
fn generate_one_zero_samples() {
    let err = generate_one(
        &mut BoomGenerator,
        &mut BoomSandbox,
        &fix_task(),
        0,
        3,
        GOLDEN_MAX_TOKENS,
        GOLDEN_TEMPERATURE,
    )
    .unwrap_err();
    assert_eq!(err, Error::ZeroSamples);
    let err_ref = agentic_ref::generate_one(
        &mut BoomGenerator,
        &mut BoomSandbox,
        &fix_task(),
        0,
        3,
        GOLDEN_MAX_TOKENS,
        GOLDEN_TEMPERATURE,
    )
    .unwrap_err();
    assert_eq!(err, err_ref);
}

#[test]
fn generate_one_zero_samples_wins_over_empty_prompt() {
    let err = generate_one(
        &mut BoomGenerator,
        &mut BoomSandbox,
        &task("t", ""),
        0,
        3,
        GOLDEN_MAX_TOKENS,
        GOLDEN_TEMPERATURE,
    )
    .unwrap_err();
    assert_eq!(err, Error::ZeroSamples);
}

#[test]
fn generate_one_empty_prompt() {
    let err = generate_one(
        &mut BoomGenerator,
        &mut BoomSandbox,
        &task("t", ""),
        2,
        3,
        GOLDEN_MAX_TOKENS,
        GOLDEN_TEMPERATURE,
    )
    .unwrap_err();
    assert_eq!(err, Error::EmptyPrompt);
    let err_ref = agentic_ref::generate_one(
        &mut BoomGenerator,
        &mut BoomSandbox,
        &task("t", ""),
        2,
        3,
        GOLDEN_MAX_TOKENS,
        GOLDEN_TEMPERATURE,
    )
    .unwrap_err();
    assert_eq!(err, err_ref);
}

#[test]
fn generate_one_whitespace_prompt_is_not_empty() {
    let t = task("t", "  \t");
    let mut gen = QueueGenerator::new(&["submit pass"]);
    let mut sbx = ScriptSandbox::new("verified", "ok");
    let got = generate_one(&mut gen, &mut sbx, &t, 1, 1, 8, 0.0).unwrap();
    let mut gen_r = QueueGenerator::new(&["submit pass"]);
    let mut sbx_r = ScriptSandbox::new("verified", "ok");
    let exp = agentic_ref::generate_one(&mut gen_r, &mut sbx_r, &t, 1, 1, 8, 0.0).unwrap();
    assert_eq!(got, exp);
    assert_eq!(got[0].outcome, Outcome::Verified);
}

// ---------------------------------------------------------------------------
// generate_one: n trajectories and outcomes
// ---------------------------------------------------------------------------

#[test]
fn generate_one_n_trajectories() {
    let mut gen = ConstGenerator::new("echo x");
    let mut sbx = ScriptSandbox::new("verified", "ok");
    let n = 5;
    let got = generate_one(&mut gen, &mut sbx, &fix_task(), n, 1, 16, 0.0).unwrap();
    assert_eq!(got.len(), n as usize);
    let mut gen_r = ConstGenerator::new("echo x");
    let mut sbx_r = ScriptSandbox::new("verified", "ok");
    let exp =
        agentic_ref::generate_one(&mut gen_r, &mut sbx_r, &fix_task(), n, 1, 16, 0.0).unwrap();
    assert_eq!(got, exp);
    for tr in &got {
        assert_eq!(tr.task_id, "t1");
        assert_eq!(tr.outcome, Outcome::Truncated);
        assert_eq!(tr.steps.len(), 1);
        assert_eq!(tr.steps[0].tool, "echo");
        assert_eq!(tr.steps[0].args, "x");
        assert_eq!(tr.steps[0].result, "ok");
    }
}

#[test]
fn generate_one_truncated_when_max_steps_hit_without_verified() {
    let mut gen = ConstGenerator::new("shell ls");
    let mut sbx = ScriptSandbox::new("verified", "ok");
    let got = generate_one(&mut gen, &mut sbx, &fix_task(), 1, 3, 16, 0.0).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].outcome, Outcome::Truncated);
    assert_eq!(got[0].steps.len(), 3);
    for s in &got[0].steps {
        assert_eq!(s.tool, "shell");
        assert_eq!(s.args, "ls");
        assert_eq!(s.result, "ok");
    }
    let mut gen_r = ConstGenerator::new("shell ls");
    let mut sbx_r = ScriptSandbox::new("verified", "ok");
    let exp =
        agentic_ref::generate_one(&mut gen_r, &mut sbx_r, &fix_task(), 1, 3, 16, 0.0).unwrap();
    assert_eq!(got, exp);
}

#[test]
fn generate_one_max_steps_zero_is_truncated_without_calls() {
    let got = generate_one(
        &mut BoomGenerator,
        &mut BoomSandbox,
        &fix_task(),
        2,
        0,
        16,
        0.0,
    )
    .unwrap();
    assert_eq!(got.len(), 2);
    for tr in &got {
        assert_eq!(tr.outcome, Outcome::Truncated);
        assert!(tr.steps.is_empty());
        assert_eq!(tr.task_id, "t1");
    }
    let exp = agentic_ref::generate_one(
        &mut BoomGenerator,
        &mut BoomSandbox,
        &fix_task(),
        2,
        0,
        16,
        0.0,
    )
    .unwrap();
    assert_eq!(got, exp);
}

#[test]
fn generate_one_verified_when_sandbox_and_generator_succeed() {
    let mut gen = QueueGenerator::new(&["shell ls", "submit pass"]);
    let mut sbx = ScriptSandbox::new("verified", "ok");
    let got = generate_one(&mut gen, &mut sbx, &fix_task(), 1, 5, 16, 0.0).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].outcome, Outcome::Verified);
    assert_eq!(got[0].steps.len(), 2);
    assert_eq!(got[0].steps[0], step("shell", "ls", "ok"));
    assert_eq!(got[0].steps[1], step("submit", "pass", "verified"));
    let mut gen_r = QueueGenerator::new(&["shell ls", "submit pass"]);
    let mut sbx_r = ScriptSandbox::new("verified", "ok");
    let exp =
        agentic_ref::generate_one(&mut gen_r, &mut sbx_r, &fix_task(), 1, 5, 16, 0.0).unwrap();
    assert_eq!(got, exp);
}

#[test]
fn generate_one_verified_stops_before_max_steps() {
    let log = CallLog::default();
    let mut gen = QueueGenerator::with_log(&["submit pass"], log.clone());
    let mut sbx = ScriptSandbox::new("verified", "ok");
    let got = generate_one(&mut gen, &mut sbx, &fix_task(), 1, 9, 16, 0.0).unwrap();
    assert_eq!(got[0].outcome, Outcome::Verified);
    assert_eq!(got[0].steps.len(), 1);
    assert_eq!(log.snapshot().len(), 1);
}

#[test]
fn generate_one_failed_on_unparseable() {
    let mut gen = QueueGenerator::new(&[""]);
    let got = generate_one(&mut gen, &mut BoomSandbox, &fix_task(), 1, 3, 16, 0.0).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].outcome, Outcome::Failed);
    assert!(got[0].steps.is_empty());
    let mut gen_r = QueueGenerator::new(&[""]);
    let exp = agentic_ref::generate_one(&mut gen_r, &mut BoomSandbox, &fix_task(), 1, 3, 16, 0.0)
        .unwrap();
    assert_eq!(got, exp);
}

#[test]
fn generate_one_failed_on_whitespace_only_completion() {
    let mut gen = QueueGenerator::new(&["  \n\t  "]);
    let got = generate_one(&mut gen, &mut BoomSandbox, &fix_task(), 1, 2, 16, 0.0).unwrap();
    assert_eq!(got[0].outcome, Outcome::Failed);
    assert!(got[0].steps.is_empty());
}

#[test]
fn generate_one_failed_on_submit_not_verified() {
    let mut gen = QueueGenerator::new(&["submit no"]);
    let mut sbx = ScriptSandbox::new("no", "ok");
    let got = generate_one(&mut gen, &mut sbx, &fix_task(), 1, 4, 16, 0.0).unwrap();
    assert_eq!(got[0].outcome, Outcome::Failed);
    assert_eq!(got[0].steps, vec![step("submit", "no", "no")]);
    let mut gen_r = QueueGenerator::new(&["submit no"]);
    let mut sbx_r = ScriptSandbox::new("no", "ok");
    let exp =
        agentic_ref::generate_one(&mut gen_r, &mut sbx_r, &fix_task(), 1, 4, 16, 0.0).unwrap();
    assert_eq!(got, exp);
}

#[test]
fn generate_one_submit_result_must_match_verified_exactly() {
    for result in ["verified ", "VERIFIED", " verified", "ok", ""] {
        let mut gen = QueueGenerator::new(&["submit pass"]);
        let mut sbx = ScriptSandbox::new(result, "ok");
        let got = generate_one(&mut gen, &mut sbx, &fix_task(), 1, 1, 16, 0.0).unwrap();
        assert_eq!(got[0].outcome, Outcome::Failed, "result={result:?}");
        assert_eq!(got[0].steps[0].result, result);
    }
}

#[test]
fn generate_one_non_submit_verified_string_does_not_verify() {
    let mut gen = ConstGenerator::new("shell verified");
    let mut sbx = ScriptSandbox::new("verified", "verified");
    let got = generate_one(&mut gen, &mut sbx, &fix_task(), 1, 2, 16, 0.0).unwrap();
    assert_eq!(got[0].outcome, Outcome::Truncated);
    assert_eq!(got[0].steps.len(), 2);
    assert_eq!(got[0].steps[0].tool, "shell");
    assert_eq!(got[0].steps[0].result, "verified");
}

#[test]
fn generate_one_submit_tool_is_case_sensitive() {
    let mut gen = ConstGenerator::new("Submit pass");
    let mut sbx = ScriptSandbox::new("verified", "ok");
    let got = generate_one(&mut gen, &mut sbx, &fix_task(), 1, 1, 16, 0.0).unwrap();
    assert_eq!(got[0].outcome, Outcome::Truncated);
    assert_eq!(got[0].steps[0].tool, "Submit");
}

#[test]
fn generate_one_copies_task_id_including_empty() {
    let t = task("", "do it");
    let mut gen = QueueGenerator::new(&["submit pass"]);
    let mut sbx = ScriptSandbox::new("verified", "ok");
    let got = generate_one(&mut gen, &mut sbx, &t, 1, 1, 16, 0.0).unwrap();
    assert_eq!(got[0].task_id, "");
    assert_eq!(got[0].outcome, Outcome::Verified);
}

#[test]
fn generate_one_length_mismatch() {
    let err = generate_one(
        &mut MismatchGenerator { n: 0 },
        &mut BoomSandbox,
        &fix_task(),
        1,
        1,
        16,
        0.0,
    )
    .unwrap_err();
    assert_eq!(err, Error::LengthMismatch { want: 1, got: 0 });
    let err = generate_one(
        &mut MismatchGenerator { n: 2 },
        &mut BoomSandbox,
        &fix_task(),
        1,
        1,
        16,
        0.0,
    )
    .unwrap_err();
    assert_eq!(err, Error::LengthMismatch { want: 1, got: 2 });
}

#[test]
fn generate_one_golden_mix_matches_reference() {
    let mut gen = QueueGenerator::new(golden_completions());
    // Mixed submit results need a queue: ScriptSandbox maps every submit the same way.
    let mut sbx = QueueSandbox::new(vec![
        Ok("ok".into()),
        Ok("verified".into()),
        Ok("ok".into()),
        Ok("no".into()),
        Ok("ok".into()),
        Ok("ok".into()),
    ]);
    let got = generate_one(
        &mut gen,
        &mut sbx,
        &fix_task(),
        GOLDEN_N,
        GOLDEN_MAX_STEPS,
        GOLDEN_MAX_TOKENS,
        GOLDEN_TEMPERATURE,
    )
    .unwrap();
    assert_eq!(got, golden_expected());

    let mut gen_r = QueueGenerator::new(golden_completions());
    let mut sbx_r = QueueSandbox::new(vec![
        Ok("ok".into()),
        Ok("verified".into()),
        Ok("ok".into()),
        Ok("no".into()),
        Ok("ok".into()),
        Ok("ok".into()),
    ]);
    let exp = agentic_ref::generate_one(
        &mut gen_r,
        &mut sbx_r,
        &fix_task(),
        GOLDEN_N,
        GOLDEN_MAX_STEPS,
        GOLDEN_MAX_TOKENS,
        GOLDEN_TEMPERATURE,
    )
    .unwrap();
    assert_eq!(got, exp);
}

#[test]
fn generate_one_history_in_subsequent_prompts() {
    let log = CallLog::default();
    let mut gen = QueueGenerator::with_log(&["shell ls", "submit pass"], log.clone());
    let mut sbx = ScriptSandbox::new("verified", "ok");
    let t = fix_task();
    let got = generate_one(&mut gen, &mut sbx, &t, 1, 4, 16, 0.25).unwrap();
    assert_eq!(got[0].outcome, Outcome::Verified);
    let calls = log.snapshot();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].0, vec![t.prompt.clone()]);
    let expected_second = agentic_ref::step_prompt(&t, &got[0].steps[..1]);
    assert_eq!(calls[1].0, vec![expected_second.clone()]);
    assert_eq!(expected_second, "fix the bug\nshell\nls\nok");
}

#[test]
fn generate_one_utf8_roundtrip() {
    let t = task("id-café", "répare le bogue");
    let mut gen = QueueGenerator::new(&["echo café", "submit passé"]);
    let mut sbx = ScriptSandbox::new("verified", "ok ✓");
    let got = generate_one(&mut gen, &mut sbx, &t, 1, 3, 16, 0.0).unwrap();
    assert_eq!(got[0].task_id, "id-café");
    assert_eq!(got[0].outcome, Outcome::Verified);
    assert_eq!(got[0].steps[0], step("echo", "café", "ok ✓"));
    assert_eq!(got[0].steps[1], step("submit", "passé", "verified"));
    let mut gen_r = QueueGenerator::new(&["echo café", "submit passé"]);
    let mut sbx_r = ScriptSandbox::new("verified", "ok ✓");
    let exp = agentic_ref::generate_one(&mut gen_r, &mut sbx_r, &t, 1, 3, 16, 0.0).unwrap();
    assert_eq!(got, exp);
}

#[test]
fn generate_one_independent_histories_across_samples() {
    let log = CallLog::default();
    let mut gen = QueueGenerator::with_log(
        &["shell a", "submit pass", "shell b", "submit pass"],
        log.clone(),
    );
    let mut sbx = ScriptSandbox::new("verified", "ok");
    let t = fix_task();
    let got = generate_one(&mut gen, &mut sbx, &t, 2, 3, 16, 0.0).unwrap();
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].steps[0].args, "a");
    assert_eq!(got[1].steps[0].args, "b");
    let calls = log.snapshot();
    assert_eq!(calls[0].0, vec![t.prompt.clone()]);
    assert_eq!(calls[2].0, vec![t.prompt.clone()]);
}

// ---------------------------------------------------------------------------
// generate_batch
// ---------------------------------------------------------------------------

#[test]
fn generate_batch_empty_batch() {
    let err = generate_batch(&mut BoomGenerator, &mut BoomSandbox, &[], 3, 2, 16, 0.0).unwrap_err();
    assert_eq!(err, Error::EmptyBatch);
    let err_ref =
        agentic_ref::generate_batch(&mut BoomGenerator, &mut BoomSandbox, &[], 3, 2, 16, 0.0)
            .unwrap_err();
    assert_eq!(err, err_ref);
}

#[test]
fn generate_batch_empty_batch_wins_over_zero_n() {
    let err = generate_batch(&mut BoomGenerator, &mut BoomSandbox, &[], 0, 2, 16, 0.0).unwrap_err();
    assert_eq!(err, Error::EmptyBatch);
}

#[test]
fn generate_batch_zero_samples() {
    let tasks = [fix_task()];
    let err =
        generate_batch(&mut BoomGenerator, &mut BoomSandbox, &tasks, 0, 2, 16, 0.0).unwrap_err();
    assert_eq!(err, Error::ZeroSamples);
}

#[test]
fn generate_batch_empty_prompt_before_generate() {
    let tasks = [fix_task(), task("t2", "")];
    let err =
        generate_batch(&mut BoomGenerator, &mut BoomSandbox, &tasks, 2, 2, 16, 0.0).unwrap_err();
    assert_eq!(err, Error::EmptyPrompt);
}

#[test]
fn generate_batch_one_vec_per_task() {
    let tasks = [task("a", "alpha"), task("b", "beta"), task("c", "gamma")];
    let mut gen = ConstGenerator::new("submit pass");
    let mut sbx = ScriptSandbox::new("verified", "ok");
    let n = 2;
    let got = generate_batch(&mut gen, &mut sbx, &tasks, n, 1, 16, 0.0).unwrap();
    assert_eq!(got.len(), tasks.len());
    for (inner, t) in got.iter().zip(tasks.iter()) {
        assert_eq!(inner.len(), n as usize);
        for tr in inner {
            assert_eq!(tr.task_id, t.task_id);
            assert_eq!(tr.outcome, Outcome::Verified);
        }
    }
    let mut gen_r = ConstGenerator::new("submit pass");
    let mut sbx_r = ScriptSandbox::new("verified", "ok");
    let exp = agentic_ref::generate_batch(&mut gen_r, &mut sbx_r, &tasks, n, 1, 16, 0.0).unwrap();
    assert_eq!(got, exp);
}

#[test]
fn generate_batch_preserves_task_order() {
    let tasks = [task("z", "one"), task("y", "two")];
    let mut gen = ConstGenerator::new("echo x");
    let mut sbx = ScriptSandbox::new("verified", "ok");
    let got = generate_batch(&mut gen, &mut sbx, &tasks, 1, 1, 8, 0.0).unwrap();
    assert_eq!(got[0][0].task_id, "z");
    assert_eq!(got[1][0].task_id, "y");
}

// ---------------------------------------------------------------------------
// outcome_verified_rate
// ---------------------------------------------------------------------------

#[test]
fn outcome_verified_rate_empty_is_zero() {
    let r = outcome_verified_rate(&[]);
    assert_eq!(r, 0.0);
    assert_eq!(r, agentic_ref::outcome_verified_rate(&[]));
}

#[test]
fn outcome_verified_rate_golden_fraction() {
    let trajs = golden_expected();
    let r = outcome_verified_rate(&trajs);
    assert_eq!(r, GOLDEN_VERIFIED_RATE);
    assert_eq!(r, agentic_ref::outcome_verified_rate(&trajs));
    assert_eq!(r, 1.0 / 4.0);
}

#[test]
fn outcome_verified_rate_all_verified_is_one() {
    let trajs = vec![
        traj("a", Outcome::Verified, vec![]),
        traj("b", Outcome::Verified, vec![]),
    ];
    assert_eq!(outcome_verified_rate(&trajs), 1.0);
    assert_eq!(
        outcome_verified_rate(&trajs),
        agentic_ref::outcome_verified_rate(&trajs)
    );
}

#[test]
fn outcome_verified_rate_none_verified_is_zero() {
    let trajs = vec![
        traj("a", Outcome::Failed, vec![]),
        traj("b", Outcome::Truncated, vec![]),
    ];
    assert_eq!(outcome_verified_rate(&trajs), 0.0);
}

#[test]
fn outcome_verified_rate_matches_reference_fractions() {
    let trajs = vec![
        traj("a", Outcome::Verified, vec![]),
        traj("b", Outcome::Failed, vec![]),
        traj("c", Outcome::Truncated, vec![]),
    ];
    let r = outcome_verified_rate(&trajs);
    assert_eq!(r, 1.0 / 3.0);
    assert_eq!(r, agentic_ref::outcome_verified_rate(&trajs));
}

// ---------------------------------------------------------------------------
// keep_verified
// ---------------------------------------------------------------------------

#[test]
fn keep_verified_drops_failed_and_truncated() {
    let trajs = golden_expected();
    let kept = keep_verified(trajs.clone());
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].outcome, Outcome::Verified);
    assert_eq!(kept[0].task_id, "t1");
    assert_eq!(kept, agentic_ref::keep_verified(trajs));
}

#[test]
fn keep_verified_empty() {
    let kept = keep_verified(Vec::new());
    assert!(kept.is_empty());
    assert_eq!(kept, agentic_ref::keep_verified(Vec::new()));
}

#[test]
fn keep_verified_preserves_order() {
    let trajs = vec![
        traj("a", Outcome::Failed, vec![]),
        traj(
            "b",
            Outcome::Verified,
            vec![step("submit", "1", "verified")],
        ),
        traj("c", Outcome::Truncated, vec![]),
        traj(
            "d",
            Outcome::Verified,
            vec![step("submit", "2", "verified")],
        ),
        traj("e", Outcome::Failed, vec![]),
    ];
    let kept = keep_verified(trajs.clone());
    assert_eq!(kept.len(), 2);
    assert_eq!(kept[0].task_id, "b");
    assert_eq!(kept[1].task_id, "d");
    assert_eq!(kept, agentic_ref::keep_verified(trajs));
}

#[test]
fn keep_verified_all_verified() {
    let trajs = vec![
        traj("a", Outcome::Verified, vec![]),
        traj("b", Outcome::Verified, vec![]),
    ];
    let kept = keep_verified(trajs.clone());
    assert_eq!(kept, trajs);
}

// ---------------------------------------------------------------------------
// max_tokens / temperature pass-through
// ---------------------------------------------------------------------------

#[test]
fn generate_one_passes_max_tokens_and_temperature() {
    let log = CallLog::default();
    let mut gen = QueueGenerator::with_log(&["submit pass", "submit pass"], log.clone());
    let mut sbx = ScriptSandbox::new("verified", "ok");
    let _ = generate_one(&mut gen, &mut sbx, &fix_task(), 2, 1, 99, 0.125).unwrap();
    let calls = log.snapshot();
    assert_eq!(calls.len(), 2);
    for (prompts, max_tokens, temperature) in &calls {
        assert_eq!(prompts.len(), 1);
        assert_eq!(*max_tokens, 99);
        assert_eq!(*temperature, 0.125);
    }
}

#[test]
fn generate_batch_passes_max_tokens_and_temperature() {
    let log = CallLog::default();
    let mut gen = ConstGenerator::with_log("submit pass", log.clone());
    let mut sbx = ScriptSandbox::new("verified", "ok");
    let tasks = [task("a", "one"), task("b", "two")];
    let _ = generate_batch(&mut gen, &mut sbx, &tasks, 1, 1, 7, 1.5).unwrap();
    let calls = log.snapshot();
    assert_eq!(calls.len(), 2);
    for (prompts, max_tokens, temperature) in &calls {
        assert_eq!(prompts.len(), 1);
        assert_eq!(*max_tokens, 7);
        assert_eq!(*temperature, 1.5);
    }
}

#[test]
fn generate_one_first_prompt_is_task_prompt() {
    let log = CallLog::default();
    let t = task("t", "exactly this prompt");
    let mut gen = QueueGenerator::with_log(&["submit pass"], log.clone());
    let mut sbx = ScriptSandbox::new("verified", "ok");
    let _ = generate_one(&mut gen, &mut sbx, &t, 1, 1, 4, 0.0).unwrap();
    let calls = log.snapshot();
    assert_eq!(calls[0].0, vec!["exactly this prompt".to_string()]);
}

// ---------------------------------------------------------------------------
// property tests
// ---------------------------------------------------------------------------

fn every_outcome_list(max_len: usize) -> Vec<Vec<Outcome>> {
    let opts = [Outcome::Verified, Outcome::Failed, Outcome::Truncated];
    let mut acc = vec![vec![]];
    let mut all = vec![vec![]];
    for _ in 0..max_len {
        let mut nxt = Vec::new();
        for prefix in &acc {
            for &o in &opts {
                let mut v = prefix.clone();
                v.push(o);
                nxt.push(v);
            }
        }
        all.extend(nxt.clone());
        acc = nxt;
    }
    all
}

fn is_subsequence(hay: &[Trajectory], needle: &[Trajectory]) -> bool {
    let mut i = 0;
    for n in needle {
        loop {
            if i >= hay.len() {
                return false;
            }
            let cur = &hay[i];
            i += 1;
            if cur == n {
                break;
            }
        }
    }
    true
}

#[test]
fn property_keep_verified_is_verified_subsequence() {
    for outcomes in every_outcome_list(4) {
        let trajs: Vec<Trajectory> = outcomes
            .iter()
            .enumerate()
            .map(|(i, o)| traj(&format!("{i}"), *o, vec![]))
            .collect();
        let kept = keep_verified(trajs.clone());
        assert!(is_subsequence(&trajs, &kept));
        assert!(kept.iter().all(|t| t.outcome == Outcome::Verified));
        let expected: Vec<Trajectory> = trajs
            .iter()
            .filter(|t| t.outcome == Outcome::Verified)
            .cloned()
            .collect();
        assert_eq!(kept, expected);
        assert_eq!(kept, agentic_ref::keep_verified(trajs));
    }
}

#[test]
fn property_rate_in_unit_interval() {
    for outcomes in every_outcome_list(4) {
        let trajs: Vec<Trajectory> = outcomes
            .iter()
            .enumerate()
            .map(|(i, o)| traj(&format!("{i}"), *o, vec![]))
            .collect();
        let r = outcome_verified_rate(&trajs);
        assert!((0.0..=1.0).contains(&r), "rate {r} out of range");
        assert_eq!(r, agentic_ref::outcome_verified_rate(&trajs));
        if trajs.is_empty() {
            assert_eq!(r, 0.0);
        }
    }
}

#[test]
fn property_batch_length_equals_tasks_len() {
    for n_tasks in 1..=3 {
        for n in 1..=3 {
            for max_steps in 0..=2 {
                let tasks: Vec<AgenticTask> = (0..n_tasks)
                    .map(|i| task(&format!("t{i}"), "do it"))
                    .collect();
                let mut gen = ConstGenerator::new("echo x");
                let mut sbx = ScriptSandbox::new("verified", "ok");
                let got = generate_batch(&mut gen, &mut sbx, &tasks, n, max_steps, 8, 0.0).unwrap();
                assert_eq!(got.len(), tasks.len());
                for inner in &got {
                    assert_eq!(inner.len(), n as usize);
                }
                let mut gen_r = ConstGenerator::new("echo x");
                let mut sbx_r = ScriptSandbox::new("verified", "ok");
                let exp = agentic_ref::generate_batch(
                    &mut gen_r, &mut sbx_r, &tasks, n, max_steps, 8, 0.0,
                )
                .unwrap();
                assert_eq!(got, exp);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// fault injection
// ---------------------------------------------------------------------------

#[test]
fn generate_one_sandbox_error_surfaces() {
    let mut gen = QueueGenerator::new(&["shell ls"]);
    let boom = Error::Message("sandbox down".into());
    let err = generate_one(
        &mut gen,
        &mut ErrSandbox { err: boom.clone() },
        &fix_task(),
        1,
        2,
        16,
        0.0,
    )
    .unwrap_err();
    assert_eq!(err, boom);
}

#[test]
fn generate_one_sandbox_error_on_later_step_surfaces() {
    let mut gen = QueueGenerator::new(&["shell ls", "shell pwd"]);
    let boom = Error::Message("later".into());
    let mut sbx = QueueSandbox::new(vec![Ok("ok".into()), Err(boom.clone())]);
    let err = generate_one(&mut gen, &mut sbx, &fix_task(), 1, 3, 16, 0.0).unwrap_err();
    assert_eq!(err, boom);
}

#[test]
fn generate_batch_sandbox_error_surfaces() {
    let tasks = [fix_task()];
    let mut gen = QueueGenerator::new(&["shell ls"]);
    let boom = Error::Message("batch sandbox".into());
    let err = generate_batch(
        &mut gen,
        &mut ErrSandbox { err: boom.clone() },
        &tasks,
        1,
        1,
        16,
        0.0,
    )
    .unwrap_err();
    assert_eq!(err, boom);
}

#[test]
fn generate_one_generator_error_surfaces() {
    let boom = Error::Message("generator down".into());
    let err = generate_one(
        &mut ErrGenerator { err: boom.clone() },
        &mut BoomSandbox,
        &fix_task(),
        1,
        2,
        16,
        0.0,
    )
    .unwrap_err();
    assert_eq!(err, boom);
}
