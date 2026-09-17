//! E2 rejection-sampling oracle tests (spec 7.1, 15.5).
//!
//! Production `extract_answer` / `RejectionSampler::sample` /
//! `RejectionSampler::sample_many` must match `common::reject_ref`.

mod common;

use std::cell::RefCell;
use std::rc::Rc;

use prometheus_synth::{
    generate_solve_prompt, verified_correct_rate, Domain, Error, Generator, Problem,
    RejectionResult, RejectionSampler, Sample, SamplerConfig, TokenCounter, Verifier,
    DEFAULT_MAX_BATCH, DEFAULT_MAX_TOKENS, DEFAULT_N_SAMPLES, DEFAULT_TEMPERATURE,
};

use common::reject_ref;

const GOLDEN_VERIFIED_CORRECT_RATE: f64 = 0.25;
const GOLDEN_N_GENERATED: u32 = 8;
const GOLDEN_N_ACCEPTED: u32 = 2;
const GOLDEN_EXPECTED_ANSWER: &str = "42";

const GOLDEN_COMPLETIONS: &[&str] = &[
    "I think the result is 41.\nAnswer: 41",
    "Work shown.\nAnswer: 42",
    "```python\nprint(0)\n```",
    "Answer: 42",
    "nope",
    "Answer: 40",
    "the last line is 99",
    "Answer: 7",
];

const EXTRACT_GOLDENS: &[(&str, &str)] = &[
    ("", ""),
    ("   \t\n", ""),
    ("   ", ""),
    ("Answer: first\nAnswer: second", "second"),
    ("Answer: first\nReasoning\nAnswer: second", "second"),
    ("ANSWER: 42", "42"),
    ("answer: 42", "42"),
    ("AnSwEr: 42", "42"),
    ("answer : 42", "42"),
    ("answer:42", "42"),
    ("Answer:\n42", "42"),
    ("Answer:   \n42", "42"),
    ("Answer:\nAnswer: 7", "7"),
    ("Answer: first\nAnswer:", "first"),
    ("Answer: first\nAnswer:   ", "first"),
    ("```\nfirst\n```\n```\nsecond\n```", "second\n"),
    ("```python\nprint(1)\n```", "print(1)\n"),
    ("thinking\n```python\nprint(1)", "print(1)"),
    ("line1\nline2\nline3", "line3"),
    ("line1\n\nline2", "line2"),
    ("Answer: 42   \n\n", "42"),
    ("```\ncode\n```\nAnswer: 99", "99"),
    ("Answer:\n```\ncode\n```", "code\n"),
    ("  Answer: 42", "  Answer: 42"),
    ("foo\n  42", "  42"),
    ("```code```", "```code```"),
    ("```\n```", ""),
    ("hello\n```\n```", ""),
    ("```python\nprint(1)```", "print(1)"),
    ("Answer: 0", "0"),
    ("Answer: false", "false"),
    ("answered: 9\n9", "9"),
    ("Answer: Café", "Café"),
    ("the answer is 42", "the answer is 42"),
    ("42", "42"),
    ("  hello", "  hello"),
    ("Answer:  foo  \nbar", "foo"),
    ("answer\t:\t42", "42"),
    ("Answer :42", "42"),
    ("ANSWER:\n42", "42"),
    ("```rust\nfn main() {}\n```", "fn main() {}\n"),
    ("see ```code``` here", "see ```code``` here"),
    ("foo\n```\nbar\n```\nbaz", "bar\n"),
    ("Answer: 1\nAnswer:\nAnswer: 3", "3"),
    ("```\nfirst\n```\n```python\nsecond\n```", "second\n"),
    ("hello\n```\nunclosed", "unclosed"),
    ("Answer:   42", "42"),
    ("\n\nAnswer: 5\n", "5"),
    ("line1\n   \nline2", "line2"),
    ("```\nbody\n```", "body\n"),
    ("``` python\nx\n```", "x\n"),
    ("lots of words here\nAnswer: 42", "42"),
    ("Answer: 42 extra", "42 extra"),
    ("answer: \nlast", "last"),
    ("```\nfirst\n```\nnot a fence", "first\n"),
    ("foo\n\nbar\n\n", "bar"),
    ("Answer:\nAnswer:\nhello", "hello"),
    ("```\n\n```", "\n"),
    ("x\n```python\n\ny\n```", "\ny\n"),
];

fn problem(pid: &str, prompt: &str, domain: Domain, verifier_id: &str) -> Problem {
    Problem {
        problem_id: pid.into(),
        domain,
        prompt: prompt.into(),
        verifier_id: verifier_id.into(),
    }
}

fn math_problem() -> Problem {
    problem("p1", "What is 6*7?", Domain::Math, GOLDEN_EXPECTED_ANSWER)
}

fn cfg_with(n_samples: u32, max_tokens: u32, temperature: f32, max_batch: u32) -> SamplerConfig {
    SamplerConfig {
        n_samples,
        max_tokens,
        temperature,
        max_batch,
    }
}

#[derive(Clone, Default)]
struct CallLog {
    calls: Rc<RefCell<Vec<(Vec<String>, u32, f32)>>>,
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
        Ok(vec!["x".into(); self.n])
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

struct MatchVerifier;

impl Verifier for MatchVerifier {
    fn verify(&mut self, problem: &Problem, answer: &str) -> prometheus_synth::Result<bool> {
        Ok(answer == problem.verifier_id)
    }
}

struct ExactVerifier {
    expected: String,
    seen: Rc<RefCell<Vec<(String, String)>>>,
}

impl Verifier for ExactVerifier {
    fn verify(&mut self, problem: &Problem, answer: &str) -> prometheus_synth::Result<bool> {
        self.seen
            .borrow_mut()
            .push((problem.problem_id.clone(), answer.to_string()));
        Ok(answer == self.expected)
    }
}

struct BoomVerifier;

impl Verifier for BoomVerifier {
    fn verify(&mut self, _problem: &Problem, _answer: &str) -> prometheus_synth::Result<bool> {
        panic!("verifier must not be called");
    }
}

struct ErrVerifier;

impl Verifier for ErrVerifier {
    fn verify(&mut self, _problem: &Problem, _answer: &str) -> prometheus_synth::Result<bool> {
        Err(Error::Message("verifier failed".into()))
    }
}

struct FakeCounter {
    n: u32,
    seen: Rc<RefCell<Vec<String>>>,
}

impl TokenCounter for FakeCounter {
    fn token_count(&self, text: &str) -> u32 {
        self.seen.borrow_mut().push(text.to_string());
        self.n
    }
}

struct LenCounter;

impl TokenCounter for LenCounter {
    fn token_count(&self, text: &str) -> u32 {
        text.len() as u32
    }
}

// ---------------------------------------------------------------------------
// Already implemented
// ---------------------------------------------------------------------------

#[test]
fn defaults_and_domains() {
    assert_eq!(DEFAULT_N_SAMPLES, 8);
    assert_eq!(DEFAULT_MAX_TOKENS, 512);
    assert!((DEFAULT_TEMPERATURE - 0.7).abs() < 1e-6);
    assert_eq!(DEFAULT_MAX_BATCH, 8);
    assert_eq!(
        Domain::ALL,
        [
            Domain::Math,
            Domain::Code,
            Domain::Lean,
            Domain::Grid,
            Domain::Market
        ]
    );
    assert_eq!(Domain::Math.as_str(), "math");
    assert_eq!(Domain::Code.as_str(), "code");
    assert_eq!(Domain::Lean.as_str(), "lean");
    assert_eq!(Domain::Grid.as_str(), "grid");
    assert_eq!(Domain::Market.as_str(), "market");
    assert_eq!(Domain::parse("math").unwrap(), Domain::Math);
    assert!(Domain::parse("nope").is_none());
    let cfg = SamplerConfig::default();
    assert_eq!(cfg.n_samples, 8);
    assert_eq!(cfg.max_tokens, 512);
    assert!((cfg.temperature - 0.7).abs() < 1e-6);
    assert_eq!(cfg.max_batch, 8);
}

#[test]
fn generate_solve_prompt_template() {
    let p = problem("p1", "What is 2+2?", Domain::Math, "4");
    let prompt = generate_solve_prompt(&p).unwrap();
    assert_eq!(
        prompt,
        "Solve the following math problem. Put the final answer after the reasoning.\n\nProblem:\nWhat is 2+2?"
    );
}

#[test]
fn generate_solve_prompt_embeds_every_domain() {
    for domain in Domain::ALL {
        let p = problem("id", "Body.", domain, "x");
        let prompt = generate_solve_prompt(&p).unwrap();
        let head = format!(
            "Solve the following {} problem. Put the final answer after the reasoning.",
            domain.as_str()
        );
        assert!(prompt.starts_with(&head), "{prompt}");
        assert!(prompt.ends_with("Problem:\nBody."), "{prompt}");
    }
}

#[test]
fn generate_solve_prompt_empty_raises() {
    let err = generate_solve_prompt(&problem("w", "", Domain::Math, "x")).unwrap_err();
    assert!(matches!(err, Error::EmptyPrompt));
}

#[test]
fn generate_solve_prompt_preserves_utf8() {
    let p = problem("cafe", "Café mass 3.14?", Domain::Math, "x");
    let prompt = generate_solve_prompt(&p).unwrap();
    assert!(prompt.contains("Café mass 3.14?"));
}

#[test]
fn verified_correct_rate_empty_or_zero_is_zero() {
    assert_eq!(verified_correct_rate(&[]), 0.0);
    let empty = RejectionResult {
        problem_id: "p".into(),
        n_generated: 0,
        n_accepted: 0,
        samples: vec![],
    };
    assert_eq!(verified_correct_rate(std::slice::from_ref(&empty)), 0.0);
    assert_eq!(empty.verified_correct_rate(), 0.0);
}

#[test]
fn verified_correct_rate_sums_accepted_over_generated() {
    let a = RejectionResult {
        problem_id: "a".into(),
        n_generated: 8,
        n_accepted: 2,
        samples: vec![],
    };
    let b = RejectionResult {
        problem_id: "b".into(),
        n_generated: 8,
        n_accepted: 0,
        samples: vec![],
    };
    assert_eq!(a.verified_correct_rate(), GOLDEN_VERIFIED_CORRECT_RATE);
    assert_eq!(verified_correct_rate(&[a, b]), 0.125);
}

// ---------------------------------------------------------------------------
// extract_answer
// ---------------------------------------------------------------------------

#[test]
fn extract_answer_goldens() {
    for (text, expected) in EXTRACT_GOLDENS {
        let got = prometheus_synth::extract_answer(text);
        assert_eq!(got, *expected, "prod {text:?}");
        assert_eq!(
            prometheus_synth::extract_answer(text),
            reject_ref::extract_answer(text),
            "ref {text:?}"
        );
    }
}

#[test]
fn extract_answer_empty_and_whitespace() {
    assert_eq!(prometheus_synth::extract_answer(""), "");
    assert_eq!(prometheus_synth::extract_answer("   \n\t"), "");
}

#[test]
fn extract_answer_last_answer_wins() {
    assert_eq!(
        prometheus_synth::extract_answer("Answer: first\nAnswer: second"),
        "second"
    );
}

#[test]
fn extract_answer_case_insensitive() {
    assert_eq!(prometheus_synth::extract_answer("ANSWER: 42"), "42");
    assert_eq!(prometheus_synth::extract_answer("AnSwEr: 42"), "42");
}

#[test]
fn extract_answer_empty_capture_falls_through() {
    assert_eq!(prometheus_synth::extract_answer("Answer:\n42"), "42");
    assert_eq!(
        prometheus_synth::extract_answer("Answer: first\nAnswer:"),
        "first"
    );
    assert_eq!(
        prometheus_synth::extract_answer("Answer:\n```\ncode\n```"),
        "code\n"
    );
}

#[test]
fn extract_answer_last_fence_wins_language_tag_stripped() {
    assert_eq!(
        prometheus_synth::extract_answer("```\nfirst\n```\n```python\nsecond\n```"),
        "second\n"
    );
}

#[test]
fn extract_answer_unclosed_fence_falls_through() {
    assert_eq!(
        prometheus_synth::extract_answer("hello\n```\nunclosed"),
        "unclosed"
    );
    assert_eq!(prometheus_synth::extract_answer("```code```"), "```code```");
}

#[test]
fn extract_answer_last_nonempty_line() {
    assert_eq!(
        prometheus_synth::extract_answer("line1\nline2\nline3"),
        "line3"
    );
    assert_eq!(prometheus_synth::extract_answer("foo\n  42"), "  42");
}

#[test]
fn extract_answer_trailing_whitespace_stripped_before_parse() {
    assert_eq!(prometheus_synth::extract_answer("Answer: 42   \n\n"), "42");
    assert_eq!(prometheus_synth::extract_answer("foo\n\nbar\n\n"), "bar");
}

// ---------------------------------------------------------------------------
// sample
// ---------------------------------------------------------------------------

#[test]
fn sample_zero_n_errors_before_generate() {
    let mut sampler =
        RejectionSampler::new(BoomGenerator, BoomVerifier, SamplerConfig::default()).unwrap();
    let err = sampler.sample(&math_problem(), 0).unwrap_err();
    assert!(matches!(err, Error::ZeroSamples));
}

#[test]
fn sample_empty_prompt_errors_before_generate() {
    let mut sampler =
        RejectionSampler::new(BoomGenerator, BoomVerifier, SamplerConfig::default()).unwrap();
    let err = sampler
        .sample(&problem("p", "", Domain::Math, "x"), 3)
        .unwrap_err();
    assert!(matches!(err, Error::EmptyPrompt));
}

#[test]
fn sample_zero_n_wins_over_empty_prompt() {
    let mut sampler =
        RejectionSampler::new(BoomGenerator, BoomVerifier, SamplerConfig::default()).unwrap();
    let err = sampler
        .sample(&problem("p", "", Domain::Math, "x"), 0)
        .unwrap_err();
    assert!(matches!(err, Error::ZeroSamples));
}

#[test]
fn sample_length_mismatch() {
    let mut sampler = RejectionSampler::new(
        MismatchGenerator { n: 0 },
        MatchVerifier,
        cfg_with(8, 512, 0.7, 8),
    )
    .unwrap();
    let err = sampler.sample(&math_problem(), 2).unwrap_err();
    assert!(matches!(err, Error::LengthMismatch { want: 2, got: 0 }));
}

#[test]
fn sample_matches_reference_golden_rate() {
    let cfg = cfg_with(8, 64, 0.5, 3);
    let problem = math_problem();
    let mut prod = RejectionSampler::new(
        QueueGenerator::new(GOLDEN_COMPLETIONS),
        MatchVerifier,
        cfg.clone(),
    )
    .unwrap();
    let got = prod.sample(&problem, GOLDEN_N_GENERATED).unwrap();
    let expected = reject_ref::sample(
        &mut QueueGenerator::new(GOLDEN_COMPLETIONS),
        &mut MatchVerifier,
        None,
        &cfg,
        &problem,
        GOLDEN_N_GENERATED,
    )
    .unwrap();
    assert_eq!(got, expected);
    assert_eq!(got.n_generated, GOLDEN_N_GENERATED);
    assert_eq!(got.n_accepted, GOLDEN_N_ACCEPTED);
    assert_eq!(got.verified_correct_rate(), GOLDEN_VERIFIED_CORRECT_RATE);
    assert_eq!(
        verified_correct_rate(std::slice::from_ref(&got)),
        GOLDEN_VERIFIED_CORRECT_RATE
    );
    let passed: Vec<bool> = got.samples.iter().map(|s| s.passed).collect();
    assert_eq!(
        passed,
        vec![false, true, false, true, false, false, false, false]
    );
    let answers: Vec<&str> = got.samples.iter().map(|s| s.answer.as_str()).collect();
    assert_eq!(
        answers,
        vec![
            "41",
            "42",
            "print(0)\n",
            "42",
            "nope",
            "40",
            "the last line is 99",
            "7",
        ]
    );
    for (s, c) in got.samples.iter().zip(GOLDEN_COMPLETIONS) {
        assert_eq!(s.problem_id, "p1");
        assert_eq!(s.text, *c);
    }
}

#[test]
fn sample_chunks_by_max_batch() {
    let cfg = cfg_with(8, 11, 0.5, 2);
    let completions = [
        "Answer: a",
        "Answer: b",
        "Answer: c",
        "Answer: d",
        "Answer: e",
    ];
    let log = CallLog::default();
    let problem = math_problem();
    let prompt = generate_solve_prompt(&problem).unwrap();
    let mut sampler = RejectionSampler::new(
        QueueGenerator::with_log(&completions, log.clone()),
        MatchVerifier,
        cfg,
    )
    .unwrap();
    let _ = sampler.sample(&problem, 5).unwrap();
    let calls = log.calls.borrow().clone();
    let sizes: Vec<usize> = calls.iter().map(|(p, _, _)| p.len()).collect();
    assert_eq!(sizes, vec![2, 2, 1]);
    for (prompts, max_tokens, temperature) in &calls {
        assert!(prompts.iter().all(|p| p == &prompt));
        assert_eq!(*max_tokens, 11);
        assert_eq!(*temperature, 0.5);
    }
}

#[test]
fn sample_keeps_rejected_traces() {
    let cfg = cfg_with(8, 512, 0.7, 8);
    let completions = ["Answer: no", "Answer: 42"];
    let mut sampler =
        RejectionSampler::new(QueueGenerator::new(&completions), MatchVerifier, cfg).unwrap();
    let got = sampler.sample(&math_problem(), 2).unwrap();
    assert_eq!(got.n_generated, 2);
    assert_eq!(got.n_accepted, 1);
    assert_eq!(got.samples.len(), 2);
    assert!(!got.samples[0].passed);
    assert!(got.samples[1].passed);
}

#[test]
fn sample_whitespace_token_count_on_raw_text() {
    let cfg = cfg_with(8, 512, 0.7, 8);
    let text = "lots of words here\nAnswer: 42";
    let problem = math_problem();
    let mut sampler =
        RejectionSampler::new(QueueGenerator::new(&[text]), MatchVerifier, cfg.clone()).unwrap();
    let got = sampler.sample(&problem, 1).unwrap();
    let expected = reject_ref::sample(
        &mut QueueGenerator::new(&[text]),
        &mut MatchVerifier,
        None,
        &cfg,
        &problem,
        1,
    )
    .unwrap();
    assert_eq!(got, expected);
    assert_eq!(got.samples[0].token_count, 6);
    assert_eq!(got.samples[0].answer, "42");
}

#[test]
fn sample_tokenizer_encode_length() {
    let cfg = cfg_with(8, 512, 0.7, 8);
    let text = "Answer: 42";
    let seen = Rc::new(RefCell::new(Vec::new()));
    let counter = FakeCounter {
        n: 9,
        seen: seen.clone(),
    };
    let mut sampler = RejectionSampler::new(QueueGenerator::new(&[text]), MatchVerifier, cfg)
        .unwrap()
        .with_tokenizer(counter);
    let got = sampler.sample(&math_problem(), 1).unwrap();
    assert_eq!(got.samples[0].token_count, 9);
    assert_eq!(*seen.borrow(), vec![text.to_string()]);
}

#[test]
fn sample_empty_completion_skips_tokenizer() {
    let cfg = cfg_with(8, 512, 0.7, 8);
    let seen = Rc::new(RefCell::new(Vec::new()));
    let counter = FakeCounter {
        n: 3,
        seen: seen.clone(),
    };
    let mut sampler = RejectionSampler::new(
        QueueGenerator::new(&["", "Answer: 42"]),
        MatchVerifier,
        cfg.clone(),
    )
    .unwrap()
    .with_tokenizer(counter);
    let got = sampler.sample(&math_problem(), 2).unwrap();
    let expected = reject_ref::sample(
        &mut QueueGenerator::new(&["", "Answer: 42"]),
        &mut MatchVerifier,
        Some(&LenCounter),
        &cfg,
        &math_problem(),
        2,
    )
    .unwrap();
    assert_eq!(got.samples[0].token_count, 0);
    assert_eq!(got.samples[1].token_count, 3);
    assert_eq!(*seen.borrow(), vec!["Answer: 42".to_string()]);
    assert_eq!(got.samples[0].answer, expected.samples[0].answer);
    assert_eq!(got.samples[1].answer, expected.samples[1].answer);
}

#[test]
fn sample_verifier_sees_extracted_answer_not_raw_text() {
    let cfg = cfg_with(8, 512, 0.7, 8);
    let seen = Rc::new(RefCell::new(Vec::new()));
    let v = ExactVerifier {
        expected: "42".into(),
        seen: seen.clone(),
    };
    let mut sampler =
        RejectionSampler::new(QueueGenerator::new(&["reasoning\nAnswer: 42"]), v, cfg).unwrap();
    let _ = sampler.sample(&math_problem(), 1).unwrap();
    assert_eq!(*seen.borrow(), vec![("p1".into(), "42".into())]);
}

#[test]
fn sample_verifier_err_becomes_message() {
    let mut sampler = RejectionSampler::new(
        QueueGenerator::new(&["Answer: 42"]),
        ErrVerifier,
        cfg_with(8, 512, 0.7, 8),
    )
    .unwrap();
    let err = sampler.sample(&math_problem(), 1).unwrap_err();
    match err {
        Error::Message(m) => assert_eq!(m, "verifier failed"),
        other => panic!("expected Message, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// sample_many
// ---------------------------------------------------------------------------

#[test]
fn sample_many_empty_batch_errors_before_generate() {
    let mut sampler =
        RejectionSampler::new(BoomGenerator, BoomVerifier, SamplerConfig::default()).unwrap();
    let err = sampler.sample_many(&[], 3).unwrap_err();
    assert!(matches!(err, Error::EmptyBatch));
}

#[test]
fn sample_many_empty_batch_wins_over_zero_n() {
    let mut sampler =
        RejectionSampler::new(BoomGenerator, BoomVerifier, SamplerConfig::default()).unwrap();
    let err = sampler.sample_many(&[], 0).unwrap_err();
    assert!(matches!(err, Error::EmptyBatch));
}

#[test]
fn sample_many_zero_n_errors_before_generate() {
    let mut sampler =
        RejectionSampler::new(BoomGenerator, BoomVerifier, SamplerConfig::default()).unwrap();
    let err = sampler.sample_many(&[math_problem()], 0).unwrap_err();
    assert!(matches!(err, Error::ZeroSamples));
}

#[test]
fn sample_many_empty_prompt_errors_before_generate() {
    let mut sampler =
        RejectionSampler::new(BoomGenerator, BoomVerifier, SamplerConfig::default()).unwrap();
    let problems = [math_problem(), problem("p2", "", Domain::Math, "x")];
    let err = sampler.sample_many(&problems, 2).unwrap_err();
    assert!(matches!(err, Error::EmptyPrompt));
}

#[test]
fn sample_many_length_mismatch() {
    let mut sampler = RejectionSampler::new(
        MismatchGenerator { n: 1 },
        MatchVerifier,
        cfg_with(8, 512, 0.7, 8),
    )
    .unwrap();
    let problems = [math_problem(), problem("p2", "two", Domain::Code, "2")];
    let err = sampler.sample_many(&problems, 2).unwrap_err();
    assert!(matches!(err, Error::LengthMismatch { want: 4, got: 1 }));
}

#[test]
fn sample_many_matches_reference_and_preserves_order() {
    let problems = [
        problem("a", "one", Domain::Math, "1"),
        problem("b", "two", Domain::Code, "2"),
    ];
    let completions = ["Answer: 1", "Answer: x", "Answer: 2", "Answer: 2"];
    let cfg = cfg_with(8, 20, 0.0, 3);
    let mut sampler = RejectionSampler::new(
        QueueGenerator::new(&completions),
        MatchVerifier,
        cfg.clone(),
    )
    .unwrap();
    let got = sampler.sample_many(&problems, 2).unwrap();
    let expected = reject_ref::sample_many(
        &mut QueueGenerator::new(&completions),
        &mut MatchVerifier,
        None,
        &cfg,
        &problems,
        2,
    )
    .unwrap();
    assert_eq!(got, expected);
    assert_eq!(got[0].problem_id, "a");
    assert_eq!(got[1].problem_id, "b");
    assert_eq!(got[0].n_generated, 2);
    assert_eq!(got[0].n_accepted, 1);
    assert_eq!(got[1].n_generated, 2);
    assert_eq!(got[1].n_accepted, 2);
    assert_eq!(got[0].samples[0].text, "Answer: 1");
    assert_eq!(got[0].samples[1].text, "Answer: x");
    assert_eq!(got[1].samples[0].text, "Answer: 2");
    assert_eq!(got[1].samples[1].text, "Answer: 2");
}

#[test]
fn sample_many_chunks_across_problems() {
    let problems = [
        problem("a", "one", Domain::Math, "42"),
        problem("b", "two", Domain::Math, "42"),
    ];
    let completions = ["A", "B", "C", "D"];
    let log = CallLog::default();
    let cfg = cfg_with(8, 5, 0.0, 3);
    let mut sampler = RejectionSampler::new(
        QueueGenerator::with_log(&completions, log.clone()),
        MatchVerifier,
        cfg,
    )
    .unwrap();
    let _ = sampler.sample_many(&problems, 2).unwrap();
    let p1 = generate_solve_prompt(&problems[0]).unwrap();
    let p2 = generate_solve_prompt(&problems[1]).unwrap();
    let calls = log.calls.borrow().clone();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].0, vec![p1.clone(), p1, p2.clone()]);
    assert_eq!(calls[1].0, vec![p2]);
    assert_eq!(calls[0].1, 5);
    assert_eq!(calls[0].2, 0.0);
}

#[test]
fn sample_many_golden_aggregate_rate() {
    let problems = [
        math_problem(),
        problem("b", "other", Domain::Math, GOLDEN_EXPECTED_ANSWER),
    ];
    let cfg = cfg_with(8, 512, 0.7, 5);
    let mut sampler = RejectionSampler::new(
        QueueGenerator::new(GOLDEN_COMPLETIONS),
        MatchVerifier,
        cfg.clone(),
    )
    .unwrap();
    let got = sampler.sample_many(&problems, 4).unwrap();
    let expected = reject_ref::sample_many(
        &mut QueueGenerator::new(GOLDEN_COMPLETIONS),
        &mut MatchVerifier,
        None,
        &cfg,
        &problems,
        4,
    )
    .unwrap();
    assert_eq!(got, expected);
    let n_accepted: u32 = got.iter().map(|r| r.n_accepted).sum();
    let n_generated: u32 = got.iter().map(|r| r.n_generated).sum();
    assert_eq!(n_accepted, GOLDEN_N_ACCEPTED);
    assert_eq!(n_generated, GOLDEN_N_GENERATED);
    assert_eq!(verified_correct_rate(&got), GOLDEN_VERIFIED_CORRECT_RATE);
}

// ---------------------------------------------------------------------------
// Property grid
// ---------------------------------------------------------------------------

#[test]
fn sample_property_grid_matches_reference() {
    let problem = problem("grid", "sum?", Domain::Math, "yes");
    let pool = [
        "Answer: yes",
        "Answer: no",
        "```\nyes\n```",
        "last line",
        "Answer:",
        "yes",
        "```python\nno\n```",
        "Answer: yes",
        "thinking\nAnswer: no",
        "```\n```",
    ];
    for n in [1u32, 2, 5] {
        for max_batch in [1u32, 2, 3, 8] {
            let completions = &pool[..n as usize];
            let cfg = cfg_with(n, 32, 0.5, max_batch);
            let mut sampler =
                RejectionSampler::new(QueueGenerator::new(completions), MatchVerifier, cfg.clone())
                    .unwrap();
            let got = sampler.sample(&problem, n).unwrap();
            let expected = reject_ref::sample(
                &mut QueueGenerator::new(completions),
                &mut MatchVerifier,
                None,
                &cfg,
                &problem,
                n,
            )
            .unwrap();
            assert_eq!(got, expected, "n={n} max_batch={max_batch}");
        }
    }
}

#[test]
fn sample_many_property_grid_matches_reference() {
    let problems = [
        problem("m1", "p1", Domain::Math, "A"),
        problem("m2", "p2", Domain::Code, "B"),
        problem("m3", "p3", Domain::Lean, "C"),
    ];
    let pool = [
        "Answer: A",
        "Answer: B",
        "Answer: C",
        "Answer: A",
        "nope",
        "```\nB\n```",
        "Answer: C",
        "last",
        "Answer: A",
        "Answer: B",
        "```python\nC\n```",
        "Answer: Z",
        "Answer: A",
        "Answer: B",
        "Answer: C",
    ];
    for n in [1u32, 2, 5] {
        for max_batch in [1u32, 2, 4, 8] {
            let completions = &pool[..(n as usize) * problems.len()];
            let cfg = cfg_with(8, 16, 1.0, max_batch);
            let mut sampler =
                RejectionSampler::new(QueueGenerator::new(completions), MatchVerifier, cfg.clone())
                    .unwrap();
            let got = sampler.sample_many(&problems, n).unwrap();
            let expected = reject_ref::sample_many(
                &mut QueueGenerator::new(completions),
                &mut MatchVerifier,
                None,
                &cfg,
                &problems,
                n,
            )
            .unwrap();
            assert_eq!(got, expected, "n={n} max_batch={max_batch}");
            assert_eq!(got[0].problem_id, "m1");
            assert_eq!(got[1].problem_id, "m2");
            assert_eq!(got[2].problem_id, "m3");
            assert!(got.iter().all(|r| r.n_generated == n));
        }
    }
}

#[test]
fn sample_result_types() {
    let mut sampler = RejectionSampler::new(
        QueueGenerator::new(&["Answer: 42"]),
        MatchVerifier,
        cfg_with(8, 512, 0.7, 1),
    )
    .unwrap();
    let got = sampler.sample(&math_problem(), 1).unwrap();
    let _: &RejectionResult = &got;
    let _: &Sample = &got.samples[0];
    assert!(got.samples[0].passed);
}
