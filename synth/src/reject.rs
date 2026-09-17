//! Rejection sampling of reasoning traces against D2 verifiers (spec 7.1, 15.5 E2).
//!
//! A problem is sent to F5 `n` times. Each completion is parsed into an answer
//! and scored by a D2 verifier. Keep the traces whose `RewardResponse.passed`
//! is true. The gate is verified-correct rate: accepted / generated,
//! fail-closed at 0.0 when nothing was generated.
//!
//! The verifier is a trait so CPU tests can inject a fake. Production wraps
//! `prometheus-verifiers` `Registry::verify`.

use serde::{Deserialize, Serialize};

use crate::{
    token_count, Error, Generator, Result, TokenCounter, DEFAULT_MAX_BATCH, DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
};

/// Default group size. Matches the GRPO group size in spec 9.2.
pub const DEFAULT_N_SAMPLES: u32 = 8;

/// D2 domain. Serde/JSON names are the values. Maps onto `VerifierKind`
/// (math → SymbolicMath, code → SandboxedTests, lean → LeanKernel,
/// grid → GridMatch, market → MarketResolution).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Domain {
    Math,
    Code,
    Lean,
    Grid,
    Market,
}

impl Domain {
    pub const ALL: [Domain; 5] = [
        Domain::Math,
        Domain::Code,
        Domain::Lean,
        Domain::Grid,
        Domain::Market,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Domain::Math => "math",
            Domain::Code => "code",
            Domain::Lean => "lean",
            Domain::Grid => "grid",
            Domain::Market => "market",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "math" => Some(Domain::Math),
            "code" => Some(Domain::Code),
            "lean" => Some(Domain::Lean),
            "grid" => Some(Domain::Grid),
            "market" => Some(Domain::Market),
            _ => None,
        }
    }
}

/// One problem to sample. `prompt` is shown to F5; `verifier_id` selects the
/// D2 verifier. Empty `prompt` is [`Error::EmptyPrompt`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Problem {
    pub problem_id: String,
    pub domain: Domain,
    pub prompt: String,
    pub verifier_id: String,
}

/// One completion. `text` is the raw generator output, `answer` is
/// [`extract_answer`] of that text, `passed` is the D2 result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sample {
    pub problem_id: String,
    pub text: String,
    pub answer: String,
    pub passed: bool,
    pub token_count: u32,
}

/// All `n` samples for one problem, accepted and rejected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectionResult {
    pub problem_id: String,
    pub n_generated: u32,
    pub n_accepted: u32,
    pub samples: Vec<Sample>,
}

impl RejectionResult {
    /// `n_accepted / n_generated`. Zero generated is 0.0.
    pub fn verified_correct_rate(&self) -> f64 {
        if self.n_generated == 0 {
            0.0
        } else {
            self.n_accepted as f64 / self.n_generated as f64
        }
    }
}

/// Batch and decoding knobs. `n_samples` is the default group size.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerConfig {
    pub n_samples: u32,
    pub max_tokens: u32,
    pub temperature: f32,
    pub max_batch: u32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            n_samples: DEFAULT_N_SAMPLES,
            max_tokens: DEFAULT_MAX_TOKENS,
            temperature: DEFAULT_TEMPERATURE,
            max_batch: DEFAULT_MAX_BATCH,
        }
    }
}

/// D2 check. True iff `RewardResponse.passed` is true.
pub trait Verifier {
    fn verify(&mut self, problem: &Problem, answer: &str) -> Result<bool>;
}

/// Prompt F5 to solve `problem`.
///
/// Empty `problem.prompt` is [`Error::EmptyPrompt`]. The template is the
/// contract between Python and Rust so both languages send the same bytes
/// to F5.
pub fn generate_solve_prompt(problem: &Problem) -> Result<String> {
    if problem.prompt.is_empty() {
        return Err(Error::EmptyPrompt);
    }
    Ok(format!(
        "Solve the following {} problem. Put the final answer after the reasoning.\n\nProblem:\n{}",
        problem.domain.as_str(),
        problem.prompt
    ))
}

/// Last complete ``` fence body, or None if none / only unclosed.
///
/// Opening fence is three backticks at the start of the text or just after a
/// newline. Rest of that line is the language tag (discarded). One leading
/// newline after the tag line is stripped. Body runs until the next ```.
/// No closer means the opening is not a fence.
fn last_fence_body(text: &str) -> Option<String> {
    let mut last = None;
    let mut i = 0;
    while i < text.len() {
        let Some(rel) = text[i..].find("```") else {
            break;
        };
        let idx = i + rel;
        if idx != 0 && text.as_bytes()[idx - 1] != b'\n' {
            i = idx + 3;
            continue;
        }
        let after = idx + 3;
        let Some(nl_rel) = text[after..].find('\n') else {
            break;
        };
        let nl = after + nl_rel;
        let Some(close_rel) = text[nl + 1..].find("```") else {
            break;
        };
        let close = nl + 1 + close_rel;
        last = Some(text[nl + 1..close].to_string());
        i = close + 3;
    }
    last
}

/// Capture of `(?i)^answer\s*:\s*(.*)$` if the capture is non-empty, trimmed.
fn answer_capture(line: &str) -> Option<String> {
    let mut chars = line.chars().peekable();
    for expect in ['a', 'n', 's', 'w', 'e', 'r'] {
        match chars.next() {
            Some(c) if c.eq_ignore_ascii_case(&expect) => {}
            _ => return None,
        }
    }
    while chars.peek().copied().is_some_and(char::is_whitespace) {
        chars.next();
    }
    match chars.next() {
        Some(':') => {}
        _ => return None,
    }
    while chars.peek().copied().is_some_and(char::is_whitespace) {
        chars.next();
    }
    let cap: String = chars.collect();
    if cap.is_empty() {
        return None;
    }
    Some(cap.trim().to_string())
}

/// Extract the answer the D2 verifier should see.
///
/// 1. Strip trailing whitespace. Empty is empty.
/// 2. If a line matches `(?i)^answer\s*:\s*(.*)$`, the capture of the last
///    such line with a non-empty capture is the answer (trimmed).
/// 3. Else if the text contains a fenced block starting with three backticks,
///    the body of the last fence is the answer (strip one leading newline
///    after the optional language tag line).
/// 4. Else the last non-empty line.
///
/// This is the only parse step. The verifier does not re-parse the reasoning.
pub fn extract_answer(text: &str) -> String {
    let text = text.trim_end();
    if text.is_empty() {
        return String::new();
    }

    let mut last_ans: Option<String> = None;
    for line in text.lines() {
        if let Some(ans) = answer_capture(line) {
            last_ans = Some(ans);
        }
    }
    if let Some(ans) = last_ans {
        return ans;
    }

    if let Some(body) = last_fence_body(text) {
        return body;
    }

    for line in text.lines().rev() {
        if !line.trim().is_empty() {
            return line.to_string();
        }
    }
    String::new()
}

/// Gate: `sum n_accepted / sum n_generated` over `results`. Empty or zero
/// generated is 0.0.
pub fn verified_correct_rate(results: &[RejectionResult]) -> f64 {
    let generated: u64 = results.iter().map(|r| r.n_generated as u64).sum();
    let accepted: u64 = results.iter().map(|r| r.n_accepted as u64).sum();
    if generated == 0 {
        0.0
    } else {
        accepted as f64 / generated as f64
    }
}

fn generate_chunked(
    generator: &mut dyn Generator,
    prompts: &[String],
    max_tokens: u32,
    temperature: f32,
    max_batch: usize,
) -> Result<Vec<String>> {
    let mut completions = Vec::with_capacity(prompts.len());
    for chunk in prompts.chunks(max_batch) {
        let outs = generator.generate(chunk, max_tokens, temperature)?;
        if outs.len() != chunk.len() {
            return Err(Error::LengthMismatch {
                want: chunk.len(),
                got: outs.len(),
            });
        }
        completions.extend(outs);
    }
    Ok(completions)
}

fn one_sample(
    verifier: &mut dyn Verifier,
    tokenizer: Option<&dyn TokenCounter>,
    problem: &Problem,
    text: &str,
) -> Result<Sample> {
    let answer = extract_answer(text);
    let passed = verifier
        .verify(problem, &answer)
        .map_err(|e| Error::Message(e.to_string()))?;
    Ok(Sample {
        problem_id: problem.problem_id.clone(),
        text: text.to_string(),
        answer,
        passed,
        token_count: token_count(text, tokenizer),
    })
}

/// Batch rejection sampling. `generator` is F5. `verifier` is D2.
/// `tokenizer` is F6 when token counts should match the frozen vocab; omit
/// it to count whitespace words.
pub struct RejectionSampler {
    generator: Box<dyn Generator>,
    verifier: Box<dyn Verifier>,
    tokenizer: Option<Box<dyn TokenCounter>>,
    config: SamplerConfig,
}

impl RejectionSampler {
    pub fn new<G, V>(generator: G, verifier: V, config: SamplerConfig) -> Result<Self>
    where
        G: Generator + 'static,
        V: Verifier + 'static,
    {
        Ok(Self {
            generator: Box::new(generator),
            verifier: Box::new(verifier),
            tokenizer: None,
            config,
        })
    }

    /// Use F6 encode length for [`Sample::token_count`].
    pub fn with_tokenizer<T>(self, tokenizer: T) -> Self
    where
        T: TokenCounter + 'static,
    {
        let mut this = self;
        this.tokenizer = Some(Box::new(tokenizer));
        this
    }

    /// `n` completions for one problem. `n == 0` is [`Error::ZeroSamples`].
    /// Empty prompt is [`Error::EmptyPrompt`].
    pub fn sample(&mut self, problem: &Problem, n: u32) -> Result<RejectionResult> {
        if n == 0 {
            return Err(Error::ZeroSamples);
        }
        let prompt = generate_solve_prompt(problem)?;
        let prompts = vec![prompt; n as usize];
        let completions = generate_chunked(
            self.generator.as_mut(),
            &prompts,
            self.config.max_tokens,
            self.config.temperature,
            self.config.max_batch as usize,
        )?;
        let tokenizer = self.tokenizer.as_deref();
        let verifier = self.verifier.as_mut();
        let mut samples = Vec::with_capacity(completions.len());
        for text in &completions {
            samples.push(one_sample(verifier, tokenizer, problem, text)?);
        }
        let n_accepted = samples.iter().filter(|s| s.passed).count() as u32;
        Ok(RejectionResult {
            problem_id: problem.problem_id.clone(),
            n_generated: n,
            n_accepted,
            samples,
        })
    }

    /// `n` completions per problem, chunked by `max_batch`. Empty `problems`
    /// is [`Error::EmptyBatch`]. Result order matches `problems`.
    pub fn sample_many(&mut self, problems: &[Problem], n: u32) -> Result<Vec<RejectionResult>> {
        if problems.is_empty() {
            return Err(Error::EmptyBatch);
        }
        if n == 0 {
            return Err(Error::ZeroSamples);
        }
        let mut prompts = Vec::with_capacity(problems.len() * n as usize);
        for problem in problems {
            let prompt = generate_solve_prompt(problem)?;
            prompts.extend(std::iter::repeat_n(prompt, n as usize));
        }
        let completions = generate_chunked(
            self.generator.as_mut(),
            &prompts,
            self.config.max_tokens,
            self.config.temperature,
            self.config.max_batch as usize,
        )?;
        let tokenizer = self.tokenizer.as_deref();
        let verifier = self.verifier.as_mut();
        let mut results = Vec::with_capacity(problems.len());
        let mut idx = 0;
        let n_us = n as usize;
        for problem in problems {
            let mut samples = Vec::with_capacity(n_us);
            for text in &completions[idx..idx + n_us] {
                samples.push(one_sample(verifier, tokenizer, problem, text)?);
            }
            idx += n_us;
            let n_accepted = samples.iter().filter(|s| s.passed).count() as u32;
            results.push(RejectionResult {
                problem_id: problem.problem_id.clone(),
                n_generated: n,
                n_accepted,
                samples,
            });
        }
        Ok(results)
    }
}
