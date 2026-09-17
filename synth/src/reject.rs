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

use crate::{Error, Generator, Result, TokenCounter, DEFAULT_MAX_BATCH, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE};

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
    let _ = text;
    unimplemented!("E2 extract_answer")
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

/// Batch rejection sampling. `generator` is F5. `verifier` is D2.
/// `tokenizer` is F6 when token counts should match the frozen vocab; omit
/// it to count whitespace words.
#[allow(dead_code)]
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
        let _ = (problem, n);
        unimplemented!("E2 RejectionSampler::sample")
    }

    /// `n` completions per problem, chunked by `max_batch`. Empty `problems`
    /// is [`Error::EmptyBatch`]. Result order matches `problems`.
    pub fn sample_many(
        &mut self,
        problems: &[Problem],
        n: u32,
    ) -> Result<Vec<RejectionResult>> {
        let _ = (problems, n);
        unimplemented!("E2 RejectionSampler::sample_many")
    }
}
