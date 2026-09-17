//! Slow, obvious E2 rejection-sampling reference.
//!
//! Must match `tests/reference/reject.py` on the same inputs. Production
//! `prometheus-synth` must match this module; production must never import
//! `tests/`.
#![allow(dead_code)]

use prometheus_synth::{
    generate_solve_prompt, Error, Generator, Problem, RejectionResult, Result, Sample,
    SamplerConfig, TokenCounter, Verifier,
};

/// Encode length if a tokenizer is given, else whitespace words.
/// Empty string is 0 without calling the tokenizer. Counted on the raw
/// completion, not the extracted answer.
pub fn token_count(text: &str, tokenizer: Option<&dyn TokenCounter>) -> u32 {
    if text.is_empty() {
        return 0;
    }
    if let Some(counter) = tokenizer {
        return counter.token_count(text);
    }
    text.split_whitespace().count() as u32
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

/// Extract the answer the D2 verifier should see. See `tests/reference/reject.py`.
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

fn generate_chunked<G: Generator>(
    generator: &mut G,
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

fn one_sample<V: Verifier>(
    verifier: &mut V,
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

pub fn sample<G: Generator, V: Verifier>(
    generator: &mut G,
    verifier: &mut V,
    tokenizer: Option<&dyn TokenCounter>,
    config: &SamplerConfig,
    problem: &Problem,
    n: u32,
) -> Result<RejectionResult> {
    if n == 0 {
        return Err(Error::ZeroSamples);
    }
    let prompt = generate_solve_prompt(problem)?;
    let prompts = vec![prompt; n as usize];
    let completions = generate_chunked(
        generator,
        &prompts,
        config.max_tokens,
        config.temperature,
        config.max_batch as usize,
    )?;
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

pub fn sample_many<G: Generator, V: Verifier>(
    generator: &mut G,
    verifier: &mut V,
    tokenizer: Option<&dyn TokenCounter>,
    config: &SamplerConfig,
    problems: &[Problem],
    n: u32,
) -> Result<Vec<RejectionResult>> {
    if problems.is_empty() {
        return Err(Error::EmptyBatch);
    }
    if n == 0 {
        return Err(Error::ZeroSamples);
    }
    let mut prompts = Vec::with_capacity(problems.len() * n as usize);
    for problem in problems {
        let prompt = generate_solve_prompt(problem)?;
        for _ in 0..n {
            prompts.push(prompt.clone());
        }
    }
    let completions = generate_chunked(
        generator,
        &prompts,
        config.max_tokens,
        config.temperature,
        config.max_batch as usize,
    )?;
    let mut results = Vec::with_capacity(problems.len());
    let mut idx = 0;
    let n_us = n as usize;
    for problem in problems {
        let texts = &completions[idx..idx + n_us];
        idx += n_us;
        let mut samples = Vec::with_capacity(n_us);
        for text in texts {
            samples.push(one_sample(verifier, tokenizer, problem, text)?);
        }
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
