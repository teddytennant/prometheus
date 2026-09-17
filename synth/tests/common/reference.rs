//! Slow, obvious E1 synth reference.
//!
//! Must match `tests/reference/synth.py` on the same inputs. Production
//! `prometheus-synth` must match this module; production must never import
//! `tests/`.
#![allow(dead_code)]

use std::collections::HashSet;

use prometheus_synth::{
    generate_prompt, Claim, Error, FactCheck, Generator, LabeledExample, OrchestratorConfig,
    Result, Rewrite, SourceDocument, Style, TokenCounter, Verdict,
};

const STOPLIST: &[&str] = &[
    "that", "this", "with", "from", "they", "them", "have", "been", "were", "will", "would",
    "could", "should", "into", "over", "under", "than", "then", "when", "what", "which", "their",
];

fn is_stop(tok: &str) -> bool {
    STOPLIST.contains(&tok)
}

fn normalize(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn content_words(tokens: &[String]) -> Vec<String> {
    tokens
        .iter()
        .filter(|t| t.chars().count() >= 4 && !is_stop(t))
        .cloned()
        .collect()
}

/// ASCII numbers matching `[0-9]+(?:\.[0-9]+)?` (the `\d+(?:\.\d+)?` rule on ASCII).
fn extract_numbers(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            if i < chars.len()
                && chars[i] == '.'
                && i + 1 < chars.len()
                && chars[i + 1].is_ascii_digit()
            {
                i += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            out.push(chars[start..i].iter().collect());
        } else {
            i += 1;
        }
    }
    out
}

/// Split on `(?<=[.!?])\s+`. Empty / whitespace-only text yields `[]`.
pub fn extract_claims(text: &str) -> Vec<String> {
    if text.split_whitespace().next().is_none() {
        return Vec::new();
    }
    let chars: Vec<char> = text.chars().collect();
    let mut pieces = Vec::new();
    let mut current = String::new();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let mut j = i + 1;
            if j < chars.len() && chars[j].is_whitespace() {
                while j < chars.len() && chars[j].is_whitespace() {
                    j += 1;
                }
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    pieces.push(trimmed.to_string());
                }
                current.clear();
                i = j;
                continue;
            }
        }
        i += 1;
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        pieces.push(trimmed.to_string());
    }
    pieces
}

pub fn check_claim(source: &str, claim: &str) -> Result<Verdict> {
    if source.is_empty() {
        return Err(Error::EmptySource);
    }
    if claim.split_whitespace().next().is_none() {
        return Ok(Verdict::NotInSource);
    }

    let src_nums = extract_numbers(source);
    if !src_nums.is_empty() {
        let src_set: HashSet<&str> = src_nums.iter().map(String::as_str).collect();
        if extract_numbers(claim)
            .iter()
            .any(|n| !src_set.contains(n.as_str()))
        {
            return Ok(Verdict::Contradicted);
        }
    }

    let src_norm = normalize(source);
    let claim_norm = normalize(claim);
    let src_tokens: Vec<String> = src_norm.split_whitespace().map(str::to_string).collect();
    let claim_tokens: Vec<String> = claim_norm.split_whitespace().map(str::to_string).collect();
    let claim_content = content_words(&claim_tokens);
    let content_set: HashSet<&str> = claim_content.iter().map(String::as_str).collect();

    for window in src_tokens.windows(2) {
        if (window[0] == "not" || window[0] == "no") && content_set.contains(window[1].as_str()) {
            return Ok(Verdict::Contradicted);
        }
    }

    if !claim_content.is_empty() {
        let src_token_set: HashSet<&str> = src_tokens.iter().map(String::as_str).collect();
        if claim_content
            .iter()
            .all(|w| src_token_set.contains(w.as_str()))
        {
            return Ok(Verdict::Supported);
        }
        return Ok(Verdict::NotInSource);
    }
    if src_norm.contains(&claim_norm) {
        return Ok(Verdict::Supported);
    }
    Ok(Verdict::NotInSource)
}

pub fn fact_check(source: &str, rewrite: &str) -> Result<FactCheck> {
    if source.is_empty() {
        return Err(Error::EmptySource);
    }
    if rewrite.is_empty() {
        return Ok(FactCheck { claims: Vec::new() });
    }
    let mut claims = Vec::new();
    for text in extract_claims(rewrite) {
        let verdict = check_claim(source, &text)?;
        claims.push(Claim { text, verdict });
    }
    Ok(FactCheck { claims })
}

pub fn accept(check: &FactCheck) -> bool {
    check.n_contradicted() == 0
}

pub fn token_count(text: &str, tokenizer: Option<&dyn TokenCounter>) -> u32 {
    if text.is_empty() {
        return 0;
    }
    if let Some(counter) = tokenizer {
        return counter.token_count(text);
    }
    text.split_whitespace().count() as u32
}

pub fn precision(examples: &[LabeledExample]) -> Result<f64> {
    let mut tp = 0u64;
    let mut fp = 0u64;
    for ex in examples {
        let pred = check_claim(&ex.source, &ex.claim)?;
        if pred == Verdict::Supported {
            if ex.gold == Verdict::Supported {
                tp += 1;
            } else {
                fp += 1;
            }
        }
    }
    if tp + fp == 0 {
        return Ok(0.0);
    }
    Ok(tp as f64 / (tp + fp) as f64)
}

pub fn rewrite_from_completion(
    doc: &SourceDocument,
    style: Style,
    completion: &str,
    tokenizer: Option<&dyn TokenCounter>,
) -> Result<Rewrite> {
    let check = fact_check(&doc.text, completion)?;
    let accepted = accept(&check);
    let n_tokens = token_count(completion, tokenizer);
    Ok(Rewrite {
        source_id: doc.source_id.clone(),
        style,
        text: completion.to_string(),
        token_count: n_tokens,
        fact_check: check,
        accepted,
    })
}

pub fn rephrase<G: Generator>(
    generator: &mut G,
    tokenizer: Option<&dyn TokenCounter>,
    config: &OrchestratorConfig,
    doc: &SourceDocument,
    style: Style,
) -> Result<Rewrite> {
    let prompt = generate_prompt(doc, style)?;
    let outs = generator.generate(
        std::slice::from_ref(&prompt),
        config.max_tokens,
        config.temperature,
    )?;
    if outs.len() != 1 {
        return Err(Error::LengthMismatch {
            want: 1,
            got: outs.len(),
        });
    }
    rewrite_from_completion(doc, style, &outs[0], tokenizer)
}

pub fn rephrase_many<G: Generator>(
    generator: &mut G,
    tokenizer: Option<&dyn TokenCounter>,
    config: &OrchestratorConfig,
    docs: &[SourceDocument],
    styles: Option<&[Style]>,
) -> Result<Vec<Rewrite>> {
    if docs.is_empty() {
        return Err(Error::EmptyBatch);
    }
    let default_styles = config.styles.clone();
    let use_styles: &[Style] = styles.unwrap_or(&default_styles);
    let mut pairs: Vec<(&SourceDocument, Style)> = Vec::new();
    for doc in docs {
        for &style in use_styles {
            pairs.push((doc, style));
        }
    }
    let mut prompts = Vec::with_capacity(pairs.len());
    for (doc, style) in &pairs {
        prompts.push(generate_prompt(doc, *style)?);
    }
    let mut completions = Vec::with_capacity(prompts.len());
    let max_batch = config.max_batch as usize;
    for chunk in prompts.chunks(max_batch) {
        let outs = generator.generate(chunk, config.max_tokens, config.temperature)?;
        if outs.len() != chunk.len() {
            return Err(Error::LengthMismatch {
                want: chunk.len(),
                got: outs.len(),
            });
        }
        completions.extend(outs);
    }
    let mut out = Vec::with_capacity(pairs.len());
    for ((doc, style), text) in pairs.iter().zip(completions.iter()) {
        out.push(rewrite_from_completion(doc, *style, text, tokenizer)?);
    }
    Ok(out)
}
