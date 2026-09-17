//! Synthetic data: rewrites (E1), rejection sampling (E2), procedural ARC (E3).
//!
//! E1: high-quality documents are rephrased in several styles (Kimi K2-style)
//! and fact-checked against the source. Gate: supported-class precision of
//! [`check_claim`] on a labeled sample.
//!
//! E2: reasoning traces sampled from F5 and kept only when a D2 verifier
//! sets `passed`. Gate: [`reject::verified_correct_rate`].
//!
//! E3: re-arc-style grid families, 2D tokenization via F6 `encode_grid`,
//! dihedral × color-perm augmentation. Gate: [`arc::diversity_stats`].
//!
//! The generator is F5's batch API behind [`Generator`]. Token counts use F6
//! encode length when a [`TokenCounter`] is provided.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_MAX_TOKENS: u32 = 512;
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_MAX_BATCH: u32 = 8;

mod reject;
pub use reject::*;
mod arc;
pub use arc::*;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    #[error("empty source")]
    EmptySource,
    #[error("empty prompt")]
    EmptyPrompt,
    #[error("n_samples must be > 0")]
    ZeroSamples,
    #[error("empty batch")]
    EmptyBatch,
    #[error("batch larger than max_batch {0}")]
    BatchTooLarge(u32),
    #[error("generator returned {got} completions for {want} prompts")]
    LengthMismatch { want: usize, got: usize },
    #[error("empty grid")]
    EmptyGrid,
    #[error("jagged grid")]
    JaggedGrid,
    #[error("ARC color {0} out of range")]
    BadColor(u8),
    #[error("grid size {rows}x{cols} out of range")]
    GridSize { rows: usize, cols: usize },
    #[error("dihedral index {0} out of range")]
    BadDihedral(u8),
    #[error("color permutation is not a permutation of 0..10")]
    BadPermutation,
    #[error("empty train")]
    EmptyTrain,
    #[error("empty tasks")]
    EmptyTasks,
    #[error("unknown family")]
    UnknownFamily,
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Rewrite styles. JSON names are snake_case values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Style {
    Encyclopedia,
    Textbook,
    Conversational,
    News,
    Technical,
    Simplified,
}

impl Style {
    pub const ALL: [Style; 6] = [
        Style::Encyclopedia,
        Style::Textbook,
        Style::Conversational,
        Style::News,
        Style::Technical,
        Style::Simplified,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Style::Encyclopedia => "encyclopedia",
            Style::Textbook => "textbook",
            Style::Conversational => "conversational",
            Style::News => "news",
            Style::Technical => "technical",
            Style::Simplified => "simplified",
        }
    }

    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "encyclopedia" => Ok(Style::Encyclopedia),
            "textbook" => Ok(Style::Textbook),
            "conversational" => Ok(Style::Conversational),
            "news" => Ok(Style::News),
            "technical" => Ok(Style::Technical),
            "simplified" => Ok(Style::Simplified),
            other => Err(Error::Message(format!("unknown style {other}"))),
        }
    }
}

/// Fact-check of one claim against the source document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Supported,
    Contradicted,
    NotInSource,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Supported => "supported",
            Verdict::Contradicted => "contradicted",
            Verdict::NotInSource => "not_in_source",
        }
    }
}

/// F5 batch generation. One completion per prompt, same order.
pub trait Generator {
    fn generate(
        &mut self,
        prompts: &[String],
        max_tokens: u32,
        temperature: f32,
    ) -> Result<Vec<String>>;
}

/// F6 encode length. Implementors wrap `Tokenizer::encode`.
pub trait TokenCounter {
    fn token_count(&self, text: &str) -> u32;
}

/// One source document to rephrase.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceDocument {
    pub source_id: String,
    pub text: String,
}

/// One atomic claim extracted from a rewrite, with a source verdict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Claim {
    pub text: String,
    pub verdict: Verdict,
}

/// All claims from a rewrite. Counts are derived from `claims`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactCheck {
    pub claims: Vec<Claim>,
}

impl FactCheck {
    pub fn n_supported(&self) -> usize {
        self.claims
            .iter()
            .filter(|c| c.verdict == Verdict::Supported)
            .count()
    }

    pub fn n_contradicted(&self) -> usize {
        self.claims
            .iter()
            .filter(|c| c.verdict == Verdict::Contradicted)
            .count()
    }

    pub fn n_not_in_source(&self) -> usize {
        self.claims
            .iter()
            .filter(|c| c.verdict == Verdict::NotInSource)
            .count()
    }
}

/// One style rewrite of a source document, after fact-check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rewrite {
    pub source_id: String,
    pub style: Style,
    pub text: String,
    pub token_count: u32,
    pub fact_check: FactCheck,
    pub accepted: bool,
}

/// Gold verdict for [`check_claim`]. Used by [`precision`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabeledExample {
    pub source: String,
    pub claim: String,
    pub gold: Verdict,
}

/// Batch and decoding knobs. `styles` is the default Cartesian factor.
#[derive(Debug, Clone, PartialEq)]
pub struct OrchestratorConfig {
    pub max_tokens: u32,
    pub temperature: f32,
    pub max_batch: u32,
    pub styles: Vec<Style>,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_MAX_TOKENS,
            temperature: DEFAULT_TEMPERATURE,
            max_batch: DEFAULT_MAX_BATCH,
            styles: Style::ALL.to_vec(),
        }
    }
}

/// Prompt the generator to rephrase `doc` in `style`.
///
/// Empty `doc.text` is [`Error::EmptySource`]. The template is the contract
/// between Python and Rust so both languages send the same bytes to F5.
pub fn generate_prompt(doc: &SourceDocument, style: Style) -> Result<String> {
    if doc.text.is_empty() {
        return Err(Error::EmptySource);
    }
    Ok(format!(
        "Rephrase the document in {} style. Preserve every fact. Do not add facts.\n\nDocument:\n{}",
        style.as_str(),
        doc.text
    ))
}

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

/// ASCII numbers matching `[0-9]+(?:\.[0-9]+)?`.
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

fn rewrite_from_completion(
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

/// Split a rewrite into atomic factual claims. Empty text is empty.
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

/// Ground `claim` in `source`. Empty source is [`Error::EmptySource`].
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

/// Extract claims from `rewrite` and ground each in `source`.
///
/// Empty source is [`Error::EmptySource`]. Empty rewrite is zero claims.
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

/// Keep a rewrite iff no claim is contradicted. `not_in_source` is kept.
pub fn accept(check: &FactCheck) -> bool {
    check.n_contradicted() == 0
}

/// `counter.token_count(text)` if given, else whitespace words. Empty is 0.
pub fn token_count(text: &str, counter: Option<&dyn TokenCounter>) -> u32 {
    if text.is_empty() {
        return 0;
    }
    if let Some(counter) = counter {
        return counter.token_count(text);
    }
    text.split_whitespace().count() as u32
}

/// Supported-class precision of [`check_claim`] vs gold.
///
/// TP = predicted supported and gold supported. FP = predicted supported
/// and gold not supported. Empty examples, or no predicted supported, is
/// 0.0 (fail-closed).
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

/// Batch rephrase over documents × styles, then fact-check.
///
/// `generator` is F5. `tokenizer` is F6 when token counts should match the
/// frozen vocab; omit it to count whitespace words.
pub struct Orchestrator {
    generator: Box<dyn Generator>,
    tokenizer: Option<Box<dyn TokenCounter>>,
    config: OrchestratorConfig,
}

impl Orchestrator {
    pub fn new<G>(generator: G, config: OrchestratorConfig) -> Result<Self>
    where
        G: Generator + 'static,
    {
        Ok(Self {
            generator: Box::new(generator),
            tokenizer: None,
            config,
        })
    }

    /// Use F6 encode length for [`Rewrite::token_count`].
    pub fn with_tokenizer<T>(mut self, tokenizer: T) -> Self
    where
        T: TokenCounter + 'static,
    {
        self.tokenizer = Some(Box::new(tokenizer));
        self
    }

    /// One document, one style. Empty source is [`Error::EmptySource`].
    pub fn rephrase(&mut self, doc: &SourceDocument, style: Style) -> Result<Rewrite> {
        let prompt = generate_prompt(doc, style)?;
        let outs = self.generator.generate(
            std::slice::from_ref(&prompt),
            self.config.max_tokens,
            self.config.temperature,
        )?;
        if outs.len() != 1 {
            return Err(Error::LengthMismatch {
                want: 1,
                got: outs.len(),
            });
        }
        rewrite_from_completion(doc, style, &outs[0], self.tokenizer.as_deref())
    }

    /// Cartesian product of docs and styles, chunked by `max_batch`.
    ///
    /// Empty `docs` is [`Error::EmptyBatch`]. Default styles are
    /// `config.styles`. Result order is docs-major, then styles.
    pub fn rephrase_many(
        &mut self,
        docs: &[SourceDocument],
        styles: Option<&[Style]>,
    ) -> Result<Vec<Rewrite>> {
        if docs.is_empty() {
            return Err(Error::EmptyBatch);
        }
        let default_styles = self.config.styles.clone();
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
        let max_batch = self.config.max_batch as usize;
        for chunk in prompts.chunks(max_batch) {
            let outs =
                self.generator
                    .generate(chunk, self.config.max_tokens, self.config.temperature)?;
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
            out.push(rewrite_from_completion(
                doc,
                *style,
                text,
                self.tokenizer.as_deref(),
            )?);
        }
        Ok(out)
    }
}
