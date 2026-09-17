//! Synthetic rewrites: generation orchestration and fact-check (spec 7.1, 15.5 E1).
//!
//! High-quality documents are rephrased in several styles (Kimi K2-style) and
//! fact-checked against the source. The checker is source-grounded and does not
//! call the generator. The generator is F5's batch API behind [`Generator`].
//! Token counts use F6 encode length when a [`TokenCounter`] is provided.
//!
//! Gate: supported-class precision of [`check_claim`] on a labeled sample.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_MAX_TOKENS: u32 = 512;
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_MAX_BATCH: u32 = 8;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    #[error("empty source")]
    EmptySource,
    #[error("empty batch")]
    EmptyBatch,
    #[error("batch larger than max_batch {0}")]
    BatchTooLarge(u32),
    #[error("generator returned {got} completions for {want} prompts")]
    LengthMismatch { want: usize, got: usize },
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

/// Split a rewrite into atomic factual claims. Empty text is empty.
pub fn extract_claims(text: &str) -> Vec<String> {
    let _ = text;
    unimplemented!("E1 extract_claims")
}

/// Ground `claim` in `source`. Empty source is [`Error::EmptySource`].
pub fn check_claim(source: &str, claim: &str) -> Result<Verdict> {
    let _ = (source, claim);
    unimplemented!("E1 check_claim")
}

/// Extract claims from `rewrite` and ground each in `source`.
///
/// Empty source is [`Error::EmptySource`]. Empty rewrite is zero claims.
pub fn fact_check(source: &str, rewrite: &str) -> Result<FactCheck> {
    let _ = (source, rewrite);
    unimplemented!("E1 fact_check")
}

/// Keep a rewrite iff no claim is contradicted. `not_in_source` is kept.
pub fn accept(check: &FactCheck) -> bool {
    let _ = check;
    unimplemented!("E1 accept")
}

/// `counter.token_count(text)` if given, else whitespace words. Empty is 0.
pub fn token_count(text: &str, counter: Option<&dyn TokenCounter>) -> u32 {
    let _ = (text, counter);
    unimplemented!("E1 token_count")
}

/// Supported-class precision of [`check_claim`] vs gold.
///
/// TP = predicted supported and gold supported. FP = predicted supported
/// and gold not supported. Empty examples, or no predicted supported, is
/// 0.0 (fail-closed).
pub fn precision(examples: &[LabeledExample]) -> Result<f64> {
    let _ = examples;
    unimplemented!("E1 precision")
}

/// Batch rephrase over documents × styles, then fact-check.
///
/// `generator` is F5. `tokenizer` is F6 when token counts should match the
/// frozen vocab; omit it to count whitespace words.
pub struct Orchestrator {
    _private: (),
}

impl Orchestrator {
    pub fn new<G>(generator: G, config: OrchestratorConfig) -> Result<Self>
    where
        G: Generator + 'static,
    {
        let _ = (generator, config);
        unimplemented!("E1 Orchestrator::new")
    }

    /// Use F6 encode length for [`Rewrite::token_count`].
    pub fn with_tokenizer<T>(self, tokenizer: T) -> Self
    where
        T: TokenCounter + 'static,
    {
        let _ = tokenizer;
        unimplemented!("E1 Orchestrator::with_tokenizer")
    }

    /// One document, one style. Empty source is [`Error::EmptySource`].
    pub fn rephrase(&mut self, doc: &SourceDocument, style: Style) -> Result<Rewrite> {
        let _ = (doc, style);
        unimplemented!("E1 rephrase")
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
        let _ = (docs, styles);
        unimplemented!("E1 rephrase_many")
    }
}
