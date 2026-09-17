//! Pretraining mixture tooling (spec 7.1, 6, 15.5 B6).
//!
//! The 7.1 table is unique tokens and epochs, not sampling weights. Mix ratios
//! come from rung 2 mixture ablations and are re-weighted in the decay phase
//! toward code, math and reasoning. `data_mix_hash` on F1 `loader_state` and
//! `checkpoint` is SHA-256 of this crate's canonical mix JSON.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Fail-closed mix errors. A zero-weight or empty mix is not a silent skip.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("mix config: {0}")]
    Config(String),
    #[error("unknown source {0}")]
    UnknownSource(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Sources in spec 7.1, serde names matching the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Web,
    Code,
    MathScienceArxiv,
    BooksPapers,
    SyntheticRewrites,
    SyntheticReasoning,
    ProceduralArc,
    AgenticTrajectories,
}

/// Training phase. Decay re-weights toward code, math and reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Pretrain,
    Decay,
}

/// Unique-token and epoch bounds from the 7.1 table.
///
/// `unique_tokens` is `None` for licensing-gated books/papers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSpec {
    pub source: Source,
    pub unique_tokens: Option<u64>,
    pub epochs_min: u32,
    pub epochs_max: u32,
}

/// One mix. `weights` must contain every source that is in the mix, each > 0,
/// and sum to 1. `mix_bucket` is the F1 shard field (e.g. `pretrain-r0`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mix {
    pub mix_id: String,
    pub mix_bucket: String,
    pub phase: Phase,
    pub weights: BTreeMap<Source, f64>,
}

/// Sources boosted in the decay phase (spec 7.1: code, math and reasoning).
pub const DECAY_SOURCES: [Source; 3] = [
    Source::Code,
    Source::MathScienceArxiv,
    Source::SyntheticReasoning,
];

/// Absolute error allowed when checking that weights sum to 1.
pub const WEIGHT_SUM_TOL: f64 = 1e-9;

/// Spec 7.1 catalog. Books/papers unique tokens stay `None`.
pub fn flagship_catalog() -> Vec<SourceSpec> {
    vec![
        SourceSpec {
            source: Source::Web,
            unique_tokens: Some(20_000_000_000_000),
            epochs_min: 2,
            epochs_max: 3,
        },
        SourceSpec {
            source: Source::Code,
            unique_tokens: Some(4_000_000_000_000),
            epochs_min: 4,
            epochs_max: 4,
        },
        SourceSpec {
            source: Source::MathScienceArxiv,
            unique_tokens: Some(1_500_000_000_000),
            epochs_min: 4,
            epochs_max: 4,
        },
        SourceSpec {
            source: Source::BooksPapers,
            unique_tokens: None,
            epochs_min: 2,
            epochs_max: 4,
        },
        SourceSpec {
            source: Source::SyntheticRewrites,
            unique_tokens: Some(40_000_000_000_000),
            epochs_min: 1,
            epochs_max: 1,
        },
        SourceSpec {
            source: Source::SyntheticReasoning,
            unique_tokens: Some(10_000_000_000_000),
            epochs_min: 1,
            epochs_max: 1,
        },
        SourceSpec {
            source: Source::ProceduralArc,
            unique_tokens: Some(2_000_000_000_000),
            epochs_min: 1,
            epochs_max: 1,
        },
        SourceSpec {
            source: Source::AgenticTrajectories,
            unique_tokens: Some(1_000_000_000_000),
            epochs_min: 1,
            epochs_max: 1,
        },
    ]
}

/// Reject empty id/bucket, non-finite or non-positive weights, duplicates
/// (impossible with BTreeMap), unknown emptiness, or a sum outside
/// `WEIGHT_SUM_TOL` of 1.
pub fn validate_mix(mix: &Mix) -> Result<()> {
    if mix.mix_id.is_empty() {
        return Err(Error::Config("empty mix_id".into()));
    }
    if mix.mix_bucket.is_empty() {
        return Err(Error::Config("empty mix_bucket".into()));
    }
    if mix.weights.is_empty() {
        return Err(Error::Config("empty weights".into()));
    }
    let mut total = 0.0;
    for (source, &w) in &mix.weights {
        if !w.is_finite() || w <= 0.0 {
            return Err(Error::Config(format!(
                "non-finite or non-positive weight {source:?}"
            )));
        }
        total += w;
    }
    if !total.is_finite() || (total - 1.0).abs() > WEIGHT_SUM_TOL {
        return Err(Error::Config("weights must sum to 1".into()));
    }
    Ok(())
}

/// Lowercase hex SHA-256 of canonical JSON (sorted keys, no extra whitespace
/// beyond serde_json's default for a BTreeMap object). Matches F1 `sha256`.
pub fn mix_hash(mix: &Mix) -> Result<String> {
    let bytes = canonical_json(mix)?;
    let digest = Sha256::digest(&bytes);
    Ok(format!("{digest:x}"))
}

/// Canonical JSON bytes hashed by `mix_hash`.
pub fn canonical_json(mix: &Mix) -> Result<Vec<u8>> {
    for &w in mix.weights.values() {
        if !w.is_finite() {
            return Err(Error::Config("non-finite weight".into()));
        }
    }
    serde_json::to_vec(mix).map_err(|e| Error::Other(e.to_string()))
}

/// Multiply `DECAY_SOURCES` weights by `factor` (> 1) and renormalize.
/// Output phase is `Decay`. Other sources stay in the mix.
pub fn decay_reweight(mix: &Mix, factor: f64) -> Result<Mix> {
    validate_mix(mix)?;
    if !factor.is_finite() || factor <= 1.0 {
        return Err(Error::Config(
            "decay factor must be finite and > 1".into(),
        ));
    }
    let mut weights = mix.weights.clone();
    for src in DECAY_SOURCES {
        if let Some(w) = weights.get_mut(&src) {
            *w *= factor;
        }
    }
    let weights = renormalize(weights)?;
    for &w in weights.values() {
        if !w.is_finite() || w <= 0.0 {
            return Err(Error::Config(
                "decay produced a non-finite or non-positive weight".into(),
            ));
        }
    }
    Ok(Mix {
        mix_id: mix.mix_id.clone(),
        mix_bucket: mix.mix_bucket.clone(),
        phase: Phase::Decay,
        weights,
    })
}

/// Drop one source and renormalize remaining weights. Errors if that was the
/// only source or it was not in the mix.
pub fn drop_source(mix: &Mix, source: Source) -> Result<Mix> {
    if !mix.weights.contains_key(&source) {
        return Err(Error::UnknownSource(source_name(source)));
    }
    if mix.weights.len() == 1 {
        return Err(Error::Config("cannot drop the last source".into()));
    }
    validate_mix(mix)?;
    let remaining: BTreeMap<Source, f64> = mix
        .weights
        .iter()
        .filter(|(k, _)| **k != source)
        .map(|(k, v)| (*k, *v))
        .collect();
    let weights = renormalize(remaining)?;
    Ok(Mix {
        mix_id: mix.mix_id.clone(),
        mix_bucket: mix.mix_bucket.clone(),
        phase: mix.phase,
        weights,
    })
}

/// Rung 2 drop-one-source ablations of `base`, one mix per source in `base`.
pub fn rung2_ablations(base: &Mix) -> Result<Vec<Mix>> {
    validate_mix(base)?;
    if base.weights.len() <= 1 {
        return Err(Error::Config("cannot drop the last source".into()));
    }
    let mut out = Vec::with_capacity(base.weights.len());
    for &src in base.weights.keys() {
        out.push(drop_source(base, src)?);
    }
    Ok(out)
}

/// Unique tokens times epochs. `None` when unique tokens are licensing-gated.
pub fn tokens_seen(unique_tokens: Option<u64>, epochs: u32) -> Option<u64> {
    unique_tokens.map(|n| n * u64::from(epochs))
}

/// Walk the CDF of `mix.weights` (BTreeMap order) and return the source for
/// `u` in `[0, 1)`. Errors if `u` is outside that range or the mix is invalid.
pub fn sample_source(mix: &Mix, u: f64) -> Result<Source> {
    if !u.is_finite() || !(0.0..1.0).contains(&u) {
        return Err(Error::Config("u must be in [0, 1)".into()));
    }
    validate_mix(mix)?;
    let mut acc = 0.0;
    let mut last = None;
    for (&src, &w) in &mix.weights {
        acc += w;
        last = Some(src);
        if u < acc {
            return Ok(src);
        }
    }
    last.ok_or_else(|| Error::Config("empty weights".into()))
}

fn renormalize(weights: BTreeMap<Source, f64>) -> Result<BTreeMap<Source, f64>> {
    let total: f64 = weights.values().copied().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::Config("cannot renormalize weights".into()));
    }
    Ok(weights
        .into_iter()
        .map(|(k, w)| (k, w / total))
        .collect())
}

fn source_name(source: Source) -> String {
    serde_json::to_value(source)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| format!("{source:?}"))
}

impl Source {
    /// All 7.1 sources in table order.
    pub fn all() -> [Source; 8] {
        [
            Source::Web,
            Source::Code,
            Source::MathScienceArxiv,
            Source::BooksPapers,
            Source::SyntheticRewrites,
            Source::SyntheticReasoning,
            Source::ProceduralArc,
            Source::AgenticTrajectories,
        ]
    }
}
