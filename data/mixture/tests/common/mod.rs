//! Shared builders and assertions for B6 mixture integration tests.
#![allow(dead_code)]

use prometheus_mixture::{Error, Mix, Phase, Result, Source, WEIGHT_SUM_TOL};
use std::collections::BTreeMap;
use std::fmt::Debug;

/// Weight comparison vs the Python reference. Spec sum tolerance is 1e-9.
pub const WEIGHT_TOL: f64 = 1e-9;

pub fn mix(id: &str, bucket: &str, phase: Phase, pairs: &[(Source, f64)]) -> Mix {
    Mix {
        mix_id: id.to_string(),
        mix_bucket: bucket.to_string(),
        phase,
        weights: pairs.iter().copied().collect(),
    }
}

/// All eight 7.1 sources at 0.125 (exactly 1.0 in binary float).
pub fn uniform8() -> Mix {
    let pairs: Vec<(Source, f64)> = Source::all().into_iter().map(|s| (s, 0.125)).collect();
    mix("flagship", "pretrain-r0", Phase::Pretrain, &pairs)
}

pub fn two_source() -> Mix {
    mix(
        "two",
        "pretrain-r0",
        Phase::Pretrain,
        &[(Source::Web, 0.75), (Source::Code, 0.25)],
    )
}

pub fn single_web() -> Mix {
    mix(
        "one",
        "pretrain-r0",
        Phase::Pretrain,
        &[(Source::Web, 1.0)],
    )
}

pub fn rung2_like() -> Mix {
    mix(
        "rung2",
        "pretrain-r2",
        Phase::Pretrain,
        &[
            (Source::Web, 0.50),
            (Source::Code, 0.20),
            (Source::MathScienceArxiv, 0.10),
            (Source::BooksPapers, 0.05),
            (Source::SyntheticRewrites, 0.05),
            (Source::SyntheticReasoning, 0.05),
            (Source::ProceduralArc, 0.03),
            (Source::AgenticTrajectories, 0.02),
        ],
    )
}

pub fn assert_config<T: Debug>(result: Result<T>) {
    match result {
        Err(Error::Config(_)) => {}
        other => panic!("expected Error::Config, got {other:?}"),
    }
}

pub fn assert_unknown<T: Debug>(result: Result<T>) {
    match result {
        Err(Error::UnknownSource(_)) => {}
        other => panic!("expected Error::UnknownSource, got {other:?}"),
    }
}

pub fn assert_sha256_hex(s: &str) {
    assert_eq!(s.len(), 64, "data_mix_hash must be 64 hex chars, got {s:?}");
    assert!(
        s.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')),
        "data_mix_hash must be lowercase hex SHA-256, got {s:?}"
    );
}

pub fn assert_weights_close(got: &BTreeMap<Source, f64>, want: &BTreeMap<Source, f64>) {
    assert_eq!(
        got.len(),
        want.len(),
        "weight keys got {:?} want {:?}",
        got.keys().collect::<Vec<_>>(),
        want.keys().collect::<Vec<_>>()
    );
    for (src, want_w) in want {
        let got_w = got
            .get(src)
            .unwrap_or_else(|| panic!("missing source {src:?}"));
        assert!(
            (got_w - want_w).abs() <= WEIGHT_TOL,
            "{src:?} weight got {got_w} want {want_w}"
        );
        assert!(
            got_w.is_finite() && *got_w > 0.0,
            "{src:?} weight must be finite and > 0, got {got_w}"
        );
    }
    let sum: f64 = got.values().copied().sum();
    assert!(
        (sum - 1.0).abs() <= WEIGHT_SUM_TOL,
        "weights must sum to 1 ± {WEIGHT_SUM_TOL}, got {sum}"
    );
}

pub fn assert_mix_close(got: &Mix, want: &Mix) {
    assert_eq!(got.mix_id, want.mix_id, "mix_id");
    assert_eq!(got.mix_bucket, want.mix_bucket, "mix_bucket");
    assert_eq!(got.phase, want.phase, "phase");
    assert_weights_close(&got.weights, &want.weights);
}

pub fn decay_share(mix: &Mix) -> f64 {
    prometheus_mixture::DECAY_SOURCES
        .iter()
        .filter_map(|s| mix.weights.get(s).copied())
        .sum()
}

pub fn source_snake(source: Source) -> String {
    serde_json::to_value(source)
        .expect("Source serde")
        .as_str()
        .expect("Source string")
        .to_string()
}
