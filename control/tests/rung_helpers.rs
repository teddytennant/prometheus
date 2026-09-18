//! Group: associated constants, `RungError` Display, `RungId` serde, type traits.
//!
//! These inspect frozen iface data only. They may pass against the A7 stub.

use prometheus_control::rung::{
    RungCheckpoint, RungConfig, RungError, RungId, RungRun, RungSpec, RungStepReport,
    RUNG_0_ACTIVE_PARAMS, RUNG_0_GPUS, RUNG_0_TOKENS, RUNG_0_TOTAL_PARAMS, RUNG_1_ACTIVE_PARAMS,
    RUNG_1_GPUS, RUNG_1_TOKENS, RUNG_1_TOTAL_PARAMS, RUNG_2_ACTIVE_PARAMS, RUNG_2_GPUS,
    RUNG_2_TOKENS, RUNG_2_TOTAL_PARAMS, RUNG_3_ACTIVE_PARAMS, RUNG_3_GPUS, RUNG_3_TOKENS,
    RUNG_3_TOTAL_PARAMS, V5_H200_GPUS,
};
use std::collections::HashSet;

fn assert_copy_eq_hash<T: Copy + Eq + std::hash::Hash>() {}
fn assert_debug_clone_partial_eq<T: std::fmt::Debug + Clone + PartialEq>() {}
fn assert_eq_trait<T: Eq>() {}
fn assert_error<T: std::error::Error>() {}
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn ladder_constants_match_spec_6_table() {
    let _: u64 = RUNG_0_ACTIVE_PARAMS;
    assert_eq!(RUNG_0_ACTIVE_PARAMS, 100_000_000);
    assert_eq!(RUNG_0_TOTAL_PARAMS, 1_000_000_000);
    assert_eq!(RUNG_0_TOKENS, 20_000_000_000);
    assert_eq!(RUNG_0_GPUS, 64);

    assert_eq!(RUNG_1_ACTIVE_PARAMS, 1_000_000_000);
    assert_eq!(RUNG_1_TOTAL_PARAMS, 15_000_000_000);
    assert_eq!(RUNG_1_TOKENS, 200_000_000_000);
    assert_eq!(RUNG_1_GPUS, 1_000);

    assert_eq!(RUNG_2_ACTIVE_PARAMS, 8_000_000_000);
    assert_eq!(RUNG_2_TOTAL_PARAMS, 120_000_000_000);
    assert_eq!(RUNG_2_TOKENS, 1_500_000_000_000);
    assert_eq!(RUNG_2_GPUS, 5_000);

    assert_eq!(RUNG_3_ACTIVE_PARAMS, 40_000_000_000);
    assert_eq!(RUNG_3_TOTAL_PARAMS, 700_000_000_000);
    assert_eq!(RUNG_3_TOKENS, 6_000_000_000_000);
    assert_eq!(RUNG_3_GPUS, 15_000);
}

#[test]
fn v5_h200_gpus_is_8_not_the_ladder_row() {
    assert_eq!(V5_H200_GPUS, 8);
    assert_ne!(V5_H200_GPUS, RUNG_0_GPUS);
    assert_ne!(RUNG_0_GPUS, 8);
}

#[test]
fn rung_error_display_strings_match_iface() {
    assert_eq!(
        RungError::NotRung0.to_string(),
        "RungRun is rung 0 only (I1 owns rungs 1-3)"
    );
    assert_eq!(
        RungError::InvalidBudget.to_string(),
        "token_budget must be > 0 and <= spec.tokens"
    );
    assert_eq!(
        RungError::InvalidTokensPerStep.to_string(),
        "tokens_per_step must be > 0"
    );
    assert_eq!(
        RungError::EmptyTokenizerHash.to_string(),
        "tokenizer_hash must be non-empty"
    );
    assert_eq!(
        RungError::TokenizerChanged.to_string(),
        "tokenizer hash changed; the tokenizer is frozen before rung 0"
    );
    assert_eq!(
        RungError::AlreadyDone.to_string(),
        "rung already consumed its token_budget"
    );
    assert_eq!(RungError::NonFiniteLoss.to_string(), "loss must be finite");
    assert_eq!(
        RungError::CheckpointMismatch.to_string(),
        "checkpoint does not match this config"
    );
    assert_eq!(
        RungError::ResumePastBudget.to_string(),
        "checkpoint tokens_seen exceeds token_budget"
    );
}

#[test]
fn rung_id_serde_snake_case_names() {
    assert_eq!(serde_json::to_string(&RungId::Zero).unwrap(), "\"zero\"");
    assert_eq!(serde_json::to_string(&RungId::One).unwrap(), "\"one\"");
    assert_eq!(serde_json::to_string(&RungId::Two).unwrap(), "\"two\"");
    assert_eq!(serde_json::to_string(&RungId::Three).unwrap(), "\"three\"");

    assert_eq!(
        serde_json::from_str::<RungId>("\"zero\"").unwrap(),
        RungId::Zero
    );
    assert_eq!(
        serde_json::from_str::<RungId>("\"one\"").unwrap(),
        RungId::One
    );
    assert_eq!(
        serde_json::from_str::<RungId>("\"two\"").unwrap(),
        RungId::Two
    );
    assert_eq!(
        serde_json::from_str::<RungId>("\"three\"").unwrap(),
        RungId::Three
    );

    assert!(serde_json::from_str::<RungId>("\"Zero\"").is_err());
    assert!(serde_json::from_str::<RungId>("\"rung_0\"").is_err());
    assert!(serde_json::from_str::<RungId>("\"0\"").is_err());
    assert!(serde_json::from_str::<RungId>("\"four\"").is_err());
}

#[test]
fn type_traits_on_public_surface() {
    assert_copy_eq_hash::<RungId>();
    assert_debug_clone_partial_eq::<RungId>();
    assert_debug_clone_partial_eq::<RungSpec>();
    assert_eq_trait::<RungSpec>();
    assert_debug_clone_partial_eq::<RungConfig>();
    assert_eq_trait::<RungConfig>();
    assert_debug_clone_partial_eq::<RungStepReport>();
    assert_debug_clone_partial_eq::<RungCheckpoint>();
    fn assert_debug_partial_eq<T: std::fmt::Debug + PartialEq>() {}
    assert_debug_partial_eq::<RungError>();
    assert_eq_trait::<RungError>();
    assert_error::<RungError>();
    fn assert_debug_clone<T: std::fmt::Debug + Clone>() {}
    assert_debug_clone::<RungRun>();
    assert_send_sync::<RungId>();
    assert_send_sync::<RungError>();
}

#[test]
fn rung_id_hash_and_equality() {
    let mut set = HashSet::new();
    set.insert(RungId::Zero);
    set.insert(RungId::One);
    set.insert(RungId::Two);
    set.insert(RungId::Three);
    assert_eq!(set.len(), 4);
    assert!(set.contains(&RungId::Zero));
    assert_ne!(RungId::Zero, RungId::One);
    assert_eq!(RungId::Zero, RungId::Zero);
}
