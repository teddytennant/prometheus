//! Group: `rung_spec` / `rung_0_spec` / `validate_rung_config` / `RungRun::new` errors.
//!
//! Check order is first-hit-wins as documented on `validate_rung_config`.

mod reference;

use prometheus_control::rung::{rung_0_spec, rung_spec};
use prometheus_control::rung::{
    validate_rung_config, RungConfig, RungError, RungId, RungRun, RungSpec, RUNG_0_ACTIVE_PARAMS,
    RUNG_0_GPUS, RUNG_0_TOKENS, RUNG_0_TOTAL_PARAMS, RUNG_1_ACTIVE_PARAMS, RUNG_1_GPUS,
    RUNG_1_TOKENS, RUNG_1_TOTAL_PARAMS, RUNG_2_ACTIVE_PARAMS, RUNG_2_GPUS, RUNG_2_TOKENS,
    RUNG_2_TOTAL_PARAMS, RUNG_3_ACTIVE_PARAMS, RUNG_3_GPUS, RUNG_3_TOKENS, RUNG_3_TOTAL_PARAMS,
    V5_H200_GPUS,
};
use reference::rung as ref_rung;

fn rung0_spec_literal() -> RungSpec {
    RungSpec {
        id: RungId::Zero,
        active_params: 100_000_000,
        total_params: 1_000_000_000,
        tokens: 20_000_000_000,
        gpus: 64,
    }
}

fn cfg(spec: RungSpec, token_budget: u64, tokens_per_step: u64, hash: &str) -> RungConfig {
    RungConfig {
        spec,
        token_budget,
        tokens_per_step,
        tokenizer_hash: hash.to_string(),
        seed: 1,
    }
}

#[test]
fn rung_spec_zero_is_spec_6_not_v5_gpu_count() {
    let s = rung_spec(RungId::Zero);
    assert_eq!(s.id, RungId::Zero);
    assert_eq!(s.active_params, 100_000_000);
    assert_eq!(s.total_params, 1_000_000_000);
    assert_eq!(s.tokens, 20_000_000_000);
    assert_eq!(s.gpus, 64);
    assert_eq!(s.active_params, RUNG_0_ACTIVE_PARAMS);
    assert_eq!(s.total_params, RUNG_0_TOTAL_PARAMS);
    assert_eq!(s.tokens, RUNG_0_TOKENS);
    assert_eq!(s.gpus, RUNG_0_GPUS);
    assert_ne!(s.gpus, V5_H200_GPUS);
    assert_ne!(s.gpus, 8);
}

#[test]
fn rung_spec_one_two_three_match_spec_6() {
    let one = rung_spec(RungId::One);
    assert_eq!(one.id, RungId::One);
    assert_eq!(one.active_params, RUNG_1_ACTIVE_PARAMS);
    assert_eq!(one.total_params, RUNG_1_TOTAL_PARAMS);
    assert_eq!(one.tokens, RUNG_1_TOKENS);
    assert_eq!(one.gpus, RUNG_1_GPUS);
    assert_eq!(one.active_params, 1_000_000_000);
    assert_eq!(one.total_params, 15_000_000_000);
    assert_eq!(one.tokens, 200_000_000_000);
    assert_eq!(one.gpus, 1_000);

    let two = rung_spec(RungId::Two);
    assert_eq!(two.id, RungId::Two);
    assert_eq!(two.active_params, RUNG_2_ACTIVE_PARAMS);
    assert_eq!(two.total_params, RUNG_2_TOTAL_PARAMS);
    assert_eq!(two.tokens, RUNG_2_TOKENS);
    assert_eq!(two.gpus, RUNG_2_GPUS);
    assert_eq!(two.active_params, 8_000_000_000);
    assert_eq!(two.total_params, 120_000_000_000);
    assert_eq!(two.tokens, 1_500_000_000_000);
    assert_eq!(two.gpus, 5_000);

    let three = rung_spec(RungId::Three);
    assert_eq!(three.id, RungId::Three);
    assert_eq!(three.active_params, RUNG_3_ACTIVE_PARAMS);
    assert_eq!(three.total_params, RUNG_3_TOTAL_PARAMS);
    assert_eq!(three.tokens, RUNG_3_TOKENS);
    assert_eq!(three.gpus, RUNG_3_GPUS);
    assert_eq!(three.active_params, 40_000_000_000);
    assert_eq!(three.total_params, 700_000_000_000);
    assert_eq!(three.tokens, 6_000_000_000_000);
    assert_eq!(three.gpus, 15_000);
}

#[test]
fn rung_0_spec_equals_rung_spec_zero() {
    assert_eq!(rung_0_spec(), rung_spec(RungId::Zero));
    assert_eq!(rung_0_spec().gpus, 64);
    assert_ne!(rung_0_spec().gpus, V5_H200_GPUS);
}

#[test]
fn no_ladder_row_uses_v5_h200_gpu_count() {
    for id in [RungId::Zero, RungId::One, RungId::Two, RungId::Three] {
        assert_ne!(rung_spec(id).gpus, V5_H200_GPUS);
    }
}

#[test]
fn validate_not_rung0_wins_even_if_budget_zero_and_hash_empty() {
    for id in [RungId::One, RungId::Two, RungId::Three] {
        let spec = RungSpec {
            id,
            active_params: 1,
            total_params: 1,
            tokens: 1,
            gpus: 1,
        };
        let c = cfg(spec, 0, 0, "");
        assert_eq!(validate_rung_config(&c), Err(RungError::NotRung0));
        assert_eq!(RungRun::new(c).unwrap_err(), RungError::NotRung0);
    }
}

#[test]
fn validate_invalid_budget_wins_over_tokens_per_step_and_hash() {
    let spec = rung0_spec_literal();
    let zero_budget = cfg(spec.clone(), 0, 0, "");
    assert_eq!(
        validate_rung_config(&zero_budget),
        Err(RungError::InvalidBudget)
    );
    assert_eq!(
        RungRun::new(zero_budget).unwrap_err(),
        RungError::InvalidBudget
    );

    let over = cfg(spec, spec_tokens() + 1, 0, "");
    assert_eq!(validate_rung_config(&over), Err(RungError::InvalidBudget));
    assert_eq!(RungRun::new(over).unwrap_err(), RungError::InvalidBudget);
}

fn spec_tokens() -> u64 {
    20_000_000_000
}

#[test]
fn validate_budget_equal_to_spec_tokens_is_ok() {
    let c = cfg(rung0_spec_literal(), 20_000_000_000, 1, "h");
    assert_eq!(validate_rung_config(&c), Ok(()));
}

#[test]
fn validate_budget_compared_against_config_spec_tokens_not_a_side_table() {
    let mut spec = rung0_spec_literal();
    spec.tokens = 100;
    assert_eq!(
        validate_rung_config(&cfg(spec.clone(), 100, 1, "h")),
        Ok(())
    );
    assert_eq!(
        validate_rung_config(&cfg(spec, 101, 1, "h")),
        Err(RungError::InvalidBudget)
    );
}

#[test]
fn validate_invalid_tokens_per_step_wins_over_empty_hash() {
    let c = cfg(rung0_spec_literal(), 10, 0, "");
    assert_eq!(
        validate_rung_config(&c),
        Err(RungError::InvalidTokensPerStep)
    );
    assert_eq!(
        RungRun::new(c).unwrap_err(),
        RungError::InvalidTokensPerStep
    );
}

#[test]
fn validate_empty_tokenizer_hash_is_last() {
    let c = cfg(rung0_spec_literal(), 10, 1, "");
    assert_eq!(validate_rung_config(&c), Err(RungError::EmptyTokenizerHash));
    assert_eq!(RungRun::new(c).unwrap_err(), RungError::EmptyTokenizerHash);
}

#[test]
fn validate_accepts_whitespace_hash_and_zero_seed() {
    let mut c = cfg(rung0_spec_literal(), 1, 1, " ");
    c.seed = 0;
    assert_eq!(validate_rung_config(&c), Ok(()));
    assert!(RungRun::new(c).is_ok());
}

#[test]
fn validate_accepts_tokens_per_step_larger_than_budget() {
    let c = cfg(rung0_spec_literal(), 3, 100, "hash");
    assert_eq!(validate_rung_config(&c), Ok(()));
    assert!(RungRun::new(c).is_ok());
}

#[test]
fn validate_ignores_unused_spec_fields_when_id_is_zero() {
    let spec = RungSpec {
        id: RungId::Zero,
        active_params: 0,
        total_params: 0,
        tokens: 50,
        gpus: V5_H200_GPUS,
    };
    let c = cfg(spec, 50, 2, "tok");
    assert_eq!(validate_rung_config(&c), Ok(()));
}

#[test]
fn new_rejects_same_errors_as_validate() {
    let not_rung0 = cfg(
        RungSpec {
            id: RungId::One,
            ..rung0_spec_literal()
        },
        10,
        1,
        "h",
    );
    assert_eq!(validate_rung_config(&not_rung0), Err(RungError::NotRung0));
    assert_eq!(RungRun::new(not_rung0).unwrap_err(), RungError::NotRung0);

    let budget = cfg(rung0_spec_literal(), 0, 1, "h");
    assert_eq!(validate_rung_config(&budget), Err(RungError::InvalidBudget));
    assert_eq!(RungRun::new(budget).unwrap_err(), RungError::InvalidBudget);

    let tps = cfg(rung0_spec_literal(), 10, 0, "h");
    assert_eq!(
        validate_rung_config(&tps),
        Err(RungError::InvalidTokensPerStep)
    );
    assert_eq!(
        RungRun::new(tps).unwrap_err(),
        RungError::InvalidTokensPerStep
    );

    let hash = cfg(rung0_spec_literal(), 10, 1, "");
    assert_eq!(
        validate_rung_config(&hash),
        Err(RungError::EmptyTokenizerHash)
    );
    assert_eq!(
        RungRun::new(hash).unwrap_err(),
        RungError::EmptyTokenizerHash
    );
}

#[test]
fn production_spec_matches_independent_reference() {
    for id in [RungId::Zero, RungId::One, RungId::Two, RungId::Three] {
        assert_eq!(rung_spec(id), ref_rung::rung_spec(id));
    }
    assert_eq!(rung_0_spec(), ref_rung::rung_0_spec());
}
