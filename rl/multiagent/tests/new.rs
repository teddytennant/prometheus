//! Group 1: `Multiagent::new` errors and Default config == spec 9.5 constants.
//!
//! The Default-field test does not call `Multiagent::new` (Default is already
//! implemented on the stub). Every other test drives production `new` and the
//! reference.

mod common;
mod reference;

use common::*;
use prometheus_multiagent::{
    MultiConfig, MultiError, DEFAULT_N_SUBS, LATENT_MEMO_KEEP_DENOM, LATENT_MEMO_KEEP_NUMER,
    MAX_N_SUBS, TOPOLOGY_WEIGHT_DENOM, WEIGHT_AUTHOR_REVIEWER, WEIGHT_ORCHESTRATOR,
    WEIGHT_PARALLEL, WEIGHT_PROPOSER_SOLVER, WEIGHT_SINGLE,
};

/// Allowed to pass against the stub: `MultiConfig::default` is implemented.
#[test]
fn default_config_matches_spec_9_5_constants() {
    let d = MultiConfig::default();
    assert_eq!(d.weight_single, 50);
    assert_eq!(d.weight_orchestrator, 20);
    assert_eq!(d.weight_parallel, 15);
    assert_eq!(d.weight_proposer_solver, 10);
    assert_eq!(d.weight_author_reviewer, 5);
    assert_eq!(d.latent_keep_numer, 50);
    assert_eq!(d.latent_keep_denom, 100);
    assert_eq!(d.default_n_subs, 2);

    assert_eq!(d.weight_single, WEIGHT_SINGLE);
    assert_eq!(d.weight_orchestrator, WEIGHT_ORCHESTRATOR);
    assert_eq!(d.weight_parallel, WEIGHT_PARALLEL);
    assert_eq!(d.weight_proposer_solver, WEIGHT_PROPOSER_SOLVER);
    assert_eq!(d.weight_author_reviewer, WEIGHT_AUTHOR_REVIEWER);
    assert_eq!(d.latent_keep_numer, LATENT_MEMO_KEEP_NUMER);
    assert_eq!(d.latent_keep_denom, LATENT_MEMO_KEEP_DENOM);
    assert_eq!(d.default_n_subs, DEFAULT_N_SUBS);
    assert_eq!(
        WEIGHT_SINGLE
            + WEIGHT_ORCHESTRATOR
            + WEIGHT_PARALLEL
            + WEIGHT_PROPOSER_SOLVER
            + WEIGHT_AUTHOR_REVIEWER,
        TOPOLOGY_WEIGHT_DENOM
    );
    assert_eq!(TOPOLOGY_WEIGHT_DENOM, 100);
    assert_eq!(MAX_N_SUBS, 99);
}

#[test]
fn new_accepts_default_config() {
    let _ = pair_default();
}

#[test]
fn new_rejects_weights_not_summing_to_100() {
    // Other fields valid; sum 99.
    let mut cfg = default_config();
    cfg.weight_author_reviewer = 4;
    match new_err_both(cfg) {
        MultiError::BadMix => {}
        other => panic!("expected BadMix, got {other:?}"),
    }
}

#[test]
fn new_rejects_weights_summing_over_100() {
    let mut cfg = default_config();
    cfg.weight_author_reviewer = 6;
    match new_err_both(cfg) {
        MultiError::BadMix => {}
        other => panic!("expected BadMix, got {other:?}"),
    }
}

#[test]
fn new_rejects_weight_sum_overflow() {
    let cfg = MultiConfig {
        weight_single: u32::MAX,
        weight_orchestrator: u32::MAX,
        weight_parallel: 0,
        weight_proposer_solver: 0,
        weight_author_reviewer: 0,
        latent_keep_numer: LATENT_MEMO_KEEP_NUMER,
        latent_keep_denom: LATENT_MEMO_KEEP_DENOM,
        default_n_subs: DEFAULT_N_SUBS,
    };
    match new_err_both(cfg) {
        MultiError::BadMix => {}
        other => panic!("expected BadMix, got {other:?}"),
    }
}

#[test]
fn new_rejects_weight_single_below_floor() {
    // 49+21+15+10+5 = 100, but single < 50.
    match new_err_both(mix(49, 21, 15, 10, 5)) {
        MultiError::BadMix => {}
        other => panic!("expected BadMix, got {other:?}"),
    }
}

#[test]
fn new_rejects_latent_keep_denom_zero() {
    let mut cfg = default_config();
    cfg.latent_keep_denom = 0;
    cfg.latent_keep_numer = 0;
    match new_err_both(cfg) {
        MultiError::BadMix => {}
        other => panic!("expected BadMix, got {other:?}"),
    }
}

#[test]
fn new_rejects_latent_keep_denom_zero_even_if_numer_positive() {
    let mut cfg = default_config();
    cfg.latent_keep_denom = 0;
    cfg.latent_keep_numer = 1;
    match new_err_both(cfg) {
        MultiError::BadMix => {}
        other => panic!("expected BadMix, got {other:?}"),
    }
}

#[test]
fn new_rejects_latent_numer_greater_than_denom() {
    let mut cfg = default_config();
    cfg.latent_keep_numer = 101;
    cfg.latent_keep_denom = 100;
    match new_err_both(cfg) {
        MultiError::BadMix => {}
        other => panic!("expected BadMix, got {other:?}"),
    }
}

#[test]
fn new_rejects_default_n_subs_zero() {
    let mut cfg = default_config();
    cfg.default_n_subs = 0;
    match new_err_both(cfg) {
        MultiError::BadMix => {}
        other => panic!("expected BadMix, got {other:?}"),
    }
}

#[test]
fn new_rejects_default_n_subs_above_max() {
    let mut cfg = default_config();
    cfg.default_n_subs = MAX_N_SUBS + 1;
    match new_err_both(cfg) {
        MultiError::BadMix => {}
        other => panic!("expected BadMix, got {other:?}"),
    }
}

#[test]
fn new_accepts_weight_single_exactly_floor() {
    let _ = pair(mix(50, 20, 15, 10, 5));
}

#[test]
fn new_accepts_weight_single_100_rest_zero() {
    let _ = pair(mix(100, 0, 0, 0, 0));
}

#[test]
fn new_accepts_custom_mix_with_zero_buckets() {
    let _ = pair(mix(50, 0, 0, 0, 50));
    let _ = pair(mix(70, 0, 15, 0, 15));
    let _ = pair(mix(50, 50, 0, 0, 0));
}

#[test]
fn new_accepts_numer_equal_denom() {
    let mut cfg = default_config();
    cfg.latent_keep_numer = 100;
    cfg.latent_keep_denom = 100;
    let _ = pair(cfg);
}

#[test]
fn new_accepts_numer_zero_positive_denom() {
    let mut cfg = default_config();
    cfg.latent_keep_numer = 0;
    cfg.latent_keep_denom = 100;
    let _ = pair(cfg);
}

#[test]
fn new_accepts_default_n_subs_bounds() {
    let mut lo = default_config();
    lo.default_n_subs = 1;
    let _ = pair(lo);
    let mut hi = default_config();
    hi.default_n_subs = MAX_N_SUBS;
    let _ = pair(hi);
}

#[test]
fn new_accepts_weight_single_above_floor() {
    let _ = pair(mix(51, 19, 15, 10, 5));
    let _ = pair(mix(80, 5, 5, 5, 5));
}
