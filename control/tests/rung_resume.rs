//! Group: `RungRun::resume`, job-boundary checkpoint, fault-injected snapshots.
//!
//! CPU analog of spec 16.2 V5: a checkpoint then resume continues the same
//! loss curve and token count. Tests use a tiny `token_budget`, not 20B.

mod reference;

use prometheus_control::rung::{RungCheckpoint, RungConfig, RungError, RungId, RungRun, RungSpec};
use reference::rung::{tiny_rung0_config, RefRungRun};

fn tiny(budget: u64, per_step: u64) -> RungConfig {
    tiny_rung0_config(budget, per_step)
}

fn crafted(cfg: &RungConfig, step: u64, tokens_seen: u64, curve: Vec<f64>) -> RungCheckpoint {
    RungCheckpoint {
        step,
        tokens_seen,
        loss_curve: curve,
        tokenizer_hash: cfg.tokenizer_hash.clone(),
        seed: cfg.seed,
        token_budget: cfg.token_budget,
        tokens_per_step: cfg.tokens_per_step,
        rung: cfg.spec.id,
    }
}

#[test]
fn v5_cpu_analog_checkpoint_resume_continues_curve_and_tokens() {
    let cfg = tiny(10, 3);
    let mut job1 = RungRun::new(cfg.clone()).unwrap();
    job1.step(1.0).unwrap();
    job1.step(0.8).unwrap();
    let ckpt = job1.checkpoint();
    assert_eq!(ckpt.tokens_seen, 6);
    assert_eq!(ckpt.loss_curve, vec![1.0, 0.8]);

    let mut job2 = RungRun::resume(cfg.clone(), &ckpt).unwrap();
    assert_eq!(job2.tokens_seen(), 6);
    assert_eq!(job2.step_index(), 2);
    assert_eq!(job2.loss_curve(), &[1.0, 0.8]);
    assert!(!job2.done());
    assert_eq!(job2.remaining_tokens(), 4);

    let r3 = job2.step(0.5).unwrap();
    assert_eq!(r3.step, 3);
    assert_eq!(r3.tokens_seen, 9);
    assert!(!r3.done);
    let r4 = job2.step(0.3).unwrap();
    assert_eq!(r4.step, 4);
    assert_eq!(r4.tokens_seen, 10);
    assert!(r4.done);
    assert_eq!(job2.loss_curve(), &[1.0, 0.8, 0.5, 0.3]);

    let mut uninterrupted = RungRun::new(cfg).unwrap();
    for loss in [1.0, 0.8, 0.5, 0.3] {
        uninterrupted.step(loss).unwrap();
    }
    assert_eq!(job2.loss_curve(), uninterrupted.loss_curve());
    assert_eq!(job2.tokens_seen(), uninterrupted.tokens_seen());
    assert_eq!(job2.step_index(), uninterrupted.step_index());
    assert_eq!(job2.done(), uninterrupted.done());
}

#[test]
fn resume_at_budget_is_done_and_next_step_is_already_done() {
    let cfg = tiny(6, 2);
    let mut run = RungRun::new(cfg.clone()).unwrap();
    run.step(1.0).unwrap();
    run.step(1.0).unwrap();
    run.step(1.0).unwrap();
    assert!(run.done());
    let ckpt = run.checkpoint();
    assert_eq!(ckpt.tokens_seen, 6);

    let mut resumed = RungRun::resume(cfg, &ckpt).unwrap();
    assert!(resumed.done());
    assert_eq!(resumed.tokens_seen(), 6);
    assert_eq!(resumed.loss_curve(), &[1.0, 1.0, 1.0]);
    assert_eq!(resumed.step(0.1).unwrap_err(), RungError::AlreadyDone);
    assert_eq!(resumed.loss_curve(), &[1.0, 1.0, 1.0]);
}

#[test]
fn resume_validates_config_before_tokenizer_changed() {
    let mut cfg = tiny(10, 2);
    cfg.tokenizer_hash.clear();
    let ckpt = crafted(&tiny(10, 2), 1, 2, vec![1.0]);
    assert_eq!(
        RungRun::resume(cfg, &ckpt).unwrap_err(),
        RungError::EmptyTokenizerHash
    );
}

#[test]
fn resume_not_rung0_wins_even_if_hash_empty_and_budget_zero() {
    let mut cfg = tiny(0, 0);
    cfg.spec.id = RungId::One;
    cfg.tokenizer_hash.clear();
    let ckpt = crafted(&tiny(10, 2), 0, 0, vec![]);
    assert_eq!(
        RungRun::resume(cfg, &ckpt).unwrap_err(),
        RungError::NotRung0
    );
}

#[test]
fn resume_invalid_budget_before_mismatch() {
    let mut cfg = tiny(10, 2);
    cfg.token_budget = 0;
    let mut ckpt = crafted(&tiny(10, 2), 0, 0, vec![]);
    ckpt.seed = 999;
    assert_eq!(
        RungRun::resume(cfg, &ckpt).unwrap_err(),
        RungError::InvalidBudget
    );
}

#[test]
fn resume_invalid_tokens_per_step_before_mismatch() {
    let mut cfg = tiny(10, 2);
    cfg.tokens_per_step = 0;
    let mut ckpt = crafted(&tiny(10, 2), 0, 0, vec![]);
    ckpt.seed = 999;
    assert_eq!(
        RungRun::resume(cfg, &ckpt).unwrap_err(),
        RungError::InvalidTokensPerStep
    );
}

#[test]
fn resume_tokenizer_changed_before_checkpoint_mismatch() {
    let cfg = tiny(10, 2);
    let mut ckpt = crafted(&cfg, 1, 2, vec![1.0]);
    ckpt.tokenizer_hash = "other-hash".to_string();
    ckpt.seed = 0;
    ckpt.token_budget = 3;
    ckpt.tokens_per_step = 9;
    ckpt.rung = RungId::Three;
    assert_eq!(
        RungRun::resume(cfg, &ckpt).unwrap_err(),
        RungError::TokenizerChanged
    );
}

#[test]
fn resume_seed_mismatch_is_checkpoint_mismatch() {
    let cfg = tiny(10, 2);
    let mut ckpt = crafted(&cfg, 1, 2, vec![1.0]);
    ckpt.seed = cfg.seed.wrapping_add(1);
    assert_eq!(
        RungRun::resume(cfg, &ckpt).unwrap_err(),
        RungError::CheckpointMismatch
    );
}

#[test]
fn resume_token_budget_mismatch_is_checkpoint_mismatch() {
    let cfg = tiny(10, 2);
    let mut other = cfg.clone();
    other.token_budget = 8;
    let ckpt = crafted(&other, 1, 2, vec![1.0]);
    assert_eq!(
        RungRun::resume(cfg, &ckpt).unwrap_err(),
        RungError::CheckpointMismatch
    );
}

#[test]
fn resume_tokens_per_step_mismatch_is_checkpoint_mismatch() {
    let cfg = tiny(10, 2);
    let mut ckpt = crafted(&cfg, 1, 2, vec![1.0]);
    ckpt.tokens_per_step = 1;
    assert_eq!(
        RungRun::resume(cfg, &ckpt).unwrap_err(),
        RungError::CheckpointMismatch
    );
}

#[test]
fn resume_rung_mismatch_is_checkpoint_mismatch() {
    let cfg = tiny(10, 2);
    let mut ckpt = crafted(&cfg, 1, 2, vec![1.0]);
    ckpt.rung = RungId::One;
    assert_eq!(
        RungRun::resume(cfg, &ckpt).unwrap_err(),
        RungError::CheckpointMismatch
    );
}

#[test]
fn resume_checkpoint_mismatch_wins_over_past_budget() {
    let cfg = tiny(10, 2);
    let mut ckpt = crafted(&cfg, 9, 11, vec![]);
    ckpt.seed = cfg.seed.wrapping_add(1);
    assert_eq!(
        RungRun::resume(cfg, &ckpt).unwrap_err(),
        RungError::CheckpointMismatch
    );
}

#[test]
fn resume_past_budget_when_tokens_seen_exceeds_budget() {
    let cfg = tiny(10, 2);
    let ckpt = crafted(&cfg, 9, 11, vec![1.0; 9]);
    assert_eq!(
        RungRun::resume(cfg, &ckpt).unwrap_err(),
        RungError::ResumePastBudget
    );
}

#[test]
fn resume_tokens_seen_equal_budget_from_crafted_checkpoint() {
    let cfg = tiny(10, 2);
    let ckpt = crafted(&cfg, 5, 10, vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    let run = RungRun::resume(cfg, &ckpt).unwrap();
    assert!(run.done());
    assert_eq!(run.tokens_seen(), 10);
    assert_eq!(run.step_index(), 5);
    assert_eq!(run.loss_curve(), &[0.1, 0.2, 0.3, 0.4, 0.5]);
}

#[test]
fn resume_restores_mid_run_and_can_continue() {
    let cfg = tiny(8, 3);
    let ckpt = crafted(&cfg, 2, 6, vec![4.0, 3.0]);
    let mut run = RungRun::resume(cfg, &ckpt).unwrap();
    assert!(!run.done());
    let last = run.step(2.0).unwrap();
    assert_eq!(last.step, 3);
    assert_eq!(last.tokens_seen, 8);
    assert!(last.done);
    assert_eq!(run.loss_curve(), &[4.0, 3.0, 2.0]);
}

#[test]
fn resume_uses_passed_config_even_if_unused_spec_fields_differ() {
    let mut cfg = tiny(10, 5);
    cfg.spec = RungSpec {
        id: RungId::Zero,
        active_params: 1,
        total_params: 2,
        tokens: 20_000_000_000,
        gpus: 8,
    };
    let ckpt = crafted(&cfg, 1, 5, vec![1.0]);
    let run = RungRun::resume(cfg.clone(), &ckpt).unwrap();
    assert_eq!(run.config(), &cfg);
    assert_eq!(run.tokens_seen(), 5);
}

#[test]
fn older_checkpoint_restores_earlier_state() {
    let cfg = tiny(12, 4);
    let mut run = RungRun::new(cfg.clone()).unwrap();
    run.step(1.0).unwrap();
    let early = run.checkpoint();
    run.step(2.0).unwrap();
    run.step(3.0).unwrap();
    assert!(run.done());

    let mut rolled = RungRun::resume(cfg, &early).unwrap();
    assert_eq!(rolled.tokens_seen(), 4);
    assert_eq!(rolled.loss_curve(), &[1.0]);
    assert!(!rolled.done());
    rolled.step(9.0).unwrap();
    assert_eq!(rolled.loss_curve(), &[1.0, 9.0]);
}

#[test]
fn production_resume_matches_reference() {
    let cfg = tiny(10, 3);
    let mut prod = RungRun::new(cfg.clone()).unwrap();
    let mut refer = RefRungRun::new(cfg.clone()).unwrap();
    prod.step(1.0).unwrap();
    refer.step(1.0).unwrap();
    let p_ckpt = prod.checkpoint();
    let r_ckpt = refer.checkpoint();
    assert_eq!(p_ckpt, r_ckpt);

    let mut prod = RungRun::resume(cfg.clone(), &p_ckpt).unwrap();
    let mut refer = RefRungRun::resume(cfg, &r_ckpt).unwrap();
    assert_eq!(prod.tokens_seen(), refer.tokens_seen());
    assert_eq!(prod.loss_curve(), refer.loss_curve());
    assert_eq!(prod.step(0.4).unwrap(), refer.step(0.4).unwrap());
    assert_eq!(prod.checkpoint(), refer.checkpoint());
}
