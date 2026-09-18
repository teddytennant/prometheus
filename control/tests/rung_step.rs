//! Group: `RungRun::new` initial state, `step`, getters, `checkpoint` snapshot.

mod reference;

use prometheus_control::rung::{RungError, RungId, RungRun, RungStepReport};
use reference::rung::{tiny_rung0_config, RefRungRun};

fn tiny(budget: u64, per_step: u64) -> prometheus_control::rung::RungConfig {
    tiny_rung0_config(budget, per_step)
}

#[test]
fn new_starts_at_step_zero_empty_curve_not_done() {
    let cfg = tiny(10, 3);
    let run = RungRun::new(cfg.clone()).expect("new");
    assert_eq!(run.step_index(), 0);
    assert_eq!(run.tokens_seen(), 0);
    assert_eq!(run.remaining_tokens(), 10);
    assert!(run.loss_curve().is_empty());
    assert!(!run.done());
    assert_eq!(run.config(), &cfg);
    let _: u64 = run.tokens_seen();
    let _: &[f64] = run.loss_curve();
}

#[test]
fn step_is_one_based_in_the_report_and_appends_loss() {
    let mut run = RungRun::new(tiny(10, 3)).unwrap();
    let r1 = run.step(1.25).unwrap();
    assert_eq!(
        r1,
        RungStepReport {
            step: 1,
            tokens_seen: 3,
            loss: 1.25,
            done: false,
        }
    );
    assert_eq!(run.step_index(), 1);
    assert_eq!(run.tokens_seen(), 3);
    assert_eq!(run.remaining_tokens(), 7);
    assert_eq!(run.loss_curve(), &[1.25]);
    assert!(!run.done());

    let r2 = run.step(0.5).unwrap();
    assert_eq!(r2.step, 2);
    assert_eq!(r2.tokens_seen, 6);
    assert_eq!(r2.loss, 0.5);
    assert!(!r2.done);
    assert_eq!(run.loss_curve(), &[1.25, 0.5]);
}

#[test]
fn last_step_may_be_short_and_equal_budget_is_done() {
    let mut run = RungRun::new(tiny(10, 3)).unwrap();
    assert!(!run.step(1.0).unwrap().done);
    assert!(!run.step(1.0).unwrap().done);
    assert!(!run.step(1.0).unwrap().done);
    let last = run.step(0.0).unwrap();
    assert_eq!(last.step, 4);
    assert_eq!(last.tokens_seen, 10);
    assert_eq!(last.loss, 0.0);
    assert!(last.done);
    assert!(run.done());
    assert_eq!(run.tokens_seen(), 10);
    assert_eq!(run.remaining_tokens(), 0);
    assert_eq!(run.loss_curve(), &[1.0, 1.0, 1.0, 0.0]);
}

#[test]
fn tokens_per_step_larger_than_budget_finishes_in_one_step() {
    let mut run = RungRun::new(tiny(7, 100)).unwrap();
    let r = run.step(2.0).unwrap();
    assert_eq!(r.step, 1);
    assert_eq!(r.tokens_seen, 7);
    assert!(r.done);
    assert!(run.done());
    assert_eq!(run.remaining_tokens(), 0);
}

#[test]
fn exact_division_finishes_without_a_short_step() {
    let mut run = RungRun::new(tiny(9, 3)).unwrap();
    assert!(!run.step(1.0).unwrap().done);
    assert!(!run.step(1.0).unwrap().done);
    let last = run.step(1.0).unwrap();
    assert!(last.done);
    assert_eq!(last.tokens_seen, 9);
    assert_eq!(last.step, 3);
}

#[test]
fn zero_and_negative_finite_loss_are_recorded() {
    let mut run = RungRun::new(tiny(4, 2)).unwrap();
    let z = run.step(0.0).unwrap();
    assert_eq!(z.loss, 0.0);
    let n = run.step(-1.5).unwrap();
    assert_eq!(n.loss, -1.5);
    assert_eq!(run.loss_curve(), &[0.0, -1.5]);
    assert!(run.done());
}

#[test]
fn extreme_finite_losses_are_recorded() {
    let mut run = RungRun::new(tiny(2, 1)).unwrap();
    run.step(f64::MAX).unwrap();
    run.step(f64::MIN).unwrap();
    assert_eq!(run.loss_curve(), &[f64::MAX, f64::MIN]);
}

#[test]
fn non_finite_loss_does_not_change_state() {
    let mut run = RungRun::new(tiny(10, 4)).unwrap();
    run.step(1.0).unwrap();
    let seen = run.tokens_seen();
    let step = run.step_index();
    let curve = run.loss_curve().to_vec();
    let done = run.done();
    let remaining = run.remaining_tokens();

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(run.step(bad).unwrap_err(), RungError::NonFiniteLoss);
        assert_eq!(run.tokens_seen(), seen);
        assert_eq!(run.step_index(), step);
        assert_eq!(run.loss_curve(), curve.as_slice());
        assert_eq!(run.done(), done);
        assert_eq!(run.remaining_tokens(), remaining);
    }

    let ok = run.step(0.25).unwrap();
    assert_eq!(ok.step, 2);
    assert_eq!(ok.loss, 0.25);
    assert_eq!(run.loss_curve(), &[1.0, 0.25]);
}

#[test]
fn already_done_does_not_change_state_or_append_loss() {
    let mut run = RungRun::new(tiny(3, 3)).unwrap();
    run.step(9.0).unwrap();
    assert!(run.done());
    let seen = run.tokens_seen();
    let step = run.step_index();
    let curve = run.loss_curve().to_vec();

    assert_eq!(run.step(1.0).unwrap_err(), RungError::AlreadyDone);
    assert_eq!(run.step(0.0).unwrap_err(), RungError::AlreadyDone);
    assert_eq!(run.tokens_seen(), seen);
    assert_eq!(run.step_index(), step);
    assert_eq!(run.loss_curve(), curve.as_slice());
    assert!(run.done());
}

#[test]
fn non_finite_wins_over_already_done() {
    let mut run = RungRun::new(tiny(1, 1)).unwrap();
    run.step(1.0).unwrap();
    assert!(run.done());
    assert_eq!(run.step(f64::NAN).unwrap_err(), RungError::NonFiniteLoss);
    assert_eq!(
        run.step(f64::INFINITY).unwrap_err(),
        RungError::NonFiniteLoss
    );
    assert_eq!(run.loss_curve(), &[1.0]);
}

#[test]
fn checkpoint_does_not_consume_tokens() {
    let mut run = RungRun::new(tiny(10, 4)).unwrap();
    run.step(1.0).unwrap();
    let before_seen = run.tokens_seen();
    let before_step = run.step_index();
    let ckpt = run.checkpoint();
    assert_eq!(run.tokens_seen(), before_seen);
    assert_eq!(run.step_index(), before_step);
    assert_eq!(ckpt.step, 1);
    assert_eq!(ckpt.tokens_seen, 4);
    assert_eq!(ckpt.loss_curve, vec![1.0]);
    assert_eq!(ckpt.tokenizer_hash, "sha256:test-tokenizer");
    assert_eq!(ckpt.seed, 7);
    assert_eq!(ckpt.token_budget, 10);
    assert_eq!(ckpt.tokens_per_step, 4);
    assert_eq!(ckpt.rung, RungId::Zero);

    let r = run.step(2.0).unwrap();
    assert_eq!(r.tokens_seen, 8);
    assert_eq!(ckpt.tokens_seen, 4);
    assert_eq!(ckpt.loss_curve, vec![1.0]);
}

#[test]
fn checkpoint_at_fresh_run_is_zeros() {
    let run = RungRun::new(tiny(5, 1)).unwrap();
    let ckpt = run.checkpoint();
    assert_eq!(ckpt.step, 0);
    assert_eq!(ckpt.tokens_seen, 0);
    assert!(ckpt.loss_curve.is_empty());
    assert!(!run.done());
}

#[test]
fn clone_is_independent() {
    let mut a = RungRun::new(tiny(8, 2)).unwrap();
    a.step(1.0).unwrap();
    let mut b = a.clone();
    b.step(2.0).unwrap();
    assert_eq!(a.tokens_seen(), 2);
    assert_eq!(a.loss_curve(), &[1.0]);
    assert_eq!(b.tokens_seen(), 4);
    assert_eq!(b.loss_curve(), &[1.0, 2.0]);
}

#[test]
fn production_step_matches_reference_on_short_last_step() {
    let cfg = tiny(10, 3);
    let mut prod = RungRun::new(cfg.clone()).unwrap();
    let mut refer = RefRungRun::new(cfg).unwrap();
    for loss in [1.0, 0.8, 0.5, 0.3] {
        assert_eq!(prod.step(loss).unwrap(), refer.step(loss).unwrap());
        assert_eq!(prod.tokens_seen(), refer.tokens_seen());
        assert_eq!(prod.remaining_tokens(), refer.remaining_tokens());
        assert_eq!(prod.loss_curve(), refer.loss_curve());
        assert_eq!(prod.done(), refer.done());
        assert_eq!(prod.step_index(), refer.step_index());
        assert_eq!(prod.checkpoint(), refer.checkpoint());
    }
    assert!(prod.done());
}
