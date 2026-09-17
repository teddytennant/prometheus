//! Group: loss-spike policy vs the independent reference (spec 5.5 / I10).
//!
//! Checks: warmup, sliding-window baseline, strict threshold, action order
//! (Log, SkipShard, Rollback, optional Page), page window inclusive, skip /
//! spike logs, live config / checkpoint fields. Every test that calls
//! `observe` must fail on the I10 stub. Empty SpikeLog getters may pass.
//!
//! No differentiable surface; no GPU (`gpu` marker unused).

mod reference;

use prometheus_obs::{
    SpikeAction, SpikeConfig, SpikeLog, EVENT_LOSS_SPIKE, EVENT_PAGE, PAGE_WINDOW_STEPS,
};

use reference::i10::{assert_actions_close, observe_pair, page_message, sample, RefSpikeLog, TOL};

fn pair(window: usize, threshold: f64, page_window: u64, ckpt: u64) -> (SpikeLog, RefSpikeLog) {
    let cfg = SpikeConfig::new(page_window, threshold, window);
    (
        SpikeLog::new(cfg.clone(), ckpt),
        RefSpikeLog::new(cfg, ckpt),
    )
}

/// Allowed to pass the stub: constructors / getters of an empty SpikeLog.
#[test]
fn empty_spike_log_has_no_spikes_or_skips() {
    let log = SpikeLog::new(SpikeConfig::new(PAGE_WINDOW_STEPS, 0.5, 8), 0);
    assert!(
        log.skipped_shards().is_empty(),
        "fresh SpikeLog skipped_shards"
    );
    assert!(log.spikes().is_empty(), "fresh SpikeLog spikes");
}

#[test]
fn warmup_returns_empty_even_for_huge_loss() {
    let (mut prod, mut refer) = pair(3, 0.01, PAGE_WINDOW_STEPS, 0);
    for step in 0..3 {
        let actions = observe_pair(
            &mut prod,
            &mut refer,
            &sample(step, 1.0e9, "shard-a", "run_alpha"),
        )
        .expect("warmup observe");
        assert!(actions.is_empty(), "warmup must not spike, got {actions:?}");
    }
    assert!(prod.spikes().is_empty());
    assert!(prod.skipped_shards().is_empty());
}

#[test]
fn first_sample_after_warmup_uses_mean_of_prior_window() {
    let (mut prod, mut refer) = pair(3, 0.5, PAGE_WINDOW_STEPS, 11);
    for (step, loss) in [(0, 1.0), (1, 2.0), (2, 3.0)] {
        let a = observe_pair(&mut prod, &mut refer, &sample(step, loss, "s", "run_alpha"))
            .expect("warmup");
        assert!(a.is_empty());
    }
    // baseline = (1+2+3)/3 = 2.0; threshold 0.5 → spike iff loss > 3.0
    let not_spike = observe_pair(&mut prod, &mut refer, &sample(3, 3.0, "s", "run_alpha"))
        .expect("equal to threshold is not a spike");
    assert!(not_spike.is_empty());

    // history is [1,2,3,3]; baseline = mean(2,3,3)=8/3; (8/3)*1.5 = 4.0; 4.0 is not a spike
    let still = observe_pair(
        &mut prod,
        &mut refer,
        &sample(4, 4.0, "shard-z", "run_alpha"),
    )
    .expect("boundary");
    assert!(still.is_empty(), "4.0 is not strictly greater than 4.0");

    // history [1,2,3,3,4]; baseline = mean(3,3,4)=10/3; spike iff loss > 5.0
    let actions = observe_pair(
        &mut prod,
        &mut refer,
        &sample(5, 5.0 + 0.5, "shard-z", "run_alpha"),
    )
    .expect("real spike");
    assert_eq!(
        actions.len(),
        3,
        "first spike: Log, SkipShard, Rollback (no Page)"
    );
    match &actions[0] {
        SpikeAction::Log { event } => {
            assert_eq!(event.step, 5);
            assert!((event.loss - 5.5).abs() <= TOL);
            assert_eq!(event.shard_id, "shard-z");
            assert_eq!(event.run_id, "run_alpha");
            assert!((event.baseline - (10.0 / 3.0)).abs() <= TOL);
        }
        other => panic!("actions[0] must be Log, got {other:?}"),
    }
    match &actions[1] {
        SpikeAction::SkipShard { shard_id } => assert_eq!(shard_id, "shard-z"),
        other => panic!("actions[1] must be SkipShard, got {other:?}"),
    }
    match &actions[2] {
        SpikeAction::Rollback { checkpoint_step } => assert_eq!(*checkpoint_step, 11),
        other => panic!("actions[2] must be Rollback, got {other:?}"),
    }
    assert_eq!(prod.spikes().len(), 1);
    assert_eq!(prod.skipped_shards().len(), 1);
    assert_eq!(prod.skipped_shards()[0].reason, EVENT_LOSS_SPIKE);
    assert_eq!(prod.skipped_shards()[0].shard_id, "shard-z");
    assert_eq!(prod.skipped_shards()[0].step, 5);
    assert_eq!(prod.skipped_shards()[0].run_id, "run_alpha");
}

#[test]
fn sliding_window_not_expanding_mean() {
    // window=2, threshold=0.1 → spike iff loss > 1.1 * baseline.
    // 1,1,2,2,2.15: last sample is NOT a spike under sliding mean(2,2)=2
    // (2.15 > 2.2? no) but WOULD be under expanding mean(1,1,2,2)=1.5 (2.15>1.65).
    let (mut prod, mut refer) = pair(2, 0.1, PAGE_WINDOW_STEPS, 0);
    let run = "run_slide";
    let losses = [
        (0, 1.0, false),
        (1, 1.0, false),
        (2, 2.0, true),   // baseline 1.0
        (3, 2.0, true),   // baseline 1.5
        (4, 2.15, false), // sliding baseline 2.0
    ];
    for (step, loss, expect_spike) in losses {
        let actions =
            observe_pair(&mut prod, &mut refer, &sample(step, loss, "sh", run)).expect("observe");
        if expect_spike {
            assert!(
                !actions.is_empty(),
                "step {step} loss {loss} should spike, got empty"
            );
        } else {
            assert!(
                actions.is_empty(),
                "step {step} loss {loss} should not spike, got {actions:?}"
            );
        }
    }
    assert_eq!(prod.spikes().len(), 2);
}

#[test]
fn baseline_zero_or_negative_is_not_a_spike() {
    let (mut prod, mut refer) = pair(2, 0.5, PAGE_WINDOW_STEPS, 0);
    observe_pair(&mut prod, &mut refer, &sample(0, 0.0, "s", "r")).unwrap();
    observe_pair(&mut prod, &mut refer, &sample(1, 0.0, "s", "r")).unwrap();
    let a = observe_pair(&mut prod, &mut refer, &sample(2, 1.0e6, "s", "r")).unwrap();
    assert!(a.is_empty(), "baseline 0 must not spike, got {a:?}");

    let (mut prod, mut refer) = pair(2, 0.5, PAGE_WINDOW_STEPS, 0);
    observe_pair(&mut prod, &mut refer, &sample(0, -2.0, "s", "r")).unwrap();
    observe_pair(&mut prod, &mut refer, &sample(1, -2.0, "s", "r")).unwrap();
    let a = observe_pair(&mut prod, &mut refer, &sample(2, 1.0e6, "s", "r")).unwrap();
    assert!(a.is_empty(), "negative baseline must not spike, got {a:?}");
}

#[test]
fn first_spike_does_not_page() {
    let (mut prod, mut refer) = pair(1, 0.5, PAGE_WINDOW_STEPS, 3);
    observe_pair(&mut prod, &mut refer, &sample(10, 1.0, "s0", "run_p")).unwrap();
    let actions = observe_pair(&mut prod, &mut refer, &sample(11, 10.0, "s1", "run_p")).unwrap();
    assert_eq!(actions.len(), 3);
    assert!(
        !actions
            .iter()
            .any(|a| matches!(a, SpikeAction::Page { .. })),
        "first spike must not page: {actions:?}"
    );
}

#[test]
fn second_spike_within_page_window_inclusive_pages() {
    let page_window = 10_u64;
    let (mut prod, mut refer) = pair(1, 0.5, page_window, 0);
    observe_pair(&mut prod, &mut refer, &sample(100, 1.0, "a", "run_p")).unwrap();
    let first = observe_pair(&mut prod, &mut refer, &sample(100, 10.0, "a", "run_p")).unwrap();
    assert_eq!(first.len(), 3, "first spike no page");

    // distance 10 inclusive → page
    observe_pair(&mut prod, &mut refer, &sample(110, 1.0, "b", "run_p")).unwrap();
    // After first spike history is [1.0, 10.0]; warmup window=1 so this 1.0 is NOT warmup.
    // Need a spike at step 110. Baseline is last 1 sample (10.0 after first spike).
    // Observing 1.0 is not a spike. Feed a huge loss at 110.
    let second = observe_pair(&mut prod, &mut refer, &sample(110, 50.0, "b", "run_p")).unwrap();
    assert!(
        second.iter().any(|a| matches!(a, SpikeAction::Page { .. })),
        "distance 10 within window 10 must page: {second:?}"
    );
    match second.last() {
        Some(SpikeAction::Page { message }) => {
            assert!(message.contains("run_p"), "page message run_id: {message}");
            assert!(message.contains("110"), "page message step: {message}");
            assert!(
                message.contains(EVENT_PAGE),
                "page message event: {message}"
            );
            assert_eq!(message, &page_message("run_p", 110));
        }
        other => panic!("last action must be Page, got {other:?}"),
    }
}

#[test]
fn second_spike_just_outside_page_window_does_not_page() {
    let page_window = 10_u64;
    let (mut prod, mut refer) = pair(1, 0.5, page_window, 0);
    observe_pair(&mut prod, &mut refer, &sample(100, 1.0, "a", "run_p")).unwrap();
    observe_pair(&mut prod, &mut refer, &sample(100, 10.0, "a", "run_p")).unwrap();
    let second = observe_pair(&mut prod, &mut refer, &sample(111, 50.0, "c", "run_p")).unwrap();
    assert_eq!(
        second.len(),
        3,
        "distance 11 > 10 must not page: {second:?}"
    );
    assert!(
        !second.iter().any(|a| matches!(a, SpikeAction::Page { .. })),
        "must not page: {second:?}"
    );
}

#[test]
fn page_uses_any_prior_spike_in_window_not_only_the_last() {
    // spikes at 0 and 5000 page; spike at 15000 is 10000 from 5000 (inclusive) so pages
    // with default 10k window. Use window 10_000, then a later spike at 20001 from 5000
    // (15001) should not page if 0 is also out of range.
    let (mut prod, mut refer) = pair(1, 0.5, 10_000, 0);
    observe_pair(&mut prod, &mut refer, &sample(0, 1.0, "s", "run_w")).unwrap();
    observe_pair(&mut prod, &mut refer, &sample(0, 10.0, "s", "run_w")).unwrap(); // spike, no page
    let mid = observe_pair(&mut prod, &mut refer, &sample(5_000, 80.0, "s", "run_w")).unwrap();
    assert!(
        mid.iter().any(|a| matches!(a, SpikeAction::Page { .. })),
        "5000 within 10k of 0: {mid:?}"
    );
    let far = observe_pair(&mut prod, &mut refer, &sample(20_001, 800.0, "s", "run_w")).unwrap();
    assert!(
        !far.iter().any(|a| matches!(a, SpikeAction::Page { .. })),
        "20001 is >10k from both 0 and 5000: {far:?}"
    );
}

#[test]
fn rollback_reads_live_last_checkpoint_step() {
    let (mut prod, mut refer) = pair(1, 0.5, PAGE_WINDOW_STEPS, 7);
    observe_pair(&mut prod, &mut refer, &sample(1, 1.0, "s", "r")).unwrap();
    let a = observe_pair(&mut prod, &mut refer, &sample(2, 10.0, "s", "r")).unwrap();
    match &a[2] {
        SpikeAction::Rollback { checkpoint_step } => assert_eq!(*checkpoint_step, 7),
        other => panic!("{other:?}"),
    }
    prod.last_checkpoint_step = 42;
    refer.last_checkpoint_step = 42;
    let b = observe_pair(&mut prod, &mut refer, &sample(3, 100.0, "s", "r")).unwrap();
    match &b[2] {
        SpikeAction::Rollback { checkpoint_step } => assert_eq!(*checkpoint_step, 42),
        other => panic!("{other:?}"),
    }
}

#[test]
fn live_page_window_is_read_from_config_each_observe() {
    let (mut prod, mut refer) = pair(1, 0.5, 0, 0);
    observe_pair(&mut prod, &mut refer, &sample(1, 1.0, "s", "r")).unwrap();
    observe_pair(&mut prod, &mut refer, &sample(1, 10.0, "s", "r")).unwrap();
    // page_window 0: only the same step would page. Next spike at step 5 would not.
    prod.config.page_window_steps = 10;
    refer.config.page_window_steps = 10;
    let a = observe_pair(&mut prod, &mut refer, &sample(5, 80.0, "s", "r")).unwrap();
    assert!(
        a.iter().any(|x| matches!(x, SpikeAction::Page { .. })),
        "after widening the window, step 5 must page: {a:?}"
    );
}

#[test]
fn action_order_is_log_skip_rollback_page() {
    let (mut prod, mut refer) = pair(1, 0.5, 100, 9);
    observe_pair(&mut prod, &mut refer, &sample(1, 1.0, "sh-1", "run_ord")).unwrap();
    observe_pair(&mut prod, &mut refer, &sample(2, 10.0, "sh-1", "run_ord")).unwrap();
    let second = observe_pair(&mut prod, &mut refer, &sample(3, 80.0, "sh-2", "run_ord")).unwrap();
    assert_eq!(second.len(), 4);
    assert!(matches!(second[0], SpikeAction::Log { .. }));
    assert!(matches!(second[1], SpikeAction::SkipShard { .. }));
    assert!(matches!(
        second[2],
        SpikeAction::Rollback { checkpoint_step: 9 }
    ));
    match &second[3] {
        SpikeAction::Page { message } => {
            assert_eq!(message, &page_message("run_ord", 3));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn property_constant_loss_never_spikes_after_warmup() {
    for window in 1..8 {
        for &threshold in &[0.1, 0.5, 2.0] {
            let (mut prod, mut refer) = pair(window, threshold, PAGE_WINDOW_STEPS, 0);
            for step in 0..(window as u64 + 12) {
                let a = observe_pair(
                    &mut prod,
                    &mut refer,
                    &sample(step, 2.5, "const", "run_prop"),
                )
                .expect("constant loss");
                assert!(
                    a.is_empty(),
                    "constant loss spiked at step {step} window {window} thr {threshold}: {a:?}"
                );
            }
            assert!(prod.spikes().is_empty());
        }
    }
}

#[test]
fn property_spike_count_matches_skips_and_events() {
    let (mut prod, mut refer) = pair(4, 0.25, 50, 1);
    let mut spikes = 0usize;
    for step in 0..40 {
        // mostly 1.0, every 7th step a 10.0 after warmup
        let loss = if step >= 4 && step % 7 == 0 {
            10.0
        } else {
            1.0
        };
        let a = observe_pair(
            &mut prod,
            &mut refer,
            &sample(step, loss, &format!("sh{step}"), "run_c"),
        )
        .expect("observe");
        if !a.is_empty() {
            spikes += 1;
            assert!(matches!(a[0], SpikeAction::Log { .. }));
            assert!(matches!(a[1], SpikeAction::SkipShard { .. }));
            assert!(matches!(a[2], SpikeAction::Rollback { .. }));
        }
    }
    assert_eq!(prod.spikes().len(), spikes);
    assert_eq!(prod.skipped_shards().len(), spikes);
    assert_eq!(refer.spikes().len(), spikes);
}

#[test]
fn window_one_baseline_is_the_previous_sample() {
    let (mut prod, mut refer) = pair(1, 1.0, PAGE_WINDOW_STEPS, 0);
    observe_pair(&mut prod, &mut refer, &sample(0, 2.0, "s", "r")).unwrap();
    // spike iff loss > 2.0 * (1+1) = 4.0
    let eq = observe_pair(&mut prod, &mut refer, &sample(1, 4.0, "s", "r")).unwrap();
    assert!(eq.is_empty(), "strict greater-than, 4.0 == 4.0: {eq:?}");
    // previous sample is now 4.0 → need loss > 8.0
    let miss = observe_pair(&mut prod, &mut refer, &sample(2, 8.0, "s", "r")).unwrap();
    assert!(miss.is_empty(), "8.0 == 8.0 is not a spike: {miss:?}");
    // previous sample is now 8.0 → need loss > 16.0
    let hit = observe_pair(&mut prod, &mut refer, &sample(3, 16.0 + 1.0, "s", "r")).unwrap();
    assert!(!hit.is_empty(), "17.0 > 8.0 * 2 must spike");
    match &hit[0] {
        SpikeAction::Log { event } => {
            assert!(
                (event.baseline - 8.0).abs() <= TOL,
                "baseline {}",
                event.baseline
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn reference_actions_match_production_on_a_mixed_stream() {
    let (mut prod, mut refer) = pair(3, 0.5, 8, 4);
    for step in 0..25 {
        let loss = 1.0 + ((step * 3) % 5) as f64 + if step == 9 || step == 12 { 20.0 } else { 0.0 };
        let a = observe_pair(&mut prod, &mut refer, &sample(step, loss, "mix", "run_mix"))
            .expect("mixed");
        let _ = a;
    }
    assert_actions_close(
        &prod
            .spikes()
            .iter()
            .map(|e| SpikeAction::Log { event: e.clone() })
            .collect::<Vec<_>>(),
        &refer
            .spikes()
            .iter()
            .map(|e| SpikeAction::Log { event: e.clone() })
            .collect::<Vec<_>>(),
    );
}
