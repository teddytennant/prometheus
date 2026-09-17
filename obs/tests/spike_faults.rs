//! Group: fault injection for SpikeLog::observe (BadSpikeConfig, non-finite).
//!
//! Every test calls `observe` and must fail on the I10 stub.

mod reference;

use prometheus_obs::{Error, SpikeConfig, SpikeLog, PAGE_WINDOW_STEPS};

use reference::i10::{observe_pair, sample, RefSpikeLog};

fn pair(window: usize, threshold: f64) -> (SpikeLog, RefSpikeLog) {
    let cfg = SpikeConfig::new(PAGE_WINDOW_STEPS, threshold, window);
    (SpikeLog::new(cfg.clone(), 0), RefSpikeLog::new(cfg, 0))
}

fn assert_bad_cfg(err: prometheus_obs::Result<Vec<prometheus_obs::SpikeAction>>, msg: &str) {
    match err {
        Err(Error::BadSpikeConfig(m)) => {
            assert_eq!(m, msg, "BadSpikeConfig message");
        }
        other => panic!("expected BadSpikeConfig({msg:?}), got {other:?}"),
    }
}

#[test]
fn relative_threshold_zero_is_bad_spike_config() {
    let (mut prod, mut refer) = pair(4, 0.0);
    let err = observe_pair(&mut prod, &mut refer, &sample(0, 1.0, "s", "r"));
    assert_bad_cfg(err, "relative_threshold must be > 0");
    assert!(prod.spikes().is_empty());
    assert!(prod.skipped_shards().is_empty());
}

#[test]
fn relative_threshold_negative_is_bad_spike_config() {
    let (mut prod, mut refer) = pair(4, -0.1);
    let err = observe_pair(&mut prod, &mut refer, &sample(0, 1.0, "s", "r"));
    assert_bad_cfg(err, "relative_threshold must be > 0");
}

#[test]
fn baseline_window_zero_is_bad_spike_config() {
    let (mut prod, mut refer) = pair(0, 0.5);
    let err = observe_pair(&mut prod, &mut refer, &sample(0, 1.0, "s", "r"));
    assert_bad_cfg(err, "baseline_window must be > 0");
}

#[test]
fn threshold_checked_before_window() {
    let cfg = SpikeConfig::new(PAGE_WINDOW_STEPS, 0.0, 0);
    let mut prod = SpikeLog::new(cfg.clone(), 0);
    let mut refer = RefSpikeLog::new(cfg, 0);
    let err = observe_pair(&mut prod, &mut refer, &sample(0, 1.0, "s", "r"));
    assert_bad_cfg(err, "relative_threshold must be > 0");
}

#[test]
fn new_does_not_validate_bad_config() {
    // Construction succeeds; observe is where BadSpikeConfig is raised.
    let _cfg = SpikeConfig::new(PAGE_WINDOW_STEPS, 0.0, 0);
    let mut prod = SpikeLog::new(SpikeConfig::new(PAGE_WINDOW_STEPS, -1.0, 0), 0);
    let mut refer = RefSpikeLog::new(SpikeConfig::new(PAGE_WINDOW_STEPS, -1.0, 0), 0);
    let err = observe_pair(&mut prod, &mut refer, &sample(1, 1.0, "s", "r"));
    assert_bad_cfg(err, "relative_threshold must be > 0");
}

#[test]
fn nan_loss_is_bad_spike_config_not_a_spike() {
    let (mut prod, mut refer) = pair(2, 0.5);
    let err = observe_pair(&mut prod, &mut refer, &sample(0, f64::NAN, "s", "r"));
    assert_bad_cfg(err, "loss must be finite");
    assert!(prod.spikes().is_empty());
    assert!(prod.skipped_shards().is_empty());
}

#[test]
fn inf_loss_is_bad_spike_config_not_a_spike() {
    let (mut prod, mut refer) = pair(2, 0.5);
    assert_bad_cfg(
        observe_pair(&mut prod, &mut refer, &sample(0, f64::INFINITY, "s", "r")),
        "loss must be finite",
    );
    assert_bad_cfg(
        observe_pair(
            &mut prod,
            &mut refer,
            &sample(1, f64::NEG_INFINITY, "s", "r"),
        ),
        "loss must be finite",
    );
}

#[test]
fn non_finite_does_not_consume_warmup_slot() {
    let (mut prod, mut refer) = pair(2, 0.5);
    let _ = observe_pair(&mut prod, &mut refer, &sample(0, f64::NAN, "s", "r"));
    let a0 = observe_pair(&mut prod, &mut refer, &sample(1, 1.0, "s", "r")).expect("first finite");
    assert!(a0.is_empty(), "still in warmup after rejected NaN");
    let a1 = observe_pair(&mut prod, &mut refer, &sample(2, 1.0, "s", "r")).expect("second finite");
    assert!(a1.is_empty(), "second warmup sample");
    // now history is two 1.0s; a huge finite loss spikes
    let spike = observe_pair(&mut prod, &mut refer, &sample(3, 10.0, "s", "r")).expect("spike");
    assert!(!spike.is_empty());
}

#[test]
fn error_leaves_spikes_and_skips_unchanged() {
    let (mut prod, mut refer) = pair(1, 0.5);
    observe_pair(&mut prod, &mut refer, &sample(0, 1.0, "s", "r")).unwrap();
    observe_pair(&mut prod, &mut refer, &sample(1, 10.0, "s", "r")).unwrap();
    assert_eq!(prod.spikes().len(), 1);

    prod.config.relative_threshold = 0.0;
    refer.config.relative_threshold = 0.0;
    let _ = observe_pair(&mut prod, &mut refer, &sample(2, 10.0, "s", "r"));
    assert_eq!(prod.spikes().len(), 1);
    assert_eq!(prod.skipped_shards().len(), 1);

    prod.config.relative_threshold = 0.5;
    refer.config.relative_threshold = 0.5;
    // history must still be [1.0, 10.0] — the failed observe did not append
    let again = observe_pair(&mut prod, &mut refer, &sample(3, 1.0, "s", "r")).expect("ok");
    assert!(
        again.is_empty(),
        "1.0 vs baseline 10.0 is not a spike; failed observe must not have appended: {again:?}"
    );
}

#[test]
fn property_all_non_finite_payloads_error_and_stay_empty() {
    for loss in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let (mut prod, mut refer) = pair(3, 0.25);
        let err = observe_pair(&mut prod, &mut refer, &sample(0, loss, "s", "r"));
        assert_bad_cfg(err, "loss must be finite");
        assert!(prod.spikes().is_empty());
        assert!(prod.skipped_shards().is_empty());
    }
}
