//! Group: MetricLog record / series vs the independent reference.
//!
//! Checks: append, same-step overwrite, sort-by-step, labels (empty is distinct),
//! EmptyHistory on a fresh log, UnknownMetric for unknown name+labels.
//! Every test calls `record` and/or `series` and must fail on the I10 stub.

mod reference;

use prometheus_obs::{Error, MetricLog, SeriesPoint};

use reference::i10::{
    assert_points_close, empty_labels, labels, record_pair, series_pair, RefMetricLog, TOL,
};

#[test]
fn series_on_empty_log_is_empty_history() {
    let prod = MetricLog::new();
    let refer = RefMetricLog::new();
    match series_pair(&prod, &refer, "train.loss", &empty_labels()) {
        Err(Error::EmptyHistory) => {}
        other => panic!("expected EmptyHistory, got {other:?}"),
    }
}

#[test]
fn record_then_series_returns_the_point() {
    let mut prod = MetricLog::new();
    let mut refer = RefMetricLog::new();
    record_pair(
        &mut prod,
        &mut refer,
        "train.loss",
        &empty_labels(),
        3,
        1.25,
    );
    let pts = series_pair(&prod, &refer, "train.loss", &empty_labels()).expect("series");
    assert_eq!(pts.len(), 1);
    assert_eq!(pts[0].step, 3);
    assert!((pts[0].value - 1.25).abs() <= TOL);
}

#[test]
fn multiple_steps_are_kept_and_sorted() {
    let mut prod = MetricLog::new();
    let mut refer = RefMetricLog::new();
    let l = empty_labels();
    record_pair(&mut prod, &mut refer, "loss", &l, 10, 4.0);
    record_pair(&mut prod, &mut refer, "loss", &l, 2, 1.0);
    record_pair(&mut prod, &mut refer, "loss", &l, 5, 2.5);
    let pts = series_pair(&prod, &refer, "loss", &l).expect("series");
    assert_eq!(
        pts.iter().map(|p| p.step).collect::<Vec<_>>(),
        vec![2, 5, 10]
    );
    assert_points_close(
        &pts,
        &[
            SeriesPoint {
                step: 2,
                value: 1.0,
            },
            SeriesPoint {
                step: 5,
                value: 2.5,
            },
            SeriesPoint {
                step: 10,
                value: 4.0,
            },
        ],
    );
}

#[test]
fn same_step_overwrites() {
    let mut prod = MetricLog::new();
    let mut refer = RefMetricLog::new();
    let l = empty_labels();
    record_pair(&mut prod, &mut refer, "loss", &l, 7, 1.0);
    record_pair(&mut prod, &mut refer, "loss", &l, 7, 9.5);
    record_pair(&mut prod, &mut refer, "loss", &l, 8, 2.0);
    record_pair(&mut prod, &mut refer, "loss", &l, 7, 3.25);
    let pts = series_pair(&prod, &refer, "loss", &l).expect("series");
    assert_eq!(pts.len(), 2);
    assert_eq!(pts[0].step, 7);
    assert!((pts[0].value - 3.25).abs() <= TOL);
    assert_eq!(pts[1].step, 8);
}

#[test]
fn labels_distinguish_series_empty_is_distinct() {
    let mut prod = MetricLog::new();
    let mut refer = RefMetricLog::new();
    let unlabeled = empty_labels();
    let gpu0 = labels(&[("gpu", "0")]);
    let gpu1 = labels(&[("gpu", "1")]);
    record_pair(&mut prod, &mut refer, "tokens", &unlabeled, 1, 7.0);
    record_pair(&mut prod, &mut refer, "tokens", &gpu0, 1, 1.0);
    record_pair(&mut prod, &mut refer, "tokens", &gpu1, 1, 5.0);
    let u = series_pair(&prod, &refer, "tokens", &unlabeled).expect("unlabeled");
    let a = series_pair(&prod, &refer, "tokens", &gpu0).expect("gpu0");
    let b = series_pair(&prod, &refer, "tokens", &gpu1).expect("gpu1");
    assert!((u[0].value - 7.0).abs() <= TOL);
    assert!((a[0].value - 1.0).abs() <= TOL);
    assert!((b[0].value - 5.0).abs() <= TOL);
}

#[test]
fn unknown_name_after_other_records_is_unknown_metric() {
    let mut prod = MetricLog::new();
    let mut refer = RefMetricLog::new();
    record_pair(&mut prod, &mut refer, "train.loss", &empty_labels(), 0, 1.0);
    match series_pair(&prod, &refer, "missing.metric", &empty_labels()) {
        Err(Error::UnknownMetric(n)) => assert_eq!(n, "missing.metric"),
        other => panic!("expected UnknownMetric(missing.metric), got {other:?}"),
    }
}

#[test]
fn unknown_labels_on_a_known_name_is_unknown_metric() {
    let mut prod = MetricLog::new();
    let mut refer = RefMetricLog::new();
    record_pair(
        &mut prod,
        &mut refer,
        "train.loss",
        &labels(&[("run", "a")]),
        0,
        1.0,
    );
    match series_pair(&prod, &refer, "train.loss", &empty_labels()) {
        Err(Error::UnknownMetric(n)) => assert_eq!(n, "train.loss"),
        other => panic!("expected UnknownMetric(train.loss) for empty labels, got {other:?}"),
    }
    match series_pair(&prod, &refer, "train.loss", &labels(&[("run", "b")])) {
        Err(Error::UnknownMetric(n)) => assert_eq!(n, "train.loss"),
        other => panic!("expected UnknownMetric(train.loss) for other labels, got {other:?}"),
    }
}

#[test]
fn names_are_independent_series() {
    let mut prod = MetricLog::new();
    let mut refer = RefMetricLog::new();
    let l = empty_labels();
    record_pair(&mut prod, &mut refer, "a", &l, 1, 1.0);
    record_pair(&mut prod, &mut refer, "b", &l, 1, 2.0);
    record_pair(&mut prod, &mut refer, "a", &l, 2, 3.0);
    let a = series_pair(&prod, &refer, "a", &l).expect("a");
    let b = series_pair(&prod, &refer, "b", &l).expect("b");
    assert_eq!(a.len(), 2);
    assert_eq!(b.len(), 1);
    assert!((b[0].value - 2.0).abs() <= TOL);
}

#[test]
fn property_last_write_wins_and_sort() {
    let mut prod = MetricLog::new();
    let mut refer = RefMetricLog::new();
    let l = empty_labels();
    // deterministic pseudo-shuffle of steps
    let steps = [7u64, 1, 4, 9, 1, 4, 3, 9, 2];
    for (i, step) in steps.iter().enumerate() {
        record_pair(
            &mut prod,
            &mut refer,
            "m",
            &l,
            *step,
            *step as f64 + i as f64 * 0.01,
        );
    }
    let pts = series_pair(&prod, &refer, "m", &l).expect("series");
    let mut seen = pts.iter().map(|p| p.step).collect::<Vec<_>>();
    let mut sorted = seen.clone();
    sorted.sort_unstable();
    assert_eq!(seen, sorted, "series must be sorted by step");
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(pts.len(), seen.len(), "duplicate steps must be overwritten");
}

#[test]
fn record_accepts_non_finite_values() {
    let mut prod = MetricLog::new();
    let mut refer = RefMetricLog::new();
    record_pair(
        &mut prod,
        &mut refer,
        "weird",
        &empty_labels(),
        1,
        f64::INFINITY,
    );
    let pts = series_pair(&prod, &refer, "weird", &empty_labels()).expect("series");
    assert_eq!(pts.len(), 1);
    assert!(pts[0].value.is_infinite());
}
