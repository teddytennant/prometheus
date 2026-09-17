//! Group: Metrics counters, gauges, histograms, labels, KindMismatch.
//!
//! Histogram rule (locked): `observe` records samples; `get` returns the **sum**
//! of observed values for that `(name, labels)` series.

mod common;

use prometheus_obs::Metrics;

use common::{
    assert_kind_mismatch, assert_some_f64, empty_labels, labels, METRIC_TOL,
};

#[test]
fn get_missing_name_is_ok_none() {
    let m = Metrics::new();
    match m.get("train.loss.ce", &empty_labels()) {
        Ok(None) => {}
        other => panic!("expected Ok(None) for missing series, got {other:?}"),
    }
}

#[test]
fn inc_accumulates_on_same_series() {
    let mut m = Metrics::new();
    let l = empty_labels();
    m.inc("tokens", &l, 1.0).expect("inc 1");
    m.inc("tokens", &l, 2.0).expect("inc 2");
    m.inc("tokens", &l, 0.5).expect("inc 0.5");
    assert_some_f64(m.get("tokens", &l), 3.5);
}

#[test]
fn inc_zero_creates_series_at_zero() {
    let mut m = Metrics::new();
    let l = empty_labels();
    m.inc("empty.counter", &l, 0.0).expect("inc 0");
    assert_some_f64(m.get("empty.counter", &l), 0.0);
}

#[test]
fn set_overwrites_gauge() {
    let mut m = Metrics::new();
    let l = empty_labels();
    m.set("lr", &l, 1.0).expect("set 1");
    assert_some_f64(m.get("lr", &l), 1.0);
    m.set("lr", &l, 9.25).expect("set 9.25");
    assert_some_f64(m.get("lr", &l), 9.25);
}

#[test]
fn observe_histogram_get_returns_sum() {
    let mut m = Metrics::new();
    let l = empty_labels();
    m.observe("step.ms", &l, 1.5).expect("observe 1.5");
    assert_some_f64(m.get("step.ms", &l), 1.5);
    m.observe("step.ms", &l, 2.5).expect("observe 2.5");
    m.observe("step.ms", &l, 0.0).expect("observe 0");
    assert_some_f64(m.get("step.ms", &l), 4.0);
}

#[test]
fn labels_distinguish_series_of_the_same_kind() {
    let mut m = Metrics::new();
    let gpu0 = labels(&[("gpu", "0")]);
    let gpu1 = labels(&[("gpu", "1")]);
    let unlabeled = empty_labels();

    m.inc("tokens", &gpu0, 1.0).expect("gpu0");
    m.inc("tokens", &gpu1, 5.0).expect("gpu1");
    m.inc("tokens", &unlabeled, 7.0).expect("unlabeled");

    assert_some_f64(m.get("tokens", &gpu0), 1.0);
    assert_some_f64(m.get("tokens", &gpu1), 5.0);
    assert_some_f64(m.get("tokens", &unlabeled), 7.0);

    match m.get("tokens", &labels(&[("gpu", "2")])) {
        Ok(None) => {}
        other => panic!("unseen label set must be Ok(None), got {other:?}"),
    }
}

#[test]
fn label_key_order_does_not_split_a_series() {
    let mut m = Metrics::new();
    let a = labels(&[("job", "train"), ("gpu", "0")]);
    let b = labels(&[("gpu", "0"), ("job", "train")]);
    m.inc("n", &a, 3.0).expect("inc a");
    m.inc("n", &b, 4.0).expect("inc b");
    assert_some_f64(m.get("n", &a), 7.0);
    assert_some_f64(m.get("n", &b), 7.0);
}

#[test]
fn inc_then_set_same_name_is_kind_mismatch() {
    let mut m = Metrics::new();
    let l = empty_labels();
    m.inc("loss", &l, 1.0).expect("inc");
    assert_kind_mismatch(m.set("loss", &l, 2.0), "loss");
    // original counter series is unchanged
    assert_some_f64(m.get("loss", &l), 1.0);
}

#[test]
fn set_then_observe_same_name_is_kind_mismatch() {
    let mut m = Metrics::new();
    let l = empty_labels();
    m.set("temp", &l, 0.8).expect("set");
    assert_kind_mismatch(m.observe("temp", &l, 1.0), "temp");
    assert_some_f64(m.get("temp", &l), 0.8);
}

#[test]
fn observe_then_inc_same_name_is_kind_mismatch() {
    let mut m = Metrics::new();
    let l = empty_labels();
    m.observe("lat", &l, 4.0).expect("observe");
    assert_kind_mismatch(m.inc("lat", &l, 1.0), "lat");
    assert_some_f64(m.get("lat", &l), 4.0);
}

#[test]
fn kind_is_per_name_even_when_labels_differ() {
    let mut m = Metrics::new();
    m.inc("mixed", &labels(&[("a", "1")]), 1.0)
        .expect("inc labeled");
    assert_kind_mismatch(m.set("mixed", &empty_labels(), 0.0), "mixed");
    assert_kind_mismatch(m.observe("mixed", &labels(&[("a", "2")]), 9.0), "mixed");
}

#[test]
fn different_names_may_use_different_kinds() {
    let mut m = Metrics::new();
    let l = empty_labels();
    m.inc("a.counter", &l, 1.0).expect("inc");
    m.set("b.gauge", &l, 2.0).expect("set");
    m.observe("c.hist", &l, 3.0).expect("observe");
    assert_some_f64(m.get("a.counter", &l), 1.0);
    assert_some_f64(m.get("b.gauge", &l), 2.0);
    assert_some_f64(m.get("c.hist", &l), 3.0);
}

#[test]
fn property_independent_names_do_not_interfere() {
    let mut m = Metrics::new();
    let l = empty_labels();
    for i in 0..8 {
        let name = format!("n.{i}");
        m.inc(&name, &l, i as f64 + 1.0).expect("inc");
    }
    for i in 0..8 {
        let name = format!("n.{i}");
        assert_some_f64(m.get(&name, &l), i as f64 + 1.0);
    }
}

#[test]
fn get_tolerance_is_documented_1e_5() {
    // Sanity that the helper uses METRIC_TOL; still goes through Metrics.
    let mut m = Metrics::new();
    let l = empty_labels();
    m.set("x", &l, 1.0).expect("set");
    match m.get("x", &l) {
        Ok(Some(v)) => {
            assert!((v - 1.0).abs() <= METRIC_TOL);
        }
        other => panic!("expected Some(1.0), got {other:?}"),
    }
}
