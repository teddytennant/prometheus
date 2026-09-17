//! Group: Horizon::raise_if doubling vs the reference. Must fail on the stub.

mod common;
mod reference;

use prometheus_tasks::{Horizon, RAISE_THRESHOLD, START_TOOL_CALLS};

fn pin_raise(start: u32, rate: f64) {
    let mut h = Horizon::new();
    // Drive from the public start, then apply the same number of no-ops as needed
    // by reconstructing via successive raises from START_TOOL_CALLS only when
    // `start` is the default. Tests that need a raised cap first raise above 0.5.
    assert_eq!(h.tool_calls(), START_TOOL_CALLS);
    if start != START_TOOL_CALLS {
        panic!("tests must start from START_TOOL_CALLS; got {start}");
    }
    let want = reference::raise_tool_calls(h.tool_calls(), rate);
    h.raise_if(rate);
    assert_eq!(h.tool_calls(), want, "rate={rate}");
}

#[test]
fn equal_to_threshold_does_not_raise() {
    pin_raise(START_TOOL_CALLS, RAISE_THRESHOLD);
    let mut h = Horizon::new();
    h.raise_if(0.5);
    assert_eq!(h.tool_calls(), START_TOOL_CALLS);
}

#[test]
fn below_threshold_is_noop() {
    pin_raise(START_TOOL_CALLS, 0.0);
    pin_raise(START_TOOL_CALLS, 0.49);
    pin_raise(START_TOOL_CALLS, -1.0);
}

#[test]
fn above_threshold_doubles() {
    let mut h = Horizon::new();
    h.raise_if(0.51);
    assert_eq!(h.tool_calls(), START_TOOL_CALLS * 2);
    assert_eq!(h.tool_calls(), 20);
}

#[test]
fn doubling_is_pinned_10_20_40() {
    let mut h = Horizon::new();
    assert_eq!(h.tool_calls(), 10);
    h.raise_if(0.75);
    assert_eq!(h.tool_calls(), 20);
    h.raise_if(0.75);
    assert_eq!(h.tool_calls(), 40);
    h.raise_if(0.5);
    assert_eq!(h.tool_calls(), 40);
    h.raise_if(1.0);
    assert_eq!(h.tool_calls(), 80);
}

#[test]
fn nan_does_not_raise_inf_does() {
    let mut h = Horizon::new();
    h.raise_if(f64::NAN);
    assert_eq!(h.tool_calls(), START_TOOL_CALLS);
    h.raise_if(f64::NEG_INFINITY);
    assert_eq!(h.tool_calls(), START_TOOL_CALLS);
    h.raise_if(f64::INFINITY);
    assert_eq!(h.tool_calls(), START_TOOL_CALLS * 2);
}

#[test]
fn matches_reference_on_a_grid() {
    for &rate in &[
        -1.0,
        0.0,
        0.5,
        0.5000000000000001,
        0.51,
        1.0,
        2.0,
        f64::NAN,
        f64::INFINITY,
    ] {
        let mut h = Horizon::new();
        let want = reference::raise_tool_calls(h.tool_calls(), rate);
        h.raise_if(rate);
        assert_eq!(h.tool_calls(), want, "rate={rate}");
    }
}
