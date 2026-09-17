//! Group: injected clock, advance, ClockSkew bounds, D2 ±30s.

mod common;
mod reference;

use common::{assert_clock_skew, assert_no_replica, default_config, fresh_world_dir, replica};
use prometheus_chaos::{faults_for, Fault, Gate, World, CLOCK_SKEW_MS};
use reference::ref_faults_for;
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn now_starts_at_zero_advance_is_injected() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    assert_eq!(w.now(), 0);
    w.advance(50);
    assert_eq!(w.now(), 50);
    w.advance(10);
    assert_eq!(w.now(), 60);
    let wall = Instant::now();
    thread::sleep(Duration::from_millis(40));
    assert!(wall.elapsed() >= Duration::from_millis(20));
    assert_eq!(w.now(), 60, "nothing reads the wall clock");
}

#[test]
fn clock_skew_at_bound_is_ok() {
    assert_eq!(CLOCK_SKEW_MS, 30_000);
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.inject(&Fault::ClockSkew {
        replica: replica("0"),
        delta_ms: CLOCK_SKEW_MS,
    })
    .expect("+30s allowed");
    w.inject(&Fault::ClockSkew {
        replica: replica("1"),
        delta_ms: -CLOCK_SKEW_MS,
    })
    .expect("-30s allowed");
}

#[test]
fn clock_skew_over_bound_fails() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    let err = w
        .inject(&Fault::ClockSkew {
            replica: replica("0"),
            delta_ms: CLOCK_SKEW_MS + 1,
        })
        .expect_err("+30001");
    assert_clock_skew(&err, CLOCK_SKEW_MS + 1);
    let err = w
        .inject(&Fault::ClockSkew {
            replica: replica("0"),
            delta_ms: -(CLOCK_SKEW_MS + 1),
        })
        .expect_err("-30001");
    assert_clock_skew(&err, -(CLOCK_SKEW_MS + 1));
}

#[test]
fn clock_skew_unknown_replica() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    let err = w
        .inject(&Fault::ClockSkew {
            replica: replica("nope"),
            delta_ms: 1,
        })
        .expect_err("unknown");
    assert_no_replica(&err, "nope");
}

#[test]
fn d2_faults_include_plus_minus_30s() {
    let fs = faults_for(Gate::D2);
    assert_eq!(fs, ref_faults_for(Gate::D2));
    let plus = fs.iter().any(|f| {
        matches!(
            f,
            Fault::ClockSkew {
                delta_ms, ..
            } if *delta_ms == CLOCK_SKEW_MS
        )
    });
    let minus = fs.iter().any(|f| {
        matches!(
            f,
            Fault::ClockSkew {
                delta_ms, ..
            } if *delta_ms == -CLOCK_SKEW_MS
        )
    });
    assert!(plus && minus, "D2 includes clock skew of ±30s: {fs:?}");
}

#[test]
fn enqueue_uses_injected_now_not_wall() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.advance(12_345);
    let now = w.now();
    w.enqueue_task(&common::task("t"), now).unwrap();
    assert_eq!(w.now(), 12_345);
}
