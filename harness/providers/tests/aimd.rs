//! Group: AIMD grow / shrink / clamp vs the independent reference.

mod common;
mod reference;

use common::aimd_cfg;
use prometheus_providers::Aimd;
use reference::RefAimd;

fn pair(min: u32, max: u32, inc: u32, dec: f64, lat: u64) -> (Aimd, RefAimd) {
    let cfg = aimd_cfg(min, max, inc, dec, lat);
    (Aimd::new(cfg.clone()), RefAimd::new(cfg))
}

fn assert_windows(prod: &Aimd, refer: &RefAimd, want: u32) {
    assert_eq!(prod.window(), want, "production window");
    assert_eq!(refer.window(), want, "reference window");
    assert_eq!(prod.window(), refer.window());
}

#[test]
fn on_success_grows_by_increase() {
    let (mut prod, mut refer) = pair(1, 32, 1, 0.5, 1_000);
    prod.on_success(10);
    refer.on_success(10);
    assert_windows(&prod, &refer, 2);
    prod.on_success(10);
    refer.on_success(10);
    assert_windows(&prod, &refer, 3);
}

#[test]
fn on_success_clamps_at_max_window() {
    let (mut prod, mut refer) = pair(1, 4, 1, 0.5, 1_000);
    for _ in 0..8 {
        prod.on_success(1);
        refer.on_success(1);
    }
    assert_windows(&prod, &refer, 4);
}

#[test]
fn on_success_increase_two_then_clamp() {
    let (mut prod, mut refer) = pair(1, 5, 2, 0.5, 1_000);
    prod.on_success(1);
    refer.on_success(1);
    assert_windows(&prod, &refer, 3);
    prod.on_success(1);
    refer.on_success(1);
    assert_windows(&prod, &refer, 5);
    prod.on_success(1);
    refer.on_success(1);
    assert_windows(&prod, &refer, 5);
}

#[test]
fn on_429_halves_with_floor() {
    let (mut prod, mut refer) = pair(1, 32, 1, 0.5, 1_000);
    for _ in 0..4 {
        prod.on_success(1);
        refer.on_success(1);
    }
    assert_windows(&prod, &refer, 5);
    prod.on_429();
    refer.on_429();
    // floor(5 * 0.5) = 2
    assert_windows(&prod, &refer, 2);
    prod.on_429();
    refer.on_429();
    // floor(2 * 0.5) = 1
    assert_windows(&prod, &refer, 1);
}

#[test]
fn on_429_clamps_at_min_window() {
    let (mut prod, mut refer) = pair(3, 16, 1, 0.5, 1_000);
    prod.on_success(1);
    refer.on_success(1);
    assert_windows(&prod, &refer, 4);
    prod.on_429();
    refer.on_429();
    // floor(4 * 0.5) = 2, clamp min 3
    assert_windows(&prod, &refer, 3);
    prod.on_429();
    refer.on_429();
    assert_windows(&prod, &refer, 3);
}

#[test]
fn on_outage_matches_on_429() {
    let cfg = aimd_cfg(1, 32, 1, 0.5, 1_000);
    let mut a = Aimd::new(cfg.clone());
    let mut b = Aimd::new(cfg.clone());
    let mut ra = RefAimd::new(cfg.clone());
    let mut rb = RefAimd::new(cfg);
    for _ in 0..7 {
        a.on_success(1);
        b.on_success(1);
        ra.on_success(1);
        rb.on_success(1);
    }
    a.on_outage();
    b.on_429();
    ra.on_outage();
    rb.on_429();
    assert_eq!(a.window(), b.window());
    assert_eq!(a.window(), ra.window());
    assert_eq!(b.window(), rb.window());
    assert_eq!(a.window(), 4); // start 1 + 7 = 8, floor(8*0.5)=4
}

#[test]
fn latency_above_limit_is_congestion_like_429() {
    let (mut prod, mut refer) = pair(1, 32, 1, 0.5, 100);
    for _ in 0..7 {
        prod.on_success(50);
        refer.on_success(50);
    }
    assert_windows(&prod, &refer, 8);
    prod.on_success(101);
    refer.on_success(101);
    assert_windows(&prod, &refer, 4);
}

#[test]
fn latency_equal_to_limit_is_success_not_congestion() {
    let (mut prod, mut refer) = pair(1, 32, 1, 0.5, 100);
    prod.on_success(100);
    refer.on_success(100);
    assert_windows(&prod, &refer, 2);
}

#[test]
fn decrease_at_least_one_still_subtracts_one() {
    let (mut prod, mut refer) = pair(1, 32, 1, 1.0, 1_000);
    for _ in 0..4 {
        prod.on_success(1);
        refer.on_success(1);
    }
    assert_windows(&prod, &refer, 5);
    prod.on_429();
    refer.on_429();
    assert_windows(&prod, &refer, 4);
}

#[test]
fn decrease_above_one_still_subtracts_one() {
    let (mut prod, mut refer) = pair(1, 32, 1, 2.0, 1_000);
    for _ in 0..9 {
        prod.on_success(1);
        refer.on_success(1);
    }
    assert_windows(&prod, &refer, 10);
    prod.on_429();
    refer.on_429();
    assert_windows(&prod, &refer, 9);
    prod.on_outage();
    refer.on_outage();
    assert_windows(&prod, &refer, 8);
}

#[test]
fn odd_window_floor_multiply() {
    let (mut prod, mut refer) = pair(1, 32, 2, 0.5, 1_000);
    prod.on_success(1);
    refer.on_success(1);
    assert_windows(&prod, &refer, 3);
    prod.on_429();
    refer.on_429();
    // floor(3 * 0.5) = 1
    assert_windows(&prod, &refer, 1);
}

#[test]
fn congestion_uses_same_decrease_ge_one_rule() {
    let (mut prod, mut refer) = pair(1, 32, 1, 1.0, 50);
    prod.on_success(10);
    refer.on_success(10);
    assert_windows(&prod, &refer, 2);
    prod.on_success(51);
    refer.on_success(51);
    assert_windows(&prod, &refer, 1);
}
