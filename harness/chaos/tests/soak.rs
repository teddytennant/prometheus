//! Group: 72h soak of injected time, all D-gates, not wall clock.

mod common;
mod reference;

use common::{default_config, fresh_world_dir};
use prometheus_chaos::{
    run_soak, SoakConfig, World, SOAK_MS,
};
use reference::{ref_run_soak, RefWorld};
use std::time::{Duration, Instant};

#[test]
fn soak_default_is_72h_injected() {
    assert_eq!(SOAK_MS, 72 * 60 * 60 * 1000);
    assert_eq!(SoakConfig::default().duration_ms, SOAK_MS);
    assert_eq!(SoakConfig::default().seed, 0);
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    let report = run_soak(
        &mut w,
        SoakConfig {
            duration_ms: 60_000,
            seed: 0,
        },
    )
    .expect("short soak");
    assert_eq!(report.gate, prometheus_chaos::Gate::Soak);
    assert!(report.recovered);
    assert!(report.invariants.hold());
    assert!(w.now() >= 60_000, "soak advances injected time, now={}", w.now());
}

#[test]
fn soak_does_not_sleep_wall_clock() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    let cfg = SoakConfig {
        duration_ms: 3_600_000, // 1h injected
        seed: 1,
    };
    let t0 = Instant::now();
    let report = run_soak(&mut w, cfg).expect("soak");
    let wall = t0.elapsed();
    assert!(
        wall < Duration::from_secs(30),
        "soak must not wait out duration on the wall clock (elapsed {wall:?})"
    );
    assert!(w.now() >= 3_600_000);
    assert!(report.invariants.hold());
    assert!(report.faults_applied >= 1);
}

#[test]
fn soak_matches_reference() {
    let (_pw, dir_w) = fresh_world_dir();
    let (_pr, dir_r) = fresh_world_dir();
    let config = default_config();
    let soak = SoakConfig {
        duration_ms: 12_000,
        seed: 3,
    };
    let mut w = World::create(&dir_w, config.clone()).expect("w");
    let mut r = RefWorld::create(&dir_r, config).expect("r");
    let a = run_soak(&mut w, soak.clone()).expect("world");
    let b = ref_run_soak(&mut r, soak).expect("ref");
    assert_eq!(a.gate, b.gate);
    assert_eq!(a.invariants, b.invariants);
    assert_eq!(a.recovered, b.recovered);
    assert_eq!(a.faults_applied, b.faults_applied);
    assert_eq!(w.now(), r.now());
}

#[test]
fn soak_survives_every_gate_kind() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    let report = run_soak(
        &mut w,
        SoakConfig {
            duration_ms: 6_000,
            seed: 0,
        },
    )
    .expect("soak");
    assert_eq!(report.invariants.lost_tasks, 0);
    assert_eq!(report.invariants.duplicated_outputs, 0);
}
