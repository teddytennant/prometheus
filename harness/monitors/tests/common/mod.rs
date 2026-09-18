//! Shared fixtures and assertions for the L3 monitors oracle.
//!
//! Production `src/` must never import `tests/`. This module is not
//! `prometheus_monitors::Monitors`.
//!
//! CPU-only. No `gpu` marker. `NowMs` is injected; nothing may call
//! `SystemTime` / `Instant` for timestamps, and nothing may `sleep`.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::path::Path;

use prometheus_monitors::{
    Boundary, Error, KillSnapshot, MonitorConfig, Monitors, NowMs, Result, Violation,
};
use tempfile::TempDir;

use crate::reference::RefMonitors;

pub const NOW0: NowMs = 1_700_000_042;
pub const NOW1: NowMs = 1_700_000_099;
pub const NOW_MAX: NowMs = u64::MAX;

pub const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
pub const HASH_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
pub const GRADER_ID: &str = "grader-arc";

pub const ALL_BOUNDARIES: [Boundary; 4] = [
    Boundary::Kernel,
    Boundary::Graders,
    Boundary::HeldOut,
    Boundary::Monitors,
];

pub struct World {
    pub tmp: TempDir,
    pub prod: MonitorConfig,
    pub refer: MonitorConfig,
}

pub fn fresh_world() -> World {
    world_with_hashes(BTreeMap::from([(
        GRADER_ID.to_string(),
        HASH_A.to_string(),
    )]))
}

pub fn world_with_hashes(hashes: BTreeMap<String, String>) -> World {
    let tmp = tempfile::tempdir().expect("tempdir");
    let prod = MonitorConfig {
        freeze_path: tmp.path().join("prod.freeze"),
        grader_hashes: hashes.clone(),
    };
    let refer = MonitorConfig {
        freeze_path: tmp.path().join("ref.freeze"),
        grader_hashes: hashes,
    };
    World { tmp, prod, refer }
}

pub fn open_pair(world: &World) -> (Monitors, RefMonitors) {
    (
        Monitors::open(world.prod.clone()).expect("prod open"),
        RefMonitors::open(world.refer.clone()).expect("ref open"),
    )
}

pub fn unwrap_err<T: Debug>(r: Result<T>, ctx: &str) -> Error {
    match r {
        Err(e) => e,
        Ok(v) => panic!("{ctx}: expected Err, got Ok({v:?})"),
    }
}

pub fn assert_snap_eq(got: &KillSnapshot, exp: &KillSnapshot) {
    assert_eq!(got.frozen, exp.frozen, "frozen");
    assert_eq!(got.frozen_at, exp.frozen_at, "frozen_at");
    assert_eq!(got.reason, exp.reason, "reason");
    assert_eq!(got.violations, exp.violations, "violations");
}

pub fn assert_result_eq<T: Debug + PartialEq>(got: Result<T>, exp: Result<T>, ctx: &str) {
    match (got, exp) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "{ctx}: Ok mismatch"),
        (Err(a), Err(b)) => assert_eq!(a, b, "{ctx}: Err mismatch"),
        (a, b) => panic!("{ctx}: prod {a:?} reference {b:?}"),
    }
}

pub fn assert_frozen_pair(prod: &Monitors, refer: &RefMonitors) {
    assert!(prod.frozen(), "prod should be frozen");
    assert!(refer.frozen(), "reference should be frozen");
    assert_eq!(prod.frozen(), prod.snapshot().frozen);
    assert_eq!(refer.frozen(), refer.snapshot().frozen);
    assert_snap_eq(&prod.snapshot(), &refer.snapshot());
}

pub fn assert_unfrozen_pair(prod: &Monitors, refer: &RefMonitors) {
    assert!(!prod.frozen(), "prod should not be frozen");
    assert!(!refer.frozen(), "reference should not be frozen");
    assert_snap_eq(&prod.snapshot(), &refer.snapshot());
}

pub fn assert_has_violation(snap: &KillSnapshot, boundary: Boundary, now: NowMs, detail: &str) {
    let hit = snap
        .violations
        .iter()
        .any(|v| v.boundary == boundary && v.at_ms == now && v.detail == detail);
    assert!(
        hit,
        "missing violation boundary={boundary:?} at_ms={now} detail={detail:?} in {snap:?}"
    );
}

pub fn load_prod_freeze(world: &World) -> KillSnapshot {
    let bytes = std::fs::read(&world.prod.freeze_path).expect("read prod freeze");
    serde_json::from_slice(&bytes).expect("parse prod freeze")
}

pub fn freeze_exists(path: &Path) -> bool {
    path.is_file()
}

pub fn write_snap(path: &Path, snap: &KillSnapshot) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(path, serde_json::to_vec(snap).expect("serde")).expect("write freeze");
}

pub fn planted_violation(boundary: Boundary, now: NowMs, detail: &str) -> Violation {
    Violation {
        boundary,
        at_ms: now,
        detail: detail.to_string(),
    }
}
