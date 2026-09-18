//! Group: `kill`. Writes the freeze file, snapshot.frozen is true,
//! idempotent (`AlreadyFrozen` or still frozen).

mod common;
mod reference;

use common::{
    assert_frozen_pair, assert_result_eq, assert_snap_eq, freeze_exists, fresh_world,
    load_prod_freeze, open_pair, NOW0, NOW1,
};
use prometheus_monitors::{Error, Monitors};

#[test]
fn kill_writes_freeze_file_and_sets_frozen() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    let got = prod.kill("manual-stop", NOW0);
    let exp = refer.kill("manual-stop", NOW0);
    assert_result_eq(got, exp, "kill");
    assert_frozen_pair(&prod, &refer);
    assert!(prod.frozen());
    assert!(prod.snapshot().frozen);
    assert_eq!(prod.snapshot().frozen_at, Some(NOW0));
    assert_eq!(prod.snapshot().reason.as_deref(), Some("manual-stop"));
    assert!(
        freeze_exists(&world.prod.freeze_path),
        "kill must write freeze_path"
    );
    let disk = load_prod_freeze(&world);
    assert_snap_eq(&disk, &prod.snapshot());
    assert!(disk.frozen);
}

#[test]
fn kill_does_not_append_a_violation() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    prod.kill("r", NOW0).expect("prod kill");
    refer.kill("r", NOW0).expect("ref kill");
    assert!(prod.snapshot().violations.is_empty());
    assert_snap_eq(&prod.snapshot(), &refer.snapshot());
}

#[test]
fn kill_is_idempotent() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    let first = prod.kill("first", NOW0).expect("first kill");
    assert!(first.frozen);
    let second = prod.kill("second", NOW1);
    match second {
        Err(Error::AlreadyFrozen) => {
            assert!(prod.frozen());
            assert_eq!(prod.snapshot().frozen_at, Some(NOW0));
            assert_eq!(prod.snapshot().reason.as_deref(), Some("first"));
        }
        Ok(s) => {
            assert!(s.frozen && prod.frozen());
            assert_eq!(s.frozen_at, Some(NOW0), "first frozen_at must win");
            assert_eq!(s.reason.as_deref(), Some("first"));
        }
        other => panic!("expected AlreadyFrozen or still frozen, got {other:?}"),
    }
    let exp = refer.kill("first", NOW0).expect("ref first");
    assert!(exp.frozen);
    match refer.kill("second", NOW1) {
        Err(Error::AlreadyFrozen) => assert!(refer.frozen()),
        Ok(s) => assert!(s.frozen),
        other => panic!("reference second kill: {other:?}"),
    }
}

#[test]
fn kill_creates_missing_parent_dirs() {
    let world = fresh_world();
    let mut cfg = world.prod.clone();
    cfg.freeze_path = world
        .tmp
        .path()
        .join("nested")
        .join("deep")
        .join("kill.freeze");
    let mut prod = Monitors::open(cfg.clone()).expect("open");
    prod.kill("nested", NOW0).expect("kill nested");
    assert!(cfg.freeze_path.is_file());
    assert!(prod.frozen());
}

#[test]
fn freeze_file_is_kill_snapshot_json() {
    let world = fresh_world();
    let (mut prod, _) = open_pair(&world);
    prod.kill("json", NOW0).expect("kill");
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&world.prod.freeze_path).unwrap()).unwrap();
    assert_eq!(v["frozen"], serde_json::json!(true));
    assert_eq!(v["frozen_at"], serde_json::json!(NOW0));
    assert_eq!(v["reason"], serde_json::json!("json"));
    assert!(v["violations"].as_array().unwrap().is_empty());
}

#[test]
fn kill_after_trip_is_already_frozen() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    let _ = prod.plant(prometheus_monitors::Boundary::Kernel, "cap", NOW0);
    let _ = refer.plant(prometheus_monitors::Boundary::Kernel, "cap", NOW0);
    assert_result_eq(
        prod.kill("late", NOW1),
        refer.kill("late", NOW1),
        "late kill",
    );
    assert_eq!(prod.snapshot().frozen_at, Some(NOW0));
    assert_eq!(
        prod.snapshot().reason.as_deref(),
        Some(reference::freeze_reason(
            prometheus_monitors::Boundary::Kernel
        ))
    );
}
