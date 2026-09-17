//! Group: Fault::Kill, recover, D0 random kills.

mod common;
mod reference;

use common::{
    assert_no_process, coordinator, default_config, fresh_world_dir, process, replica, task,
};
use prometheus_chaos::{
    run_gate, Fault, Gate, World, D0_KILLS, EVENT_ENQUEUED, EVENT_FAULT, EVENT_RECOVER,
};
use reference::{ref_run_gate, RefWorld};

#[test]
fn kill_unknown_process_is_no_process() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    let err = w
        .inject(&Fault::Kill {
            process: process("no-such-proc"),
        })
        .expect_err("unknown");
    assert_no_process(&err, "no-such-proc");
}

#[test]
fn kill_drops_in_memory_handle_recover_reopens() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.enqueue_task(&task("t"), w.now()).unwrap();
    w.inject(&Fault::Kill {
        process: coordinator(),
    })
    .unwrap();
    assert!(
        w.replica_log(&replica("0")).is_err(),
        "replica_log must fail after Kill drops the handle"
    );
    w.recover().unwrap();
    let log = w.replica_log(&replica("0")).expect("reopened");
    assert!(log.iter().any(|e| e.event_type == EVENT_ENQUEUED));
    assert!(log.iter().any(|e| e.event_type == EVENT_RECOVER));
    assert!(log.iter().any(|e| e.event_type == EVENT_FAULT));
}

#[test]
fn enqueue_after_kill_without_recover_fails() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.inject(&Fault::Kill {
        process: coordinator(),
    })
    .unwrap();
    assert!(
        w.enqueue_task(&task("t"), w.now()).is_err(),
        "enqueue with all handles dropped"
    );
    w.recover().unwrap();
    w.enqueue_task(&task("t"), w.now()).expect("after recover");
}

#[test]
fn recover_does_not_lose_enqueued_task() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.enqueue_task(&task("t"), w.now()).unwrap();
    w.inject(&Fault::Kill {
        process: coordinator(),
    })
    .unwrap();
    w.recover().unwrap();
    w.complete_task(&task("t"), 1, b"ok", w.now()).unwrap();
    let inv = w.invariants().unwrap();
    assert!(inv.hold(), "kill+recover must not lose a durable task");
}

#[test]
fn run_gate_d0_applies_d0_kills() {
    assert_eq!(D0_KILLS, 1_000);
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    let report = run_gate(&mut w, Gate::D0).expect("D0");
    assert_eq!(report.gate, Gate::D0);
    assert_eq!(report.faults_applied, D0_KILLS);
    assert!(report.recovered);
    assert!(report.invariants.hold());
}

#[test]
fn run_gate_d0_matches_reference() {
    let (_pw, dir_w) = fresh_world_dir();
    let (_pr, dir_r) = fresh_world_dir();
    let config = default_config();
    let mut w = World::create(&dir_w, config.clone()).expect("w");
    let mut r = RefWorld::create(&dir_r, config).expect("r");
    let a = run_gate(&mut w, Gate::D0).expect("world D0");
    let b = ref_run_gate(&mut r, Gate::D0).expect("ref D0");
    assert_eq!(a.gate, b.gate);
    assert_eq!(a.faults_applied, b.faults_applied);
    assert_eq!(a.invariants, b.invariants);
    assert_eq!(a.recovered, b.recovered);
    assert_eq!(
        w.log_len(&replica("0")).unwrap(),
        r.log_len(&replica("0")).unwrap()
    );
}

#[test]
fn d0_same_seed_is_deterministic() {
    let (_p1, d1) = fresh_world_dir();
    let (_p2, d2) = fresh_world_dir();
    let config = default_config();
    let mut a = World::create(&d1, config.clone()).expect("a");
    let mut b = World::create(&d2, config).expect("b");
    let ra = run_gate(&mut a, Gate::D0).unwrap();
    let rb = run_gate(&mut b, Gate::D0).unwrap();
    assert_eq!(ra, rb);
    assert_eq!(
        a.log_len(&replica("0")).unwrap(),
        b.log_len(&replica("0")).unwrap()
    );
}

#[test]
fn worker_kill_is_valid_process() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.inject(&Fault::Kill {
        process: process("worker-0"),
    })
    .expect("kill worker-0");
    w.recover().unwrap();
}
