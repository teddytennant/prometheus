//! Group: lost tasks, duplicated outputs, Invariants::hold, Error::Invariant.

mod common;
mod reference;

use common::{assert_invariant_err, default_config, fresh_world_dir, replica, task};
use prometheus_chaos::{
    run_gate, Fault, Gate, Invariants, ProcessId, World, EVENT_COMPLETED, EVENT_ENQUEUED,
};
use reference::RefWorld;

/// Allowed to pass against the stub: hold() is implemented on the iface.
#[test]
fn hold_true_iff_both_zero() {
    assert!(Invariants {
        lost_tasks: 0,
        duplicated_outputs: 0
    }
    .hold());
    assert!(!Invariants {
        lost_tasks: 1,
        duplicated_outputs: 0
    }
    .hold());
    assert!(!Invariants {
        lost_tasks: 0,
        duplicated_outputs: 1
    }
    .hold());
    assert!(!Invariants {
        lost_tasks: 2,
        duplicated_outputs: 3
    }
    .hold());
}

#[test]
fn enqueue_without_complete_is_lost() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.enqueue_task(&task("t"), w.now()).expect("enq");
    let inv = w.invariants().expect("inv");
    assert_eq!(inv.lost_tasks, 1);
    assert_eq!(inv.duplicated_outputs, 0);
    assert!(!inv.hold());
    let log = w.replica_log(&replica("0")).expect("log");
    assert!(log.iter().any(|e| e.event_type == EVENT_ENQUEUED));
}

#[test]
fn enqueue_and_complete_holds() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.enqueue_task(&task("t"), w.now()).expect("enq");
    w.complete_task(&task("t"), 1, b"out", w.now())
        .expect("complete");
    let inv = w.invariants().expect("inv");
    assert_eq!(inv.lost_tasks, 0);
    assert_eq!(inv.duplicated_outputs, 0);
    assert!(inv.hold());
    let log = w.replica_log(&replica("0")).expect("log");
    assert!(log.iter().any(|e| e.event_type == EVENT_COMPLETED));
}

#[test]
fn second_complete_same_attempt_is_duplicate() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.enqueue_task(&task("t"), w.now()).expect("enq");
    w.complete_task(&task("t"), 1, b"a", w.now()).expect("c1");
    w.complete_task(&task("t"), 1, b"b", w.now()).expect("c2");
    let inv = w.invariants().expect("inv");
    assert_eq!(inv.duplicated_outputs, 1);
    assert_eq!(inv.lost_tasks, 0);
    assert!(!inv.hold());
}

#[test]
fn different_attempts_are_not_duplicates() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.enqueue_task(&task("t"), w.now()).expect("enq");
    w.complete_task(&task("t"), 1, b"a", w.now()).expect("c1");
    w.complete_task(&task("t"), 2, b"b", w.now()).expect("c2");
    let inv = w.invariants().expect("inv");
    assert_eq!(inv.duplicated_outputs, 0);
    assert!(inv.hold());
}

#[test]
fn run_gate_returns_invariant_if_preexisting_lost_task() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.enqueue_task(&task("orphan"), w.now()).expect("enq");
    let err = run_gate(&mut w, Gate::D3).expect_err("preexisting lost");
    assert_invariant_err(&err);
}

#[test]
fn invariants_match_reference() {
    let (_pw, dir_w) = fresh_world_dir();
    let (_pr, dir_r) = fresh_world_dir();
    let config = default_config();
    let mut w = World::create(&dir_w, config.clone()).expect("w");
    let mut r = RefWorld::create(&dir_r, config).expect("r");
    w.enqueue_task(&task("t"), 0).unwrap();
    r.enqueue_task(&task("t"), 0).unwrap();
    assert_eq!(
        w.invariants().unwrap().lost_tasks,
        r.invariants().unwrap().lost_tasks
    );
    w.complete_task(&task("t"), 1, b"x", 0).unwrap();
    r.complete_task(&task("t"), 1, b"x", 0).unwrap();
    assert_eq!(w.invariants().unwrap(), r.invariants().unwrap());
    w.complete_task(&task("t"), 1, b"y", 0).unwrap();
    r.complete_task(&task("t"), 1, b"y", 0).unwrap();
    assert_eq!(w.invariants().unwrap(), r.invariants().unwrap());
    assert!(!w.invariants().unwrap().hold());
}

#[test]
fn kill_does_not_count_as_duplicate() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    w.enqueue_task(&task("t"), w.now()).unwrap();
    w.complete_task(&task("t"), 1, b"ok", w.now()).unwrap();
    w.inject(&Fault::Kill {
        process: ProcessId("coordinator".into()),
    })
    .unwrap();
    w.recover().unwrap();
    let inv = w.invariants().unwrap();
    assert_eq!(inv.duplicated_outputs, 0);
    assert_eq!(inv.lost_tasks, 0);
    assert!(inv.hold());
}
