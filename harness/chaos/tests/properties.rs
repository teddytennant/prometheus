//! Group: World vs RefWorld on the same call sequence; error kinds; shapes.

mod common;
mod reference;

use common::{
    cfg, coordinator, default_config, err_kind_eq, fresh_world_dir, replica, task, worker,
};
use prometheus_chaos::{Fault, ReplicaId, World, CLOCK_SKEW_MS};
use reference::{Lcg, RefWorld};

#[derive(Clone)]
enum Op {
    Advance(u64),
    Enqueue(&'static str),
    Complete(&'static str, u64),
    KillCoord,
    KillWorker,
    Partition { asymmetric: bool },
    Heal,
    ClockOk,
    ClockBad,
    Outage,
    HealOutage,
    DiskFull,
    Hung,
    Unhang,
    Broker,
    TokenExpiry,
    TokenRotate,
    JobPreempt,
    NodeLoss,
    Recover,
}

fn apply_world(w: &mut World, op: &Op) -> Result<(), prometheus_chaos::Error> {
    match op {
        Op::Advance(ms) => {
            w.advance(*ms);
            Ok(())
        }
        Op::Enqueue(id) => w.enqueue_task(&task(id), w.now()),
        Op::Complete(id, a) => w.complete_task(&task(id), *a, b"out", w.now()),
        Op::KillCoord => w.inject(&Fault::Kill {
            process: coordinator(),
        }),
        Op::KillWorker => w.inject(&Fault::Kill {
            process: worker(0),
        }),
        Op::Partition { asymmetric } => w.inject(&Fault::Partition {
            from: replica("0"),
            to: replica("1"),
            asymmetric: *asymmetric,
        }),
        Op::Heal => w.inject(&Fault::HealPartition {
            from: replica("0"),
            to: replica("1"),
        }),
        Op::ClockOk => w.inject(&Fault::ClockSkew {
            replica: replica("0"),
            delta_ms: CLOCK_SKEW_MS,
        }),
        Op::ClockBad => w.inject(&Fault::ClockSkew {
            replica: replica("0"),
            delta_ms: CLOCK_SKEW_MS + 1,
        }),
        Op::Outage => w.inject(&Fault::Outage {
            provider: "xai".into(),
        }),
        Op::HealOutage => w.inject(&Fault::HealOutage {
            provider: "xai".into(),
        }),
        Op::DiskFull => w.inject(&Fault::DiskFull {
            replica: replica("1"),
        }),
        Op::Hung => w.inject(&Fault::HungSqueue),
        Op::Unhang => w.inject(&Fault::UnhangSqueue),
        Op::Broker => w.inject(&Fault::BrokerDeath),
        Op::TokenExpiry => w.inject(&Fault::TokenExpiry),
        Op::TokenRotate => w.inject(&Fault::TokenRotate),
        Op::JobPreempt => w.inject(&Fault::JobPreempt {
            job: prometheus_chaos::JobId("job-0".into()),
        }),
        Op::NodeLoss => w.inject(&Fault::NodeLoss {
            replica: replica("0"),
        }),
        Op::Recover => w.recover(),
    }
}

fn apply_ref(r: &mut RefWorld, op: &Op) -> Result<(), prometheus_chaos::Error> {
    match op {
        Op::Advance(ms) => {
            r.advance(*ms);
            Ok(())
        }
        Op::Enqueue(id) => r.enqueue_task(&task(id), r.now()),
        Op::Complete(id, a) => r.complete_task(&task(id), *a, b"out", r.now()),
        Op::KillCoord => r.inject(&Fault::Kill {
            process: coordinator(),
        }),
        Op::KillWorker => r.inject(&Fault::Kill {
            process: worker(0),
        }),
        Op::Partition { asymmetric } => r.inject(&Fault::Partition {
            from: replica("0"),
            to: replica("1"),
            asymmetric: *asymmetric,
        }),
        Op::Heal => r.inject(&Fault::HealPartition {
            from: replica("0"),
            to: replica("1"),
        }),
        Op::ClockOk => r.inject(&Fault::ClockSkew {
            replica: replica("0"),
            delta_ms: CLOCK_SKEW_MS,
        }),
        Op::ClockBad => r.inject(&Fault::ClockSkew {
            replica: replica("0"),
            delta_ms: CLOCK_SKEW_MS + 1,
        }),
        Op::Outage => r.inject(&Fault::Outage {
            provider: "xai".into(),
        }),
        Op::HealOutage => r.inject(&Fault::HealOutage {
            provider: "xai".into(),
        }),
        Op::DiskFull => r.inject(&Fault::DiskFull {
            replica: replica("1"),
        }),
        Op::Hung => r.inject(&Fault::HungSqueue),
        Op::Unhang => r.inject(&Fault::UnhangSqueue),
        Op::Broker => r.inject(&Fault::BrokerDeath),
        Op::TokenExpiry => r.inject(&Fault::TokenExpiry),
        Op::TokenRotate => r.inject(&Fault::TokenRotate),
        Op::JobPreempt => r.inject(&Fault::JobPreempt {
            job: prometheus_chaos::JobId("job-0".into()),
        }),
        Op::NodeLoss => r.inject(&Fault::NodeLoss {
            replica: replica("0"),
        }),
        Op::Recover => r.recover(),
    }
}

fn assert_pair(w: &World, r: &RefWorld, after: &str) {
    assert_eq!(w.now(), r.now(), "now after {after}");
    assert_eq!(w.replicas(), r.replicas(), "replicas after {after}");
    assert_eq!(w.processes(), r.processes(), "processes after {after}");
    let wi = w.invariants().expect("w inv");
    let ri = r.invariants().expect("r inv");
    assert_eq!(wi, ri, "invariants after {after}");
    for id in w.replicas() {
        match (w.log_len(&id), r.log_len(&id)) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "log_len {id:?} after {after}"),
            (Err(a), Err(b)) => assert!(err_kind_eq(&a, &b), "log_len err {a:?} vs {b:?}"),
            (a, b) => panic!("log_len mismatch after {after}: {a:?} vs {b:?}"),
        }
    }
}

#[test]
fn world_matches_reference_scripted_ops() {
    let (_pw, dir_w) = fresh_world_dir();
    let (_pr, dir_r) = fresh_world_dir();
    let config = default_config();
    let mut w = World::create(&dir_w, config.clone()).expect("w");
    let mut r = RefWorld::create(&dir_r, config).expect("r");
    assert_pair(&w, &r, "create");
    let ops = [
        Op::Enqueue("t0"),
        Op::Advance(5),
        Op::Complete("t0", 1),
        Op::Partition { asymmetric: true },
        Op::Enqueue("t1"),
        Op::Complete("t1", 1),
        Op::Heal,
        Op::ClockOk,
        Op::ClockBad,
        Op::Outage,
        Op::Broker,
        Op::TokenExpiry,
        Op::Hung,
        Op::JobPreempt,
        Op::KillCoord,
        Op::Recover,
        Op::Complete("t0", 2),
        Op::Unhang,
        Op::HealOutage,
        Op::TokenRotate,
        Op::DiskFull,
        Op::Enqueue("t2"),
        Op::Complete("t2", 1),
    ];
    for (i, op) in ops.iter().enumerate() {
        let aw = apply_world(&mut w, op);
        let ar = apply_ref(&mut r, op);
        match (aw, ar) {
            (Ok(()), Ok(())) => {}
            (Err(a), Err(b)) => assert!(
                err_kind_eq(&a, &b),
                "op {i} err mismatch {a:?} vs {b:?}"
            ),
            (aw, ar) => panic!("op {i} Ok/Err mismatch: {aw:?} vs {ar:?}"),
        }
        assert_pair(&w, &r, &format!("op {i}"));
    }
}

#[test]
fn world_matches_reference_random_ops() {
    let (_pw, dir_w) = fresh_world_dir();
    let (_pr, dir_r) = fresh_world_dir();
    let config = cfg(2, 4, 99);
    let mut w = World::create(&dir_w, config.clone()).expect("w");
    let mut r = RefWorld::create(&dir_r, config).expect("r");
    let catalog = [
        Op::Advance(1),
        Op::Enqueue("x"),
        Op::Complete("x", 1),
        Op::Complete("x", 1),
        Op::KillWorker,
        Op::Recover,
        Op::Partition { asymmetric: false },
        Op::Heal,
        Op::ClockOk,
        Op::Outage,
        Op::Hung,
        Op::Broker,
        Op::TokenExpiry,
        Op::NodeLoss,
    ];
    let mut rng = Lcg::new(99);
    for i in 0..40 {
        let op = &catalog[rng.pick(catalog.len())];
        let aw = apply_world(&mut w, op);
        let ar = apply_ref(&mut r, op);
        match (aw, ar) {
            (Ok(()), Ok(())) => {}
            (Err(a), Err(b)) => assert!(err_kind_eq(&a, &b), "rand {i} {a:?} vs {b:?}"),
            (aw, ar) => panic!("rand {i} Ok/Err mismatch: {aw:?} vs {ar:?}"),
        }
        // After kill, replica_log is down; recover before comparing logs.
        if matches!(op, Op::KillCoord | Op::KillWorker | Op::NodeLoss) {
            let _ = w.recover();
            let _ = r.recover();
        }
        assert_pair(&w, &r, &format!("rand {i}"));
    }
}

#[test]
fn unknown_replica_and_process_match_reference() {
    let (_pw, dir_w) = fresh_world_dir();
    let (_pr, dir_r) = fresh_world_dir();
    let config = default_config();
    let mut w = World::create(&dir_w, config.clone()).expect("w");
    let mut r = RefWorld::create(&dir_r, config).expect("r");
    let bad_r = Fault::NodeLoss {
        replica: ReplicaId("zzz".into()),
    };
    let ew = w.inject(&bad_r).unwrap_err();
    let er = r.inject(&bad_r).unwrap_err();
    assert!(err_kind_eq(&ew, &er));
    let bad_p = Fault::Kill {
        process: prometheus_chaos::ProcessId("ghost".into()),
    };
    let ew = w.inject(&bad_p).unwrap_err();
    let er = r.inject(&bad_p).unwrap_err();
    assert!(err_kind_eq(&ew, &er));
}

#[test]
fn replica_dir_shape() {
    let (_parent, dir) = fresh_world_dir();
    let w = World::create(&dir, default_config()).expect("create");
    let id = replica("0");
    assert_eq!(w.replica_dir(&id), dir.join("replicas").join("0"));
}

#[test]
fn n_processes_zero_still_creates_replicas() {
    let (_parent, dir) = fresh_world_dir();
    let w = World::create(&dir, cfg(2, 0, 0)).expect("create");
    assert!(w.processes().is_empty());
    assert_eq!(w.replicas().len(), 2);
}
