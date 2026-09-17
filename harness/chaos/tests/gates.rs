//! Group: faults_for / run_gate D0–D5 kinds and survival.

mod common;
mod reference;

use common::{coordinator, default_config, fresh_world_dir, HOSTED_PROVIDERS};
use prometheus_chaos::{faults_for, run_gate, Fault, Gate, ProcessId, ReplicaId, World, D0_KILLS};
use reference::{ref_faults_for, ref_run_gate, RefWorld};

fn kinds(fs: &[Fault]) -> Vec<&'static str> {
    fs.iter()
        .map(|f| match f {
            Fault::Kill { .. } => "kill",
            Fault::Partition { .. } => "partition",
            Fault::HealPartition { .. } => "heal_partition",
            Fault::Outage { .. } => "outage",
            Fault::HealOutage { .. } => "heal_outage",
            Fault::DiskFull { .. } => "disk_full",
            Fault::DiskCorrupt { .. } => "disk_corrupt",
            Fault::ClockSkew { .. } => "clock_skew",
            Fault::TokenExpiry => "token_expiry",
            Fault::TokenRotate => "token_rotate",
            Fault::NodeLoss { .. } => "node_loss",
            Fault::NodeReplace { .. } => "node_replace",
            Fault::JobPreempt { .. } => "job_preempt",
            Fault::WalltimeKill { .. } => "walltime_kill",
            Fault::HungSqueue => "hung_squeue",
            Fault::UnhangSqueue => "unhang_squeue",
            Fault::BrokerDeath => "broker_death",
            Fault::RateLimit { .. } => "rate_limit",
        })
        .collect()
}

#[test]
fn faults_for_matches_reference_every_gate() {
    for g in [
        Gate::D0,
        Gate::D1,
        Gate::D2,
        Gate::D3,
        Gate::D4,
        Gate::D5,
        Gate::Soak,
    ] {
        assert_eq!(
            faults_for(g),
            ref_faults_for(g),
            "faults_for({g:?}) must match the reference placeholders"
        );
    }
}

#[test]
fn faults_for_d0_is_kill() {
    let fs = faults_for(Gate::D0);
    assert!(
        fs.iter().any(|f| matches!(f, Fault::Kill { .. })),
        "D0 survives Kill: {fs:?}"
    );
}

#[test]
fn faults_for_d1_includes_coordinator_kill_and_partition() {
    let fs = faults_for(Gate::D1);
    let kill_coord = fs.iter().any(|f| {
        matches!(
            f,
            Fault::Kill {
                process
            } if process == &ProcessId("coordinator".into())
        )
    });
    let part = fs.iter().any(|f| matches!(f, Fault::Partition { .. }));
    assert!(kill_coord, "D1 kills the coordinator: {fs:?}");
    assert!(part, "D1 partitions a worker past lease expiry: {fs:?}");
}

#[test]
fn faults_for_d2_asymmetric_node_loss_clock() {
    let fs = faults_for(Gate::D2);
    assert!(fs.iter().any(|f| matches!(
        f,
        Fault::Partition {
            asymmetric: true,
            ..
        }
    )));
    assert!(fs.iter().any(|f| matches!(f, Fault::NodeLoss { .. })));
    assert!(fs.iter().any(|f| matches!(f, Fault::ClockSkew { .. })));
}

#[test]
fn faults_for_d3_slurm() {
    let fs = faults_for(Gate::D3);
    let k = kinds(&fs);
    assert!(k.contains(&"job_preempt"), "{k:?}");
    assert!(k.contains(&"walltime_kill"), "{k:?}");
    assert!(k.contains(&"hung_squeue"), "{k:?}");
}

#[test]
fn faults_for_d4_broker_token_outage() {
    let fs = faults_for(Gate::D4);
    let k = kinds(&fs);
    assert!(k.contains(&"broker_death"), "{k:?}");
    assert!(k.contains(&"token_expiry"), "{k:?}");
    assert!(k.contains(&"outage"), "{k:?}");
}

#[test]
fn faults_for_d5_outage_every_hosted_provider() {
    let fs = faults_for(Gate::D5);
    for p in HOSTED_PROVIDERS {
        assert!(
            fs.iter()
                .any(|f| matches!(f, Fault::Outage { provider } if provider == p)),
            "D5 must outage hosted provider {p}: {fs:?}"
        );
    }
}

#[test]
fn run_gate_d1_through_d5_hold() {
    for g in [Gate::D1, Gate::D2, Gate::D3, Gate::D4, Gate::D5] {
        let (_parent, dir) = fresh_world_dir();
        let mut w = World::create(&dir, default_config()).expect("create");
        assert!(
            !w.processes().is_empty(),
            "processes() nonempty after create"
        );
        assert_eq!(w.processes()[0], coordinator());
        let report = run_gate(&mut w, g).unwrap_or_else(|e| panic!("run_gate({g:?}): {e:?}"));
        assert_eq!(report.gate, g);
        assert!(report.recovered);
        assert!(
            report.invariants.hold(),
            "{g:?} must not lose tasks or duplicate outputs: {:?}",
            report.invariants
        );
        assert!(report.faults_applied >= 1, "{g:?} applied no faults");
    }
}

#[test]
fn run_gate_d5_allows_reduced_size_but_not_lost_tasks() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    let report = run_gate(&mut w, Gate::D5).expect("D5");
    assert!(report.invariants.lost_tasks == 0);
    assert!(report.invariants.hold());
}

#[test]
fn run_gate_matches_reference_d3() {
    let (_pw, dir_w) = fresh_world_dir();
    let (_pr, dir_r) = fresh_world_dir();
    let config = default_config();
    let mut w = World::create(&dir_w, config.clone()).expect("w");
    let mut r = RefWorld::create(&dir_r, config).expect("r");
    let a = run_gate(&mut w, Gate::D3).expect("world");
    let b = ref_run_gate(&mut r, Gate::D3).expect("ref");
    assert_eq!(a, b);
}

#[test]
fn d0_kills_constant_is_1000() {
    // Combined with run_gate so the test cannot pass the stub by constants alone.
    assert_eq!(D0_KILLS, 1_000);
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    let report = run_gate(&mut w, Gate::D0).expect("D0");
    assert_eq!(report.faults_applied, 1_000);
}

#[test]
fn node_loss_unknown_is_no_replica() {
    let (_parent, dir) = fresh_world_dir();
    let mut w = World::create(&dir, default_config()).expect("create");
    let err = w
        .inject(&Fault::NodeLoss {
            replica: ReplicaId("ghost".into()),
        })
        .expect_err("unknown");
    match err {
        prometheus_chaos::Error::NoReplica(id) => assert_eq!(id, "ghost"),
        other => panic!("{other:?}"),
    }
}
