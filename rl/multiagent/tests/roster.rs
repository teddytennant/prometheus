//! Group 3: `roster` for all five topologies, `BadWidth` cases, two-digit
//! padding (`n_subs=10` → `sub-09` last).

mod common;
mod reference;

use common::*;
use prometheus_multiagent::{MultiError, Role, Topology, MAX_N_SUBS};

#[test]
fn roster_single() {
    let got = roster_both(Topology::Single, 0).unwrap();
    assert_eq!(got, vec![(aid("solo"), Role::Solo)]);
}

#[test]
fn roster_orchestrator_subs_default_width() {
    let got = roster_both(Topology::OrchestratorSubs, 2).unwrap();
    assert_eq!(
        got,
        vec![
            (aid("orchestrator"), Role::Orchestrator),
            (aid("sub-00"), Role::SubAgent),
            (aid("sub-01"), Role::SubAgent),
        ]
    );
}

#[test]
fn roster_orchestrator_n_subs_10_last_is_sub_09() {
    let got = roster_both(Topology::OrchestratorSubs, 10).unwrap();
    assert_eq!(got.len(), 11);
    assert_eq!(got[0], (aid("orchestrator"), Role::Orchestrator));
    assert_eq!(got[1], (aid("sub-00"), Role::SubAgent));
    assert_eq!(got[10], (aid("sub-09"), Role::SubAgent));
    for (i, (id, role)) in got.iter().enumerate().skip(1) {
        assert_eq!(*role, Role::SubAgent);
        assert_eq!(id.0, format!("sub-{:02}", i - 1));
        assert!(id.0.as_bytes().contains(&b'-'), "ASCII hyphen");
        assert!(!id.0.contains('_'));
    }
}

#[test]
fn roster_orchestrator_n_subs_99_last_is_sub_98() {
    let got = roster_both(Topology::OrchestratorSubs, MAX_N_SUBS).unwrap();
    assert_eq!(got.len(), 100);
    assert_eq!(got[99], (aid("sub-98"), Role::SubAgent));
}

#[test]
fn roster_parallel_aggregate() {
    let got = roster_both(Topology::ParallelAggregate, 2).unwrap();
    assert_eq!(
        got,
        vec![
            (aid("attempt-00"), Role::Attempt),
            (aid("attempt-01"), Role::Attempt),
            (aid("aggregator"), Role::Aggregator),
        ]
    );
}

#[test]
fn roster_parallel_n_subs_10_last_attempt_is_attempt_09() {
    let got = roster_both(Topology::ParallelAggregate, 10).unwrap();
    assert_eq!(got.len(), 11);
    assert_eq!(got[0], (aid("attempt-00"), Role::Attempt));
    assert_eq!(got[9], (aid("attempt-09"), Role::Attempt));
    assert_eq!(got[10], (aid("aggregator"), Role::Aggregator));
}

#[test]
fn roster_parallel_n_subs_1() {
    let got = roster_both(Topology::ParallelAggregate, 1).unwrap();
    assert_eq!(
        got,
        vec![
            (aid("attempt-00"), Role::Attempt),
            (aid("aggregator"), Role::Aggregator),
        ]
    );
}

#[test]
fn roster_proposer_solver() {
    let got = roster_both(Topology::ProposerSolver, 0).unwrap();
    assert_eq!(
        got,
        vec![
            (aid("proposer"), Role::Proposer),
            (aid("solver"), Role::Solver),
        ]
    );
}

#[test]
fn roster_author_reviewer() {
    let got = roster_both(Topology::AuthorReviewer, 0).unwrap();
    assert_eq!(
        got,
        vec![
            (aid("author"), Role::Author),
            (aid("reviewer"), Role::Reviewer),
        ]
    );
}

#[test]
fn roster_bad_width_single_nonzero() {
    for n in [1u32, 2, 10, 99, 100] {
        match roster_both(Topology::Single, n) {
            Err(MultiError::BadWidth) => {}
            other => panic!("Single n_subs={n}: {other:?}"),
        }
    }
}

#[test]
fn roster_bad_width_fixed_pair_topologies() {
    for topo in [Topology::ProposerSolver, Topology::AuthorReviewer] {
        for n in [1u32, 2, 99, 100] {
            match roster_both(topo, n) {
                Err(MultiError::BadWidth) => {}
                other => panic!("{topo:?} n_subs={n}: {other:?}"),
            }
        }
    }
}

#[test]
fn roster_bad_width_variable_zero_and_over_max() {
    for topo in [Topology::OrchestratorSubs, Topology::ParallelAggregate] {
        match roster_both(topo, 0) {
            Err(MultiError::BadWidth) => {}
            other => panic!("{topo:?} n_subs=0: {other:?}"),
        }
        match roster_both(topo, MAX_N_SUBS + 1) {
            Err(MultiError::BadWidth) => {}
            other => panic!("{topo:?} n_subs=100: {other:?}"),
        }
        match roster_both(topo, u32::MAX) {
            Err(MultiError::BadWidth) => {}
            other => panic!("{topo:?} n_subs=MAX: {other:?}"),
        }
    }
}

#[test]
fn roster_ids_are_exact_ascii_hyphen_strings() {
    let orch = roster_both(Topology::OrchestratorSubs, 1).unwrap();
    assert_eq!(orch[0].0 .0, "orchestrator");
    assert_eq!(orch[1].0 .0, "sub-00");
    let par = roster_both(Topology::ParallelAggregate, 1).unwrap();
    assert_eq!(par[0].0 .0, "attempt-00");
    assert_eq!(par[1].0 .0, "aggregator");
    assert_ne!(orch[1].0 .0, "sub-0");
    assert_ne!(orch[1].0 .0, "sub_00");
    assert_ne!(par[0].0 .0, "attempt_00");
}
