//! Group 8: charge_tokens: n=0, at budget, over budget unchanged, per-agent vs total.

mod common;
mod reference;

use common::*;
use prometheus_multiagent::{MultiError, Topology};

#[test]
fn charge_n_zero_success_total_unchanged() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 10),
    );
    let p = prod.charge_tokens(&eid("e"), &aid("solo"), 0).unwrap();
    let r = refer.charge_tokens(&eid("e"), &aid("solo"), 0).unwrap();
    assert_eq!(p, 0);
    assert_eq!(r, 0);
    assert_eq!(prod.tokens_used(&eid("e")).unwrap(), 0);
    assert_eq!(prod.tokens_used_by(&eid("e"), &aid("solo")).unwrap(), 0);
}

#[test]
fn charge_n_zero_allowed_at_budget() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 5),
    );
    assert_eq!(prod.charge_tokens(&eid("e"), &aid("solo"), 5).unwrap(), 5);
    assert_eq!(refer.charge_tokens(&eid("e"), &aid("solo"), 5).unwrap(), 5);
    assert_eq!(prod.charge_tokens(&eid("e"), &aid("solo"), 0).unwrap(), 5);
    assert_eq!(refer.charge_tokens(&eid("e"), &aid("solo"), 0).unwrap(), 5);
}

#[test]
fn charge_equal_to_budget_allowed() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 7),
    );
    assert_eq!(prod.charge_tokens(&eid("e"), &aid("solo"), 7).unwrap(), 7);
    assert_eq!(refer.charge_tokens(&eid("e"), &aid("solo"), 7).unwrap(), 7);
    assert_eq!(prod.tokens_used(&eid("e")).unwrap(), 7);
}

#[test]
fn charge_over_budget_unchanged() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::OrchestratorSubs, 2, 10),
    );
    assert_eq!(
        prod.charge_tokens(&eid("e"), &aid("orchestrator"), 6)
            .unwrap(),
        6
    );
    assert_eq!(
        refer
            .charge_tokens(&eid("e"), &aid("orchestrator"), 6)
            .unwrap(),
        6
    );
    let e = assert_both_err(
        prod.charge_tokens(&eid("e"), &aid("sub-00"), 5),
        refer.charge_tokens(&eid("e"), &aid("sub-00"), 5),
    );
    match e {
        MultiError::BudgetExceeded => {}
        other => panic!("{other:?}"),
    }
    assert_eq!(prod.tokens_used(&eid("e")).unwrap(), 6);
    assert_eq!(refer.tokens_used(&eid("e")).unwrap(), 6);
    assert_eq!(
        prod.tokens_used_by(&eid("e"), &aid("orchestrator"))
            .unwrap(),
        6
    );
    assert_eq!(prod.tokens_used_by(&eid("e"), &aid("sub-00")).unwrap(), 0);
    assert_eq!(refer.tokens_used_by(&eid("e"), &aid("sub-00")).unwrap(), 0);
}

#[test]
fn charge_one_over_budget_from_zero() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 10),
    );
    match assert_both_err(
        prod.charge_tokens(&eid("e"), &aid("solo"), 11),
        refer.charge_tokens(&eid("e"), &aid("solo"), 11),
    ) {
        MultiError::BudgetExceeded => {}
        other => panic!("{other:?}"),
    }
    assert_eq!(prod.tokens_used(&eid("e")).unwrap(), 0);
    assert_eq!(refer.tokens_used(&eid("e")).unwrap(), 0);
}

#[test]
fn charge_per_agent_vs_total() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::ProposerSolver, 0, 100),
    );
    assert_eq!(
        prod.charge_tokens(&eid("e"), &aid("proposer"), 10).unwrap(),
        10
    );
    assert_eq!(
        refer
            .charge_tokens(&eid("e"), &aid("proposer"), 10)
            .unwrap(),
        10
    );
    assert_eq!(
        prod.charge_tokens(&eid("e"), &aid("solver"), 20).unwrap(),
        30
    );
    assert_eq!(
        refer.charge_tokens(&eid("e"), &aid("solver"), 20).unwrap(),
        30
    );
    assert_eq!(prod.tokens_used(&eid("e")).unwrap(), 30);
    assert_eq!(refer.tokens_used(&eid("e")).unwrap(), 30);
    assert_eq!(
        prod.tokens_used_by(&eid("e"), &aid("proposer")).unwrap(),
        10
    );
    assert_eq!(prod.tokens_used_by(&eid("e"), &aid("solver")).unwrap(), 20);
    assert_tokens_sum(&prod, &refer, &eid("e"));
}

#[test]
fn charge_returns_new_episode_total_not_agent_total() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::AuthorReviewer, 0, 50),
    );
    prod.charge_tokens(&eid("e"), &aid("author"), 8).unwrap();
    refer.charge_tokens(&eid("e"), &aid("author"), 8).unwrap();
    let p = prod.charge_tokens(&eid("e"), &aid("reviewer"), 3).unwrap();
    let r = refer.charge_tokens(&eid("e"), &aid("reviewer"), 3).unwrap();
    assert_eq!(p, 11);
    assert_eq!(r, 11);
    assert_eq!(prod.tokens_used_by(&eid("e"), &aid("reviewer")).unwrap(), 3);
}

#[test]
fn charge_unknown_agent() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 10),
    );
    let e = assert_both_err(
        prod.charge_tokens(&eid("e"), &aid("ghost"), 1),
        refer.charge_tokens(&eid("e"), &aid("ghost"), 1),
    );
    assert_unknown_agent(&e, &aid("ghost"));
    assert_eq!(prod.tokens_used(&eid("e")).unwrap(), 0);
}

#[test]
fn charge_unknown_episode_before_agent_even_for_n_zero() {
    let (mut prod, mut refer) = pair_default();
    let e = assert_both_err(
        prod.charge_tokens(&eid("nope"), &aid("ghost"), 0),
        refer.charge_tokens(&eid("nope"), &aid("ghost"), 0),
    );
    assert_unknown_episode(&e, &eid("nope"));
}

#[test]
fn charge_unknown_agent_even_for_n_zero() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 10),
    );
    let e = assert_both_err(
        prod.charge_tokens(&eid("e"), &aid("ghost"), 0),
        refer.charge_tokens(&eid("e"), &aid("ghost"), 0),
    );
    assert_unknown_agent(&e, &aid("ghost"));
}

#[test]
fn tokens_used_by_unknown_agent_does_not_invent_zero() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 10),
    );
    let e = assert_both_err(
        prod.tokens_used_by(&eid("e"), &aid("ghost")),
        refer.tokens_used_by(&eid("e"), &aid("ghost")),
    );
    assert_unknown_agent(&e, &aid("ghost"));
}

#[test]
fn charge_does_not_cross_episodes() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("a", "t", Topology::Single, 0, 10),
    );
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("b", "t", Topology::Single, 0, 100),
    );
    prod.charge_tokens(&eid("a"), &aid("solo"), 10).unwrap();
    refer.charge_tokens(&eid("a"), &aid("solo"), 10).unwrap();
    assert_eq!(prod.tokens_used(&eid("b")).unwrap(), 0);
    assert_eq!(refer.tokens_used(&eid("b")).unwrap(), 0);
}

#[test]
fn charge_overflow_is_budget_exceeded_unchanged() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, u64::MAX),
    );
    assert_eq!(
        prod.charge_tokens(&eid("e"), &aid("solo"), u64::MAX)
            .unwrap(),
        u64::MAX
    );
    assert_eq!(
        refer
            .charge_tokens(&eid("e"), &aid("solo"), u64::MAX)
            .unwrap(),
        u64::MAX
    );
    match assert_both_err(
        prod.charge_tokens(&eid("e"), &aid("solo"), 1),
        refer.charge_tokens(&eid("e"), &aid("solo"), 1),
    ) {
        MultiError::BudgetExceeded => {}
        other => panic!("{other:?}"),
    }
    assert_eq!(prod.tokens_used(&eid("e")).unwrap(), u64::MAX);
    assert_eq!(refer.tokens_used(&eid("e")).unwrap(), u64::MAX);
}

#[test]
fn uncharged_agent_reads_zero_not_error() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::ProposerSolver, 0, 10),
    );
    prod.charge_tokens(&eid("e"), &aid("proposer"), 4).unwrap();
    refer.charge_tokens(&eid("e"), &aid("proposer"), 4).unwrap();
    assert_eq!(prod.tokens_used_by(&eid("e"), &aid("solver")).unwrap(), 0);
    assert_eq!(refer.tokens_used_by(&eid("e"), &aid("solver")).unwrap(), 0);
}
