//! Group 7: credit records, empty used, unknown ids.

mod common;
mod reference;

use common::*;
use prometheus_multiagent::{CreditRecord, Topology};

#[test]
fn credit_starts_empty() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::OrchestratorSubs, 2, 8),
    );
    assert_eq!(prod.credit(&eid("e")).unwrap(), Vec::new());
    assert_eq!(refer.credit(&eid("e")).unwrap(), Vec::new());
}

#[test]
fn credit_records_append() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::OrchestratorSubs, 2, 8),
    );
    assert_both_ok(
        prod.record_credit(&eid("e"), cred("orchestrator", vec![aid("sub-00")])),
        refer.record_credit(&eid("e"), cred("orchestrator", vec![aid("sub-00")])),
    );
    assert_both_ok(
        prod.record_credit(
            &eid("e"),
            cred("orchestrator", vec![aid("sub-01"), aid("sub-00")]),
        ),
        refer.record_credit(
            &eid("e"),
            cred("orchestrator", vec![aid("sub-01"), aid("sub-00")]),
        ),
    );
    let want = vec![
        CreditRecord {
            parent: aid("orchestrator"),
            used: vec![aid("sub-00")],
        },
        CreditRecord {
            parent: aid("orchestrator"),
            used: vec![aid("sub-01"), aid("sub-00")],
        },
    ];
    assert_eq!(prod.credit(&eid("e")).unwrap(), want);
    assert_eq!(refer.credit(&eid("e")).unwrap(), want);
}

#[test]
fn credit_empty_used_allowed() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    assert_both_ok(
        prod.record_credit(&eid("e"), cred("solo", vec![])),
        refer.record_credit(&eid("e"), cred("solo", vec![])),
    );
    assert_eq!(
        prod.credit(&eid("e")).unwrap(),
        vec![CreditRecord {
            parent: aid("solo"),
            used: vec![],
        }]
    );
    assert_eq!(refer.credit(&eid("e")).unwrap()[0].used, Vec::new());
}

#[test]
fn credit_duplicates_in_used_stored_as_given() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::OrchestratorSubs, 2, 8),
    );
    let used = vec![aid("sub-00"), aid("sub-00"), aid("sub-01")];
    assert_both_ok(
        prod.record_credit(&eid("e"), cred("orchestrator", used.clone())),
        refer.record_credit(&eid("e"), cred("orchestrator", used.clone())),
    );
    assert_eq!(prod.credit(&eid("e")).unwrap()[0].used, used);
    assert_eq!(refer.credit(&eid("e")).unwrap()[0].used, used);
}

#[test]
fn credit_unknown_parent() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::OrchestratorSubs, 2, 8),
    );
    let e = assert_both_err(
        prod.record_credit(&eid("e"), cred("ghost", vec![])),
        refer.record_credit(&eid("e"), cred("ghost", vec![])),
    );
    assert_unknown_agent(&e, &aid("ghost"));
    assert_eq!(prod.credit(&eid("e")).unwrap(), Vec::new());
}

#[test]
fn credit_unknown_parent_before_used() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::OrchestratorSubs, 2, 8),
    );
    let e = assert_both_err(
        prod.record_credit(&eid("e"), cred("ghost", vec![aid("also-ghost")])),
        refer.record_credit(&eid("e"), cred("ghost", vec![aid("also-ghost")])),
    );
    assert_unknown_agent(&e, &aid("ghost"));
}

#[test]
fn credit_first_unknown_used_in_order() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::OrchestratorSubs, 2, 8),
    );
    let e = assert_both_err(
        prod.record_credit(
            &eid("e"),
            cred(
                "orchestrator",
                vec![aid("sub-00"), aid("ghost-1"), aid("ghost-2")],
            ),
        ),
        refer.record_credit(
            &eid("e"),
            cred(
                "orchestrator",
                vec![aid("sub-00"), aid("ghost-1"), aid("ghost-2")],
            ),
        ),
    );
    assert_unknown_agent(&e, &aid("ghost-1"));
    assert_eq!(prod.credit(&eid("e")).unwrap(), Vec::new());
    assert_eq!(refer.credit(&eid("e")).unwrap(), Vec::new());
}

#[test]
fn credit_unknown_episode_before_parent() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let e = assert_both_err(
        prod.record_credit(&eid("nope"), cred("ghost", vec![])),
        refer.record_credit(&eid("nope"), cred("ghost", vec![])),
    );
    assert_unknown_episode(&e, &eid("nope"));
}

#[test]
fn credit_parent_may_appear_in_used() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::ParallelAggregate, 2, 8),
    );
    assert_both_ok(
        prod.record_credit(
            &eid("e"),
            cred("aggregator", vec![aid("aggregator"), aid("attempt-00")]),
        ),
        refer.record_credit(
            &eid("e"),
            cred("aggregator", vec![aid("aggregator"), aid("attempt-00")]),
        ),
    );
    assert_eq!(prod.credit(&eid("e")).unwrap().len(), 1);
    assert_eq!(
        refer.credit(&eid("e")).unwrap()[0].parent,
        aid("aggregator")
    );
}

#[test]
fn credit_does_not_cross_episodes() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("a", "t", Topology::Single, 0, 8),
    );
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("b", "t", Topology::Single, 0, 8),
    );
    assert_both_ok(
        prod.record_credit(&eid("a"), cred("solo", vec![aid("solo")])),
        refer.record_credit(&eid("a"), cred("solo", vec![aid("solo")])),
    );
    assert_eq!(prod.credit(&eid("b")).unwrap(), Vec::new());
    assert_eq!(refer.credit(&eid("b")).unwrap(), Vec::new());
}

#[test]
fn credit_agent_from_other_episode_roster_is_unknown() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("a", "t", Topology::Single, 0, 8),
    );
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("b", "t", Topology::AuthorReviewer, 0, 8),
    );
    let e = assert_both_err(
        prod.record_credit(&eid("a"), cred("author", vec![])),
        refer.record_credit(&eid("a"), cred("author", vec![])),
    );
    assert_unknown_agent(&e, &aid("author"));
}
