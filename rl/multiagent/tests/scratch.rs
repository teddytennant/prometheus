//! Group 5: `scratch_append` / `scratch_read` concatenation and UnknownAgent.

mod common;
mod reference;

use common::*;
use prometheus_multiagent::Topology;

#[test]
fn scratch_starts_empty() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    assert_eq!(prod.scratch_read(&eid("e")).unwrap(), "");
    assert_eq!(refer.scratch_read(&eid("e")).unwrap(), "");
}

#[test]
fn scratch_append_concatenates_without_separator() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    assert_both_ok(
        prod.scratch_append(&eid("e"), &aid("solo"), "hello"),
        refer.scratch_append(&eid("e"), &aid("solo"), "hello"),
    );
    assert_eq!(prod.scratch_read(&eid("e")).unwrap(), "hello");
    assert_eq!(refer.scratch_read(&eid("e")).unwrap(), "hello");
    assert_both_ok(
        prod.scratch_append(&eid("e"), &aid("solo"), "world"),
        refer.scratch_append(&eid("e"), &aid("solo"), "world"),
    );
    assert_eq!(prod.scratch_read(&eid("e")).unwrap(), "helloworld");
    assert_eq!(refer.scratch_read(&eid("e")).unwrap(), "helloworld");
}

#[test]
fn scratch_empty_text_allowed() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    assert_both_ok(
        prod.scratch_append(&eid("e"), &aid("solo"), ""),
        refer.scratch_append(&eid("e"), &aid("solo"), ""),
    );
    assert_eq!(prod.scratch_read(&eid("e")).unwrap(), "");
    assert_both_ok(
        prod.scratch_append(&eid("e"), &aid("solo"), "x"),
        refer.scratch_append(&eid("e"), &aid("solo"), "x"),
    );
    assert_both_ok(
        prod.scratch_append(&eid("e"), &aid("solo"), ""),
        refer.scratch_append(&eid("e"), &aid("solo"), ""),
    );
    assert_eq!(prod.scratch_read(&eid("e")).unwrap(), "x");
    assert_eq!(refer.scratch_read(&eid("e")).unwrap(), "x");
}

#[test]
fn scratch_is_shared_across_agents_call_order() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::OrchestratorSubs, 2, 8),
    );
    assert_both_ok(
        prod.scratch_append(&eid("e"), &aid("orchestrator"), "A"),
        refer.scratch_append(&eid("e"), &aid("orchestrator"), "A"),
    );
    assert_both_ok(
        prod.scratch_append(&eid("e"), &aid("sub-00"), "B"),
        refer.scratch_append(&eid("e"), &aid("sub-00"), "B"),
    );
    assert_both_ok(
        prod.scratch_append(&eid("e"), &aid("sub-01"), "C"),
        refer.scratch_append(&eid("e"), &aid("sub-01"), "C"),
    );
    assert_eq!(prod.scratch_read(&eid("e")).unwrap(), "ABC");
    assert_eq!(refer.scratch_read(&eid("e")).unwrap(), "ABC");
}

#[test]
fn scratch_append_unknown_agent() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let e = assert_both_err(
        prod.scratch_append(&eid("e"), &aid("ghost"), "nope"),
        refer.scratch_append(&eid("e"), &aid("ghost"), "nope"),
    );
    assert_unknown_agent(&e, &aid("ghost"));
    assert_eq!(prod.scratch_read(&eid("e")).unwrap(), "");
    assert_eq!(refer.scratch_read(&eid("e")).unwrap(), "");
}

#[test]
fn scratch_append_unknown_agent_case_sensitive() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let e = assert_both_err(
        prod.scratch_append(&eid("e"), &aid("Solo"), "x"),
        refer.scratch_append(&eid("e"), &aid("Solo"), "x"),
    );
    assert_unknown_agent(&e, &aid("Solo"));
}

#[test]
fn scratch_append_unknown_episode_before_unknown_agent() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let e = assert_both_err(
        prod.scratch_append(&eid("nope"), &aid("ghost"), "x"),
        refer.scratch_append(&eid("nope"), &aid("ghost"), "x"),
    );
    assert_unknown_episode(&e, &eid("nope"));
}

#[test]
fn scratch_does_not_bleed_across_episodes() {
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
        prod.scratch_append(&eid("a"), &aid("solo"), "only-a"),
        refer.scratch_append(&eid("a"), &aid("solo"), "only-a"),
    );
    assert_eq!(prod.scratch_read(&eid("b")).unwrap(), "");
    assert_eq!(refer.scratch_read(&eid("b")).unwrap(), "");
    assert_eq!(prod.scratch_read(&eid("a")).unwrap(), "only-a");
}

#[test]
fn scratch_unknown_agent_with_empty_text_still_errors() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let e = assert_both_err(
        prod.scratch_append(&eid("e"), &aid("missing"), ""),
        refer.scratch_append(&eid("e"), &aid("missing"), ""),
    );
    assert_unknown_agent(&e, &aid("missing"));
}

#[test]
fn scratch_agent_from_another_topology_is_unknown() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let e = assert_both_err(
        prod.scratch_append(&eid("e"), &aid("orchestrator"), "x"),
        refer.scratch_append(&eid("e"), &aid("orchestrator"), "x"),
    );
    assert_unknown_agent(&e, &aid("orchestrator"));
}
