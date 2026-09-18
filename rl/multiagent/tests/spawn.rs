//! Group 4: spawn EmptyId, DuplicateEpisode, BadWidth, zero budget Message,
//! returned roster, initial empty scratch/inboxes/credit/tokens.
//! Unknown episode on every episode-keyed method.

mod common;
mod reference;

use common::*;
use prometheus_multiagent::{MultiError, Role, Topology};

#[test]
fn spawn_single_returns_roster_and_empty_state() {
    let (mut prod, mut refer) = pair_default();
    let roster = spawn_ok(
        &mut prod,
        &mut refer,
        spec("ep-1", "task-a", Topology::Single, 0, 32),
    );
    assert_eq!(roster, vec![(aid("solo"), Role::Solo)]);
    assert_eq!(prod.topology_of(&eid("ep-1")).unwrap(), Topology::Single);
    assert_eq!(refer.topology_of(&eid("ep-1")).unwrap(), Topology::Single);
}

#[test]
fn spawn_orchestrator_uses_spec_n_subs_not_default() {
    let mut cfg = default_config();
    cfg.default_n_subs = 5;
    let (mut prod, mut refer) = pair(cfg);
    let roster = spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::OrchestratorSubs, 2, 64),
    );
    assert_eq!(roster.len(), 3);
    assert_eq!(roster[0], (aid("orchestrator"), Role::Orchestrator));
    assert_eq!(roster[2], (aid("sub-01"), Role::SubAgent));
}

#[test]
fn spawn_all_five_topologies() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("s", "t", Topology::Single, 0, 8),
    );
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("o", "t", Topology::OrchestratorSubs, 3, 8),
    );
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("p", "t", Topology::ParallelAggregate, 4, 8),
    );
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("ps", "t", Topology::ProposerSolver, 0, 8),
    );
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("ar", "t", Topology::AuthorReviewer, 0, 8),
    );
    assert_eq!(prod.agents(&eid("p")).unwrap().len(), 5);
    assert_eq!(refer.agents(&eid("p")).unwrap().len(), 5);
}

#[test]
fn spawn_empty_episode_id_is_empty_id_even_if_task_empty() {
    let (mut prod, mut refer) = pair_default();
    let e = assert_both_err(
        prod.spawn_episode(spec("", "", Topology::Single, 0, 8)),
        refer.spawn_episode(spec("", "", Topology::Single, 0, 8)),
    );
    match e {
        MultiError::EmptyId => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn spawn_empty_episode_id_beats_empty_task_and_bad_width_and_zero_budget() {
    let (mut prod, mut refer) = pair_default();
    let e = assert_both_err(
        prod.spawn_episode(spec("", "task", Topology::Single, 7, 0)),
        refer.spawn_episode(spec("", "task", Topology::Single, 7, 0)),
    );
    match e {
        MultiError::EmptyId => {}
        other => panic!("episode-empty should be EmptyId first, got {other:?}"),
    }
}

#[test]
fn spawn_empty_task_id_is_empty_id() {
    let (mut prod, mut refer) = pair_default();
    let e = assert_both_err(
        prod.spawn_episode(spec("ep", "", Topology::Single, 0, 8)),
        refer.spawn_episode(spec("ep", "", Topology::Single, 0, 8)),
    );
    match e {
        MultiError::EmptyId => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn spawn_empty_task_beats_duplicate_check() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("ep", "t", Topology::Single, 0, 8),
    );
    // Same episode, empty task → EmptyId (task check is before duplicate).
    let e = assert_both_err(
        prod.spawn_episode(spec("ep", "", Topology::Single, 0, 8)),
        refer.spawn_episode(spec("ep", "", Topology::Single, 0, 8)),
    );
    match e {
        MultiError::EmptyId => {}
        other => panic!("empty task before duplicate, got {other:?}"),
    }
}

#[test]
fn spawn_duplicate_episode() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("ep", "t1", Topology::Single, 0, 8),
    );
    let e = assert_both_err(
        prod.spawn_episode(spec("ep", "t2", Topology::AuthorReviewer, 0, 99)),
        refer.spawn_episode(spec("ep", "t2", Topology::AuthorReviewer, 0, 99)),
    );
    match e {
        MultiError::DuplicateEpisode(id) => assert_eq!(id, eid("ep")),
        other => panic!("{other:?}"),
    }
    // First episode untouched.
    assert_eq!(prod.topology_of(&eid("ep")).unwrap(), Topology::Single);
    assert_prod_matches_ref(&prod, &refer, &eid("ep"));
}

#[test]
fn spawn_duplicate_beats_bad_width_and_zero_budget() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("ep", "t", Topology::Single, 0, 8),
    );
    let e = assert_both_err(
        prod.spawn_episode(spec("ep", "t", Topology::Single, 3, 0)),
        refer.spawn_episode(spec("ep", "t", Topology::Single, 3, 0)),
    );
    match e {
        MultiError::DuplicateEpisode(id) => assert_eq!(id, eid("ep")),
        other => panic!("duplicate before BadWidth/budget, got {other:?}"),
    }
}

#[test]
fn spawn_bad_width() {
    let (mut prod, mut refer) = pair_default();
    let e = assert_both_err(
        prod.spawn_episode(spec("ep", "t", Topology::Single, 1, 8)),
        refer.spawn_episode(spec("ep", "t", Topology::Single, 1, 8)),
    );
    match e {
        MultiError::BadWidth => {}
        other => panic!("{other:?}"),
    }
    // Failed spawn must not create the episode.
    assert_unknown_episode(&prod.agents(&eid("ep")).unwrap_err(), &eid("ep"));
    assert_unknown_episode(&refer.agents(&eid("ep")).unwrap_err(), &eid("ep"));
}

#[test]
fn spawn_bad_width_beats_zero_budget() {
    let (mut prod, mut refer) = pair_default();
    let e = assert_both_err(
        prod.spawn_episode(spec("ep", "t", Topology::OrchestratorSubs, 0, 0)),
        refer.spawn_episode(spec("ep", "t", Topology::OrchestratorSubs, 0, 0)),
    );
    match e {
        MultiError::BadWidth => {}
        other => panic!("BadWidth before zero budget, got {other:?}"),
    }
}

#[test]
fn spawn_zero_budget_is_message_containing_budget() {
    let (mut prod, mut refer) = pair_default();
    let e = assert_both_err(
        prod.spawn_episode(spec("ep", "t", Topology::Single, 0, 0)),
        refer.spawn_episode(spec("ep", "t", Topology::Single, 0, 0)),
    );
    assert_message_contains(&e, "budget");
    assert_unknown_episode(&prod.tokens_used(&eid("ep")).unwrap_err(), &eid("ep"));
}

#[test]
fn spawn_two_episodes_are_independent() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("a", "t", Topology::Single, 0, 10),
    );
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("b", "t", Topology::ProposerSolver, 0, 99),
    );
    assert_eq!(prod.topology_of(&eid("a")).unwrap(), Topology::Single);
    assert_eq!(
        prod.topology_of(&eid("b")).unwrap(),
        Topology::ProposerSolver
    );
    assert_prod_matches_ref(&prod, &refer, &eid("a"));
    assert_prod_matches_ref(&prod, &refer, &eid("b"));
}

#[test]
fn unknown_episode_on_every_keyed_method_before_any_spawn() {
    let (mut prod, mut refer) = pair_default();
    assert_unknown_episode_on_all_keyed(&mut prod, &mut refer, &eid("ghost"));
}

#[test]
fn unknown_episode_on_every_keyed_method_after_spawning_a_different_id() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("live", "t", Topology::Single, 0, 8),
    );
    assert_unknown_episode_on_all_keyed(&mut prod, &mut refer, &eid("other"));
    // live episode still intact
    assert_prod_matches_ref(&prod, &refer, &eid("live"));
}

#[test]
fn episode_ids_are_case_sensitive() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("E", "t", Topology::Single, 0, 8),
    );
    assert_unknown_episode(&prod.agents(&eid("e")).unwrap_err(), &eid("e"));
    assert_unknown_episode(&refer.agents(&eid("e")).unwrap_err(), &eid("e"));
}
