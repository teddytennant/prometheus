//! Group 2: `topology_from_draw` production CDF (including wrap),
//! `sample_topology` with a custom mix (still single>=50, sum=100),
//! `topology_weight`.

mod common;
mod reference;

use common::*;
use prometheus_multiagent::{
    Topology, TOPOLOGY_WEIGHT_DENOM, WEIGHT_AUTHOR_REVIEWER, WEIGHT_ORCHESTRATOR, WEIGHT_PARALLEL,
    WEIGHT_PROPOSER_SOLVER, WEIGHT_SINGLE,
};

#[test]
fn topology_weight_is_production_constants() {
    assert_eq!(weight_both(Topology::Single), WEIGHT_SINGLE);
    assert_eq!(weight_both(Topology::OrchestratorSubs), WEIGHT_ORCHESTRATOR);
    assert_eq!(weight_both(Topology::ParallelAggregate), WEIGHT_PARALLEL);
    assert_eq!(
        weight_both(Topology::ProposerSolver),
        WEIGHT_PROPOSER_SOLVER
    );
    assert_eq!(
        weight_both(Topology::AuthorReviewer),
        WEIGHT_AUTHOR_REVIEWER
    );
    let sum = weight_both(Topology::Single)
        + weight_both(Topology::OrchestratorSubs)
        + weight_both(Topology::ParallelAggregate)
        + weight_both(Topology::ProposerSolver)
        + weight_both(Topology::AuthorReviewer);
    assert_eq!(sum, TOPOLOGY_WEIGHT_DENOM);
}

#[test]
fn topology_from_draw_production_cdf_boundaries() {
    // Half-open ranges: 0..50, 50..70, 70..85, 85..95, 95..100.
    assert_eq!(draw_both(0), Topology::Single);
    assert_eq!(draw_both(49), Topology::Single);
    assert_eq!(draw_both(50), Topology::OrchestratorSubs);
    assert_eq!(draw_both(69), Topology::OrchestratorSubs);
    assert_eq!(draw_both(70), Topology::ParallelAggregate);
    assert_eq!(draw_both(84), Topology::ParallelAggregate);
    assert_eq!(draw_both(85), Topology::ProposerSolver);
    assert_eq!(draw_both(94), Topology::ProposerSolver);
    assert_eq!(draw_both(95), Topology::AuthorReviewer);
    assert_eq!(draw_both(99), Topology::AuthorReviewer);
}

#[test]
fn topology_from_draw_wraps_mod_100() {
    assert_eq!(draw_both(100), Topology::Single);
    assert_eq!(draw_both(149), Topology::Single);
    assert_eq!(draw_both(150), Topology::OrchestratorSubs);
    assert_eq!(draw_both(170), Topology::ParallelAggregate);
    assert_eq!(draw_both(185), Topology::ProposerSolver);
    assert_eq!(draw_both(195), Topology::AuthorReviewer);
    assert_eq!(draw_both(199), Topology::AuthorReviewer);
    assert_eq!(draw_both(1000), Topology::Single);
    assert_eq!(draw_both(u64::MAX), draw_both(u64::MAX % 100));
}

#[test]
fn topology_from_draw_covers_every_residue() {
    for d in 0u64..100 {
        let t = draw_both(d);
        let expected = if d < 50 {
            Topology::Single
        } else if d < 70 {
            Topology::OrchestratorSubs
        } else if d < 85 {
            Topology::ParallelAggregate
        } else if d < 95 {
            Topology::ProposerSolver
        } else {
            Topology::AuthorReviewer
        };
        assert_eq!(t, expected, "draw {d}");
    }
}

#[test]
fn sample_topology_default_matches_production_cdf() {
    let (prod, refer) = pair_default();
    for d in 0u64..200 {
        let p = prod.sample_topology(d);
        let r = refer.sample_topology(d);
        assert_eq!(p, r, "sample_topology({d})");
        assert_eq!(p, draw_both(d), "default sample vs topology_from_draw");
    }
}

#[test]
fn sample_topology_custom_mix_still_single_floor_sum_100() {
    // 60, 10, 10, 10, 10. Ranges: 0..60 S, 60..70 O, 70..80 P, 80..90 PS, 90..100 AR.
    let (prod, refer) = pair(mix(60, 10, 10, 10, 10));
    let check = |d: u64, expected: Topology| {
        let p = prod.sample_topology(d);
        let r = refer.sample_topology(d);
        assert_eq!(p, r, "sample {d}");
        assert_eq!(p, expected, "sample {d}");
    };
    check(0, Topology::Single);
    check(59, Topology::Single);
    check(60, Topology::OrchestratorSubs);
    check(69, Topology::OrchestratorSubs);
    check(70, Topology::ParallelAggregate);
    check(79, Topology::ParallelAggregate);
    check(80, Topology::ProposerSolver);
    check(89, Topology::ProposerSolver);
    check(90, Topology::AuthorReviewer);
    check(99, Topology::AuthorReviewer);
    check(160, Topology::OrchestratorSubs);
}

#[test]
fn sample_topology_skips_zero_weight_buckets() {
    // 50, 0, 0, 0, 50. 0..50 Single, 50..100 AuthorReviewer. Middle three skipped.
    let (prod, refer) = pair(mix(50, 0, 0, 0, 50));
    for d in 0u64..50 {
        assert_eq!(prod.sample_topology(d), Topology::Single);
        assert_eq!(refer.sample_topology(d), Topology::Single);
    }
    for d in 50u64..100 {
        assert_eq!(prod.sample_topology(d), Topology::AuthorReviewer);
        assert_eq!(refer.sample_topology(d), Topology::AuthorReviewer);
    }
    assert_eq!(prod.sample_topology(50), Topology::AuthorReviewer);
    assert_eq!(refer.sample_topology(99), Topology::AuthorReviewer);

    // 70, 0, 15, 0, 15. 0..70 S, orch skipped, 70..85 Parallel, proposer skipped, 85..100 AR.
    let (prod, refer) = pair(mix(70, 0, 15, 0, 15));
    assert_eq!(prod.sample_topology(69), Topology::Single);
    assert_eq!(refer.sample_topology(69), Topology::Single);
    assert_eq!(prod.sample_topology(70), Topology::ParallelAggregate);
    assert_eq!(refer.sample_topology(70), Topology::ParallelAggregate);
    assert_eq!(prod.sample_topology(84), Topology::ParallelAggregate);
    assert_eq!(prod.sample_topology(85), Topology::AuthorReviewer);
    assert_eq!(refer.sample_topology(85), Topology::AuthorReviewer);
    assert_eq!(prod.sample_topology(99), Topology::AuthorReviewer);
}

#[test]
fn sample_topology_all_mass_on_single_and_orchestrator() {
    let (prod, refer) = pair(mix(50, 50, 0, 0, 0));
    assert_eq!(prod.sample_topology(49), Topology::Single);
    assert_eq!(refer.sample_topology(49), Topology::Single);
    assert_eq!(prod.sample_topology(50), Topology::OrchestratorSubs);
    assert_eq!(refer.sample_topology(50), Topology::OrchestratorSubs);
    assert_eq!(prod.sample_topology(99), Topology::OrchestratorSubs);
    assert_eq!(refer.sample_topology(99), Topology::OrchestratorSubs);
}

#[test]
fn sample_topology_does_not_mutate_and_ignores_episodes() {
    let (mut prod, mut refer) = pair_default();
    let roster = spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 16),
    );
    let _ = prod.sample_topology(77);
    let _ = refer.sample_topology(77);
    assert_eq!(prod.agents(&eid("e")).unwrap(), roster);
    assert_prod_matches_ref(&prod, &refer, &eid("e"));
}

#[test]
fn topology_from_draw_ignores_instance_mix() {
    // Free function is production constants even if we constructed a custom mix.
    let (_prod, _refer) = pair(mix(100, 0, 0, 0, 0));
    assert_eq!(draw_both(50), Topology::OrchestratorSubs);
    assert_ne!(draw_both(50), Topology::Single);
}
