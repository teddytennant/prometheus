//! Group 10: property tests.
//!
//! - lockstep prod vs ref
//! - after every successful spawn, roster len matches topology
//! - tokens_used == sum of tokens_used_by
//! - inbox isolation
//! - mix CDF histogram over 10000 draws is exact counts for production weights

mod common;
mod reference;

use common::*;
use prometheus_multiagent::{
    AgentId, DistillPair, Multiagent, Role, Topology, WEIGHT_AUTHOR_REVIEWER, WEIGHT_ORCHESTRATOR,
    WEIGHT_PARALLEL, WEIGHT_PROPOSER_SOLVER, WEIGHT_SINGLE,
};
use reference::RefMultiagent;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn bounded(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next() % n
        }
    }
}

fn topo_index(t: Topology) -> usize {
    match t {
        Topology::Single => 0,
        Topology::OrchestratorSubs => 1,
        Topology::ParallelAggregate => 2,
        Topology::ProposerSolver => 3,
        Topology::AuthorReviewer => 4,
    }
}

fn histogram(n: u64, mut f: impl FnMut(u64) -> Topology) -> [u64; 5] {
    let mut h = [0u64; 5];
    for d in 0..n {
        h[topo_index(f(d))] += 1;
    }
    h
}

#[test]
fn production_cdf_histogram_10000_exact_counts() {
    let h = histogram(10_000, |d| draw_both(d));
    assert_eq!(h[0], WEIGHT_SINGLE as u64 * 100);
    assert_eq!(h[1], WEIGHT_ORCHESTRATOR as u64 * 100);
    assert_eq!(h[2], WEIGHT_PARALLEL as u64 * 100);
    assert_eq!(h[3], WEIGHT_PROPOSER_SOLVER as u64 * 100);
    assert_eq!(h[4], WEIGHT_AUTHOR_REVIEWER as u64 * 100);
    assert_eq!(h, [5000, 2000, 1500, 1000, 500]);
}

#[test]
fn sample_topology_default_histogram_10000_exact_counts() {
    let (prod, refer) = pair_default();
    let hp = histogram(10_000, |d| prod.sample_topology(d));
    let hr = histogram(10_000, |d| refer.sample_topology(d));
    assert_eq!(hp, hr);
    assert_eq!(hp, [5000, 2000, 1500, 1000, 500]);
}

#[test]
fn sample_topology_custom_zero_bucket_histogram_10000_exact() {
    // 60, 25, 0, 10, 5 → 6000, 2500, 0, 1000, 500 over 10000 sequential draws.
    let (prod, refer) = pair(mix(60, 25, 0, 10, 5));
    let hp = histogram(10_000, |d| prod.sample_topology(d));
    let hr = histogram(10_000, |d| refer.sample_topology(d));
    assert_eq!(hp, hr);
    assert_eq!(hp, [6000, 2500, 0, 1000, 500]);
}

fn valid_n_subs(topology: Topology, rng: &mut Lcg) -> u32 {
    match topology {
        Topology::Single | Topology::ProposerSolver | Topology::AuthorReviewer => 0,
        Topology::OrchestratorSubs | Topology::ParallelAggregate => 1 + rng.bounded(8) as u32,
    }
}

fn pick_agent(roster: &[(AgentId, Role)], rng: &mut Lcg) -> AgentId {
    roster[rng.bounded(roster.len() as u64) as usize].0.clone()
}

fn apply_op(
    rng: &mut Lcg,
    prod: &mut Multiagent,
    refer: &mut RefMultiagent,
    live: &mut Vec<(String, Vec<(AgentId, Role)>, Topology, u32)>,
    next_id: &mut u64,
) {
    match rng.bounded(7) {
        0 => {
            let topologies = [
                Topology::Single,
                Topology::OrchestratorSubs,
                Topology::ParallelAggregate,
                Topology::ProposerSolver,
                Topology::AuthorReviewer,
            ];
            let topology = topologies[rng.bounded(5) as usize];
            let n_subs = valid_n_subs(topology, rng);
            let id = format!("ep{}", *next_id);
            *next_id += 1;
            let budget = 1 + rng.bounded(80);
            let s = spec(&id, "task", topology, n_subs, budget);
            match spawn_both(prod, refer, s) {
                Ok(roster) => {
                    assert_eq!(roster.len(), expected_roster_len(topology, n_subs));
                    live.push((id, roster, topology, n_subs));
                }
                Err(_) => panic!("valid spawn failed"),
            }
        }
        1 => {
            if live.is_empty() {
                return;
            }
            let i = rng.bounded(live.len() as u64) as usize;
            let ep = eid(&live[i].0);
            let agent = pick_agent(&live[i].1, rng);
            let text = if rng.bounded(4) == 0 {
                String::new()
            } else {
                format!("t{}", rng.next())
            };
            assert_both_ok(
                prod.scratch_append(&ep, &agent, &text),
                refer.scratch_append(&ep, &agent, &text),
            );
        }
        2 => {
            if live.is_empty() {
                return;
            }
            let i = rng.bounded(live.len() as u64) as usize;
            let ep = eid(&live[i].0);
            let from = pick_agent(&live[i].1, rng);
            let to = pick_agent(&live[i].1, rng);
            let memo = if rng.bounded(3) == 0 {
                None
            } else {
                latent(vec![rng.bounded(8) as f64 * 0.25])
            };
            let m = prometheus_multiagent::Message {
                from,
                to,
                text: format!("m{}", rng.next()),
                latent_memo: memo,
            };
            let draw = rng.next();
            let p = prod.send_message(&ep, m.clone(), draw);
            let r = refer.send_message(&ep, m, draw);
            match (p, r) {
                (Ok(a), Ok(b)) => assert_eq!(a, b, "send"),
                (Err(e), Err(rf)) => assert_err_eq(&e, &rf),
                other => panic!("send mismatch {other:?}"),
            }
        }
        3 => {
            if live.is_empty() {
                return;
            }
            let i = rng.bounded(live.len() as u64) as usize;
            let ep = eid(&live[i].0);
            let parent = pick_agent(&live[i].1, rng);
            let n_used = rng.bounded(3) as usize;
            let used: Vec<AgentId> = (0..n_used).map(|_| pick_agent(&live[i].1, rng)).collect();
            let rec = prometheus_multiagent::CreditRecord { parent, used };
            assert_both_ok(
                prod.record_credit(&ep, rec.clone()),
                refer.record_credit(&ep, rec),
            );
        }
        4 => {
            if live.is_empty() {
                return;
            }
            let i = rng.bounded(live.len() as u64) as usize;
            let ep = eid(&live[i].0);
            let agent = pick_agent(&live[i].1, rng);
            let n = rng.bounded(40);
            let p = prod.charge_tokens(&ep, &agent, n);
            let r = refer.charge_tokens(&ep, &agent, n);
            match (p, r) {
                (Ok(a), Ok(b)) => assert_eq!(a, b, "charge"),
                (Err(e), Err(rf)) => assert_err_eq(&e, &rf),
                other => panic!("charge mismatch {other:?}"),
            }
        }
        5 => {
            let d = DistillPair {
                task: tid("prop"),
                token_budget: rng.bounded(200),
                multi_reward: (rng.bounded(7) as i64 - 3) as f64,
                single_reward: (rng.bounded(7) as i64 - 3) as f64,
            };
            let p = prod.distill_back(d.clone());
            let r = refer.distill_back(d);
            match (p, r) {
                (Ok(a), Ok(b)) => assert_eq!(a, b, "distill"),
                (Err(e), Err(rf)) => assert_err_eq(&e, &rf),
                other => panic!("distill mismatch {other:?}"),
            }
        }
        _ => {
            // Fault injection: unknown ids / bad spawn.
            let ghost = eid("no-such");
            let _ = assert_both_err(
                prod.inbox(&ghost, &aid("solo")),
                refer.inbox(&ghost, &aid("solo")),
            );
            if let Some((id, roster, _, _)) = live.first() {
                let ep = eid(id);
                let _ = assert_both_err(
                    prod.send_message(&ep, msg("ghost", roster[0].0 .0.as_str(), "x", None), 0),
                    refer.send_message(&ep, msg("ghost", roster[0].0 .0.as_str(), "x", None), 0),
                );
            }
            let bad = spec("dup-bad", "t", Topology::Single, 3, 0);
            let _ = spawn_both(prod, refer, bad);
        }
    }
}

#[test]
fn lockstep_prod_vs_ref_invariants() {
    let (mut prod, mut refer) = pair_default();
    let mut rng = Lcg(0x17_0a_c1e);
    let mut live: Vec<(String, Vec<(AgentId, Role)>, Topology, u32)> = Vec::new();
    let mut next_id = 0u64;

    for _ in 0..400 {
        apply_op(&mut rng, &mut prod, &mut refer, &mut live, &mut next_id);
        for (id, roster, topology, n_subs) in &live {
            let ep = eid(id);
            assert_eq!(roster.len(), expected_roster_len(*topology, *n_subs));
            assert_eq!(prod.agents(&ep).unwrap().len(), roster.len());
            assert_eq!(refer.agents(&ep).unwrap().len(), roster.len());
            assert_eq!(prod.topology_of(&ep).unwrap(), *topology);
            assert_prod_matches_ref(&prod, &refer, &ep);
            assert_inbox_isolation(&prod, &refer, &ep);
            assert_tokens_sum(&prod, &refer, &ep);
        }
    }
    assert!(
        !live.is_empty(),
        "property walk should have spawned at least one episode"
    );
}

#[test]
fn lockstep_error_paths_unknown_and_bad_width() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("live", "t", Topology::OrchestratorSubs, 2, 20),
    );
    // Duplicate
    let e = spawn_both(
        &mut prod,
        &mut refer,
        spec("live", "t2", Topology::Single, 0, 20),
    )
    .unwrap_err();
    match e {
        prometheus_multiagent::MultiError::DuplicateEpisode(id) => assert_eq!(id, eid("live")),
        other => panic!("{other:?}"),
    }
    // Bad width does not create
    let _ = spawn_both(
        &mut prod,
        &mut refer,
        spec("wide", "t", Topology::Single, 2, 20),
    )
    .unwrap_err();
    assert_unknown_episode(&prod.agents(&eid("wide")).unwrap_err(), &eid("wide"));
    assert_unknown_episode(&refer.agents(&eid("wide")).unwrap_err(), &eid("wide"));
    assert_prod_matches_ref(&prod, &refer, &eid("live"));
}

#[test]
fn roster_len_matches_topology_after_each_successful_spawn() {
    let (mut prod, mut refer) = pair_default();
    let cases = [
        (Topology::Single, 0u32),
        (Topology::OrchestratorSubs, 1),
        (Topology::OrchestratorSubs, 10),
        (Topology::ParallelAggregate, 1),
        (Topology::ParallelAggregate, 7),
        (Topology::ProposerSolver, 0),
        (Topology::AuthorReviewer, 0),
    ];
    for (i, (topo, n)) in cases.iter().enumerate() {
        let id = format!("r{i}");
        let roster = spawn_ok(&mut prod, &mut refer, spec(&id, "t", *topo, *n, 16));
        assert_eq!(roster.len(), expected_roster_len(*topo, *n));
        assert_eq!(prod.agents(&eid(&id)).unwrap().len(), roster.len());
        assert_eq!(refer.agents(&eid(&id)).unwrap().len(), roster.len());
    }
}
