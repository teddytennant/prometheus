//! Shared builders and assertions for I7 `prometheus-multiagent` oracle tests.
//!
//! Documented rules the implementer must match (also see `tests/reference`):
//!
//! # Construction (`Multiagent::new`)
//! - Five weights must sum to `TOPOLOGY_WEIGHT_DENOM` (100) else `BadMix`.
//!   Overflow of the u32 sum is not 100 → `BadMix`.
//! - `weight_single >= WEIGHT_SINGLE` (50) else `BadMix`.
//! - `latent_keep_denom == 0` or `latent_keep_numer > latent_keep_denom` →
//!   `BadMix`. `numer == denom` and `numer == 0` with `denom > 0` are allowed.
//! - `default_n_subs` not in `1..=MAX_N_SUBS` (99) → `BadMix`.
//! - Check order: sum → single floor → latent denom/numer → default_n_subs.
//! - `spawn_episode` uses `EpisodeSpec.n_subs`, never `default_n_subs`.
//!
//! # `topology_from_draw(draw)` uses production constants, not instance config
//! - `u = draw % 100`
//! - CDF order: Single, OrchestratorSubs, ParallelAggregate, ProposerSolver,
//!   AuthorReviewer
//! - Production ranges: `0..50`, `50..70`, `70..85`, `85..95`, `95..100`
//!   (half-open)
//!
//! # `Multiagent::sample_topology`
//! - Instance weights, same CDF order, `draw % 100`.
//! - A zero-weight bucket is skipped (empty range).
//! - Does not mutate episode state.
//!
//! # `topology_weight` returns the production constants (`WEIGHT_*`).
//!
//! # `roster(topology, n_subs)`
//! - Single: n_subs must be 0. Roster: `(solo, Solo)`
//! - OrchestratorSubs: n_subs in 1..=99. Roster: `(orchestrator,
//!   Orchestrator)` then `(sub-00, SubAgent)` .. `(sub-{n-1}, SubAgent)`
//!   with two-digit zero-pad
//! - ParallelAggregate: n_subs in 1..=99. Roster: `(attempt-00, Attempt)`
//!   .. `(attempt-{n-1}, Attempt)` then `(aggregator, Aggregator)`
//! - ProposerSolver: n_subs must be 0. Roster: `(proposer, Proposer)`,
//!   `(solver, Solver)`
//! - AuthorReviewer: n_subs must be 0. Roster: `(author, Author)`,
//!   `(reviewer, Reviewer)`
//! - Else `BadWidth`. Ids are the exact strings above (ASCII hyphen).
//!
//! # Spawn
//! - Empty episode id or empty task id → `EmptyId` (check episode id first).
//! - Duplicate episode id → `DuplicateEpisode`.
//! - Then `roster` width check (`BadWidth`).
//! - `token_budget == 0` → `Message` containing `"budget"` (case-insensitive).
//! - Returns the roster. Scratch empty string. All inboxes empty. Credit
//!   empty. tokens_used 0. tokens_used_by each agent 0.
//! - Failed spawn does not create an episode. Failed mutators leave state
//!   unchanged.
//!
//! # Scratch
//! - Unknown episode → UnknownEpisode.
//! - `scratch_append` unknown agent → UnknownAgent.
//! - Empty text allowed. Append is concatenation (no extra separator).
//!   Order is call order. Shared pad for the episode.
//!
//! # Messages
//! - Unknown episode → UnknownEpisode.
//! - Unknown from or to → UnknownAgent (check from first).
//! - Empty text allowed. from == to allowed (self message still lands in
//!   that inbox).
//! - If latent_memo is None, stored None (`keep_draw` ignored).
//! - If Some: keep when `keep_draw % latent_keep_denom < latent_keep_numer`,
//!   else drop to None. Instance `latent_keep_*`, not production constants.
//! - Return the stored message (post-dropout). FIFO per inbox.
//! - `inbox` of agent A never includes messages sent to B. That is isolation
//!   (spec 12 private inbox). Sender does not get a copy unless from == to.
//!
//! # Credit
//! - Unknown parent → UnknownAgent.
//! - Any used id unknown → UnknownAgent (first unknown in `used` order).
//! - `used` may be empty. Duplicates in `used` are stored as given.
//!   Records append.
//!
//! # Tokens
//! - Unknown episode / agent as usual (episode first).
//! - n==0 is success, total unchanged.
//! - If tokens_used + n > token_budget (or add overflows u64) →
//!   BudgetExceeded, charges unchanged (saturating would be wrong).
//! - Equal to budget is allowed.
//! - tokens_used is the sum across agents. tokens_used_by is that agent's
//!   charge.
//! - Messaging / scratch / credit do not consult the token budget.
//!
//! # Distill (`distill_back`)
//! - Non-finite multi_reward or single_reward → InvalidReward (check multi
//!   first).
//! - multi_reward > single_reward → Some(DistillTarget { task, token_budget })
//! - else (including equal) → None
//! - Does not look up episodes. Does not require a live Multiagent episode.
//!   Does not mutate. IS weighting is `rl.loss`, not this crate.
//!
//! Unknown episode on every episode-keyed method. Reads never invent data.
//!
//! Production must never import this module.

#![allow(dead_code)]

use crate::reference::RefMultiagent;
use prometheus_multiagent::{
    AgentId, CreditRecord, EpisodeId, EpisodeSpec, LatentMemo, Message, MultiConfig, MultiError,
    Multiagent, Result, Role, TaskId, TokenCount, Topology, DEFAULT_N_SUBS, LATENT_MEMO_KEEP_DENOM,
    LATENT_MEMO_KEEP_NUMER, MAX_N_SUBS, TOPOLOGY_WEIGHT_DENOM, WEIGHT_AUTHOR_REVIEWER,
    WEIGHT_ORCHESTRATOR, WEIGHT_PARALLEL, WEIGHT_PROPOSER_SOLVER, WEIGHT_SINGLE,
};

pub fn eid(s: &str) -> EpisodeId {
    EpisodeId(s.to_string())
}

pub fn aid(s: &str) -> AgentId {
    AgentId(s.to_string())
}

pub fn tid(s: &str) -> TaskId {
    TaskId(s.to_string())
}

pub fn default_config() -> MultiConfig {
    MultiConfig::default()
}

/// Custom mix. Caller is responsible for a legal mix when used with `pair`.
pub fn mix(
    weight_single: u32,
    weight_orchestrator: u32,
    weight_parallel: u32,
    weight_proposer_solver: u32,
    weight_author_reviewer: u32,
) -> MultiConfig {
    MultiConfig {
        weight_single,
        weight_orchestrator,
        weight_parallel,
        weight_proposer_solver,
        weight_author_reviewer,
        latent_keep_numer: LATENT_MEMO_KEEP_NUMER,
        latent_keep_denom: LATENT_MEMO_KEEP_DENOM,
        default_n_subs: DEFAULT_N_SUBS,
    }
}

pub fn spec(
    episode: &str,
    task: &str,
    topology: Topology,
    n_subs: u32,
    token_budget: TokenCount,
) -> EpisodeSpec {
    EpisodeSpec {
        id: eid(episode),
        task: tid(task),
        topology,
        n_subs,
        token_budget,
    }
}

pub fn latent(values: Vec<f64>) -> Option<LatentMemo> {
    Some(LatentMemo { values })
}

pub fn msg(from: &str, to: &str, text: &str, memo: Option<LatentMemo>) -> Message {
    Message {
        from: aid(from),
        to: aid(to),
        text: text.to_string(),
        latent_memo: memo,
    }
}

pub fn cred(parent: &str, used: Vec<AgentId>) -> CreditRecord {
    CreditRecord {
        parent: aid(parent),
        used,
    }
}

pub fn pair(cfg: MultiConfig) -> (Multiagent, RefMultiagent) {
    let prod = match Multiagent::new(cfg.clone()) {
        Ok(m) => m,
        Err(e) => panic!("Multiagent::new failed: {e:?}"),
    };
    let refer = match RefMultiagent::new(cfg) {
        Ok(m) => m,
        Err(e) => panic!("RefMultiagent::new failed: {e:?}"),
    };
    (prod, refer)
}

pub fn pair_default() -> (Multiagent, RefMultiagent) {
    pair(default_config())
}

/// `Multiagent` is not `Debug`, so `Result::expect_err` cannot be used on `new`.
pub fn new_err_both(cfg: MultiConfig) -> MultiError {
    let refer = match RefMultiagent::new(cfg.clone()) {
        Ok(_) => panic!("RefMultiagent::new succeeded, expected error"),
        Err(e) => e,
    };
    let prod = match Multiagent::new(cfg) {
        Ok(_) => panic!("Multiagent::new succeeded, expected error"),
        Err(e) => e,
    };
    assert_err_eq(&prod, &refer);
    prod
}

pub fn err_tag(err: &MultiError) -> &'static str {
    match err {
        MultiError::UnknownEpisode(_) => "UnknownEpisode",
        MultiError::DuplicateEpisode(_) => "DuplicateEpisode",
        MultiError::UnknownAgent(_) => "UnknownAgent",
        MultiError::EmptyId => "EmptyId",
        MultiError::BadMix => "BadMix",
        MultiError::BadWidth => "BadWidth",
        MultiError::BudgetExceeded => "BudgetExceeded",
        MultiError::InvalidReward => "InvalidReward",
        MultiError::Message(_) => "Message",
    }
}

pub fn assert_err_eq(prod: &MultiError, refer: &MultiError) {
    match (prod, refer) {
        (MultiError::UnknownEpisode(a), MultiError::UnknownEpisode(b)) => assert_eq!(a, b),
        (MultiError::DuplicateEpisode(a), MultiError::DuplicateEpisode(b)) => assert_eq!(a, b),
        (MultiError::UnknownAgent(a), MultiError::UnknownAgent(b)) => assert_eq!(a, b),
        (MultiError::EmptyId, MultiError::EmptyId) => {}
        (MultiError::BadMix, MultiError::BadMix) => {}
        (MultiError::BadWidth, MultiError::BadWidth) => {}
        (MultiError::BudgetExceeded, MultiError::BudgetExceeded) => {}
        (MultiError::InvalidReward, MultiError::InvalidReward) => {}
        (MultiError::Message(_), MultiError::Message(_)) => {}
        (a, b) => panic!(
            "prod error {a:?} != ref error {b:?} ({} vs {})",
            err_tag(a),
            err_tag(b)
        ),
    }
}

pub fn assert_both_ok<T, U>(prod: Result<T>, refer: Result<U>) {
    match (prod, refer) {
        (Ok(_), Ok(_)) => {}
        (Err(e), Err(r)) => panic!("both erred: prod {e:?} ref {r:?}"),
        (Ok(_), Err(r)) => panic!("prod Ok, ref Err {r:?}"),
        (Err(e), Ok(_)) => panic!("prod Err {e:?}, ref Ok"),
    }
}

pub fn assert_both_err<T: std::fmt::Debug, U: std::fmt::Debug>(
    prod: Result<T>,
    refer: Result<U>,
) -> MultiError {
    match (prod, refer) {
        (Err(e), Err(r)) => {
            assert_err_eq(&e, &r);
            e
        }
        (Ok(v), Ok(w)) => panic!("both Ok: prod {v:?} ref {w:?}"),
        (Ok(v), Err(r)) => panic!("prod Ok {v:?}, ref Err {r:?}"),
        (Err(e), Ok(w)) => panic!("prod Err {e:?}, ref Ok {w:?}"),
    }
}

pub fn assert_message_contains(err: &MultiError, needle: &str) {
    match err {
        MultiError::Message(s) => {
            assert!(
                s.to_lowercase().contains(&needle.to_lowercase()),
                "Message {s:?} does not contain {needle:?}"
            );
        }
        other => panic!("expected Message containing {needle:?}, got {other:?}"),
    }
}

pub fn assert_unknown_episode(err: &MultiError, episode: &EpisodeId) {
    match err {
        MultiError::UnknownEpisode(id) => assert_eq!(id, episode),
        other => panic!("expected UnknownEpisode({episode:?}), got {other:?}"),
    }
}

pub fn assert_unknown_agent(err: &MultiError, agent: &AgentId) {
    match err {
        MultiError::UnknownAgent(id) => assert_eq!(id, agent),
        other => panic!("expected UnknownAgent({agent:?}), got {other:?}"),
    }
}

pub fn draw_both(draw: u64) -> Topology {
    let prod = prometheus_multiagent::topology_from_draw(draw);
    let refer = crate::reference::topology_from_draw(draw);
    assert_eq!(prod, refer, "topology_from_draw({draw})");
    prod
}

pub fn weight_both(topology: Topology) -> u32 {
    let prod = prometheus_multiagent::topology_weight(topology);
    let refer = crate::reference::topology_weight(topology);
    assert_eq!(prod, refer, "topology_weight({topology:?})");
    prod
}

pub fn roster_both(topology: Topology, n_subs: u32) -> Result<Vec<(AgentId, Role)>> {
    let refer = crate::reference::roster(topology, n_subs);
    let prod = prometheus_multiagent::roster(topology, n_subs);
    match (prod, refer) {
        (Ok(a), Ok(b)) => {
            assert_eq!(a, b, "roster({topology:?}, {n_subs})");
            Ok(a)
        }
        (Err(e), Err(r)) => {
            assert_err_eq(&e, &r);
            Err(e)
        }
        (Ok(a), Err(r)) => panic!("prod roster Ok {a:?}, ref Err {r:?}"),
        (Err(e), Ok(b)) => panic!("prod roster Err {e:?}, ref Ok {b:?}"),
    }
}

pub fn expected_roster_len(topology: Topology, n_subs: u32) -> usize {
    match topology {
        Topology::Single => 1,
        Topology::OrchestratorSubs | Topology::ParallelAggregate => 1 + n_subs as usize,
        Topology::ProposerSolver | Topology::AuthorReviewer => 2,
    }
}

pub fn spawn_both(
    prod: &mut Multiagent,
    refer: &mut RefMultiagent,
    spec: EpisodeSpec,
) -> Result<Vec<(AgentId, Role)>> {
    let p = prod.spawn_episode(spec.clone());
    let r = refer.spawn_episode(spec);
    match (p, r) {
        (Ok(a), Ok(b)) => {
            assert_eq!(a, b, "spawn roster");
            Ok(a)
        }
        (Err(e), Err(rf)) => {
            assert_err_eq(&e, &rf);
            Err(e)
        }
        (Ok(a), Err(rf)) => panic!("prod spawn Ok {a:?}, ref Err {rf:?}"),
        (Err(e), Ok(b)) => panic!("prod spawn Err {e:?}, ref Ok {b:?}"),
    }
}

pub fn spawn_ok(
    prod: &mut Multiagent,
    refer: &mut RefMultiagent,
    spec: EpisodeSpec,
) -> Vec<(AgentId, Role)> {
    let ep = spec.id.clone();
    let topology = spec.topology;
    let n_subs = spec.n_subs;
    match spawn_both(prod, refer, spec) {
        Ok(roster) => {
            assert_eq!(
                roster.len(),
                expected_roster_len(topology, n_subs),
                "roster len vs topology"
            );
            assert_episode_initial(prod, refer, &ep, &roster);
            roster
        }
        Err(e) => panic!("spawn_ok failed: {e:?}"),
    }
}

pub fn assert_episode_initial(
    prod: &Multiagent,
    refer: &RefMultiagent,
    episode: &EpisodeId,
    roster: &[(AgentId, Role)],
) {
    assert_eq!(prod.scratch_read(episode).unwrap(), "");
    assert_eq!(refer.scratch_read(episode).unwrap(), "");
    assert_eq!(prod.credit(episode).unwrap(), Vec::new());
    assert_eq!(refer.credit(episode).unwrap(), Vec::new());
    assert_eq!(prod.tokens_used(episode).unwrap(), 0);
    assert_eq!(refer.tokens_used(episode).unwrap(), 0);
    assert_eq!(prod.agents(episode).unwrap(), roster);
    assert_eq!(refer.agents(episode).unwrap(), roster);
    for (id, _) in roster {
        assert_eq!(prod.inbox(episode, id).unwrap(), Vec::new(), "inbox {id:?}");
        assert_eq!(refer.inbox(episode, id).unwrap(), Vec::new());
        assert_eq!(prod.tokens_used_by(episode, id).unwrap(), 0);
        assert_eq!(refer.tokens_used_by(episode, id).unwrap(), 0);
    }
}

pub fn assert_inbox_isolation(prod: &Multiagent, refer: &RefMultiagent, episode: &EpisodeId) {
    let agents = prod.agents(episode).expect("prod agents");
    assert_eq!(agents, refer.agents(episode).expect("ref agents"));
    for (id, _) in &agents {
        let pin = prod.inbox(episode, id).expect("prod inbox");
        let rin = refer.inbox(episode, id).expect("ref inbox");
        assert_eq!(pin, rin, "inbox {id:?}");
        for m in &pin {
            assert_eq!(
                &m.to, id,
                "inbox isolation: mail for {:?} landed in {:?}",
                m.to, id
            );
        }
    }
}

pub fn assert_tokens_sum(prod: &Multiagent, refer: &RefMultiagent, episode: &EpisodeId) {
    let agents = prod.agents(episode).expect("prod agents");
    let mut psum: TokenCount = 0;
    let mut rsum: TokenCount = 0;
    for (id, _) in &agents {
        let p = prod.tokens_used_by(episode, id).expect("prod by");
        let r = refer.tokens_used_by(episode, id).expect("ref by");
        assert_eq!(p, r, "tokens_used_by {id:?}");
        psum = psum.checked_add(p).expect("psum");
        rsum = rsum.checked_add(r).expect("rsum");
    }
    assert_eq!(prod.tokens_used(episode).expect("prod total"), psum);
    assert_eq!(refer.tokens_used(episode).expect("ref total"), rsum);
    assert_eq!(psum, rsum);
}

pub fn assert_prod_matches_ref(prod: &Multiagent, refer: &RefMultiagent, episode: &EpisodeId) {
    assert_eq!(
        prod.topology_of(episode).expect("prod topo"),
        refer.topology_of(episode).expect("ref topo"),
        "topology_of"
    );
    assert_eq!(
        prod.agents(episode).expect("prod agents"),
        refer.agents(episode).expect("ref agents"),
        "agents"
    );
    assert_eq!(
        prod.scratch_read(episode).expect("prod scratch"),
        refer.scratch_read(episode).expect("ref scratch"),
        "scratch"
    );
    assert_eq!(
        prod.credit(episode).expect("prod credit"),
        refer.credit(episode).expect("ref credit"),
        "credit"
    );
    assert_inbox_isolation(prod, refer, episode);
    assert_tokens_sum(prod, refer, episode);
}

pub fn assert_unknown_episode_on_all_keyed(
    prod: &mut Multiagent,
    refer: &mut RefMultiagent,
    ghost: &EpisodeId,
) {
    assert_unknown_episode(&prod.agents(ghost).unwrap_err(), ghost);
    assert_unknown_episode(&refer.agents(ghost).unwrap_err(), ghost);

    assert_unknown_episode(&prod.topology_of(ghost).unwrap_err(), ghost);
    assert_unknown_episode(&refer.topology_of(ghost).unwrap_err(), ghost);

    assert_unknown_episode(&prod.scratch_read(ghost).unwrap_err(), ghost);
    assert_unknown_episode(&refer.scratch_read(ghost).unwrap_err(), ghost);

    let e = assert_both_err(
        prod.scratch_append(ghost, &aid("solo"), "x"),
        refer.scratch_append(ghost, &aid("solo"), "x"),
    );
    assert_unknown_episode(&e, ghost);

    let e = assert_both_err(
        prod.send_message(ghost, msg("solo", "solo", "hi", None), 0),
        refer.send_message(ghost, msg("solo", "solo", "hi", None), 0),
    );
    assert_unknown_episode(&e, ghost);

    let e = assert_both_err(
        prod.inbox(ghost, &aid("solo")),
        refer.inbox(ghost, &aid("solo")),
    );
    assert_unknown_episode(&e, ghost);

    let e = assert_both_err(
        prod.record_credit(ghost, cred("solo", vec![])),
        refer.record_credit(ghost, cred("solo", vec![])),
    );
    assert_unknown_episode(&e, ghost);

    let e = assert_both_err(prod.credit(ghost), refer.credit(ghost));
    assert_unknown_episode(&e, ghost);

    let e = assert_both_err(
        prod.charge_tokens(ghost, &aid("solo"), 0),
        refer.charge_tokens(ghost, &aid("solo"), 0),
    );
    assert_unknown_episode(&e, ghost);

    let e = assert_both_err(prod.tokens_used(ghost), refer.tokens_used(ghost));
    assert_unknown_episode(&e, ghost);

    let e = assert_both_err(
        prod.tokens_used_by(ghost, &aid("solo")),
        refer.tokens_used_by(ghost, &aid("solo")),
    );
    assert_unknown_episode(&e, ghost);
}

/// Silence unused-import warnings for constants tests/common re-exports.
pub fn _constants_are_9_5() {
    let _ = (
        WEIGHT_SINGLE,
        WEIGHT_ORCHESTRATOR,
        WEIGHT_PARALLEL,
        WEIGHT_PROPOSER_SOLVER,
        WEIGHT_AUTHOR_REVIEWER,
        TOPOLOGY_WEIGHT_DENOM,
        LATENT_MEMO_KEEP_NUMER,
        LATENT_MEMO_KEEP_DENOM,
        DEFAULT_N_SUBS,
        MAX_N_SUBS,
    );
}
