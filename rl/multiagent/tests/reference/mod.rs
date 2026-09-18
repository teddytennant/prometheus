//! Independent I7 multi-agent reference: slow, obvious CPU state machine.
//!
//! Production (`prometheus_multiagent`) must never import this module. Tests
//! drive `Multiagent` and `RefMultiagent` on every behavior assertion.
//!
//! There is no tensor math in this crate (no NumPy/PyTorch). The "reference"
//! is this state machine. Mix sampling is an integer CDF; latent dropout is
//! integer modulo; token budgets are checked (not saturating) addition.
//!
//! # Locked rules (implementers must match; also see `tests/common`)
//!
//! ## Construction (`Multiagent::new` / `RefMultiagent::new`)
//! - Five weights must sum to `TOPOLOGY_WEIGHT_DENOM` (100) else `BadMix`.
//!   Overflow of the u32 sum is not 100 → `BadMix`.
//! - `weight_single >= WEIGHT_SINGLE` (50) else `BadMix`.
//! - `latent_keep_denom == 0` or `latent_keep_numer > latent_keep_denom` →
//!   `BadMix`. `numer == denom` (always keep) and `numer == 0` with
//!   `denom > 0` (always drop) are allowed.
//! - `default_n_subs` not in `1..=MAX_N_SUBS` (99) → `BadMix`.
//! - Check order: sum → single floor → latent denom/numer → default_n_subs.
//! - `spawn_episode` uses `EpisodeSpec.n_subs`, never `default_n_subs`.
//!
//! ## `topology_from_draw(draw)` (production constants, not instance config)
//! - `u = draw % 100` (`TOPOLOGY_WEIGHT_DENOM`).
//! - CDF order: Single, OrchestratorSubs, ParallelAggregate,
//!   ProposerSolver, AuthorReviewer.
//! - Production ranges (half-open): `0..50`, `50..70`, `70..85`, `85..95`,
//!   `95..100`.
//!
//! ## `sample_topology`
//! - Instance weights, same CDF order, `draw % 100`.
//! - A zero-weight bucket is skipped (empty range: acc does not advance).
//! - Does not mutate episode state.
//!
//! ## `topology_weight`
//! - Always the production `WEIGHT_*` constants, never instance config.
//!
//! ## `roster(topology, n_subs)`
//! - Single: `n_subs` must be 0. Roster: `(solo, Solo)`.
//! - OrchestratorSubs: `n_subs` in `1..=99`. Roster: `(orchestrator,
//!   Orchestrator)` then `(sub-00, SubAgent)` .. `(sub-{n-1}, SubAgent)`
//!   with two-digit zero-pad (`n_subs=10` → last `sub-09`).
//! - ParallelAggregate: `n_subs` in `1..=99`. Roster: `(attempt-00,
//!   Attempt)` .. `(attempt-{n-1}, Attempt)` then `(aggregator, Aggregator)`.
//! - ProposerSolver: `n_subs` must be 0. Roster: `(proposer, Proposer)`,
//!   `(solver, Solver)`.
//! - AuthorReviewer: `n_subs` must be 0. Roster: `(author, Author)`,
//!   `(reviewer, Reviewer)`.
//! - Else `BadWidth`. Ids are those exact ASCII-hyphen strings.
//!
//! ## Spawn
//! - Empty episode id or empty task id → `EmptyId` (episode id first).
//! - Duplicate episode id → `DuplicateEpisode`.
//! - Then `roster` width check (`BadWidth`).
//! - `token_budget == 0` → `Message` containing `"budget"` (case-insensitive).
//! - Returns the roster. Scratch `""`. All inboxes empty. Credit empty.
//!   `tokens_used` 0. `tokens_used_by` each agent 0.
//! - Failed spawn does not create an episode.
//!
//! ## Scratch
//! - Unknown episode → `UnknownEpisode`.
//! - `scratch_append` unknown agent → `UnknownAgent`.
//! - Empty text allowed. Append is concatenation (no extra separator).
//!   Order is call order. Shared across agents of the episode.
//! - Failed append leaves scratch unchanged.
//!
//! ## Messages
//! - Unknown episode → `UnknownEpisode`.
//! - Unknown from or to → `UnknownAgent` (from first).
//! - Empty text allowed. `from == to` allowed (self message still lands
//!   in that inbox only).
//! - `latent_memo == None` stays `None` (`keep_draw` ignored).
//! - `Some`: keep when `keep_draw % latent_keep_denom < latent_keep_numer`,
//!   else drop to `None`. Only the memo field is dropped; text/from/to stay.
//!   Uses instance `latent_keep_*`, not the production constants.
//! - Return the stored message (post-dropout). FIFO per inbox.
//! - Inbox of A never includes messages sent to B (spec 12 isolation).
//!   Outgoing mail is not copied into the sender inbox unless `from == to`.
//! - Failed send leaves inboxes unchanged.
//!
//! ## Credit
//! - Unknown episode → `UnknownEpisode`.
//! - Unknown parent → `UnknownAgent`.
//! - Any used id unknown → `UnknownAgent` (first unknown in `used` order).
//! - `used` may be empty. Duplicates in `used` are stored as given.
//!   Records append. Failed record leaves credit unchanged.
//!
//! ## Tokens
//! - Unknown episode / agent as usual (episode first).
//! - `n == 0` is success, total unchanged (allowed even at budget).
//! - If `tokens_used + n > token_budget` (or the add overflows u64) →
//!   `BudgetExceeded`, charges unchanged. Saturating would be wrong.
//! - Equal to budget is allowed.
//! - `tokens_used` is the sum across agents. `tokens_used_by` is that
//!   agent's charge.
//! - `send_message` / scratch / credit do not consult the token budget.
//!
//! ## Distill (`distill_back`)
//! - Non-finite `multi_reward` or `single_reward` → `InvalidReward`
//!   (check multi first).
//! - `multi_reward > single_reward` → `Some(DistillTarget { task,
//!   token_budget })`.
//! - Else (including equal) → `None`.
//! - Does not look up episodes. Does not require a live episode. Does
//!   not mutate. IS weighting is `rl.loss`, not this crate.
//!
//! Unknown episode on every episode-keyed method. Reads never invent
//! data: unknown agent on `inbox` / `tokens_used_by` is `UnknownAgent`,
//! not an empty/zero placeholder.
//!
//! Production must never import this module.

#![allow(dead_code)]

use prometheus_multiagent::{
    AgentId, CreditRecord, DistillPair, DistillTarget, EpisodeId, EpisodeSpec, Message,
    MultiConfig, MultiError, Result, Role, TaskId, TokenCount, Topology, MAX_N_SUBS,
    TOPOLOGY_WEIGHT_DENOM, WEIGHT_AUTHOR_REVIEWER, WEIGHT_ORCHESTRATOR, WEIGHT_PARALLEL,
    WEIGHT_PROPOSER_SOLVER, WEIGHT_SINGLE,
};
use std::collections::HashMap;

/// Independent copy of the free function. Tests must also call
/// `prometheus_multiagent::topology_weight`.
pub fn topology_weight(topology: Topology) -> u32 {
    match topology {
        Topology::Single => WEIGHT_SINGLE,
        Topology::OrchestratorSubs => WEIGHT_ORCHESTRATOR,
        Topology::ParallelAggregate => WEIGHT_PARALLEL,
        Topology::ProposerSolver => WEIGHT_PROPOSER_SOLVER,
        Topology::AuthorReviewer => WEIGHT_AUTHOR_REVIEWER,
    }
}

/// Independent copy. Uses production constants, not instance config.
pub fn topology_from_draw(draw: u64) -> Topology {
    cdf_from_weights(
        draw,
        WEIGHT_SINGLE,
        WEIGHT_ORCHESTRATOR,
        WEIGHT_PARALLEL,
        WEIGHT_PROPOSER_SOLVER,
        WEIGHT_AUTHOR_REVIEWER,
    )
}

fn cdf_from_weights(
    draw: u64,
    w_single: u32,
    w_orch: u32,
    w_parallel: u32,
    w_ps: u32,
    w_ar: u32,
) -> Topology {
    let u = draw % (TOPOLOGY_WEIGHT_DENOM as u64);
    let buckets = [
        (Topology::Single, w_single),
        (Topology::OrchestratorSubs, w_orch),
        (Topology::ParallelAggregate, w_parallel),
        (Topology::ProposerSolver, w_ps),
        (Topology::AuthorReviewer, w_ar),
    ];
    let mut acc: u64 = 0;
    for (topo, w) in buckets {
        // Zero-weight bucket: empty range, acc does not advance, skipped.
        acc = acc.saturating_add(w as u64);
        if u < acc {
            return topo;
        }
    }
    // Construction requires the five weights to sum to 100, so `u in 0..100`
    // always hits. A zero-weight tail never runs because a prior bucket
    // already consumed the mass.
    unreachable!("CDF weights must sum to TOPOLOGY_WEIGHT_DENOM")
}

/// Independent copy of the free function. Tests must also call
/// `prometheus_multiagent::roster`.
pub fn roster(topology: Topology, n_subs: u32) -> Result<Vec<(AgentId, Role)>> {
    match topology {
        Topology::Single => {
            if n_subs != 0 {
                return Err(MultiError::BadWidth);
            }
            Ok(vec![(AgentId("solo".to_string()), Role::Solo)])
        }
        Topology::OrchestratorSubs => {
            if !(1..=MAX_N_SUBS).contains(&n_subs) {
                return Err(MultiError::BadWidth);
            }
            let mut out = Vec::with_capacity(1 + n_subs as usize);
            out.push((AgentId("orchestrator".to_string()), Role::Orchestrator));
            for i in 0..n_subs {
                out.push((AgentId(format!("sub-{i:02}")), Role::SubAgent));
            }
            Ok(out)
        }
        Topology::ParallelAggregate => {
            if !(1..=MAX_N_SUBS).contains(&n_subs) {
                return Err(MultiError::BadWidth);
            }
            let mut out = Vec::with_capacity(1 + n_subs as usize);
            for i in 0..n_subs {
                out.push((AgentId(format!("attempt-{i:02}")), Role::Attempt));
            }
            out.push((AgentId("aggregator".to_string()), Role::Aggregator));
            Ok(out)
        }
        Topology::ProposerSolver => {
            if n_subs != 0 {
                return Err(MultiError::BadWidth);
            }
            Ok(vec![
                (AgentId("proposer".to_string()), Role::Proposer),
                (AgentId("solver".to_string()), Role::Solver),
            ])
        }
        Topology::AuthorReviewer => {
            if n_subs != 0 {
                return Err(MultiError::BadWidth);
            }
            Ok(vec![
                (AgentId("author".to_string()), Role::Author),
                (AgentId("reviewer".to_string()), Role::Reviewer),
            ])
        }
    }
}

struct EpisodeState {
    topology: Topology,
    token_budget: TokenCount,
    roster: Vec<(AgentId, Role)>,
    scratch: String,
    inboxes: HashMap<AgentId, Vec<Message>>,
    credit: Vec<CreditRecord>,
    tokens_used: TokenCount,
    tokens_by: HashMap<AgentId, TokenCount>,
}

impl EpisodeState {
    fn has_agent(&self, id: &AgentId) -> bool {
        self.roster.iter().any(|(a, _)| a == id)
    }
}

/// Slow reference multi-agent. Same public method names as `Multiagent`.
pub struct RefMultiagent {
    config: MultiConfig,
    episodes: HashMap<EpisodeId, EpisodeState>,
}

impl RefMultiagent {
    pub fn new(config: MultiConfig) -> Result<Self> {
        let sum = config
            .weight_single
            .checked_add(config.weight_orchestrator)
            .and_then(|s| s.checked_add(config.weight_parallel))
            .and_then(|s| s.checked_add(config.weight_proposer_solver))
            .and_then(|s| s.checked_add(config.weight_author_reviewer));
        if sum != Some(TOPOLOGY_WEIGHT_DENOM) {
            return Err(MultiError::BadMix);
        }
        if config.weight_single < WEIGHT_SINGLE {
            return Err(MultiError::BadMix);
        }
        if config.latent_keep_denom == 0 || config.latent_keep_numer > config.latent_keep_denom {
            return Err(MultiError::BadMix);
        }
        if !(1..=MAX_N_SUBS).contains(&config.default_n_subs) {
            return Err(MultiError::BadMix);
        }
        Ok(Self {
            config,
            episodes: HashMap::new(),
        })
    }

    pub fn sample_topology(&self, draw: u64) -> Topology {
        cdf_from_weights(
            draw,
            self.config.weight_single,
            self.config.weight_orchestrator,
            self.config.weight_parallel,
            self.config.weight_proposer_solver,
            self.config.weight_author_reviewer,
        )
    }

    pub fn spawn_episode(&mut self, spec: EpisodeSpec) -> Result<Vec<(AgentId, Role)>> {
        if spec.id.0.is_empty() {
            return Err(MultiError::EmptyId);
        }
        if spec.task.0.is_empty() {
            return Err(MultiError::EmptyId);
        }
        if self.episodes.contains_key(&spec.id) {
            return Err(MultiError::DuplicateEpisode(spec.id));
        }
        let roster = roster(spec.topology, spec.n_subs)?;
        if spec.token_budget == 0 {
            return Err(MultiError::Message(
                "token_budget must be > 0 (budget parity charges every agent against one cap)"
                    .to_string(),
            ));
        }

        let mut inboxes = HashMap::new();
        let mut tokens_by = HashMap::new();
        for (id, _) in &roster {
            inboxes.insert(id.clone(), Vec::new());
            tokens_by.insert(id.clone(), 0);
        }
        let returned = roster.clone();
        self.episodes.insert(
            spec.id,
            EpisodeState {
                topology: spec.topology,
                token_budget: spec.token_budget,
                roster,
                scratch: String::new(),
                inboxes,
                credit: Vec::new(),
                tokens_used: 0,
                tokens_by,
            },
        );
        let _task: TaskId = spec.task;
        Ok(returned)
    }

    pub fn agents(&self, episode: &EpisodeId) -> Result<Vec<(AgentId, Role)>> {
        let ep = self
            .episodes
            .get(episode)
            .ok_or_else(|| MultiError::UnknownEpisode(episode.clone()))?;
        Ok(ep.roster.clone())
    }

    pub fn topology_of(&self, episode: &EpisodeId) -> Result<Topology> {
        let ep = self
            .episodes
            .get(episode)
            .ok_or_else(|| MultiError::UnknownEpisode(episode.clone()))?;
        Ok(ep.topology)
    }

    pub fn scratch_read(&self, episode: &EpisodeId) -> Result<String> {
        let ep = self
            .episodes
            .get(episode)
            .ok_or_else(|| MultiError::UnknownEpisode(episode.clone()))?;
        Ok(ep.scratch.clone())
    }

    pub fn scratch_append(
        &mut self,
        episode: &EpisodeId,
        agent: &AgentId,
        text: &str,
    ) -> Result<()> {
        let ep = self
            .episodes
            .get_mut(episode)
            .ok_or_else(|| MultiError::UnknownEpisode(episode.clone()))?;
        if !ep.has_agent(agent) {
            return Err(MultiError::UnknownAgent(agent.clone()));
        }
        ep.scratch.push_str(text);
        Ok(())
    }

    pub fn send_message(
        &mut self,
        episode: &EpisodeId,
        message: Message,
        keep_draw: u64,
    ) -> Result<Message> {
        let denom = self.config.latent_keep_denom as u64;
        let numer = self.config.latent_keep_numer as u64;
        let ep = self
            .episodes
            .get_mut(episode)
            .ok_or_else(|| MultiError::UnknownEpisode(episode.clone()))?;
        if !ep.has_agent(&message.from) {
            return Err(MultiError::UnknownAgent(message.from));
        }
        if !ep.has_agent(&message.to) {
            return Err(MultiError::UnknownAgent(message.to));
        }
        let mut stored = message;
        if stored.latent_memo.is_some() {
            let keep = keep_draw % denom < numer;
            if !keep {
                stored.latent_memo = None;
            }
        }
        ep.inboxes
            .get_mut(&stored.to)
            .expect("inbox exists for roster agent")
            .push(stored.clone());
        Ok(stored)
    }

    pub fn inbox(&self, episode: &EpisodeId, agent: &AgentId) -> Result<Vec<Message>> {
        let ep = self
            .episodes
            .get(episode)
            .ok_or_else(|| MultiError::UnknownEpisode(episode.clone()))?;
        if !ep.has_agent(agent) {
            return Err(MultiError::UnknownAgent(agent.clone()));
        }
        Ok(ep.inboxes.get(agent).cloned().unwrap_or_default())
    }

    pub fn record_credit(&mut self, episode: &EpisodeId, credit: CreditRecord) -> Result<()> {
        let ep = self
            .episodes
            .get_mut(episode)
            .ok_or_else(|| MultiError::UnknownEpisode(episode.clone()))?;
        if !ep.has_agent(&credit.parent) {
            return Err(MultiError::UnknownAgent(credit.parent));
        }
        for u in &credit.used {
            if !ep.has_agent(u) {
                return Err(MultiError::UnknownAgent(u.clone()));
            }
        }
        ep.credit.push(credit);
        Ok(())
    }

    pub fn credit(&self, episode: &EpisodeId) -> Result<Vec<CreditRecord>> {
        let ep = self
            .episodes
            .get(episode)
            .ok_or_else(|| MultiError::UnknownEpisode(episode.clone()))?;
        Ok(ep.credit.clone())
    }

    pub fn charge_tokens(
        &mut self,
        episode: &EpisodeId,
        agent: &AgentId,
        n: TokenCount,
    ) -> Result<TokenCount> {
        let ep = self
            .episodes
            .get_mut(episode)
            .ok_or_else(|| MultiError::UnknownEpisode(episode.clone()))?;
        if !ep.has_agent(agent) {
            return Err(MultiError::UnknownAgent(agent.clone()));
        }
        let new_total = match ep.tokens_used.checked_add(n) {
            Some(t) => t,
            None => return Err(MultiError::BudgetExceeded),
        };
        if new_total > ep.token_budget {
            return Err(MultiError::BudgetExceeded);
        }
        ep.tokens_used = new_total;
        let slot = ep
            .tokens_by
            .get_mut(agent)
            .expect("tokens_by exists for roster agent");
        *slot = slot.checked_add(n).expect("per-agent fits if total fits");
        Ok(new_total)
    }

    pub fn tokens_used(&self, episode: &EpisodeId) -> Result<TokenCount> {
        let ep = self
            .episodes
            .get(episode)
            .ok_or_else(|| MultiError::UnknownEpisode(episode.clone()))?;
        Ok(ep.tokens_used)
    }

    pub fn tokens_used_by(&self, episode: &EpisodeId, agent: &AgentId) -> Result<TokenCount> {
        let ep = self
            .episodes
            .get(episode)
            .ok_or_else(|| MultiError::UnknownEpisode(episode.clone()))?;
        if !ep.has_agent(agent) {
            return Err(MultiError::UnknownAgent(agent.clone()));
        }
        Ok(ep.tokens_by.get(agent).copied().unwrap_or(0))
    }

    pub fn distill_back(&self, pair: DistillPair) -> Result<Option<DistillTarget>> {
        if !pair.multi_reward.is_finite() {
            return Err(MultiError::InvalidReward);
        }
        if !pair.single_reward.is_finite() {
            return Err(MultiError::InvalidReward);
        }
        if pair.multi_reward > pair.single_reward {
            Ok(Some(DistillTarget {
                task: pair.task,
                token_budget: pair.token_budget,
            }))
        } else {
            Ok(None)
        }
    }
}
