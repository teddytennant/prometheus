//! Multi-agent topologies, messaging, credit, distill-back (spec 9.3, 9.5, 15.5 I7).
//!
//! CPU analog: simulated episodes, text messages, a shared scratch-pad, integer
//! token budgets. No GPUs, no JAX, no SGLang, no Firecracker, no live internet.
//! Isolation (spec 12): each agent has a private inbox; other agents cannot
//! read it. The scratch-pad is the shared channel. GSPO/DAPO math lives in
//! Python `rl.loss`; this crate does not import it. The coordinator, rewards,
//! and env sandbox are not crate dependencies.
//!
//! Spec 9.3 names debate, committee, hierarchical, adversarial. Spec 9.5 is
//! the training mix this crate samples. Mapping:
//! - [`Topology::OrchestratorSubs`] = hierarchical (20%)
//! - [`Topology::ParallelAggregate`] = committee (15%)
//! - [`Topology::ProposerSolver`] = adversarial (10%)
//! - [`Topology::AuthorReviewer`] = debate (5%)
//! - [`Topology::Single`] = single-agent baseline (≥50%, locked at 50 in production)
//!
//! Spec 9.5, CPU contract:
//! - Each RL episode samples a topology from the mix.
//! - Messages are text. An optional latent memo is dropped
//!   [`LATENT_MEMO_KEEP_NUMER`]/[`LATENT_MEMO_KEEP_DENOM`] of the time.
//! - Credit assignment records which sub-agent outputs the parent used.
//! - Distill-back: orchestrated solutions that beat single-agent on the same
//!   task at the same total token budget become off-policy positives. IS
//!   weighting is `rl.loss`, not this crate.
//! - Budget parity: tokens are charged across all agents against one cap.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tokens charged against an episode's shared budget (spec 9.5 budget parity).
pub type TokenCount = u64;

/// Spec 9.5: single-agent share of episodes. Locked at 50 of 100 (the floor).
pub const WEIGHT_SINGLE: u32 = 50;

/// Spec 9.5: orchestrator + parallel sub-agents.
pub const WEIGHT_ORCHESTRATOR: u32 = 20;

/// Spec 9.5: N parallel attempts + aggregator.
pub const WEIGHT_PARALLEL: u32 = 15;

/// Spec 9.5: proposer ↔ solver self-play.
pub const WEIGHT_PROPOSER_SOLVER: u32 = 10;

/// Spec 9.5: author ↔ reviewer.
pub const WEIGHT_AUTHOR_REVIEWER: u32 = 5;

/// Denominator for the 9.5 mix. The five weights sum to this.
pub const TOPOLOGY_WEIGHT_DENOM: u32 = 100;

/// Keep rate for the optional latent memo (dropped the rest of the time).
pub const LATENT_MEMO_KEEP_NUMER: u32 = 50;

/// Denominator for [`LATENT_MEMO_KEEP_NUMER`].
pub const LATENT_MEMO_KEEP_DENOM: u32 = 100;

/// Default parallel width for orchestrator-subs and parallel-aggregate.
pub const DEFAULT_N_SUBS: u32 = 2;

/// Maximum `n_subs` for variable-width topologies. Roster ids are two-digit.
pub const MAX_N_SUBS: u32 = 99;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EpisodeId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub String);

/// Spec 9.5 episode topologies. 9.3 names are in the crate docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Topology {
    Single,
    OrchestratorSubs,
    ParallelAggregate,
    ProposerSolver,
    AuthorReviewer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Solo,
    Orchestrator,
    SubAgent,
    Attempt,
    Aggregator,
    Proposer,
    Solver,
    Author,
    Reviewer,
}

/// CPU analog of a few vectors passed alongside a text message (spec 9.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatentMemo {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub from: AgentId,
    pub to: AgentId,
    pub text: String,
    pub latent_memo: Option<LatentMemo>,
}

/// Mix and dropout knobs. Production is the 9.5 constants.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiConfig {
    pub weight_single: u32,
    pub weight_orchestrator: u32,
    pub weight_parallel: u32,
    pub weight_proposer_solver: u32,
    pub weight_author_reviewer: u32,
    pub latent_keep_numer: u32,
    pub latent_keep_denom: u32,
    pub default_n_subs: u32,
}

impl Default for MultiConfig {
    fn default() -> Self {
        Self {
            weight_single: WEIGHT_SINGLE,
            weight_orchestrator: WEIGHT_ORCHESTRATOR,
            weight_parallel: WEIGHT_PARALLEL,
            weight_proposer_solver: WEIGHT_PROPOSER_SOLVER,
            weight_author_reviewer: WEIGHT_AUTHOR_REVIEWER,
            latent_keep_numer: LATENT_MEMO_KEEP_NUMER,
            latent_keep_denom: LATENT_MEMO_KEEP_DENOM,
            default_n_subs: DEFAULT_N_SUBS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeSpec {
    pub id: EpisodeId,
    pub task: TaskId,
    pub topology: Topology,
    /// Parallel width. Must be 0 for [`Topology::Single`],
    /// [`Topology::ProposerSolver`], and [`Topology::AuthorReviewer`].
    /// Must be in 1..=[`MAX_N_SUBS`] for [`Topology::OrchestratorSubs`] and
    /// [`Topology::ParallelAggregate`].
    pub n_subs: u32,
    /// Total token budget charged across all agents (spec 9.5 budget parity).
    pub token_budget: TokenCount,
}

/// Spec 9.3: which sub-agent outputs the parent used.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreditRecord {
    pub parent: AgentId,
    pub used: Vec<AgentId>,
}

/// Two completed attempts on the same task at the same token budget.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistillPair {
    pub task: TaskId,
    pub token_budget: TokenCount,
    pub multi_reward: f64,
    pub single_reward: f64,
}

/// Off-policy positive for the single agent (spec 9.5 distill-back).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistillTarget {
    pub task: TaskId,
    pub token_budget: TokenCount,
}

#[derive(Debug, thiserror::Error)]
pub enum MultiError {
    #[error("episode not found: {0:?}")]
    UnknownEpisode(EpisodeId),
    #[error("duplicate episode id: {0:?}")]
    DuplicateEpisode(EpisodeId),
    #[error("agent not found: {0:?}")]
    UnknownAgent(AgentId),
    #[error("empty id")]
    EmptyId,
    #[error("topology mix is invalid")]
    BadMix,
    #[error("n_subs does not match the topology")]
    BadWidth,
    #[error("token budget exceeded")]
    BudgetExceeded,
    #[error("invalid reward")]
    InvalidReward,
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, MultiError>;

/// Production 9.5 weight for `topology`.
pub fn topology_weight(topology: Topology) -> u32 {
    match topology {
        Topology::Single => WEIGHT_SINGLE,
        Topology::OrchestratorSubs => WEIGHT_ORCHESTRATOR,
        Topology::ParallelAggregate => WEIGHT_PARALLEL,
        Topology::ProposerSolver => WEIGHT_PROPOSER_SOLVER,
        Topology::AuthorReviewer => WEIGHT_AUTHOR_REVIEWER,
    }
}

/// Map `draw % TOPOLOGY_WEIGHT_DENOM` through the production 9.5 CDF.
///
/// Order: Single, OrchestratorSubs, ParallelAggregate, ProposerSolver,
/// AuthorReviewer. Ranges at production weights: `0..50`, `50..70`,
/// `70..85`, `85..95`, `95..100`.
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
        acc = acc.saturating_add(w as u64);
        if u < acc {
            return topo;
        }
    }
    unreachable!("CDF weights must sum to TOPOLOGY_WEIGHT_DENOM")
}

/// Deterministic roster for a topology.
///
/// Ids (zero-padded two-digit indices):
/// - Single: `solo`
/// - OrchestratorSubs: `orchestrator`, `sub-00` .. `sub-{n_subs-1}`
/// - ParallelAggregate: `attempt-00` .. `attempt-{n_subs-1}`, `aggregator`
/// - ProposerSolver: `proposer`, `solver`
/// - AuthorReviewer: `author`, `reviewer`
///
/// Errors [`MultiError::BadWidth`] when `n_subs` does not match the topology.
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

/// Multi-agent episode book: roster, scratch-pad, inboxes, credit, tokens.
pub struct Multiagent {
    config: MultiConfig,
    episodes: HashMap<EpisodeId, EpisodeState>,
}

impl Multiagent {
    /// Reject [`MultiError::BadMix`] unless the five weights sum to
    /// [`TOPOLOGY_WEIGHT_DENOM`], `weight_single >= WEIGHT_SINGLE`,
    /// `latent_keep_denom > 0`, `latent_keep_numer <= latent_keep_denom`,
    /// and `default_n_subs` is in 1..=[`MAX_N_SUBS`].
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

    /// CDF walk of this instance's weights. `draw % TOPOLOGY_WEIGHT_DENOM`.
    /// Same topology order as [`topology_from_draw`].
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

    /// Spawn an episode. Returns the roster. Empty episode or task id is
    /// [`MultiError::EmptyId`]. Duplicate episode id is
    /// [`MultiError::DuplicateEpisode`]. `n_subs` must match the topology
    /// ([`roster`]). `token_budget == 0` is [`MultiError::Message`] whose
    /// text contains "budget" (case-insensitive).
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

    /// Shared scratch-pad. Empty string until the first append.
    pub fn scratch_read(&self, episode: &EpisodeId) -> Result<String> {
        let ep = self
            .episodes
            .get(episode)
            .ok_or_else(|| MultiError::UnknownEpisode(episode.clone()))?;
        Ok(ep.scratch.clone())
    }

    /// Append `text` to the shared scratch-pad. Unknown agent is
    /// [`MultiError::UnknownAgent`]. Empty `text` is allowed.
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

    /// Store `msg` in `msg.to`'s inbox. If `msg.latent_memo` is `Some`, keep
    /// it when `keep_draw % latent_keep_denom < latent_keep_numer`, else
    /// drop it to `None`. Returns the stored message (post-dropout).
    ///
    /// `from` and `to` must be agents of this episode. Empty `text` is
    /// allowed. There is no method that reads another agent's inbox:
    /// [`Self::inbox`] is per-agent (spec 12 isolation).
    pub fn send_message(
        &mut self,
        episode: &EpisodeId,
        msg: Message,
        keep_draw: u64,
    ) -> Result<Message> {
        let denom = self.config.latent_keep_denom as u64;
        let numer = self.config.latent_keep_numer as u64;
        let ep = self
            .episodes
            .get_mut(episode)
            .ok_or_else(|| MultiError::UnknownEpisode(episode.clone()))?;
        if !ep.has_agent(&msg.from) {
            return Err(MultiError::UnknownAgent(msg.from));
        }
        if !ep.has_agent(&msg.to) {
            return Err(MultiError::UnknownAgent(msg.to));
        }
        let mut stored = msg;
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

    /// Private inbox for `agent`. FIFO. Unknown agent is
    /// [`MultiError::UnknownAgent`].
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

    /// Record that `credit.parent` used `credit.used`. Parent and every used
    /// id must be agents of this episode. `used` may be empty. Duplicate
    /// records append.
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

    /// Add `n` tokens to `agent`'s charge and the episode total. Returns the
    /// new episode total. If the total would exceed `token_budget`, return
    /// [`MultiError::BudgetExceeded`] and leave charges unchanged. `n == 0`
    /// is a no-op success.
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

    /// Spec 9.5 distill-back. Does not require a live episode.
    ///
    /// - Non-finite reward: [`MultiError::InvalidReward`].
    /// - `multi_reward > single_reward`: `Ok(Some(DistillTarget))`.
    /// - Else `Ok(None)` (tie does not distill).
    ///
    /// Same `task` and `token_budget` are the caller's contract; this method
    /// does not look up episodes. IS weighting is `rl.loss`.
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
