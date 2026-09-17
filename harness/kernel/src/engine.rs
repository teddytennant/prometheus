//! Kernel state machine: agents, budgets, jobs, bus, patches, tickets, ledger.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::mirror;
use crate::query;
use crate::{
    AgentHandle, AgentId, Error, JobHandle, JobId, JobRequest, KernelConfig, Message, MessageId,
    NowMs, Patch, PatchId, PatchState, PatchTarget, Quota, Record, RecordId, Result, Rung,
    Signature, SpawnRequest, Ticket, TicketId, WeightPromoteState, CANARY_MS, KERNEL_INVARIANT,
    MAX_SPAWN_DEPTH,
};

const WEIGHT_SIGN_OFF: usize = 2;

#[derive(Clone, Serialize, Deserialize)]
struct DiskAgent {
    handle: AgentHandle,
    remaining: Quota,
}

#[derive(Clone, Serialize, Deserialize, Default)]
struct DiskMeta {
    next_agent: u64,
}

#[derive(Clone)]
struct Agent {
    handle: AgentHandle,
    remaining: Quota,
}

struct StoredPatch {
    patch: Patch,
    canary_start: Option<NowMs>,
}

pub struct Engine {
    log_dir: PathBuf,
    mirror_root: PathBuf,
    ledger: prometheus_ledger::Ledger,
    records: Vec<Record>,
    agents: HashMap<AgentId, Agent>,
    inbox: HashMap<AgentId, VecDeque<Message>>,
    patches: HashMap<PatchId, StoredPatch>,
    tickets: HashMap<TicketId, Ticket>,
    weights: HashMap<String, WeightPromoteState>,
    next_agent: u64,
    next_job: u64,
    next_msg: u64,
    next_patch: u64,
    next_ticket: u64,
}

impl Engine {
    pub fn open(cfg: KernelConfig) -> Result<Self> {
        if let Some(parent) = cfg.ledger_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(io_err)?;
            }
        }
        fs::create_dir_all(&cfg.log_dir).map_err(io_err)?;

        let ledger = prometheus_ledger::Ledger::open(&cfg.ledger_path).map_err(ledger_err)?;
        let records = ledger
            .query("1 = 1 ORDER BY rowid", &[])
            .map_err(ledger_err)?;

        let (agents, mut next_agent) = load_agents(&cfg.log_dir)?;
        if let Some(n) = load_next_agent(&cfg.log_dir)? {
            next_agent = next_agent.max(n);
        }
        if next_agent == 0 {
            next_agent = 1;
        }

        let mut inbox = HashMap::new();
        for id in agents.keys() {
            inbox.insert(id.clone(), VecDeque::new());
        }

        Ok(Self {
            log_dir: cfg.log_dir,
            mirror_root: cfg.mirror_root,
            ledger,
            records,
            agents,
            inbox,
            patches: HashMap::new(),
            tickets: HashMap::new(),
            weights: HashMap::new(),
            next_agent,
            next_job: 1,
            next_msg: 1,
            next_patch: 1,
            next_ticket: 1,
        })
    }

    pub fn spawn(&mut self, req: SpawnRequest, _now: NowMs) -> Result<AgentHandle> {
        let depth = match &req.parent {
            None => 0,
            Some(pid) => {
                let parent = self.agent(pid)?;
                parent.handle.depth + 1
            }
        };
        if depth > MAX_SPAWN_DEPTH {
            return Err(Error::SpawnDepth { depth });
        }
        if let Some(pid) = &req.parent {
            self.charge(pid, req.budget)?;
        }

        let id = AgentId(format!("a-{}", self.next_agent));
        self.next_agent += 1;
        self.persist_meta()?;

        let handle = AgentHandle {
            id: id.clone(),
            role: req.role,
            program: req.program,
            parent: req.parent,
            depth,
        };
        write_instructions(&self.log_dir, &id, &req.instructions)?;
        let agent = Agent {
            handle: handle.clone(),
            remaining: req.budget,
        };
        self.persist_agent(&agent)?;
        self.inbox.insert(id.clone(), VecDeque::new());
        self.agents.insert(id, agent);
        Ok(handle)
    }

    pub fn send(
        &mut self,
        from: &AgentId,
        to: &AgentId,
        body: &str,
        now: NowMs,
    ) -> Result<MessageId> {
        self.agent(from)?;
        self.agent(to)?;
        let id = MessageId(format!("m-{}", self.next_msg));
        self.next_msg += 1;
        let msg = Message {
            id: id.clone(),
            from: from.clone(),
            to: to.clone(),
            body: body.to_string(),
            created_ms: now,
        };
        self.inbox.entry(to.clone()).or_default().push_back(msg);
        Ok(id)
    }

    pub fn recv(&mut self, to: &AgentId, _timeout_ms: u64, _now: NowMs) -> Result<Option<Message>> {
        self.agent(to)?;
        Ok(self.inbox.entry(to.clone()).or_default().pop_front())
    }

    pub fn submit_job(&mut self, req: JobRequest, _now: NowMs) -> Result<JobHandle> {
        self.agent(&req.agent)?;
        if req.rung != Rung::R0 {
            let key = req.tags.first().map(String::as_str).unwrap_or("");
            if key.is_empty() || !self.has_rung_rows(key) {
                return Err(Error::RungNotCleared {
                    rung: req.rung.as_u32(),
                });
            }
        }
        self.charge(&req.agent, req.budget)?;
        let id = JobId(format!("j-{}", self.next_job));
        self.next_job += 1;
        Ok(JobHandle {
            id,
            agent: req.agent,
            gpus: req.gpus,
            wall_ms: req.wall_ms,
            rung: req.rung,
        })
    }

    pub fn remaining(&self, agent: &AgentId) -> Result<Quota> {
        Ok(self.agent(agent)?.remaining)
    }

    pub fn ledger_append(&mut self, record: Record) -> Result<RecordId> {
        match self.ledger.append(record.clone()) {
            Ok(id) => {
                self.records.push(record);
                Ok(id)
            }
            Err(prometheus_ledger::Error::Duplicate(id)) => {
                Err(Error::Message(format!("duplicate experiment_id {id}")))
            }
            Err(e) => Err(ledger_err(e)),
        }
    }

    pub fn ledger_query(&self, sql_or_embedding: &str) -> Result<Vec<Record>> {
        query::run(&self.records, sql_or_embedding)
    }

    pub fn fetch(&self, url: &str) -> Result<Vec<u8>> {
        let path = mirror::file_path(&self.mirror_root, url)
            .ok_or_else(|| Error::NotMirrored(url.to_string()))?;
        mirror::read_bytes(&self.mirror_root, &path)
            .ok_or_else(|| Error::NotMirrored(url.to_string()))
    }

    pub fn propose_patch(
        &mut self,
        author: &AgentId,
        diff: &str,
        rationale: &str,
        target: PatchTarget,
        _now: NowMs,
    ) -> Result<PatchId> {
        self.agent(author)?;
        let id = PatchId(format!("p-{}", self.next_patch));
        self.next_patch += 1;
        let patch = Patch {
            id: id.clone(),
            target,
            diff: diff.to_string(),
            rationale: rationale.to_string(),
            author: author.clone(),
            state: PatchState::Proposed,
        };
        self.patches.insert(
            id.clone(),
            StoredPatch {
                patch,
                canary_start: None,
            },
        );
        Ok(id)
    }

    pub fn patch(&self, id: &PatchId) -> Result<Patch> {
        self.patches
            .get(id)
            .map(|s| s.patch.clone())
            .ok_or_else(|| Error::PatchNotFound(id.0.clone()))
    }

    pub fn promote_genome(&mut self, id: &PatchId, now: NowMs) -> Result<PatchState> {
        let stored = self
            .patches
            .get_mut(id)
            .ok_or_else(|| Error::PatchNotFound(id.0.clone()))?;
        let next = match stored.patch.state {
            PatchState::Proposed => {
                if stored.patch.diff.contains("FAIL_SMOKE") {
                    PatchState::Refused
                } else {
                    PatchState::SmokePassed
                }
            }
            PatchState::SmokePassed => {
                if stored.patch.diff.contains("FAIL_GATE") {
                    PatchState::Refused
                } else {
                    PatchState::GatePassed
                }
            }
            PatchState::GatePassed => {
                stored.canary_start = Some(now);
                PatchState::Canary
            }
            PatchState::Canary => {
                let start = stored.canary_start.unwrap_or(now);
                if now.saturating_sub(start) < CANARY_MS {
                    PatchState::Canary
                } else if stored.patch.diff.contains("FAIL_CANARY") {
                    PatchState::Reverted
                } else {
                    PatchState::RolledOut
                }
            }
            PatchState::RolledOut | PatchState::Reverted => stored.patch.state,
            PatchState::Refused => {
                return Err(Error::PromoteRefused(id.0.clone()));
            }
        };
        stored.patch.state = next;
        Ok(next)
    }

    pub fn promote_weights(
        &mut self,
        checkpoint: &str,
        signatures: &[Signature],
        _now: NowMs,
    ) -> Result<WeightPromoteState> {
        if let Some(WeightPromoteState::Serving) = self.weights.get(checkpoint) {
            return Ok(WeightPromoteState::Serving);
        }
        if checkpoint.is_empty() {
            return Err(Error::PromoteRefused("empty checkpoint".into()));
        }
        if checkpoint.contains("FAIL_EVALS") || checkpoint.contains("FAIL_BENCH") {
            self.weights
                .insert(checkpoint.to_string(), WeightPromoteState::Refused);
            return Ok(WeightPromoteState::Refused);
        }
        let mut humans = HashSet::new();
        for sig in signatures {
            if !sig.bytes.is_empty() {
                humans.insert(sig.human.clone());
            }
        }
        if humans.len() < WEIGHT_SIGN_OFF {
            return Err(Error::NoQuorum {
                have: humans.len(),
                need: WEIGHT_SIGN_OFF,
            });
        }
        self.weights
            .insert(checkpoint.to_string(), WeightPromoteState::Serving);
        Ok(WeightPromoteState::Serving)
    }

    pub fn escalate(
        &mut self,
        author: &AgentId,
        summary: &str,
        evidence: &str,
        _now: NowMs,
    ) -> Result<TicketId> {
        self.agent(author)?;
        let id = TicketId(format!("t-{}", self.next_ticket));
        self.next_ticket += 1;
        let ticket = Ticket {
            id: id.clone(),
            summary: summary.to_string(),
            evidence: evidence.to_string(),
            author: author.clone(),
        };
        self.tickets.insert(id.clone(), ticket);
        Ok(id)
    }

    pub fn ticket(&self, id: &TicketId) -> Result<Ticket> {
        self.tickets
            .get(id)
            .cloned()
            .ok_or_else(|| Error::TicketNotFound(id.0.clone()))
    }

    fn agent(&self, id: &AgentId) -> Result<&Agent> {
        self.agents
            .get(id)
            .ok_or_else(|| Error::AgentNotFound(id.0.clone()))
    }

    fn has_rung_rows(&self, key: &str) -> bool {
        let check = format!("CHECK:{key}");
        let prereg = format!("PREREG:{key}");
        let has_check = self
            .records
            .iter()
            .any(|r| r.hypothesis.starts_with(&check));
        let has_prereg = self
            .records
            .iter()
            .any(|r| r.hypothesis.starts_with(&prereg));
        has_check && has_prereg
    }

    fn charge(&mut self, id: &AgentId, need: Quota) -> Result<()> {
        let remaining = {
            let agent = self.agent(id)?;
            if let Some(dim) = first_short_dim(agent.remaining, need) {
                if agent.remaining.exhausted() {
                    return Err(Error::BudgetExhausted(dim.to_string()));
                }
                return Err(Error::QuotaRefused(dim.to_string()));
            }
            sub_quota(agent.remaining, need)
        };
        let agent = self
            .agents
            .get_mut(id)
            .ok_or_else(|| Error::AgentNotFound(id.0.clone()))?;
        agent.remaining = remaining;
        let snapshot = agent.clone();
        self.persist_agent(&snapshot)?;
        Ok(())
    }

    fn persist_agent(&self, agent: &Agent) -> Result<()> {
        let dir = self.log_dir.join("agents").join(&agent.handle.id.0);
        fs::create_dir_all(&dir).map_err(io_err)?;
        let disk = DiskAgent {
            handle: agent.handle.clone(),
            remaining: agent.remaining,
        };
        let json = serde_json::to_string(&disk).map_err(|e| Error::Message(e.to_string()))?;
        fs::write(dir.join("state.json"), json).map_err(io_err)
    }

    fn persist_meta(&self) -> Result<()> {
        let meta = DiskMeta {
            next_agent: self.next_agent,
        };
        let json = serde_json::to_string(&meta).map_err(|e| Error::Message(e.to_string()))?;
        fs::write(self.log_dir.join("kernel_meta.json"), json).map_err(io_err)
    }
}

fn first_short_dim(have: Quota, need: Quota) -> Option<&'static str> {
    if have.tokens < need.tokens {
        Some("tokens")
    } else if have.gpu_ms < need.gpu_ms {
        Some("gpu_ms")
    } else if have.wall_ms < need.wall_ms {
        Some("wall_ms")
    } else {
        None
    }
}

fn sub_quota(have: Quota, need: Quota) -> Quota {
    Quota {
        tokens: have.tokens - need.tokens,
        gpu_ms: have.gpu_ms - need.gpu_ms,
        wall_ms: have.wall_ms - need.wall_ms,
    }
}

fn write_instructions(log_dir: &Path, id: &AgentId, caller: &str) -> Result<()> {
    let dir = log_dir.join("agents").join(&id.0);
    fs::create_dir_all(&dir).map_err(io_err)?;
    let body = format!("{KERNEL_INVARIANT}\n{caller}");
    fs::write(dir.join("instructions.txt"), body).map_err(io_err)
}

fn load_agents(log_dir: &Path) -> Result<(HashMap<AgentId, Agent>, u64)> {
    let mut agents = HashMap::new();
    let mut next_agent = 1u64;
    let agents_dir = log_dir.join("agents");
    let entries = match fs::read_dir(&agents_dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((agents, next_agent)),
        Err(e) => return Err(io_err(e)),
    };
    for entry in entries {
        let entry = entry.map_err(io_err)?;
        let file_type = entry.file_type().map_err(io_err)?;
        if !file_type.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let state_path = entry.path().join("state.json");
        let Ok(raw) = fs::read_to_string(&state_path) else {
            continue;
        };
        let disk: DiskAgent = match serde_json::from_str(&raw) {
            Ok(d) => d,
            Err(_) => continue,
        };
        if let Some(n) = parse_agent_num(&name) {
            next_agent = next_agent.max(n + 1);
        }
        let id = AgentId(name);
        agents.insert(
            id,
            Agent {
                handle: disk.handle,
                remaining: disk.remaining,
            },
        );
    }
    Ok((agents, next_agent))
}

fn load_next_agent(log_dir: &Path) -> Result<Option<u64>> {
    let path = log_dir.join("kernel_meta.json");
    match fs::read_to_string(&path) {
        Ok(raw) => {
            let meta: DiskMeta =
                serde_json::from_str(&raw).map_err(|e| Error::Message(e.to_string()))?;
            Ok(Some(meta.next_agent))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(io_err(e)),
    }
}

fn parse_agent_num(id: &str) -> Option<u64> {
    id.strip_prefix("a-")?.parse().ok()
}

fn io_err(err: std::io::Error) -> Error {
    Error::Message(err.to_string())
}

fn ledger_err(err: prometheus_ledger::Error) -> Error {
    Error::Message(err.to_string())
}
