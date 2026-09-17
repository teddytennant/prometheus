//! Slow, obvious L1 kernel reference. **Not** `prometheus_kernel::Kernel`.
//! Production code must never import this module.

#![allow(dead_code)]

use crate::common::{first_short_dim, mirror_file_path, sub_quota, WEIGHT_SIGN_OFF};
use prometheus_kernel::{
    AgentHandle, AgentId, Error, JobHandle, JobId, JobRequest, KernelConfig, Message, MessageId,
    NowMs, Patch, PatchId, PatchState, Quota, Result, SpawnRequest, Ticket, TicketId,
    WeightPromoteState, CANARY_MS, KERNEL_INVARIANT, MAX_SPAWN_DEPTH,
};
use prometheus_ledger::{Record, RecordId};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

struct Agent {
    handle: AgentHandle,
    remaining: Quota,
}

struct PatchRec {
    patch: Patch,
    canary_start: Option<NowMs>,
}

pub struct RefKernel {
    mirror_root: PathBuf,
    log_dir: PathBuf,
    agents: HashMap<AgentId, Agent>,
    inbox: HashMap<AgentId, VecDeque<Message>>,
    records: Vec<Record>,
    patches: HashMap<PatchId, PatchRec>,
    tickets: HashMap<TicketId, Ticket>,
    weights: HashMap<String, WeightPromoteState>,
    next_agent: u64,
    next_job: u64,
    next_msg: u64,
    next_patch: u64,
    next_ticket: u64,
}

impl RefKernel {
    pub fn open(cfg: &KernelConfig) -> Self {
        let _ = std::fs::create_dir_all(&cfg.log_dir);
        Self {
            mirror_root: cfg.mirror_root.clone(),
            log_dir: cfg.log_dir.clone(),
            agents: HashMap::new(),
            inbox: HashMap::new(),
            records: Vec::new(),
            patches: HashMap::new(),
            tickets: HashMap::new(),
            weights: HashMap::new(),
            next_agent: 1,
            next_job: 1,
            next_msg: 1,
            next_patch: 1,
            next_ticket: 1,
        }
    }

    pub fn spawn(&mut self, req: SpawnRequest, _now: NowMs) -> Result<AgentHandle> {
        let depth = match &req.parent {
            None => 0,
            Some(pid) => {
                let parent = self
                    .agents
                    .get(pid)
                    .ok_or_else(|| Error::AgentNotFound(pid.0.clone()))?;
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
        let handle = AgentHandle {
            id: id.clone(),
            role: req.role,
            parent: req.parent.clone(),
            depth,
            program: req.program.clone(),
        };
        let stored = format!("{KERNEL_INVARIANT}\n{}", req.instructions);
        write_instructions(&self.log_dir, &id, &stored)?;
        self.agents.insert(
            id.clone(),
            Agent {
                handle: handle.clone(),
                remaining: req.budget,
            },
        );
        self.inbox.entry(id).or_default();
        Ok(handle)
    }

    pub fn send(
        &mut self,
        from: &AgentId,
        to: &AgentId,
        body: &str,
        now: NowMs,
    ) -> Result<MessageId> {
        if !self.agents.contains_key(from) {
            return Err(Error::AgentNotFound(from.0.clone()));
        }
        if !self.agents.contains_key(to) {
            return Err(Error::AgentNotFound(to.0.clone()));
        }
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
        if !self.agents.contains_key(to) {
            return Err(Error::AgentNotFound(to.0.clone()));
        }
        Ok(self.inbox.get_mut(to).and_then(|q| q.pop_front()))
    }

    pub fn submit_job(&mut self, req: JobRequest, _now: NowMs) -> Result<JobHandle> {
        if !self.agents.contains_key(&req.agent) {
            return Err(Error::AgentNotFound(req.agent.0.clone()));
        }
        if req.rung != prometheus_kernel::Rung::R0 {
            let key = match req.tags.first() {
                Some(k) => k.as_str(),
                None => {
                    return Err(Error::RungNotCleared {
                        rung: req.rung.as_u32(),
                    })
                }
            };
            if !has_rung_rows(&self.records, key) {
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

    pub fn remaining(&self, id: &AgentId) -> Result<Quota> {
        self.agents
            .get(id)
            .map(|a| a.remaining)
            .ok_or_else(|| Error::AgentNotFound(id.0.clone()))
    }

    pub fn ledger_append(&mut self, record: Record) -> Result<RecordId> {
        if self
            .records
            .iter()
            .any(|r| r.experiment_id == record.experiment_id)
        {
            return Err(Error::Message(format!(
                "duplicate experiment_id {}",
                record.experiment_id
            )));
        }
        let id = RecordId(record.experiment_id.clone());
        self.records.push(record);
        Ok(id)
    }

    pub fn ledger_query(&self, sql_or_embedding: &str) -> Result<Vec<Record>> {
        parse_and_filter(&self.records, sql_or_embedding).map_err(Error::Message)
    }

    pub fn fetch(&self, url: &str) -> Result<Vec<u8>> {
        let not = || Error::NotMirrored(url.to_string());
        let path = mirror_file_path(&self.mirror_root, url).ok_or_else(not)?;
        read_mirrored(&self.mirror_root, &path).ok_or_else(not)
    }

    pub fn propose_patch(
        &mut self,
        author: &AgentId,
        diff: &str,
        rationale: &str,
        target: prometheus_kernel::PatchTarget,
        _now: NowMs,
    ) -> Result<PatchId> {
        if !self.agents.contains_key(author) {
            return Err(Error::AgentNotFound(author.0.clone()));
        }
        let id = PatchId(format!("p-{}", self.next_patch));
        self.next_patch += 1;
        let patch = Patch {
            id: id.clone(),
            target,
            diff: diff.to_string(),
            rationale: rationale.to_string(),
            state: PatchState::Proposed,
            author: author.clone(),
        };
        self.patches.insert(
            id.clone(),
            PatchRec {
                patch,
                canary_start: None,
            },
        );
        Ok(id)
    }

    pub fn patch(&self, id: &PatchId) -> Result<Patch> {
        self.patches
            .get(id)
            .map(|p| p.patch.clone())
            .ok_or_else(|| Error::PatchNotFound(id.0.clone()))
    }

    pub fn promote_genome(&mut self, id: &PatchId, now: NowMs) -> Result<PatchState> {
        let rec = self
            .patches
            .get_mut(id)
            .ok_or_else(|| Error::PatchNotFound(id.0.clone()))?;
        match rec.patch.state {
            PatchState::Proposed => {
                if rec.patch.diff.contains("FAIL_SMOKE") {
                    rec.patch.state = PatchState::Refused;
                    return Ok(PatchState::Refused);
                }
                rec.patch.state = PatchState::SmokePassed;
                Ok(PatchState::SmokePassed)
            }
            PatchState::SmokePassed => {
                if rec.patch.diff.contains("FAIL_GATE") {
                    rec.patch.state = PatchState::Refused;
                    return Ok(PatchState::Refused);
                }
                rec.patch.state = PatchState::GatePassed;
                Ok(PatchState::GatePassed)
            }
            PatchState::GatePassed => {
                rec.patch.state = PatchState::Canary;
                rec.canary_start = Some(now);
                Ok(PatchState::Canary)
            }
            PatchState::Canary => {
                let start = rec.canary_start.unwrap_or(now);
                if now.saturating_sub(start) < CANARY_MS {
                    return Ok(PatchState::Canary);
                }
                if rec.patch.diff.contains("FAIL_CANARY") {
                    rec.patch.state = PatchState::Reverted;
                    return Ok(PatchState::Reverted);
                }
                rec.patch.state = PatchState::RolledOut;
                Ok(PatchState::RolledOut)
            }
            PatchState::RolledOut => Ok(PatchState::RolledOut),
            PatchState::Reverted => Ok(PatchState::Reverted),
            PatchState::Refused => Err(Error::PromoteRefused(id.0.clone())),
        }
    }

    pub fn promote_weights(
        &mut self,
        checkpoint: &str,
        signatures: &[prometheus_kernel::Signature],
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
                humans.insert(sig.human.0.clone());
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
        if !self.agents.contains_key(author) {
            return Err(Error::AgentNotFound(author.0.clone()));
        }
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

    fn charge(&mut self, id: &AgentId, need: Quota) -> Result<()> {
        let agent = self
            .agents
            .get_mut(id)
            .ok_or_else(|| Error::AgentNotFound(id.0.clone()))?;
        if let Some(dim) = first_short_dim(agent.remaining, need) {
            if agent.remaining.exhausted() {
                return Err(Error::BudgetExhausted(dim.to_string()));
            }
            return Err(Error::QuotaRefused(dim.to_string()));
        }
        agent.remaining = sub_quota(agent.remaining, need);
        Ok(())
    }
}

fn write_instructions(log_dir: &Path, id: &AgentId, text: &str) -> Result<()> {
    let dir = log_dir.join("agents").join(&id.0);
    std::fs::create_dir_all(&dir).map_err(|e| Error::Message(e.to_string()))?;
    std::fs::write(dir.join("instructions.txt"), text)
        .map_err(|e| Error::Message(e.to_string()))?;
    Ok(())
}

fn has_rung_rows(records: &[Record], key: &str) -> bool {
    let check = format!("CHECK:{key}");
    let prereg = format!("PREREG:{key}");
    let has_check = records.iter().any(|r| r.hypothesis.starts_with(&check));
    let has_prereg = records.iter().any(|r| r.hypothesis.starts_with(&prereg));
    has_check && has_prereg
}

fn read_mirrored(mirror_root: &Path, path: &Path) -> Option<Vec<u8>> {
    if !path.is_file() {
        return None;
    }
    let canon = path.canonicalize().ok()?;
    let root = mirror_root.canonicalize().ok()?;
    if !canon.starts_with(&root) {
        return None;
    }
    std::fs::read(path).ok()
}

#[derive(Debug)]
enum Pred {
    ExperimentId(String),
    AuthorRole(String),
    Rung(Option<u32>),
    HypothesisContains(String),
    KindCheck,
    KindPrereg,
}

impl Pred {
    fn matches(&self, r: &Record) -> bool {
        match self {
            Pred::ExperimentId(v) => r.experiment_id == *v,
            Pred::AuthorRole(v) => r.author_role == *v,
            Pred::Rung(v) => r.rung == *v,
            Pred::HypothesisContains(v) => r.hypothesis.contains(v),
            Pred::KindCheck => r.hypothesis.starts_with("CHECK:"),
            Pred::KindPrereg => r.hypothesis.starts_with("PREREG:"),
        }
    }
}

pub fn parse_and_filter(
    records: &[Record],
    query: &str,
) -> std::result::Result<Vec<Record>, String> {
    let q = query.trim();
    if q.is_empty() || q == "*" || q == "all" {
        return Ok(records.to_vec());
    }
    let upper = q.to_ascii_uppercase();
    if upper.contains("SELECT ")
        || upper.contains("DROP ")
        || upper.contains("INSERT ")
        || upper.contains("DELETE ")
    {
        return Err("query language is not SQL".into());
    }
    let parts = split_and(q)?;
    let mut preds = Vec::new();
    for p in parts {
        preds.push(parse_pred(&p)?);
    }
    Ok(records
        .iter()
        .filter(|r| preds.iter().all(|p| p.matches(r)))
        .cloned()
        .collect())
}

fn split_and(q: &str) -> std::result::Result<Vec<String>, String> {
    let chars: Vec<char> = q.chars().collect();
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '"' {
            in_quotes = !in_quotes;
            cur.push('"');
            i += 1;
            continue;
        }
        if !in_quotes && chars[i] == ' ' && i + 5 <= chars.len() {
            let slice: String = chars[i..i + 5].iter().collect();
            if slice == " AND " {
                parts.push(cur.trim().to_string());
                cur.clear();
                i += 5;
                continue;
            }
        }
        cur.push(chars[i]);
        i += 1;
    }
    if in_quotes {
        return Err("unterminated quote".into());
    }
    parts.push(cur.trim().to_string());
    if parts.iter().any(|p| p.is_empty()) {
        return Err("empty predicate".into());
    }
    Ok(parts)
}

fn parse_pred(p: &str) -> std::result::Result<Pred, String> {
    let p = p.trim();
    if let Some(v) = p.strip_prefix("kind=") {
        return match unquote(v)?.as_str() {
            "check" => Ok(Pred::KindCheck),
            "prereg" => Ok(Pred::KindPrereg),
            other => Err(format!("unknown kind {other}")),
        };
    }
    if let Some(v) = p.strip_prefix("experiment_id=") {
        return Ok(Pred::ExperimentId(unquote(v)?));
    }
    if let Some(v) = p.strip_prefix("author_role=") {
        return Ok(Pred::AuthorRole(unquote(v)?));
    }
    if let Some(v) = p.strip_prefix("rung=") {
        let v = unquote(v)?;
        if v == "none" {
            return Ok(Pred::Rung(None));
        }
        let n: u32 = v.parse().map_err(|_| format!("bad rung {v}"))?;
        return Ok(Pred::Rung(Some(n)));
    }
    if let Some(v) = p.strip_prefix("hypothesis~") {
        return Ok(Pred::HypothesisContains(unquote(v)?));
    }
    Err(format!("unknown predicate {p}"))
}

fn unquote(s: &str) -> std::result::Result<String, String> {
    let s = s.trim();
    if let Some(inner) = s.strip_prefix('"').and_then(|x| x.strip_suffix('"')) {
        return Ok(inner.to_string());
    }
    if s.split_whitespace().nth(1).is_some() {
        return Err("unquoted whitespace".into());
    }
    Ok(s.to_string())
}
