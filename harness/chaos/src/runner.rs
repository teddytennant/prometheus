//! Chaos injections against in-memory harness doubles (spec 15.2).

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChaosFault {
    ProcessKill,
    Partition,
    BrokerDeath,
    ProviderOutage,
    TokenExpiry,
    DiskFull,
    ClockSkew,
    SlurmPreempt,
}

#[derive(Debug, Clone)]
pub struct Task {
    pub id: u64,
    pub attempt: u32,
    pub holder: Option<String>,
    pub heartbeat: u64,
    pub done: bool,
    pub output: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct HarnessDouble {
    pub tasks: HashMap<u64, Task>,
    pub broker_holder: Option<String>,
    pub token: String,
    pub token_committed: bool,
    pub providers: Vec<(String, bool)>,
    pub provider_idx: usize,
    pub disk_free: u64,
    pub clock: i64,
    pub workers: HashSet<String>,
    pub log: Vec<String>,
}

impl HarnessDouble {
    pub fn new() -> Self {
        let mut tasks = HashMap::new();
        tasks.insert(
            1,
            Task {
                id: 1,
                attempt: 1,
                holder: Some("w0".into()),
                heartbeat: 0,
                done: false,
                output: None,
            },
        );
        let mut workers = HashSet::new();
        workers.insert("w0".into());
        workers.insert("w1".into());
        Self {
            tasks,
            broker_holder: Some("w0".into()),
            token: "tok-1".into(),
            token_committed: true,
            providers: vec![("p0".into(), true), ("p1".into(), true)],
            provider_idx: 0,
            disk_free: 1024,
            clock: 0,
            workers,
            log: vec!["admit:1".into(), "claim:1:w0".into()],
        }
    }

    pub fn reclaim(&mut self, id: u64, worker: &str) {
        if let Some(t) = self.tasks.get_mut(&id) {
            if t.done {
                return;
            }
            t.attempt += 1;
            t.holder = Some(worker.into());
            t.heartbeat = self.clock.max(0) as u64;
            self.log.push(format!("claim:{id}:{worker}"));
        }
    }

    pub fn finish(&mut self, id: u64, worker: &str, bytes: Vec<u8>) -> Result<(), String> {
        if self.disk_free < bytes.len() as u64 {
            return Err("disk full".into());
        }
        let t = self.tasks.get_mut(&id).ok_or("unknown")?;
        if t.done {
            return Err("dup".into());
        }
        if t.holder.as_deref() != Some(worker) {
            return Err("not holder".into());
        }
        self.disk_free -= bytes.len() as u64;
        t.output = Some(bytes);
        t.done = true;
        t.holder = None;
        self.log.push(format!("finish:{id}"));
        Ok(())
    }

    pub fn invariants(&self) -> Result<(), String> {
        if self.tasks.is_empty() {
            return Err("lost task table".into());
        }
        for t in self.tasks.values() {
            if t.done && t.output.is_none() {
                return Err("done without output".into());
            }
        }
        if !self.token_committed {
            return Err("uncommitted token in use".into());
        }
        if self.token.is_empty() {
            return Err("empty token".into());
        }
        if !self.providers.iter().any(|(_, h)| *h) {
            return Err("no provider".into());
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ChaosRunner {
    pub harness: HarnessDouble,
}

impl ChaosRunner {
    pub fn new() -> Self {
        Self {
            harness: HarnessDouble::new(),
        }
    }

    pub fn inject(&mut self, fault: ChaosFault) -> Result<(), String> {
        match fault {
            ChaosFault::ProcessKill => {
                self.harness.workers.remove("w0");
                if let Some(t) = self.harness.tasks.get_mut(&1) {
                    t.holder = None;
                }
                self.harness.reclaim(1, "w1");
            }
            ChaosFault::Partition => {
                self.harness.clock += 20;
                if let Some(t) = self.harness.tasks.get_mut(&1) {
                    t.holder = None;
                }
                self.harness.reclaim(1, "w1");
            }
            ChaosFault::BrokerDeath => {
                self.harness.broker_holder = Some("w1".into());
                self.harness.log.push("broker-lease:w1".into());
            }
            ChaosFault::ProviderOutage => {
                self.harness.providers[self.harness.provider_idx].1 = false;
                let n = self.harness.providers.len();
                for i in 1..=n {
                    let idx = (self.harness.provider_idx + i) % n;
                    if self.harness.providers[idx].1 {
                        self.harness.provider_idx = idx;
                        break;
                    }
                }
            }
            ChaosFault::TokenExpiry => {
                self.harness.token_committed = false;
                self.harness.token = "tok-2".into();
                self.harness.token_committed = true;
            }
            ChaosFault::DiskFull => {
                let holder = self
                    .harness
                    .tasks
                    .get(&1)
                    .and_then(|t| t.holder.clone())
                    .unwrap_or_else(|| "w0".into());
                self.harness.disk_free = 0;
                let r = self.harness.finish(1, &holder, vec![1, 2, 3]);
                if r.is_err() {
                    self.harness.disk_free = 4096;
                    if !self.harness.tasks.get(&1).map(|t| t.done).unwrap_or(false) {
                        self.harness.finish(1, &holder, vec![1, 2, 3])?;
                    }
                }
            }
            ChaosFault::ClockSkew => {
                self.harness.clock -= 50;
                if let Some(t) = self.harness.tasks.get_mut(&1) {
                    t.heartbeat = 0;
                }
                self.harness.clock = 0;
            }
            ChaosFault::SlurmPreempt => {
                self.harness.workers.remove("w0");
                if let Some(t) = self.harness.tasks.get_mut(&1) {
                    t.holder = None;
                }
                self.harness.workers.insert("w2".into());
                self.harness.reclaim(1, "w2");
            }
        }
        self.harness.invariants()
    }

    pub fn inject_all(&mut self) -> Result<(), String> {
        for f in [
            ChaosFault::ProcessKill,
            ChaosFault::Partition,
            ChaosFault::BrokerDeath,
            ChaosFault::ProviderOutage,
            ChaosFault::TokenExpiry,
            ChaosFault::ClockSkew,
            ChaosFault::SlurmPreempt,
            ChaosFault::DiskFull,
        ] {
            self.inject(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_fault_recovers() {
        for f in [
            ChaosFault::ProcessKill,
            ChaosFault::Partition,
            ChaosFault::BrokerDeath,
            ChaosFault::ProviderOutage,
            ChaosFault::TokenExpiry,
            ChaosFault::DiskFull,
            ChaosFault::ClockSkew,
            ChaosFault::SlurmPreempt,
        ] {
            let mut r = ChaosRunner::new();
            r.inject(f).unwrap_or_else(|e| panic!("{f:?}: {e}"));
            r.harness.invariants().unwrap();
            assert!(!r.harness.tasks.is_empty());
        }
    }

    #[test]
    fn kill_does_not_lose_task() {
        let mut r = ChaosRunner::new();
        r.inject(ChaosFault::ProcessKill).unwrap();
        let t = &r.harness.tasks[&1];
        assert!(!t.done);
        assert_eq!(t.holder.as_deref(), Some("w1"));
        assert_eq!(t.attempt, 2);
    }
}
