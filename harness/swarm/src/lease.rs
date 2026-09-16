//! Tasks, leases (expire at 2 missed heartbeats), CAS artifacts (spec 15.2).

use crate::log::EventLog;
use crate::TaskId;
use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lease {
    pub holder: String,
    pub task: TaskId,
    pub attempt: u32,
    pub heartbeat_at: u64,
    pub missed: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    pub attempt: u32,
    pub payload: String,
    pub done: bool,
    pub lease: Option<Lease>,
}

#[derive(Debug, Clone, Default)]
pub struct ArtifactCas {
    store: HashMap<String, Vec<u8>>,
    outputs: HashMap<u64, (u32, String)>,
}

impl ArtifactCas {
    pub fn put(&mut self, task: TaskId, attempt: u32, bytes: &[u8]) -> Result<String, String> {
        if let Some((best, _)) = self.outputs.get(&task.0) {
            if attempt < *best {
                return Err("zombie attempt cannot overwrite".into());
            }
        }
        let mut h = Sha256::new();
        h.update(bytes);
        let hash = format!("{:x}", h.finalize());
        self.store.insert(hash.clone(), bytes.to_vec());
        self.outputs.insert(task.0, (attempt, hash.clone()));
        Ok(hash)
    }

    pub fn get(&self, hash: &str) -> Option<&[u8]> {
        self.store.get(hash).map(|b| b.as_slice())
    }

    pub fn output_of(&self, task: TaskId) -> Option<(u32, &str)> {
        self.outputs
            .get(&task.0)
            .map(|(a, h)| (*a, h.as_str()))
    }
}

#[derive(Debug, Clone)]
pub struct Cluster {
    pub log: EventLog,
    pub tasks: HashMap<u64, Task>,
    pub cas: ArtifactCas,
    pub now: u64,
    pub heartbeat_iv: u64,
}

impl Cluster {
    pub fn new(heartbeat_iv: u64) -> Self {
        Self {
            log: EventLog::new(),
            tasks: HashMap::new(),
            cas: ArtifactCas::default(),
            now: 0,
            heartbeat_iv: heartbeat_iv.max(1),
        }
    }

    pub fn tick(&mut self, dt: u64) {
        self.now += dt;
    }

    pub fn admit(&mut self, id: TaskId, payload: &str) -> Result<(), String> {
        if self.tasks.contains_key(&id.0) {
            return Err("dup".into());
        }
        self.tasks.insert(
            id.0,
            Task {
                id,
                attempt: 0,
                payload: payload.into(),
                done: false,
                lease: None,
            },
        );
        self.log.append("admit", &format!("{}:{}", id.0, payload));
        Ok(())
    }

    pub fn claim(&mut self, id: TaskId, holder: &str) -> Result<Lease, String> {
        let t = self.tasks.get_mut(&id.0).ok_or("unknown")?;
        if t.done {
            return Err("done".into());
        }
        if let Some(l) = &t.lease {
            if l.missed < 2 {
                return Err("leased".into());
            }
        }
        t.attempt += 1;
        let lease = Lease {
            holder: holder.into(),
            task: id,
            attempt: t.attempt,
            heartbeat_at: self.now,
            missed: 0,
        };
        t.lease = Some(lease.clone());
        self.log
            .append("claim", &format!("{}:{}:{}", id.0, holder, t.attempt));
        Ok(lease)
    }

    pub fn heartbeat(&mut self, id: TaskId, holder: &str) -> Result<(), String> {
        let t = self.tasks.get_mut(&id.0).ok_or("unknown")?;
        let l = t.lease.as_mut().ok_or("no lease")?;
        if l.holder != holder {
            return Err("not holder".into());
        }
        l.heartbeat_at = self.now;
        l.missed = 0;
        self.log.append("heartbeat", &format!("{}:{}", id.0, holder));
        Ok(())
    }

    /// Expire leases that missed two heartbeats.
    pub fn expire(&mut self) -> Vec<TaskId> {
        let iv = self.heartbeat_iv;
        let now = self.now;
        let mut out = Vec::new();
        for t in self.tasks.values_mut() {
            if t.done {
                continue;
            }
            if let Some(l) = &mut t.lease {
                let missed = now.saturating_sub(l.heartbeat_at) / iv;
                l.missed = missed as u32;
                if l.missed >= 2 {
                    out.push(t.id);
                    t.lease = None;
                }
            }
        }
        for id in &out {
            self.log.append("expire", &id.0.to_string());
        }
        out
    }

    pub fn finish(&mut self, id: TaskId, holder: &str, bytes: &[u8]) -> Result<String, String> {
        let attempt = {
            let t = self.tasks.get(&id.0).ok_or("unknown")?;
            let l = t.lease.as_ref().ok_or("no lease")?;
            if l.holder != holder {
                return Err("not holder".into());
            }
            l.attempt
        };
        let hash = self.cas.put(id, attempt, bytes)?;
        if let Some(t) = self.tasks.get_mut(&id.0) {
            t.done = true;
            t.lease = None;
        }
        self.log.append("finish", &format!("{}:{}:{}", id.0, attempt, hash));
        Ok(hash)
    }

    pub fn recover_from_log(log: &EventLog, heartbeat_iv: u64) -> Self {
        let mut c = Self::new(heartbeat_iv);
        c.log = log.clone();
        for e in &log.events {
            match e.kind.as_str() {
                "admit" => {
                    let mut parts = e.body.splitn(2, ':');
                    if let (Some(id), Some(payload)) = (parts.next(), parts.next()) {
                        if let Ok(n) = id.parse::<u64>() {
                            c.tasks.insert(
                                n,
                                Task {
                                    id: TaskId(n),
                                    attempt: 0,
                                    payload: payload.into(),
                                    done: false,
                                    lease: None,
                                },
                            );
                        }
                    }
                }
                "claim" => {
                    let p: Vec<&str> = e.body.split(':').collect();
                    if p.len() == 3 {
                        if let (Ok(n), Ok(att)) = (p[0].parse::<u64>(), p[2].parse::<u32>()) {
                            if let Some(t) = c.tasks.get_mut(&n) {
                                t.attempt = att;
                                t.lease = Some(Lease {
                                    holder: p[1].into(),
                                    task: TaskId(n),
                                    attempt: att,
                                    heartbeat_at: 0,
                                    missed: 0,
                                });
                            }
                        }
                    }
                }
                "expire" => {
                    if let Ok(n) = e.body.parse::<u64>() {
                        if let Some(t) = c.tasks.get_mut(&n) {
                            t.lease = None;
                        }
                    }
                }
                "finish" => {
                    let p: Vec<&str> = e.body.split(':').collect();
                    if p.len() == 3 {
                        if let (Ok(n), Ok(att)) = (p[0].parse::<u64>(), p[1].parse::<u32>()) {
                            if let Some(t) = c.tasks.get_mut(&n) {
                                t.done = true;
                                t.lease = None;
                            }
                            let _ = c.cas.put(TaskId(n), att, p[2].as_bytes());
                        }
                    }
                }
                _ => {}
            }
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_expires_after_two_misses() {
        let mut c = Cluster::new(10);
        c.admit(TaskId(1), "work").unwrap();
        c.claim(TaskId(1), "n0").unwrap();
        c.tick(10);
        assert!(c.expire().is_empty());
        c.tick(10);
        assert_eq!(c.expire(), vec![TaskId(1)]);
        c.claim(TaskId(1), "n1").unwrap();
        assert_eq!(c.tasks[&1].attempt, 2);
    }

    #[test]
    fn zombie_cannot_overwrite() {
        let mut cas = ArtifactCas::default();
        cas.put(TaskId(9), 2, b"new").unwrap();
        assert!(cas.put(TaskId(9), 1, b"old").is_err());
        assert_eq!(cas.output_of(TaskId(9)).unwrap().0, 2);
    }

    #[test]
    fn recover_from_log_rebuilds() {
        let mut c = Cluster::new(5);
        c.admit(TaskId(3), "p").unwrap();
        c.claim(TaskId(3), "n0").unwrap();
        c.finish(TaskId(3), "n0", b"out").unwrap();
        let rebuilt = Cluster::recover_from_log(&c.log, 5);
        assert!(rebuilt.log.verify());
        assert!(rebuilt.tasks[&3].done);
        assert_eq!(rebuilt.tasks[&3].attempt, 1);
    }
}
