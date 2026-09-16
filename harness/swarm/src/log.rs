//! Append-only hash-chained event log: source of truth (spec 15.2).

use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    pub seq: u64,
    pub kind: String,
    pub body: String,
    pub prev: String,
    pub hash: String,
}

#[derive(Debug, Clone, Default)]
pub struct EventLog {
    pub events: Vec<Event>,
    last: String,
}

impl EventLog {
    pub fn new() -> Self {
        Self {
            events: vec![],
            last: "0".repeat(64),
        }
    }

    pub fn genesis() -> String {
        "0".repeat(64)
    }

    pub fn append(&mut self, kind: &str, body: &str) -> Event {
        let seq = self.events.len() as u64;
        let mut h = Sha256::new();
        h.update(self.last.as_bytes());
        h.update(seq.to_le_bytes());
        h.update(kind.as_bytes());
        h.update(body.as_bytes());
        let hash = format!("{:x}", h.finalize());
        let e = Event {
            seq,
            kind: kind.into(),
            body: body.into(),
            prev: self.last.clone(),
            hash: hash.clone(),
        };
        self.events.push(e.clone());
        self.last = hash;
        e
    }

    pub fn verify(&self) -> bool {
        let mut prev = Self::genesis();
        for (i, e) in self.events.iter().enumerate() {
            if e.prev != prev || e.seq != i as u64 {
                return false;
            }
            let mut h = Sha256::new();
            h.update(prev.as_bytes());
            h.update((i as u64).to_le_bytes());
            h.update(e.kind.as_bytes());
            h.update(e.body.as_bytes());
            if format!("{:x}", h.finalize()) != e.hash {
                return false;
            }
            prev = e.hash.clone();
        }
        true
    }

    pub fn last_hash(&self) -> &str {
        &self.last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_tamper() {
        let mut log = EventLog::new();
        log.append("admit", "1");
        log.append("claim", "1:n0");
        assert!(log.verify());
        log.events[1].body = "mutated".into();
        assert!(!log.verify());
    }
}
