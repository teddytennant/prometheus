//! Event log, SDC hash, spike rollback (spec 5.4, 14, 15.5 A5).
//! Elastic DP, health, stragglers, watchdog (spec 5.5).

mod health;
mod scheduler;
mod spike;
mod watchdog;

pub use health::{Health, StragglerDetector};
pub use scheduler::{Replica, Scheduler};
pub use spike::{SpikeAction, SpikePolicy};
pub use watchdog::Watchdog;

use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub kind: String,
    pub body: String,
    pub prev: String,
    pub hash: String,
}

#[derive(Debug, Default)]
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

    pub fn append(&mut self, kind: &str, body: &str) -> String {
        let mut h = Sha256::new();
        h.update(self.last.as_bytes());
        h.update(kind.as_bytes());
        h.update(body.as_bytes());
        let hash = format!("{:x}", h.finalize());
        self.events.push(Event {
            kind: kind.into(),
            body: body.into(),
            prev: self.last.clone(),
            hash: hash.clone(),
        });
        self.last = hash.clone();
        hash
    }

    pub fn verify(&self) -> bool {
        let mut prev = "0".repeat(64);
        for e in &self.events {
            if e.prev != prev {
                return false;
            }
            let mut h = Sha256::new();
            h.update(prev.as_bytes());
            h.update(e.kind.as_bytes());
            h.update(e.body.as_bytes());
            if format!("{:x}", h.finalize()) != e.hash {
                return false;
            }
            prev = e.hash.clone();
        }
        true
    }
}

pub fn sdc_hash(shard: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(shard);
    format!("{:x}", h.finalize())
}

pub fn spike_skip(shards: &[Vec<u8>], bad: usize) -> Vec<Vec<u8>> {
    shards
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != bad)
        .map(|(_, s)| s.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_and_sdc() {
        let mut log = EventLog::new();
        log.append("step", "1");
        log.append("step", "2");
        assert!(log.verify());
        let a = vec![1u8, 2, 3];
        let mut b = a.clone();
        b[0] ^= 1;
        assert_ne!(sdc_hash(&a), sdc_hash(&b));
        let kept = spike_skip(&[vec![0], vec![1], vec![2]], 1);
        assert_eq!(kept.len(), 2);
    }
}
