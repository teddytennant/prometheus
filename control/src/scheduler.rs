//! Elastic DP scheduler: drop/rejoin replicas, hold tokens/step (spec 5.5).

use crate::health::Health;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replica {
    pub id: u32,
    pub live: bool,
    pub health: Health,
}

#[derive(Debug, Clone)]
pub struct Scheduler {
    pub replicas: Vec<Replica>,
    pub target_tokens: u64,
    pub microbatch_tokens: u64,
    pub grad_accum: u64,
}

impl Scheduler {
    pub fn new(n: u32, target_tokens: u64, microbatch_tokens: u64) -> Self {
        let replicas = (0..n)
            .map(|id| Replica {
                id,
                live: true,
                health: Health::ok(),
            })
            .collect();
        let mut s = Self {
            replicas,
            target_tokens,
            microbatch_tokens: microbatch_tokens.max(1),
            grad_accum: 1,
        };
        s.recompute_accum();
        s
    }

    pub fn live_count(&self) -> u64 {
        self.replicas.iter().filter(|r| r.live).count() as u64
    }

    pub fn tokens_per_step(&self) -> u64 {
        self.live_count() * self.microbatch_tokens * self.grad_accum
    }

    fn recompute_accum(&mut self) {
        let live = self.live_count().max(1);
        let den = live * self.microbatch_tokens;
        self.grad_accum = (self.target_tokens + den - 1) / den;
    }

    /// Failed replica drops out; grad accum rises so tokens/step stay at target.
    pub fn drop_replica(&mut self, id: u32) -> Result<u64, String> {
        let r = self
            .replicas
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or_else(|| "unknown replica".to_string())?;
        if !r.live {
            return Err("already down".into());
        }
        r.live = false;
        self.recompute_accum();
        Ok(self.grad_accum)
    }

    /// Spare rack rejoins at the next step boundary.
    pub fn rejoin_replica(&mut self, id: u32) -> Result<u64, String> {
        let r = self
            .replicas
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or_else(|| "unknown replica".to_string())?;
        if r.live {
            return Err("already live".into());
        }
        r.live = true;
        r.health = Health::ok();
        self.recompute_accum();
        Ok(self.grad_accum)
    }

    pub fn drain_unhealthy(&mut self) -> Vec<u32> {
        let ids: Vec<u32> = self
            .replicas
            .iter()
            .filter(|r| r.live && r.health.unhealthy())
            .map(|r| r.id)
            .collect();
        for id in &ids {
            let _ = self.drop_replica(*id);
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drop_raises_accum_holds_tokens() {
        let mut s = Scheduler::new(4, 64, 8);
        assert_eq!(s.live_count(), 4);
        let before = s.tokens_per_step();
        s.drop_replica(2).unwrap();
        assert_eq!(s.live_count(), 3);
        assert!(s.grad_accum >= 1);
        assert!(s.tokens_per_step() >= before || s.tokens_per_step() >= s.target_tokens);
        s.rejoin_replica(2).unwrap();
        assert_eq!(s.live_count(), 4);
    }

    #[test]
    fn drain_unhealthy_replica() {
        let mut s = Scheduler::new(2, 16, 4);
        s.replicas[0].health.xid = true;
        let drained = s.drain_unhealthy();
        assert_eq!(drained, vec![0]);
        assert!(!s.replicas[0].live);
        assert!(s.replicas[1].live);
    }
}
