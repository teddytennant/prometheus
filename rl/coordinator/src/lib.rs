//! Async RL coordinator: staleness k<=4, weight versions (spec 9.2, D3).

mod policy;

pub use policy::{truncated_is, Batch, ParityHalt};

use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct WeightVersion {
    pub id: u64,
}

#[derive(Debug)]
pub struct Coordinator {
    pub trainer: u64,
    pub max_stale: u64,
    pub rollout_q: VecDeque<Batch>,
    pub trainer_q: VecDeque<Batch>,
    pub rollout_racks: u32,
    pub trainer_racks: u32,
}

impl Coordinator {
    pub fn new() -> Self {
        Self {
            trainer: 0,
            max_stale: 4,
            rollout_q: VecDeque::new(),
            trainer_q: VecDeque::new(),
            rollout_racks: 65,
            trainer_racks: 35,
        }
    }

    pub fn publish(&mut self) -> WeightVersion {
        self.trainer += 1;
        WeightVersion { id: self.trainer }
    }

    pub fn admit(&self, rollout: u64) -> Result<(), String> {
        if self.trainer.saturating_sub(rollout) > self.max_stale {
            return Err("stale".into());
        }
        Ok(())
    }

    pub fn admit_batch(&mut self, batch: Batch) -> Result<(), String> {
        self.admit(batch.policy_version)?;
        self.trainer_q.push_back(batch);
        Ok(())
    }

    pub fn enqueue_rollout(&mut self, batch: Batch) {
        self.rollout_q.push_back(batch);
    }

    pub fn split(rollout_frac: f64) -> (f64, f64) {
        (rollout_frac, 1.0 - rollout_frac)
    }

    /// Move racks toward 65/35, biased by queue depth.
    pub fn move_racks(&mut self) -> (u32, u32) {
        let total = (self.rollout_racks + self.trainer_racks).max(2);
        let rd = self.rollout_q.len() as f64;
        let td = self.trainer_q.len() as f64;
        let pressure = if rd + td == 0.0 { 0.65 } else { rd / (rd + td) };
        let frac = 0.5 * 0.65 + 0.5 * pressure;
        let r = ((frac * total as f64).round() as u32).clamp(1, total - 1);
        self.rollout_racks = r;
        self.trainer_racks = total - r;
        (self.rollout_racks, self.trainer_racks)
    }

    pub fn parity_check(trainer: &[f32], sglang: &[f32], threshold: f32) -> Result<(), ParityHalt> {
        if trainer.len() != sglang.len() {
            return Err(ParityHalt {
                max_abs: f32::INFINITY,
                threshold,
            });
        }
        let mut max_abs = 0.0f32;
        for (a, b) in trainer.iter().zip(sglang) {
            max_abs = max_abs.max((a - b).abs());
        }
        if max_abs > threshold {
            Err(ParityHalt {
                max_abs,
                threshold,
            })
        } else {
            Ok(())
        }
    }
}

impl Default for Coordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_k5() {
        let mut c = Coordinator::new();
        let v = c.publish();
        for _ in 0..5 {
            c.publish();
        }
        assert!(c.admit(v.id).is_err());
        assert!(c.admit(c.trainer).is_ok());
        let (r, t) = Coordinator::split(0.65);
        assert!((r + t - 1.0).abs() < 1e-12);
    }

    #[test]
    fn admit_batch_stale_and_racks() {
        let mut c = Coordinator::new();
        let v = c.publish();
        for _ in 0..5 {
            c.publish();
        }
        let stale = Batch {
            policy_version: v.id,
            tokens: vec![1],
            logprobs: vec![0.0],
            routing_ids: vec![3],
        };
        assert!(c.admit_batch(stale).is_err());
        let fresh = Batch {
            policy_version: c.trainer,
            tokens: vec![1],
            logprobs: vec![-0.2],
            routing_ids: vec![3],
        };
        c.admit_batch(fresh).unwrap();
        for _ in 0..10 {
            c.enqueue_rollout(Batch {
                policy_version: c.trainer,
                tokens: vec![2],
                logprobs: vec![-0.1],
                routing_ids: vec![1],
            });
        }
        let (rr, tr) = c.move_racks();
        assert_eq!(rr + tr, 100);
        assert!(rr > tr);
    }

    #[test]
    fn parity_halts_on_drift() {
        assert!(Coordinator::parity_check(&[0.1, 0.2], &[0.1, 0.2], 0.01).is_ok());
        assert!(Coordinator::parity_check(&[0.1], &[0.5], 0.01).is_err());
    }
}
