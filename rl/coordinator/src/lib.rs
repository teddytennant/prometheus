//! Async RL coordinator: staleness k<=4, weight versions (spec 9.2, D3).

#[derive(Debug, Clone)]
pub struct WeightVersion {
    pub id: u64,
}

#[derive(Debug)]
pub struct Coordinator {
    pub trainer: u64,
    pub max_stale: u64,
}

impl Coordinator {
    pub fn new() -> Self {
        Self { trainer: 0, max_stale: 4 }
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

    pub fn split(rollout_frac: f64) -> (f64, f64) {
        (rollout_frac, 1.0 - rollout_frac)
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
}
