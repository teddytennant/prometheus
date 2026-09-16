//! Loss-spike policy: rollback, skip shard, page on 2nd spike in 10k steps.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpikeAction {
    Rollback { to_step: u64, skip_shard: bool },
    RollbackAndPage { to_step: u64 },
}

#[derive(Debug, Clone)]
pub struct SpikePolicy {
    pub last_ckpt_step: u64,
    pub window: u64,
    spikes: Vec<u64>,
}

impl SpikePolicy {
    pub fn new(last_ckpt_step: u64) -> Self {
        Self {
            last_ckpt_step,
            window: 10_000,
            spikes: Vec::new(),
        }
    }

    pub fn note_ckpt(&mut self, step: u64) {
        self.last_ckpt_step = step;
    }

    pub fn on_spike(&mut self, step: u64) -> SpikeAction {
        self.spikes
            .retain(|s| step.saturating_sub(*s) <= self.window);
        self.spikes.push(step);
        if self.spikes.len() >= 2 {
            SpikeAction::RollbackAndPage {
                to_step: self.last_ckpt_step,
            }
        } else {
            SpikeAction::Rollback {
                to_step: self.last_ckpt_step,
                skip_shard: true,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_rollback_second_pages() {
        let mut p = SpikePolicy::new(40);
        match p.on_spike(100) {
            SpikeAction::Rollback {
                to_step,
                skip_shard,
            } => {
                assert_eq!(to_step, 40);
                assert!(skip_shard);
            }
            _ => panic!("expected rollback"),
        }
        match p.on_spike(200) {
            SpikeAction::RollbackAndPage { to_step } => assert_eq!(to_step, 40),
            _ => panic!("expected page"),
        }
        let mut p2 = SpikePolicy::new(0);
        p2.on_spike(1);
        p2.window = 10;
        match p2.on_spike(10_000) {
            SpikeAction::Rollback { skip_shard, .. } => assert!(skip_shard),
            _ => panic!("outside window is a first spike"),
        }
    }
}
