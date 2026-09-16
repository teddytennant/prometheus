//! Monitor ring: each watched by two others; tripwires; signed freeze (spec 14.10).

use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct Monitor {
    pub id: u32,
    pub alive: bool,
    pub watchers: [u32; 2],
}

#[derive(Debug, Clone)]
pub struct MonitorRing {
    pub monitors: Vec<Monitor>,
}

impl MonitorRing {
    pub fn new(n: usize) -> Self {
        let n = n.max(3) as u32;
        let monitors = (0..n)
            .map(|i| Monitor {
                id: i,
                alive: true,
                watchers: [(i + 1) % n, (i + 2) % n],
            })
            .collect();
        Self { monitors }
    }

    pub fn watchers_of(&self, id: u32) -> Option<[u32; 2]> {
        self.monitors.get(id as usize).map(|m| m.watchers)
    }

    pub fn mark_dead(&mut self, id: u32) {
        if let Some(m) = self.monitors.get_mut(id as usize) {
            m.alive = false;
        }
    }

    /// A monitor is covered iff both of its watchers are alive.
    pub fn covered(&self, id: u32) -> bool {
        match self.watchers_of(id) {
            Some([a, b]) => {
                self.monitors.get(a as usize).map(|m| m.alive).unwrap_or(false)
                    && self.monitors.get(b as usize).map(|m| m.alive).unwrap_or(false)
            }
            None => false,
        }
    }

    pub fn quorum_alive(&self) -> bool {
        let n = self.monitors.len();
        self.monitors.iter().filter(|m| m.alive).count() * 2 > n
    }
}

#[derive(Debug, Clone)]
pub struct Tripwires {
    pub rci_max: f64,
    pub last_rci: f64,
    pub hidden_eval: f64,
    pub hidden_floor: f64,
}

impl Tripwires {
    pub fn new() -> Self {
        Self {
            rci_max: 0.02,
            last_rci: 0.0,
            hidden_eval: 1.0,
            hidden_floor: 0.9,
        }
    }

    pub fn rci_trip(&mut self, next: f64) -> bool {
        let d = (next - self.last_rci).abs();
        self.last_rci = next;
        d > self.rci_max
    }

    pub fn hidden_trip(&self) -> bool {
        self.hidden_eval + f64::EPSILON < self.hidden_floor
    }
}

#[derive(Debug, Clone)]
pub struct KillSwitch {
    pub frozen: bool,
    key: String,
}

impl KillSwitch {
    pub fn new(key: &str) -> Self {
        Self {
            frozen: false,
            key: key.into(),
        }
    }

    pub fn sign(key: &str) -> String {
        let mut h = Sha256::new();
        h.update(key.as_bytes());
        h.update(b"freeze");
        format!("{:x}", h.finalize())
    }

    pub fn freeze(&mut self, sig: &str) -> Result<(), String> {
        if sig != Self::sign(&self.key) {
            return Err("bad freeze signature".into());
        }
        self.frozen = true;
        Ok(())
    }

    pub fn honor(&self) -> bool {
        self.frozen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_two_watchers_and_signed_freeze() {
        let mut ring = MonitorRing::new(5);
        for m in &ring.monitors {
            assert_ne!(m.watchers[0], m.id);
            assert_ne!(m.watchers[1], m.id);
            assert_ne!(m.watchers[0], m.watchers[1]);
        }
        assert!(ring.covered(0));
        ring.mark_dead(1);
        assert!(!ring.covered(0));
        assert!(ring.quorum_alive());
        let mut tw = Tripwires::new();
        assert!(!tw.rci_trip(0.01));
        assert!(tw.rci_trip(0.05));
        tw.hidden_eval = 0.5;
        assert!(tw.hidden_trip());
        let mut ks = KillSwitch::new("human");
        assert!(ks.freeze("nope").is_err());
        ks.freeze(&KillSwitch::sign("human")).unwrap();
        assert!(ks.honor());
    }
}
