//! Per-rank health flags and straggler detection (spec 5.5).

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Health {
    pub xid: bool,
    pub ecc: bool,
    pub nvlink: bool,
    pub nic: bool,
    pub thermal: bool,
}

impl Health {
    pub fn ok() -> Self {
        Self::default()
    }

    pub fn unhealthy(&self) -> bool {
        self.xid || self.ecc || self.nvlink || self.nic || self.thermal
    }
}

#[derive(Debug, Clone)]
pub struct StragglerDetector {
    timings: HashMap<u32, Vec<u64>>,
    window: usize,
    factor: f64,
}

impl StragglerDetector {
    pub fn new(window: usize, factor: f64) -> Self {
        Self {
            timings: HashMap::new(),
            window: window.max(1),
            factor,
        }
    }

    pub fn record(&mut self, rank: u32, micros: u64) {
        let v = self.timings.entry(rank).or_default();
        v.push(micros);
        if v.len() > self.window {
            v.remove(0);
        }
    }

    fn median(vals: &mut [u64]) -> f64 {
        if vals.is_empty() {
            return 0.0;
        }
        vals.sort_unstable();
        let n = vals.len();
        if n % 2 == 1 {
            vals[n / 2] as f64
        } else {
            (vals[n / 2 - 1] as f64 + vals[n / 2] as f64) / 2.0
        }
    }

    /// Ranks whose latest step is slower than `factor` times the fleet median.
    pub fn slow_ranks(&self) -> Vec<u32> {
        let mut lasts: Vec<(u32, u64)> = self
            .timings
            .iter()
            .filter_map(|(&r, v)| v.last().copied().map(|t| (r, t)))
            .collect();
        if lasts.len() < 2 {
            return vec![];
        }
        let mut times: Vec<u64> = lasts.iter().map(|(_, t)| *t).collect();
        let med = Self::median(&mut times);
        if med <= 0.0 {
            return vec![];
        }
        lasts.sort_by_key(|(r, _)| *r);
        lasts
            .into_iter()
            .filter(|(_, t)| *t as f64 > med * self.factor)
            .map(|(r, _)| r)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_and_stragglers() {
        let mut h = Health::ok();
        assert!(!h.unhealthy());
        h.thermal = true;
        assert!(h.unhealthy());
        let mut d = StragglerDetector::new(8, 2.0);
        for r in 0..4u32 {
            d.record(r, 1000);
        }
        d.record(3, 5000);
        assert_eq!(d.slow_ranks(), vec![3]);
    }
}
