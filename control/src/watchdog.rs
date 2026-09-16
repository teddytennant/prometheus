//! Collective watchdog timeout (spec 5.5).

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Watchdog {
    pub timeout_ms: u64,
    inflight: HashMap<String, u64>,
}

impl Watchdog {
    pub fn new(timeout_ms: u64) -> Self {
        Self {
            timeout_ms,
            inflight: HashMap::new(),
        }
    }

    pub fn start(&mut self, name: &str, now_ms: u64) {
        self.inflight.insert(name.to_string(), now_ms);
    }

    pub fn finish(&mut self, name: &str) -> bool {
        self.inflight.remove(name).is_some()
    }

    pub fn timed_out(&self, now_ms: u64) -> Vec<String> {
        let mut names: Vec<String> = self
            .inflight
            .iter()
            .filter(|(_, start)| now_ms.saturating_sub(**start) > self.timeout_ms)
            .map(|(n, _)| n.clone())
            .collect();
        names.sort();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_out_collectives() {
        let mut w = Watchdog::new(100);
        w.start("allreduce", 0);
        w.start("allgather", 0);
        assert!(w.timed_out(50).is_empty());
        w.finish("allgather");
        assert_eq!(w.timed_out(101), vec!["allreduce".to_string()]);
        assert!(w.finish("allreduce"));
        assert!(w.timed_out(1000).is_empty());
    }
}
