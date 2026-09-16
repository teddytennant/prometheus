//! Worker swarm: spawn N, collect, no duplicated output (spec 15.2 H2).

use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

#[derive(Debug)]
pub struct Swarm {
    pub inflight: HashSet<TaskId>,
    pub done: HashSet<TaskId>,
}

impl Swarm {
    pub fn new() -> Self {
        Self { inflight: HashSet::new(), done: HashSet::new() }
    }

    pub fn admit(&mut self, id: TaskId) -> Result<(), String> {
        if self.inflight.contains(&id) || self.done.contains(&id) {
            return Err("duplicate".into());
        }
        self.inflight.insert(id);
        Ok(())
    }

    pub fn finish(&mut self, id: TaskId) -> Result<(), String> {
        if !self.inflight.remove(&id) {
            return Err("unknown".into());
        }
        self.done.insert(id);
        Ok(())
    }
}

impl Default for Swarm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_dup() {
        let mut s = Swarm::new();
        s.admit(TaskId(1)).unwrap();
        assert!(s.admit(TaskId(1)).is_err());
        s.finish(TaskId(1)).unwrap();
        assert!(s.done.contains(&TaskId(1)));
    }
}
