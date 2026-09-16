//! Crash-only swarm: idempotent admit/finish (spec 15.2, 15.5 H2).

mod lease;
mod log;

pub use lease::{ArtifactCas, Cluster, Lease, Task};
pub use log::{Event, EventLog};

use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TaskId(pub u64);

#[derive(Debug, Default)]
pub struct Swarm {
    admitted: HashSet<u64>,
    finished: HashSet<u64>,
}

impl Swarm {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn admit(&mut self, id: TaskId) -> Result<(), String> {
        if !self.admitted.insert(id.0) {
            return Err("dup".into());
        }
        Ok(())
    }

    pub fn finish(&mut self, id: TaskId) -> Result<(), String> {
        if !self.admitted.contains(&id.0) {
            return Err("unknown".into());
        }
        if !self.finished.insert(id.0) {
            return Err("dup".into());
        }
        Ok(())
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
        assert!(s.finish(TaskId(1)).is_err());
    }
}
