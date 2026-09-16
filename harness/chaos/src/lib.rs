//! Chaos faults for soak (spec 15.2, 15.5 H11).

mod runner;

pub use runner::{ChaosFault, ChaosRunner, HarnessDouble, Task};

#[derive(Debug, Clone)]
pub enum Fault {
    Drop { task: u64 },
    Dup { task: u64 },
}

pub fn apply(f: Fault, seen: &mut std::collections::HashSet<u64>) -> Result<usize, String> {
    match f {
        Fault::Drop { task } => {
            seen.remove(&task);
            Err("lost".into())
        }
        Fault::Dup { task } => {
            if !seen.insert(task) {
                return Err("dup".into());
            }
            Ok(seen.len())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn drop_is_error() {
        let mut s = HashSet::from([1]);
        assert!(apply(Fault::Drop { task: 1 }, &mut s).is_err());
    }
}
