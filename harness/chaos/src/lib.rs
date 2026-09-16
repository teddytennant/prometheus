//! Fault injection for soak (spec 15.2 H6, V10).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    KillRank,
    BitFlip,
    BadShard,
    DropTask,
}

pub fn apply(fault: Fault, n_tasks: usize) -> Result<usize, String> {
    match fault {
        Fault::DropTask => Err("lost task".into()),
        Fault::KillRank | Fault::BitFlip | Fault::BadShard => Ok(n_tasks),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drop_is_error() {
        assert!(apply(Fault::DropTask, 4).is_err());
        assert_eq!(apply(Fault::KillRank, 4).unwrap(), 4);
    }
}
