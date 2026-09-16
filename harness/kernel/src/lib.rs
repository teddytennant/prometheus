//! Kernel capability tokens (spec 14.3, 15.5 L1).

mod caps;

pub use caps::{sign_update, Kernel};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cap {
    Fs,
    Net,
    Exec,
}

#[derive(Debug, Clone)]
pub struct Grant {
    pub cap: Cap,
    pub budget: u64,
}

pub fn grant(g: &Grant, used: u64) -> Result<u64, String> {
    if used > g.budget {
        return Err("budget".into());
    }
    Ok(g.budget - used)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_budget_denied() {
        let g = Grant {
            cap: Cap::Net,
            budget: 0,
        };
        assert!(grant(&g, 1).is_err());
        assert_eq!(grant(&g, 0).unwrap(), 0);
    }
}
