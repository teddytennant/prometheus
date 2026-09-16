//! Capability + budget kernel (spec 15.2 H10). Never grant without a budget.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cap {
    Gpu,
    Net,
    FsWrite,
}

#[derive(Debug)]
pub struct Grant {
    pub cap: Cap,
    pub budget: u64,
}

pub fn grant(cap: Cap, budget: u64) -> Result<Grant, String> {
    if budget == 0 {
        return Err("no budget".into());
    }
    Ok(Grant { cap, budget })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_budget_denied() {
        assert!(grant(Cap::Gpu, 0).is_err());
        assert!(grant(Cap::Gpu, 1).is_ok());
    }
}
