//! Token broker / spend ledger (spec 15.2 H3).

#[derive(Debug)]
pub struct Broker {
    pub remaining: u64,
}

impl Broker {
    pub fn new(budget: u64) -> Self {
        Self { remaining: budget }
    }

    pub fn spend(&mut self, n: u64) -> Result<(), String> {
        if n > self.remaining {
            return Err("budget".into());
        }
        self.remaining -= n;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_overspend() {
        let mut b = Broker::new(10);
        b.spend(7).unwrap();
        assert!(b.spend(4).is_err());
        b.spend(3).unwrap();
        assert_eq!(b.remaining, 0);
    }
}
