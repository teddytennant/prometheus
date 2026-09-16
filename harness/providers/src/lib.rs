//! Token broker + spend accounting (spec 15.2, 15.5 H3).

mod aimd;
mod broker;
mod validate;

pub use aimd::{Aimd, Provider, ProviderSet};
pub use broker::TokenBroker;
pub use validate::{validate_response, ValidateError};

#[derive(Debug)]
pub struct Broker {
    pub remaining: u64,
}

impl Broker {
    pub fn new(budget: u64) -> Self {
        Self { remaining: budget }
    }

    pub fn spend(&mut self, n: u64) -> Result<u64, String> {
        if n > self.remaining {
            return Err("overspend".into());
        }
        self.remaining -= n;
        Ok(self.remaining)
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
        assert_eq!(b.remaining, 3);
    }
}
