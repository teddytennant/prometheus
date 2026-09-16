//! Single token broker by lease; rotate commits before use (spec 15.2).

use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct TokenBroker {
    pub lease_holder: Option<String>,
    refresh: String,
    access: String,
    committed: bool,
}

impl TokenBroker {
    pub fn new(refresh: &str) -> Self {
        Self {
            lease_holder: None,
            refresh: refresh.into(),
            access: derive_access(refresh, 0),
            committed: true,
        }
    }

    pub fn take_lease(&mut self, node: &str) -> Result<(), String> {
        if let Some(h) = &self.lease_holder {
            if h != node {
                return Err("broker leased".into());
            }
        }
        self.lease_holder = Some(node.into());
        Ok(())
    }

    pub fn release(&mut self, node: &str) -> Result<(), String> {
        match &self.lease_holder {
            Some(h) if h == node => {
                self.lease_holder = None;
                Ok(())
            }
            _ => Err("not holder".into()),
        }
    }

    /// Commit the rotated refresh token first, then derive a new access token.
    pub fn rotate(&mut self, new_refresh: &str) {
        self.committed = false;
        self.refresh = new_refresh.into();
        self.committed = true;
        self.access = derive_access(&self.refresh, 1);
    }

    pub fn access(&self) -> Result<&str, String> {
        if !self.committed {
            return Err("uncommitted token".into());
        }
        if self.lease_holder.is_none() {
            return Err("no lease".into());
        }
        Ok(&self.access)
    }

    pub fn refresh_committed(&self) -> &str {
        &self.refresh
    }
}

fn derive_access(refresh: &str, gen: u64) -> String {
    let mut h = Sha256::new();
    h.update(refresh.as_bytes());
    h.update(gen.to_le_bytes());
    format!("{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_lease_and_commit_before_use() {
        let mut b = TokenBroker::new("r0");
        assert!(b.access().is_err());
        b.take_lease("n0").unwrap();
        let a0 = b.access().unwrap().to_string();
        assert!(b.take_lease("n1").is_err());
        b.rotate("r1");
        let a1 = b.access().unwrap().to_string();
        assert_ne!(a0, a1);
        assert_eq!(b.refresh_committed(), "r1");
    }
}
