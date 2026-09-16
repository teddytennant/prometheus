//! Capability API, budgets, fetch-from-mirror, promote, signed updates (spec 14.3/14.10).

use crate::Cap;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Kernel {
    caps: HashSet<Cap>,
    pub token_budget: u64,
    pub time_budget_ms: u64,
    pub tokens_used: u64,
    pub time_used_ms: u64,
    mirror: HashMap<String, Vec<u8>>,
    pub version: u32,
    pub update_keys: Vec<String>,
    promoted: Vec<String>,
}

impl Kernel {
    pub fn new(token_budget: u64, time_budget_ms: u64) -> Self {
        Self {
            caps: HashSet::new(),
            token_budget,
            time_budget_ms,
            tokens_used: 0,
            time_used_ms: 0,
            mirror: HashMap::new(),
            version: 1,
            update_keys: vec!["human-0".into(), "human-1".into()],
            promoted: vec![],
        }
    }

    pub fn grant_cap(&mut self, cap: Cap) {
        self.caps.insert(cap);
    }

    pub fn has(&self, cap: Cap) -> bool {
        self.caps.contains(&cap)
    }

    pub fn spend_tokens(&mut self, n: u64) -> Result<u64, String> {
        if !self.has(Cap::Net) && n > 0 {
            return Err("no net cap".into());
        }
        let next = self.tokens_used.saturating_add(n);
        if next > self.token_budget {
            return Err("token budget".into());
        }
        self.tokens_used = next;
        Ok(self.token_budget - self.tokens_used)
    }

    pub fn spend_time(&mut self, ms: u64) -> Result<u64, String> {
        let next = self.time_used_ms.saturating_add(ms);
        if next > self.time_budget_ms {
            return Err("time budget".into());
        }
        self.time_used_ms = next;
        Ok(self.time_budget_ms - self.time_used_ms)
    }

    pub fn mirror_put(&mut self, url: &str, body: Vec<u8>) {
        self.mirror.insert(url.into(), body);
    }

    pub fn fetch(&self, url: &str) -> Result<&[u8], String> {
        if !self.has(Cap::Net) {
            return Err("no net cap".into());
        }
        self.mirror.get(url).map(|b| b.as_slice()).ok_or_else(|| "not in mirror".into())
    }

    pub fn promote(&mut self, artifact: &str) -> Result<(), String> {
        if !self.has(Cap::Exec) {
            return Err("no exec cap".into());
        }
        if artifact.is_empty() {
            return Err("empty artifact".into());
        }
        self.promoted.push(artifact.into());
        Ok(())
    }

    pub fn update_kernel(&mut self, new_ver: u32, signatures: &[String]) -> Result<(), String> {
        if signatures.is_empty() {
            return Err("unsigned kernel update".into());
        }
        let msg = format!("kernel:{new_ver}");
        let mut ok = 0usize;
        for k in &self.update_keys {
            let mut h = Sha256::new();
            h.update(k.as_bytes());
            h.update(msg.as_bytes());
            let sig = format!("{:x}", h.finalize());
            if signatures.iter().any(|s| s == &sig) {
                ok += 1;
            }
        }
        if ok < self.update_keys.len() {
            return Err("unsigned kernel update".into());
        }
        self.version = new_ver;
        Ok(())
    }

    pub fn promoted(&self) -> &[String] {
        &self.promoted
    }
}

pub fn sign_update(key: &str, ver: u32) -> String {
    let mut h = Sha256::new();
    h.update(key.as_bytes());
    h.update(format!("kernel:{ver}").as_bytes());
    format!("{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_mirror_unsigned_refused() {
        let mut k = Kernel::new(10, 100);
        k.grant_cap(Cap::Net);
        k.grant_cap(Cap::Exec);
        k.spend_tokens(4).unwrap();
        assert!(k.spend_tokens(7).is_err());
        k.mirror_put("https://m/a", b"bin".to_vec());
        assert_eq!(k.fetch("https://m/a").unwrap(), b"bin");
        assert!(k.fetch("https://evil").is_err());
        k.promote("art-1").unwrap();
        assert!(k.update_kernel(2, &[]).is_err());
        assert!(k.update_kernel(2, &["deadbeef".into()]).is_err());
        let sigs: Vec<String> = k.update_keys.iter().map(|key| sign_update(key, 2)).collect();
        k.update_kernel(2, &sigs).unwrap();
        assert_eq!(k.version, 2);
    }
}
