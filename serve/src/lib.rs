//! Session router + KV tiering stand-in (spec 8, 15.5 C5).
//! Request state machine, buckets, routing capture, MTP (spec 13).

mod engine;

pub use engine::{
    argmax, mtp_accept, DecodeKind, Engine, KvBlock, KvTier, Request, RECURRENCE_BUCKETS,
};

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub tokens: Vec<u32>,
    pub kv_tier: Tier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Gpu,
    Host,
    Nvme,
}

#[derive(Debug, Default)]
pub struct Router {
    sessions: HashMap<String, Session>,
}

impl Router {
    pub fn open(&mut self, id: &str) -> &Session {
        self.sessions.entry(id.to_string()).or_insert(Session {
            id: id.into(),
            tokens: vec![],
            kv_tier: Tier::Gpu,
        })
    }

    pub fn append(&mut self, id: &str, tok: u32) {
        self.open(id);
        self.sessions.get_mut(id).unwrap().tokens.push(tok);
    }

    pub fn demote(&mut self, id: &str, tier: Tier) {
        if let Some(s) = self.sessions.get_mut(id) {
            s.kv_tier = tier;
        }
    }

    pub fn restore_matches(a: &Session, b: &Session) -> bool {
        a.tokens == b.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_restore() {
        let mut r = Router::default();
        r.append("s", 1);
        r.append("s", 2);
        r.demote("s", Tier::Nvme);
        let snap = r.open("s").clone();
        assert!(Router::restore_matches(&snap, r.open("s")));
        assert_eq!(snap.kv_tier, Tier::Nvme);
    }
}
