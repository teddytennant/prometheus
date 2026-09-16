//! Request state machine: verbal/latent, recurrence buckets, KV tiers, routing, MTP.

use std::collections::HashMap;

pub const RECURRENCE_BUCKETS: [u32; 5] = [1, 2, 4, 8, 16];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecodeKind {
    Verbal,
    Latent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvTier {
    Hbm,
    Grace,
    Nvme,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvBlock {
    pub tokens: Vec<u32>,
    pub state: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub id: String,
    pub kind: DecodeKind,
    pub recurrence: u32,
    pub kv_tier: KvTier,
    pub tokens: Vec<u32>,
    pub latent: Option<Vec<f32>>,
    pub routing_ids: Vec<u32>,
    pub mtp_draft: Vec<u32>,
    pub ttt_adapter: Option<String>,
}

#[derive(Debug, Default)]
pub struct Engine {
    requests: HashMap<String, Request>,
    kv: HashMap<(String, KvTier), KvBlock>,
    ttt: Option<String>,
}

impl Engine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bucket_r(r: u32) -> u32 {
        RECURRENCE_BUCKETS
            .iter()
            .copied()
            .find(|&b| b >= r.max(1))
            .unwrap_or(16)
    }

    pub fn open_verbal(&mut self, id: &str, r: u32) -> &Request {
        let recurrence = Self::bucket_r(r);
        self.requests.entry(id.to_string()).or_insert(Request {
            id: id.into(),
            kind: DecodeKind::Verbal,
            recurrence,
            kv_tier: KvTier::Hbm,
            tokens: vec![],
            latent: None,
            routing_ids: vec![],
            mtp_draft: vec![],
            ttt_adapter: None,
        })
    }

    pub fn open_latent(&mut self, id: &str, r: u32, embed: Vec<f32>) -> &Request {
        let recurrence = Self::bucket_r(r);
        self.requests.entry(id.to_string()).or_insert(Request {
            id: id.into(),
            kind: DecodeKind::Latent,
            recurrence,
            kv_tier: KvTier::Hbm,
            tokens: vec![],
            latent: Some(embed),
            routing_ids: vec![],
            mtp_draft: vec![],
            ttt_adapter: None,
        })
    }

    pub fn append_token(&mut self, id: &str, tok: u32) {
        if let Some(r) = self.requests.get_mut(id) {
            r.tokens.push(tok);
        }
    }

    pub fn capture_routing(&mut self, id: &str, expert: u32) {
        if let Some(r) = self.requests.get_mut(id) {
            r.routing_ids.push(expert);
        }
    }

    pub fn routing_ids(&self, id: &str) -> &[u32] {
        self.requests
            .get(id)
            .map(|r| r.routing_ids.as_slice())
            .unwrap_or(&[])
    }

    pub fn set_ttt(&mut self, sidecar: &str) {
        self.ttt = Some(sidecar.into());
    }

    pub fn ttt_hook(&mut self, id: &str, adapter: &str) -> bool {
        if self.ttt.is_none() {
            return false;
        }
        if let Some(r) = self.requests.get_mut(id) {
            r.ttt_adapter = Some(adapter.into());
            true
        } else {
            false
        }
    }

    fn snapshot_kv(req: &Request) -> KvBlock {
        KvBlock {
            tokens: req.tokens.clone(),
            state: req
                .tokens
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect(),
        }
    }

    /// Swap KV from the request's current tier onto `to`, then park there.
    pub fn swap_kv(&mut self, id: &str, to: KvTier) -> Result<(), String> {
        let req = self.requests.get_mut(id).ok_or("unknown request")?;
        let block = Self::snapshot_kv(req);
        self.kv.insert((id.to_string(), req.kv_tier), block.clone());
        self.kv.insert((id.to_string(), to), block);
        req.kv_tier = to;
        Ok(())
    }

    pub fn restore_kv(&mut self, id: &str, from: KvTier) -> Result<(), String> {
        let block = self
            .kv
            .get(&(id.to_string(), from))
            .cloned()
            .ok_or("no kv at tier")?;
        let req = self.requests.get_mut(id).ok_or("unknown request")?;
        req.tokens = block.tokens.clone();
        req.kv_tier = KvTier::Hbm;
        self.kv.insert((id.to_string(), KvTier::Hbm), block);
        Ok(())
    }

    /// Verbal and latent share one batch conceptually; returned grouped by kind then r.
    pub fn batch(&self) -> Vec<Vec<&Request>> {
        let mut groups: HashMap<(DecodeKind, u32), Vec<&Request>> = HashMap::new();
        for r in self.requests.values() {
            groups
                .entry((r.kind, r.recurrence))
                .or_default()
                .push(r);
        }
        let mut keys: Vec<(DecodeKind, u32)> = groups.keys().copied().collect();
        keys.sort_by_key(|(k, r)| (*k == DecodeKind::Latent, *r));
        keys.into_iter().map(|k| groups.remove(&k).unwrap()).collect()
    }

    pub fn speculate_mtp(&mut self, id: &str, head_logits: &[Vec<f32>]) -> Vec<u32> {
        let draft: Vec<u32> = head_logits.iter().map(|h| argmax(h)).collect();
        if let Some(r) = self.requests.get_mut(id) {
            r.mtp_draft = draft.clone();
        }
        draft
    }

    pub fn get(&self, id: &str) -> Option<&Request> {
        self.requests.get(id)
    }
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

pub fn mtp_accept(draft: &[u32], actual: &[u32]) -> usize {
    draft
        .iter()
        .zip(actual)
        .take_while(|(d, a)| d == a)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_are_regular() {
        assert_eq!(Engine::bucket_r(1), 1);
        assert_eq!(Engine::bucket_r(3), 4);
        assert_eq!(Engine::bucket_r(8), 8);
        assert_eq!(Engine::bucket_r(9), 16);
        assert_eq!(Engine::bucket_r(99), 16);
        let mut e = Engine::new();
        e.open_verbal("a", 3);
        e.open_verbal("b", 4);
        e.open_latent("c", 2, vec![0.1, 0.2]);
        let batches = e.batch();
        for g in &batches {
            let r0 = g[0].recurrence;
            let k0 = g[0].kind;
            assert!(g.iter().all(|x| x.recurrence == r0 && x.kind == k0));
            assert!(RECURRENCE_BUCKETS.contains(&r0));
        }
    }

    #[test]
    fn swap_restore_and_routing() {
        let mut e = Engine::new();
        e.open_verbal("s", 1);
        e.append_token("s", 7);
        e.append_token("s", 8);
        e.capture_routing("s", 42);
        e.capture_routing("s", 7);
        e.swap_kv("s", KvTier::Grace).unwrap();
        assert_eq!(e.get("s").unwrap().kv_tier, KvTier::Grace);
        e.swap_kv("s", KvTier::Nvme).unwrap();
        e.append_token("s", 99);
        e.restore_kv("s", KvTier::Grace).unwrap();
        let r = e.get("s").unwrap();
        assert_eq!(r.tokens, vec![7, 8]);
        assert_eq!(r.kv_tier, KvTier::Hbm);
        assert_eq!(e.routing_ids("s"), &[42, 7]);
    }

    #[test]
    fn mtp_and_ttt() {
        let mut e = Engine::new();
        e.open_verbal("s", 1);
        e.set_ttt("arc-lora");
        assert!(e.ttt_hook("s", "task-9"));
        let draft = e.speculate_mtp("s", &[vec![0.1, 0.9, 0.0], vec![0.8, 0.1, 0.0]]);
        assert_eq!(draft, vec![1, 0]);
        assert_eq!(mtp_accept(&draft, &[1, 0, 3]), 2);
        assert_eq!(mtp_accept(&draft, &[1, 2]), 1);
        assert_eq!(e.get("s").unwrap().ttt_adapter.as_deref(), Some("task-9"));
    }
}
