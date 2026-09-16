//! Overlay FS, network policy, time-cut web snapshot, tool schemas, fleet.

use crate::Snapshot;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Default)]
pub struct OverlayFs {
    pub base: Snapshot,
    overlay: HashMap<String, String>,
    pub flagged: Vec<String>,
}

impl OverlayFs {
    pub fn new(base: Snapshot) -> Self {
        Self {
            base,
            overlay: HashMap::new(),
            flagged: Vec::new(),
        }
    }

    pub fn read(&self, path: &str) -> Option<&str> {
        if let Some(s) = self.overlay.get(path) {
            return Some(s.as_str());
        }
        if self.base.files.contains(path) {
            Some("")
        } else {
            None
        }
    }

    pub fn write(&mut self, path: &str, body: &str) -> Result<(), String> {
        if path.contains("test") {
            self.flagged.push(path.into());
            return Err("test-file write blocked".into());
        }
        self.overlay.insert(path.into(), body.into());
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct NetworkPolicy {
    allow: HashSet<String>,
}

impl NetworkPolicy {
    pub fn deny_all() -> Self {
        Self::default()
    }

    pub fn allow(&mut self, host: &str) {
        self.allow.insert(host.to_string());
    }

    pub fn check(&self, host: &str) -> bool {
        self.allow.contains(host)
    }
}

#[derive(Debug, Clone)]
pub struct WebSnapshot {
    pub cutoff: u64,
    pages: HashMap<String, (u64, String)>,
}

impl WebSnapshot {
    pub fn new(cutoff: u64) -> Self {
        Self {
            cutoff,
            pages: HashMap::new(),
        }
    }

    pub fn insert(&mut self, url: &str, fetched_at: u64, body: String) {
        self.pages.insert(url.into(), (fetched_at, body));
    }

    pub fn get(&self, url: &str) -> Result<&str, String> {
        match self.pages.get(url) {
            None => Err("miss".into()),
            Some((t, _)) if *t > self.cutoff => Err("past cutoff".into()),
            Some((_, body)) => Ok(body),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolParam {
    pub name: String,
    pub ty: String,
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolSchema {
    pub name: String,
    pub params: Vec<ToolParam>,
}

impl ToolSchema {
    pub fn validate(&self, args: &serde_json::Value) -> Result<(), String> {
        let obj = args.as_object().ok_or("args not object")?;
        for p in &self.params {
            if p.required && !obj.contains_key(&p.name) {
                return Err(format!("missing {}", p.name));
            }
            if let Some(v) = obj.get(&p.name) {
                let ok = match p.ty.as_str() {
                    "string" => v.is_string(),
                    "number" => v.is_number(),
                    "bool" => v.is_boolean(),
                    _ => true,
                };
                if !ok {
                    return Err(format!("type {}", p.name));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct EnvFleet {
    snaps: Vec<Snapshot>,
}

impl EnvFleet {
    pub fn new(n: usize, seed: Snapshot) -> Self {
        Self {
            snaps: vec![seed; n.max(1)],
        }
    }

    pub fn len(&self) -> usize {
        self.snaps.len()
    }

    pub fn fork(&self, i: usize) -> Result<Snapshot, String> {
        self.snaps.get(i).map(|s| s.fork()).ok_or_else(|| "oob".into())
    }

    pub fn get_mut(&mut self, i: usize) -> Option<&mut Snapshot> {
        self.snaps.get_mut(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn overlay_flags_test_writes() {
        let mut base = Snapshot {
            files: HashSet::new(),
        };
        base.files.insert("src/lib.rs".into());
        let mut ov = OverlayFs::new(base);
        ov.write("src/main.rs", "fn main(){}").unwrap();
        assert!(ov.write("tests/foo.py", "pass").is_err());
        assert_eq!(ov.flagged, vec!["tests/foo.py".to_string()]);
        assert_eq!(ov.read("src/main.rs"), Some("fn main(){}"));
    }

    #[test]
    fn net_deny_and_web_cutoff() {
        let mut p = NetworkPolicy::deny_all();
        assert!(!p.check("example.com"));
        p.allow("example.com");
        assert!(p.check("example.com"));
        let mut w = WebSnapshot::new(100);
        w.insert("https://a", 50, "old".into());
        w.insert("https://b", 150, "new".into());
        assert_eq!(w.get("https://a").unwrap(), "old");
        assert!(w.get("https://b").is_err());
    }

    #[test]
    fn tool_schema_and_fleet() {
        let t = ToolSchema {
            name: "read".into(),
            params: vec![ToolParam {
                name: "path".into(),
                ty: "string".into(),
                required: true,
            }],
        };
        t.validate(&serde_json::json!({"path": "a.rs"})).unwrap();
        assert!(t.validate(&serde_json::json!({})).is_err());
        let fleet = EnvFleet::new(3, Snapshot { files: HashSet::new() });
        assert_eq!(fleet.len(), 3);
        let f = fleet.fork(1).unwrap();
        assert!(f.files.is_empty());
    }
}
