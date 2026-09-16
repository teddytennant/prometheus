//! Oracle / reviewer / implementer records (spec 15.2 H5).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Record {
    pub role: String,
    pub artifact: String,
    pub ok: bool,
}

#[derive(Debug, Default)]
pub struct Pipeline {
    pub records: Vec<Record>,
}

impl Pipeline {
    pub fn push(&mut self, r: Record) {
        self.records.push(r);
    }

    pub fn last_ok(&self, role: &str) -> bool {
        self.records.iter().rev().find(|r| r.role == role).map(|r| r.ok).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_roles() {
        let mut p = Pipeline::default();
        p.push(Record { role: "oracle".into(), artifact: "t".into(), ok: true });
        p.push(Record { role: "reviewer".into(), artifact: "t".into(), ok: false });
        assert!(p.last_ok("oracle"));
        assert!(!p.last_ok("reviewer"));
    }
}
