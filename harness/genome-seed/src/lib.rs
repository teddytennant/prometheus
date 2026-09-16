//! Seed genome (spec 14.3, 15.5 L4).

mod genome;

pub use genome::{Genome, Program, Role, Skill, Tool};

#[derive(Debug, Clone)]
pub struct Seed {
    pub name: String,
    pub claim: String,
}

pub fn parse(md: &str) -> Seed {
    let mut name = "unknown".to_string();
    let mut claim = String::new();
    for line in md.lines() {
        if let Some(rest) = line.strip_prefix("# ") {
            name = rest.trim().to_string();
        }
        if let Some(rest) = line.strip_prefix("claim:") {
            claim = rest.trim().to_string();
        }
    }
    Seed { name, claim }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_seed() {
        let s = parse("# muon-clip\nclaim: QK clip stops logit explosion\n");
        assert_eq!(s.name, "muon-clip");
        assert!(s.claim.contains("QK"));
    }
}
