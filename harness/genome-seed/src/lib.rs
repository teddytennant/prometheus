//! Parse a genome-seed markdown block (spec 15.2 L1-L4).

#[derive(Debug, PartialEq, Eq)]
pub struct Seed {
    pub name: String,
    pub claim: String,
}

pub fn parse(md: &str) -> Option<Seed> {
    let mut name = None;
    let mut claim = None;
    for line in md.lines() {
        if let Some(rest) = line.strip_prefix("# ") {
            name = Some(rest.trim().to_string());
        }
        if let Some(rest) = line.strip_prefix("claim: ") {
            claim = Some(rest.trim().to_string());
        }
    }
    Some(Seed { name: name?, claim: claim? })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_seed() {
        let s = parse("# muon-clip\nclaim: QK clip stops logit explosion\n").unwrap();
        assert_eq!(s.name, "muon-clip");
        assert!(s.claim.contains("QK"));
    }
}
