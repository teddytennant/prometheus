//! Domain mixture weights (spec 7, 15.5 B8).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Mixture {
    pub domains: Vec<(String, f64)>,
}

impl Mixture {
    pub fn normalize(mut self) -> Result<Self, String> {
        let s: f64 = self.domains.iter().map(|(_, w)| *w).sum();
        if s <= 0.0 {
            return Err("empty mixture".into());
        }
        for (_, w) in &mut self.domains {
            *w /= s;
        }
        Ok(self)
    }

    pub fn sample<'a>(&'a self, u: f64) -> &'a str {
        let mut acc = 0.0;
        for (name, w) in &self.domains {
            acc += *w;
            if u <= acc {
                return name;
            }
        }
        &self.domains.last().unwrap().0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_one() {
        let m = Mixture {
            domains: vec![("math".into(), 2.0), ("code".into(), 2.0)],
        }
        .normalize()
        .unwrap();
        let s: f64 = m.domains.iter().map(|(_, w)| *w).sum();
        assert!((s - 1.0).abs() < 1e-12);
        assert_eq!(m.sample(0.1), "math");
        assert_eq!(m.sample(0.9), "code");
    }
}
