//! AIMD swarm size and provider failover (spec 15.2).

#[derive(Debug, Clone)]
pub struct Aimd {
    pub size: u32,
    pub min: u32,
    pub max: u32,
}

impl Aimd {
    pub fn new(size: u32, min: u32, max: u32) -> Self {
        let min = min.max(1);
        let max = max.max(min);
        Self {
            size: size.clamp(min, max),
            min,
            max,
        }
    }

    pub fn increase(&mut self) {
        self.size = (self.size + 1).min(self.max);
    }

    pub fn decrease(&mut self) {
        self.size = (self.size / 2).max(self.min);
    }
}

#[derive(Debug, Clone)]
pub struct Provider {
    pub name: String,
    pub healthy: bool,
}

#[derive(Debug, Clone)]
pub struct ProviderSet {
    providers: Vec<Provider>,
    current: usize,
}

impl ProviderSet {
    pub fn new(names: &[&str]) -> Self {
        Self {
            providers: names
                .iter()
                .map(|n| Provider {
                    name: (*n).into(),
                    healthy: true,
                })
                .collect(),
            current: 0,
        }
    }

    pub fn current(&self) -> Option<&Provider> {
        self.providers.get(self.current)
    }

    pub fn mark_unhealthy(&mut self, name: &str) {
        if let Some(p) = self.providers.iter_mut().find(|p| p.name == name) {
            p.healthy = false;
        }
    }

    pub fn failover(&mut self) -> Result<&Provider, String> {
        if self.providers.is_empty() {
            return Err("empty".into());
        }
        let n = self.providers.len();
        for i in 1..=n {
            let idx = (self.current + i) % n;
            if self.providers[idx].healthy {
                self.current = idx;
                return Ok(&self.providers[idx]);
            }
        }
        Err("all providers down".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aimd_and_failover() {
        let mut a = Aimd::new(4, 1, 8);
        a.increase();
        assert_eq!(a.size, 5);
        a.decrease();
        assert_eq!(a.size, 2);
        let mut ps = ProviderSet::new(&["p0", "p1", "p2"]);
        assert_eq!(ps.current().unwrap().name, "p0");
        ps.mark_unhealthy("p1");
        assert_eq!(ps.failover().unwrap().name, "p2");
        ps.mark_unhealthy("p2");
        assert_eq!(ps.failover().unwrap().name, "p0");
        ps.mark_unhealthy("p0");
        assert!(ps.failover().is_err());
    }
}
