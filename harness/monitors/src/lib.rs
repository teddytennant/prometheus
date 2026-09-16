//! Integrity / hacking / boundary monitors (spec 14.10, 15.5 L3).

mod ring;

pub use ring::{KillSwitch, Monitor, MonitorRing, Tripwires};

#[derive(Debug, Clone)]
pub struct Trip {
    pub name: String,
    pub fired: bool,
}

pub fn watch(prev: f64, next: f64, thresh: f64) -> Trip {
    Trip {
        name: "rci".into(),
        fired: (next - prev).abs() > thresh,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trips() {
        assert!(watch(0.1, 0.5, 0.2).fired);
        assert!(!watch(0.1, 0.15, 0.2).fired);
    }
}
