//! Ops CLI status (spec 15.2, 15.5 H6).

mod cli;

pub use cli::{parse, sign, verify_sig, Command, OpsState};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Status {
    pub nodes: u32,
    pub missions: u32,
    pub paused: bool,
}

impl Status {
    pub fn render(&self, json: bool) -> String {
        if json {
            serde_json::to_string(self).unwrap_or_default()
        } else {
            format!(
                "nodes={} missions={} paused={}",
                self.nodes, self.missions, self.paused
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip() {
        let s = Status {
            nodes: 2,
            missions: 1,
            paused: false,
        };
        let j = s.render(true);
        let back: Status = serde_json::from_str(&j).unwrap();
        assert_eq!(s, back);
    }
}
