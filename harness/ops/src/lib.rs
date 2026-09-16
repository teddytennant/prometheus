//! Status JSON for the lab (spec 15.2 H8).

use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Status {
    pub live_jobs: u32,
    pub tokens_remaining: u64,
    pub last_eval: String,
}

impl Status {
    pub fn render(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip() {
        let s = Status { live_jobs: 2, tokens_remaining: 9, last_eval: "ok".into() };
        let v: serde_json::Value = serde_json::from_str(&s.render()).unwrap();
        assert_eq!(v["live_jobs"], 2);
    }
}
