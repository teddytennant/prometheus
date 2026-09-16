//! Snapshot-fork env with test-file write block (spec 9.4, 15.5 D4).

mod overlay;

pub use overlay::{EnvFleet, NetworkPolicy, OverlayFs, ToolParam, ToolSchema, WebSnapshot};

use std::collections::HashSet;

#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    pub files: HashSet<String>,
}

impl Snapshot {
    pub fn fork(&self) -> Self {
        self.clone()
    }

    pub fn write(&mut self, path: &str, _body: &str) -> Result<(), String> {
        if path.contains("test") {
            return Err("test-file write blocked".into());
        }
        self.files.insert(path.into());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_test_write() {
        let mut s = Snapshot {
            files: HashSet::new(),
        };
        assert!(s.write("src/foo.rs", "x").is_ok());
        assert!(s.write("tests/test_foo.py", "x").is_err());
        let f = s.fork();
        assert!(f.files.contains("src/foo.rs"));
    }
}
