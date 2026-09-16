//! Snapshot-fork envs + test-write flag (spec 9.4, D4).
//! Firecracker is cluster-conditional (V0 kvm probe).

use std::collections::HashSet;

#[derive(Debug, Clone)]
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
        let mut s = Snapshot { files: HashSet::new() };
        assert!(s.write("src/lib.rs", "ok").is_ok());
        assert!(s.write("tests/foo.py", "pass").is_err());
        let f = s.fork();
        assert_eq!(f.files, s.files);
    }
}
