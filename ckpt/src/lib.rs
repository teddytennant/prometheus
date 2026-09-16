//! In-memory checkpoint + content hash (spec 5.4, 15.5 A6).
//! 90TB sharded ckpt and Grace offload are S3 (cluster-conditional).

use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    pub step: u64,
    pub payload: Vec<u8>,
    pub hash: String,
}

impl Checkpoint {
    pub fn new(step: u64, payload: Vec<u8>) -> Self {
        let hash = sha(&payload);
        Self { step, payload, hash }
    }

    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        fs::create_dir_all(dir)?;
        fs::write(dir.join("payload.bin"), &self.payload)?;
        fs::write(dir.join("meta.json"), format!(r#"{{"step":{},"hash":"{}"}}"#, self.step, self.hash))?;
        Ok(())
    }

    pub fn load(dir: &Path) -> std::io::Result<Self> {
        let payload = fs::read(dir.join("payload.bin"))?;
        let meta = fs::read_to_string(dir.join("meta.json"))?;
        let v: serde_json::Value = serde_json::from_str(&meta).map_err(std::io::Error::other)?;
        let step = v["step"].as_u64().unwrap_or(0);
        let hash = v["hash"].as_str().unwrap_or("").to_string();
        if sha(&payload) != hash {
            return Err(std::io::Error::other("hash mismatch"));
        }
        Ok(Self { step, payload, hash })
    }
}

pub fn sha(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

pub fn bitwise_equal(a: &Checkpoint, b: &Checkpoint) -> bool {
    a.payload == b.payload && a.step == b.step
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn roundtrip_bitwise() {
        let dir = tempdir().unwrap();
        let ck = Checkpoint::new(3, b"abc123".to_vec());
        ck.save(dir.path()).unwrap();
        let loaded = Checkpoint::load(dir.path()).unwrap();
        assert!(bitwise_equal(&ck, &loaded));
    }

    #[test]
    fn detects_corruption() {
        let dir = tempdir().unwrap();
        let ck = Checkpoint::new(1, b"payload".to_vec());
        ck.save(dir.path()).unwrap();
        fs::write(dir.path().join("payload.bin"), b"tampered").unwrap();
        assert!(Checkpoint::load(dir.path()).is_err());
    }
}
