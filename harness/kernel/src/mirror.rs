//! Local fetch-mirror path mapping. Never percent-decodes; never talks to the network.

use std::path::{Path, PathBuf};

use crate::is_mirror_url;

/// Map a mirror URL to a path under `mirror_root`.
/// Host directory is lowercase; query/fragment/port are stripped; no percent-decode.
pub fn file_path(mirror_root: &Path, url: &str) -> Option<PathBuf> {
    if url.contains('\0') {
        return None;
    }
    if !is_mirror_url(url) {
        return None;
    }
    let rest = url.strip_prefix("https://")?;
    let (hostport, path) = match rest.split_once('/') {
        Some((h, p)) => (h, p),
        None => (rest, ""),
    };
    let host = hostport.split(':').next().unwrap_or(hostport);
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host.is_empty() {
        return None;
    }
    let path = path.split('?').next().unwrap_or(path);
    let path = path.split('#').next().unwrap_or(path);
    let mut out = mirror_root.join(host);
    if !path.is_empty() {
        for seg in path.split('/') {
            if seg.is_empty() {
                continue;
            }
            if seg == "." || seg == ".." || seg.contains('\0') {
                return None;
            }
            out.push(seg);
        }
    }
    Some(out)
}

pub fn read_bytes(mirror_root: &Path, path: &Path) -> Option<Vec<u8>> {
    if !path.is_file() {
        return None;
    }
    let canon = path.canonicalize().ok()?;
    let root = mirror_root.canonicalize().ok()?;
    if !canon.starts_with(&root) {
        return None;
    }
    std::fs::read(path).ok()
}
