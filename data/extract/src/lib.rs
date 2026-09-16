//! Ingest and extraction for HTML, PDF, and LaTeX (spec 7, 15.5 B1).
//!
//! The pipeline is Rust. This crate turns a raw byte blob into UTF-8 text
//! plus provenance. LaTeX math is preserved as source (`$...$`, `$$...$$`,
//! and environments), not converted to Unicode approximations.
//!
//! Gate: golden outputs on a fixed corpus. Nothing here extracts yet.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const PARQUET_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum Error {
    #[error("empty input")]
    Empty,
    #[error("unparseable {format:?}")]
    Unparseable { format: Format },
    #[error("unsupported format")]
    Unsupported,
    #[error("{0}")]
    Message(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Html,
    Pdf,
    Latex,
}

impl Format {
    pub fn from_media_type(media_type: &str) -> Option<Self> {
        match media_type {
            "text/html" | "application/xhtml+xml" => Some(Self::Html),
            "application/pdf" => Some(Self::Pdf),
            "application/x-latex" | "application/x-tex" | "text/x-latex" | "text/x-tex" => {
                Some(Self::Latex)
            }
            _ => None,
        }
    }

    pub fn from_path(path: &str) -> Option<Self> {
        let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
        match ext.as_str() {
            "html" | "htm" | "xhtml" => Some(Self::Html),
            "pdf" => Some(Self::Pdf),
            "tex" | "latex" => Some(Self::Latex),
            _ => None,
        }
    }
}

/// Extracted document. `text` is UTF-8. `content_hash` is SHA-256 of `text`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedDocument {
    pub text: String,
    pub format: Format,
    pub source: Option<String>,
    pub content_hash: String,
    pub bytes_in: u64,
}

impl ExtractedDocument {
    pub fn from_text(text: String, format: Format, source: Option<String>, bytes_in: u64) -> Self {
        let content_hash = sha256_hex(text.as_bytes());
        Self {
            text,
            format,
            source,
            content_hash,
            bytes_in,
        }
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex_encode(&digest)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// Extract UTF-8 text. Empty input is `Error::Empty`. Corrupt or undecodable
/// input is `Error::Unparseable`. LaTeX math must remain as source.
pub fn extract(bytes: &[u8], format: Format) -> Result<ExtractedDocument, Error> {
    let _ = (bytes, format);
    unimplemented!("B1 extract")
}

/// Guess format from path or media type, then extract.
pub fn extract_from(
    bytes: &[u8],
    path: Option<&str>,
    media_type: Option<&str>,
) -> Result<ExtractedDocument, Error> {
    let _ = (bytes, path, media_type);
    unimplemented!("B1 extract_from")
}

/// True if `text` still contains at least one LaTeX math span (`$`, `$$`, or
/// `\begin{...}` for a math environment). Used by goldens that feed LaTeX.
pub fn preserves_latex_math(text: &str) -> bool {
    let _ = text;
    unimplemented!("B1 preserves_latex_math")
}
