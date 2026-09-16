//! Shared assertions for B1 extract integration tests.
#![allow(dead_code)]

use prometheus_extract::{sha256_hex, Error, ExtractedDocument, Format};

pub const PAGE_HTML: &[u8] = include_bytes!("../goldens/page.html");
pub const PAGE_EXPECTED: &str = include_str!("../goldens/page.expected.txt");
pub const TINY_PDF: &[u8] = include_bytes!("../goldens/tiny.pdf");
pub const TINY_EXPECTED: &str = include_str!("../goldens/tiny.expected.txt");
pub const MATH_TEX: &[u8] = include_bytes!("../goldens/math.tex");
pub const MATH_EXPECTED: &str = include_str!("../goldens/math.expected.txt");

pub fn phrases_from_expected(expected: &str) -> Vec<&str> {
    expected
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}

/// Require each phrase to occur in order in `text` (whitespace inside a phrase
/// is exact; surrounding whitespace is free).
pub fn assert_phrases_in_order(text: &str, phrases: &[&str]) {
    let mut rest = text;
    for phrase in phrases {
        match rest.find(phrase) {
            Some(i) => rest = &rest[i + phrase.len()..],
            None => panic!(
                "extracted text is missing phrase {phrase:?}\nfull text: {text:?}\nremaining: {rest:?}"
            ),
        }
    }
}

pub fn assert_forbidden(text: &str, needles: &[&str]) {
    for needle in needles {
        assert!(
            !text.contains(needle),
            "extracted text must not contain {needle:?}, got {text:?}"
        );
    }
}

pub fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn assert_err_empty(result: Result<ExtractedDocument, Error>) {
    match result {
        Err(Error::Empty) => {}
        Ok(doc) => panic!("expected Error::Empty, got Ok(text={:?})", doc.text),
        Err(other) => panic!("expected Error::Empty, got Err({other:?})"),
    }
}

pub fn assert_err_unparseable(result: Result<ExtractedDocument, Error>, expected: Format) {
    match result {
        Err(Error::Unparseable { format }) => {
            assert_eq!(format, expected, "Unparseable format tag");
        }
        Ok(doc) => panic!(
            "expected Error::Unparseable {{ format: {expected:?} }}, got Ok(text={:?})",
            doc.text
        ),
        Err(other) => {
            panic!("expected Error::Unparseable {{ format: {expected:?} }}, got Err({other:?})")
        }
    }
}

pub fn assert_err_unsupported(result: Result<ExtractedDocument, Error>) {
    match result {
        Err(Error::Unsupported) => {}
        Ok(doc) => panic!("expected Error::Unsupported, got Ok(text={:?})", doc.text),
        Err(other) => panic!("expected Error::Unsupported, got Err({other:?})"),
    }
}

pub fn assert_content_hash(doc: &ExtractedDocument) {
    assert_eq!(
        doc.content_hash,
        sha256_hex(doc.text.as_bytes()),
        "content_hash must be SHA-256 hex of UTF-8 text bytes"
    );
    assert_eq!(doc.content_hash.len(), 64, "SHA-256 hex is 64 chars");
    assert!(
        doc.content_hash.bytes().all(|b| b.is_ascii_hexdigit()),
        "content_hash must be hex, got {:?}",
        doc.content_hash
    );
    assert_eq!(
        doc.content_hash,
        doc.content_hash.to_ascii_lowercase(),
        "content_hash must be lowercase hex"
    );
}

pub fn assert_meta(doc: &ExtractedDocument, format: Format, source: Option<&str>, bytes_in: u64) {
    assert_eq!(doc.format, format, "format");
    assert_eq!(doc.source.as_deref(), source, "source");
    assert_eq!(doc.bytes_in, bytes_in, "bytes_in");
    assert_content_hash(doc);
}
