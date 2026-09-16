//! Group: sha256_hex and ExtractedDocument::from_text (implemented; may pass).

use prometheus_extract::{sha256_hex, ExtractedDocument, Format};

#[test]
fn sha256_hex_matches_known_vectors() {
    assert_eq!(
        sha256_hex(b"hello"),
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    );
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn from_text_sets_fields_and_hashes_text_not_input_bytes() {
    let doc = ExtractedDocument::from_text(
        "hello".to_string(),
        Format::Html,
        Some("page.html".to_string()),
        99,
    );
    assert_eq!(doc.text, "hello");
    assert_eq!(doc.format, Format::Html);
    assert_eq!(doc.source.as_deref(), Some("page.html"));
    assert_eq!(doc.bytes_in, 99);
    assert_eq!(doc.content_hash, sha256_hex(b"hello"));
    assert_eq!(
        doc.content_hash,
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    );
}
