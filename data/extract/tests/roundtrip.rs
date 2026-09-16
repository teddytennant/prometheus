//! Group: extract then content_hash matches sha256_hex(text).

mod common;

use prometheus_extract::{extract, sha256_hex, Format};

#[test]
fn html_hash_is_sha256_of_extracted_text() {
    let doc = extract(common::PAGE_HTML, Format::Html).expect("html extract");
    assert_eq!(doc.content_hash, sha256_hex(doc.text.as_bytes()));
    common::assert_content_hash(&doc);
}

#[test]
fn pdf_hash_is_sha256_of_extracted_text() {
    let doc = extract(common::TINY_PDF, Format::Pdf).expect("pdf extract");
    assert_eq!(doc.content_hash, sha256_hex(doc.text.as_bytes()));
    common::assert_content_hash(&doc);
}

#[test]
fn latex_hash_is_sha256_of_extracted_text() {
    let doc = extract(common::MATH_TEX, Format::Latex).expect("latex extract");
    assert_eq!(doc.content_hash, sha256_hex(doc.text.as_bytes()));
    common::assert_content_hash(&doc);
}

#[test]
fn extract_is_deterministic() {
    let a = extract(common::PAGE_HTML, Format::Html).expect("first");
    let b = extract(common::PAGE_HTML, Format::Html).expect("second");
    assert_eq!(a.text, b.text);
    assert_eq!(a.content_hash, b.content_hash);
    assert_eq!(a.bytes_in, b.bytes_in);
}
